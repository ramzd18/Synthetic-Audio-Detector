import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import os
import random
from pathlib import Path
import json
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import PreTrainedModel, PretrainedConfig
from huggingface_hub import HfApi, Repository
import math
from scipy import signal
from hifigan_vocoder import hifigan, get_sample_mel
from waveglow_vocoder import WaveGlowVocoder
from vocos import Vocos
from speechbrain.inference.vocoders import HIFIGAN
from datasets import load_dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile as sf
import librosa
from scipy.io import wavfile
from huggingface_hub import login
import worldvocoder as wv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :].transpose(0, 1)

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, 
                              padding=dilation*(kernel_size-1)//2, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, 
                              padding=dilation*(kernel_size-1)//2, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out += residual
        return F.relu(out)

class MultiScaleCNN(nn.Module):
    def __init__(self, input_channels=1, output_channels=256):
        super(MultiScaleCNN, self).__init__()
        
        # Multiple branches with different kernel sizes
        branch_channels = output_channels // 4
        
        self.branch_3 = nn.Sequential(
            nn.Conv1d(input_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        
        self.branch_7 = nn.Sequential(
            nn.Conv1d(input_channels, branch_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        
        self.branch_15 = nn.Sequential(
            nn.Conv1d(input_channels, branch_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        
        self.branch_31 = nn.Sequential(
            nn.Conv1d(input_channels, branch_channels, kernel_size=31, padding=15),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        branch_3_out = self.branch_3(x)
        branch_7_out = self.branch_7(x)
        branch_15_out = self.branch_15(x)
        branch_31_out = self.branch_31(x)
        
        # Concatenate all branches
        out = torch.cat([branch_3_out, branch_7_out, branch_15_out, branch_31_out], dim=1)
        return out

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 1)
        )
        
    def forward(self, x):
        # x: (batch_size, seq_len, hidden_dim)
        attention_weights = self.attention(x)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        pooled = torch.sum(x * attention_weights, dim=1)  # (batch_size, hidden_dim)
        return pooled

class EnhancedRawNet2Config(PretrainedConfig):
    # model_type = "enhanced_rawnet2"
    
    def __init__(
        self,
        cnn_channels=256,
        num_residual_blocks=4,
        transformer_layers=6,
        transformer_heads=8,
        transformer_dim=256,
        transformer_ff_dim=1024,
        num_classes=2,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cnn_channels = cnn_channels
        self.num_residual_blocks = num_residual_blocks
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_dim = transformer_dim
        self.transformer_ff_dim = transformer_ff_dim
        self.num_classes = num_classes
        self.dropout = dropout

class EnhancedRawNet2(PreTrainedModel):
    config_class = EnhancedRawNet2Config
    
    def __init__(self, config):
        super().__init__(config)
        
        self.multi_scale_cnn = MultiScaleCNN(input_channels=1, output_channels=config.cnn_channels)
        
        # Additional CNN layers for feature extraction
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(config.cnn_channels, config.cnn_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(config.cnn_channels),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            
            nn.Conv1d(config.cnn_channels, config.cnn_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(config.cnn_channels),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(config.cnn_channels, dilation=2**i) 
            for i in range(config.num_residual_blocks)
        ])
        
        # Positional encoding for transformer
        self.positional_encoding = PositionalEncoding(config.transformer_dim)
        
        # Project CNN features to transformer dimension if needed
        if config.cnn_channels != config.transformer_dim:
            self.feature_projection = nn.Linear(config.cnn_channels, config.transformer_dim)
        else:
            self.feature_projection = nn.Identity()
        
        # Multi-layer transformer (replacing GRU)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.transformer_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-layer normalization
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers
        )
        
        # Attention-based pooling
        self.attention_pooling = AttentionPooling(config.transformer_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.transformer_dim, config.transformer_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.transformer_dim // 2, config.num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x, labels=None):
        # x: (batch_size, 1, sequence_length)
        
        # Multi-scale CNN feature extraction
        x = self.multi_scale_cnn(x)  # (batch_size, cnn_channels, seq_len)
        
        # Additional CNN processing
        x = self.cnn_layers(x)  # (batch_size, cnn_channels, reduced_seq_len)
        
        # Residual blocks
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        # Prepare for transformer: (batch_size, seq_len, hidden_dim)
        x = x.transpose(1, 2)  # (batch_size, seq_len, cnn_channels)
        
        # Project to transformer dimension
        x = self.feature_projection(x)  # (batch_size, seq_len, transformer_dim)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer processing
        x = self.transformer(x)  # (batch_size, seq_len, transformer_dim)
        
        # Attention-based pooling
        x = self.attention_pooling(x)  # (batch_size, transformer_dim)
        
        # Classification
        logits = self.classifier(x)  # (batch_size, num_classes)
        
        outputs = {"logits": logits}
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            outputs["loss"] = loss
        
        return outputs

class AudioDataset(Dataset):
    def __init__(self, data_dir, max_length=64000, sample_rate=16000):
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.sample_rate = sample_rate
        
        # Collect all audio files
        self.audio_files = []
        self.labels = []
        
        # Real audio files (label 0)
        real_dir = self.data_dir / "real"
        if real_dir.exists():
            for audio_file in real_dir.glob("*.wav"):
                self.audio_files.append(audio_file)
                self.labels.append(0)
        
        # Synthetic audio files (label 1)
        synthetic_dir = self.data_dir / "synthetic"
        if synthetic_dir.exists():
            for audio_file in synthetic_dir.glob("*.wav"):
                self.audio_files.append(audio_file)
                self.labels.append(1)
        
        logger.info(f"Found {len(self.audio_files)} audio files")
        logger.info(f"Real: {self.labels.count(0)}, Synthetic: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Pad or truncate to max_length
        if waveform.shape[1] > self.max_length:
            # Random crop for training, center crop for validation
            start = random.randint(0, waveform.shape[1] - self.max_length)
            waveform = waveform[:, start:start + self.max_length]
        elif waveform.shape[1] < self.max_length:
            # Pad with zeros
            padding = self.max_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        
        return {
            'input': waveform.squeeze(0),  # Remove channel dimension
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_synthetic_data(real_audio_dir, synthetic_audio_dir, dataset_name="common_voice", language="en", num_samples=1000):
    """
    Create synthetic data by loading real audio from HuggingFace datasets,
    processing half through neural vocoders, and organizing into directories.
    """
    logger.info(f"Creating synthetic data from {dataset_name} dataset")
    
    # Create directories
    os.makedirs(real_audio_dir, exist_ok=True)
    os.makedirs(synthetic_audio_dir, exist_ok=True)
    
    try:
        # Load dataset from HuggingFace
        logger.info(f"Loading {dataset_name} dataset...")
        if dataset_name == "common_voice":
            dataset = load_dataset("mozilla-foundation/common_voice_13_0", language, split="train", streaming=True)
            dataset = dataset.take(num_samples)
        elif dataset_name == "librispeech":
            dataset = load_dataset("librispeech_asr", "clean", split="train.100", streaming=True)
            dataset = dataset.take(num_samples)
        elif dataset_name == "vctk":
            dataset = load_dataset("vctk", split="train", streaming=True)
            dataset = dataset.take(num_samples)
        else:
            # Fallback to LibriSpeech
            dataset = load_dataset("librispeech_asr", "clean", split="train.100", streaming=True)
            dataset = dataset.take(num_samples)
        
        # Convert streaming dataset to list for easier manipulation
        dataset_list = list(dataset)
        logger.info(f"Loaded {len(dataset_list)} audio samples")
        
        # Split dataset in half
        split_point = len(dataset_list) // 2
        real_samples = dataset_list[:split_point]
        synthetic_samples = dataset_list[split_point:]
        
        # Save real audio samples
        logger.info("Saving real audio samples...")
        for i, sample in enumerate(tqdm(real_samples, desc="Processing real audio")):
            try:
                # Extract audio data
                if 'audio' in sample:
                    audio_data = sample['audio']
                    audio_array = audio_data['array']
                    sample_rate = audio_data['sampling_rate']
                else:
                    continue
                
                # Ensure audio is 16kHz mono
                if sample_rate != 16000:
                    audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                    sample_rate = 16000
                
                # Save as WAV file
                output_path = os.path.join(real_audio_dir, f"real_{i:05d}.wav")
                sf.write(output_path, audio_array, sample_rate)
                
            except Exception as e:
                logger.warning(f"Error processing real sample {i}: {e}")
                continue
        
        # Initialize vocoders for synthetic audio generation
        logger.info("Loading neural vocoders...")
        vocoders = load_vocoders()
        
        # Generate synthetic audio samples
        logger.info("Generating synthetic audio samples...")
        for i, sample in enumerate(tqdm(synthetic_samples, desc="Generating synthetic audio")):
            try:
                # Extract audio data
                if 'audio' in sample:
                    audio_data = sample['audio']
                    audio_array = audio_data['array']
                    sample_rate = audio_data['sampling_rate']
                else:
                    continue
                
                # Ensure audio is 16kHz mono
                if sample_rate != 16000:
                    audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                    sample_rate = 16000
                
                # Randomly select a vocoder
                vocoder_name = random.choice(list(vocoders.keys()))
                vocoder_models = vocoders[vocoder_name]
                
                # Generate synthetic audio using the selected vocoder
                synthetic_audio = generate_synthetic_audio(
                    audio_array, sample_rate, vocoder_models, vocoder_name
                )
                
                if synthetic_audio is not None:
                    output_path = os.path.join(synthetic_audio_dir, f"synthetic_{vocoder_name}_{i:05d}.wav")
                    sf.write(output_path, synthetic_audio, 16000)
                
            except Exception as e:
                logger.warning(f"Error processing synthetic sample {i}: {e}")
                continue
        
        logger.info(f"Data generation complete!")
        logger.info(f"Real samples: {len(os.listdir(real_audio_dir))}")
        logger.info(f"Synthetic samples: {len(os.listdir(synthetic_audio_dir))}")
        
    except Exception as e:
        logger.error(f"Error in data generation: {e}")
        raise

def load_vocoders():
    """Load five specific neural vocoders"""
    vocoders = {}
    
    try:
        # 1. HiFi-GAN (from hifigan_vocoder)
        logger.info("Loading HiFi-GAN...")
        hifigan_model = hifigan(dataset='uni', device='cpu')
        vocoders['hifigan'] = {
            'model': hifigan_model,
            'type': 'hifigan'
        }
    except Exception as e:
        logger.warning(f"Failed to load HiFi-GAN: {e}")
    
    try:
        # 2. Tacotron2 with Griffin-Lim
        logger.info("Loading Tacotron2 with Griffin-Lim...")
        bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
        processor = bundle.get_text_processor()
        tacotron2 = bundle.get_tacotron2()
        vocoder = bundle.get_vocoder()
        
        vocoders['tacotron2_griffinlim'] = {
            'model': vocoder,
            'processor': processor,
            'tacotron2': tacotron2,
            'type': 'tacotron2_griffinlim'
        }
    except Exception as e:
        logger.warning(f"Failed to load Tacotron2 with Griffin-Lim: {e}")
    
    try:
        # 3. WorldVocoder
        logger.info("Loading WorldVocoder...")
        vocoder = wv.World()
        vocoders['worldvocoder'] = {
            'model': vocoder,
            'type': 'worldvocoder'
        }
    except Exception as e:
        logger.warning(f"Failed to load WorldVocoder: {e}")
    
    try:
        # 4. Vocos
        logger.info("Loading Vocos...")
        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        vocoders['vocos'] = {
            'model': vocos,
            'type': 'vocos'
        }
    except Exception as e:
        logger.warning(f"Failed to load Vocos: {e}")
    
    try:
        # 5. SpeechBrain's HiFi-GAN
        logger.info("Loading SpeechBrain's HiFi-GAN...")
        hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech")
        vocoders['speechbrain_hifigan'] = {
            'model': hifi_gan,
            'type': 'speechbrain_hifigan'
        }
    except Exception as e:
        logger.warning(f"Failed to load SpeechBrain's HiFi-GAN: {e}")
    
    if not vocoders:
        logger.error("No vocoders could be loaded!")
        raise RuntimeError("Failed to load any vocoders")
    
    logger.info(f"Loaded {len(vocoders)} vocoders: {list(vocoders.keys())}")
    return vocoders

def generate_synthetic_audio(audio_array, sample_rate, vocoder_models, vocoder_name):
    """Generate synthetic audio using the specified vocoder"""
    
    try:
        if vocoder_name == 'hifigan':
            return generate_with_hifigan(audio_array, sample_rate, vocoder_models)
        elif vocoder_name == 'tacotron2_griffinlim':
            return generate_with_tacotron2_griffinlim(audio_array, sample_rate, vocoder_models)
        elif vocoder_name == 'worldvocoder':
            return generate_with_worldvocoder(audio_array, sample_rate, vocoder_models)
        elif vocoder_name == 'vocos':
            return generate_with_vocos(audio_array, sample_rate, vocoder_models)
        elif vocoder_name == 'speechbrain_hifigan':
            return generate_with_speechbrain_hifigan(audio_array, sample_rate, vocoder_models)
        else:
            logger.warning(f"Unknown vocoder type: {vocoder_name}")
            return add_vocoder_artifacts(audio_array, vocoder_name)
            
    except Exception as e:
        logger.warning(f"Error generating with {vocoder_name}: {e}")
        return add_vocoder_artifacts(audio_array, vocoder_name)

def audio_to_mel(audio_array, sample_rate, n_mels=80, n_fft=1024, hop_length=256):
    """
    Convert a numpy audio array to a mel-spectrogram tensor.
    """
    # Ensure audio is a torch tensor, shape (1, N)
    if not isinstance(audio_array, torch.Tensor):
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
    else:
        audio_tensor = audio_array.float()
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    # Normalize if needed
    if audio_tensor.abs().max() > 1.0:
        audio_tensor = audio_tensor / audio_tensor.abs().max()
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
        power=1.0,
        norm="slaney",
        mel_scale="htk"
    )
    mel = mel_transform(audio_tensor)
    # (1, n_mels, T)
    return mel

def generate_with_hifigan(audio_array, sample_rate, models):
    """Generate synthetic audio using HiFi-GAN"""
    try:
        model = models['model']
        if isinstance(audio_array, np.ndarray):
            audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
        else:
            audio_tensor = audio_array.float()
            
        mel = audio_to_mel(audio_tensor, sample_rate)  # (1, 80, T)
        with torch.no_grad():
            audio = model.infer(mel)
            if isinstance(audio, torch.Tensor):
                audio = audio.squeeze().detach().cpu().numpy()
            else:
                audio = np.array(audio).squeeze()
        return audio.astype(np.float32)
    except Exception as e:
        logger.warning(f"HiFi-GAN generation failed: {e}")
        return add_vocoder_artifacts(audio_array, 'hifigan')
def generate_with_tacotron2_griffinlim(audio_array, sample_rate, models):
    """Generate synthetic audio using Tacotron2 with Griffin-Lim"""
    try:
        vocoder = models['model']
        # print("TACOTRON2", dir(vocoder))

        if isinstance(audio_array, np.ndarray):
            audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
        else:
            audio_tensor = audio_array.float()
        mel_from_audio = audio_to_mel(audio_tensor, sample_rate)
        with torch.no_grad():
            # Use vocoder.generate() instead of infer()
            audio = vocoder.forward(mel_from_audio)
            if isinstance(audio, torch.Tensor):
                audio = audio.squeeze().detach().cpu().numpy()
            else:
                audio = np.array(audio).squeeze()
            
            # Handle inhomogeneous array shape by ensuring consistent dimensions
            if isinstance(audio, np.ndarray) and len(audio.shape) > 1:
                # Take first channel if multi-channel
                audio = audio[0] if audio.shape[0] <= 2 else audio.mean(axis=0)
            
        return audio.astype(np.float32)
    except Exception as e:
        logger.warning(f"Tacotron2 with Griffin-Lim generation failed: {e}")
        return add_vocoder_artifacts(audio_array, 'tacotron2_griffinlim')

def generate_with_worldvocoder(audio_array, sample_rate, models):
    """Generate synthetic audio using WorldVocoder"""
    try:
        vocoder = models['model']
        
        # Convert to numpy array if needed
        if isinstance(audio_array, torch.Tensor):
            audio_array = audio_array.detach().cpu().numpy()
        
        # Ensure sample_rate is a Python int, not np.int
        sample_rate = int(sample_rate)
        
        # Encode audio using World vocoder
        dat = vocoder.encode(sample_rate, audio_array, f0_method='harvest')        
        dat = vocoder.decode(dat)
        audio = dat["out"]
        
        return audio.astype(np.float32)
    except Exception as e:
        logger.warning(f"WorldVocoder generation failed: {e}")
        return add_vocoder_artifacts(audio_array, 'worldvocoder')

def generate_with_vocos(audio_array, sample_rate, models):
    """Generate synthetic audio using Vocos"""
    try:
        model = models['model']
        # Convert to tensor if needed
        if isinstance(audio_array, np.ndarray):
            audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
        else:
            audio_tensor = audio_array.float()
            
        mel = audio_to_mel(audio_tensor, sample_rate, n_mels=100)  # (1, 80, T)
        print("MEL SHAPE", mel.shape)
        device = next(model.parameters()).device
        mel = mel.to(device)
        
        with torch.no_grad():
            # Vocos expects mel in format (batch, n_mels, time)
            audio = model.decode(mel)
            if isinstance(audio, torch.Tensor):
                audio = audio.squeeze().detach().cpu().numpy()
            else:
                audio = np.array(audio).squeeze()
        
        # Ensure audio is in valid range
        audio = np.clip(audio, -1.0, 1.0)
        return audio.astype(np.float32)
    except Exception as e:
        logger.warning(f"Vocos generation failed: {e}")
        return add_vocoder_artifacts(audio_array, 'vocos')

def generate_with_speechbrain_hifigan(audio_array, sample_rate, models):
    """Generate synthetic audio using SpeechBrain's HiFi-GAN"""
    try:
        model = models['model']
        
        # Convert to tensor if needed
        if isinstance(audio_array, np.ndarray):
            audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
        else:
            audio_tensor = audio_array.float()
            
        mel = audio_to_mel(audio_tensor, sample_rate)  # (1, 80, T)
        
        with torch.no_grad():
            # SpeechBrain HiFi-GAN expects mel spectrograms
            waveforms = model.decode_batch(mel)
            if isinstance(waveforms, (list, tuple)):
                audio = waveforms[0].squeeze().detach().cpu().numpy()
            else:
                audio = waveforms.squeeze().detach().cpu().numpy()
        
        # Ensure audio is in valid range and format
        audio = np.clip(audio, -1.0, 1.0)
        audio = audio.astype(np.float32)
        
        return audio
    except Exception as e:
        logger.warning(f"SpeechBrain's HiFi-GAN generation failed: {e}")
        return add_vocoder_artifacts(audio_array, 'speechbrain_hifigan')

def add_vocoder_artifacts(audio_array, vocoder_type):
    """Add synthetic artifacts to audio to simulate vocoder processing"""
    try:
        # Ensure audio_array is numpy array
        if isinstance(audio_array, torch.Tensor):
            synthetic_audio = audio_array.detach().cpu().numpy()
        else:
            synthetic_audio = audio_array.copy()
            
        # Ensure it's 1D
        if synthetic_audio.ndim > 1:
            synthetic_audio = synthetic_audio.squeeze()
            
        # Add various artifacts that different vocoders typically introduce
        if vocoder_type == 'hifigan':
            # HiFi-GAN artifacts: slight high-frequency emphasis and phase distortion
            try:
                nyquist = 16000 / 2
                high_freq = min(7000, nyquist - 100)  # Ensure frequency is below Nyquist
                b, a = signal.butter(4, high_freq / nyquist, btype='high')
                high_emphasis = signal.filtfilt(b, a, synthetic_audio)
                synthetic_audio = high_emphasis * 0.15 + synthetic_audio * 0.85
                # Add slight aliasing artifacts
                synthetic_audio = np.clip(synthetic_audio * 1.05, -1.0, 1.0)
            except Exception as e:
                logger.warning(f"Error adding HiFi-GAN artifacts: {e}")
            
        elif vocoder_type == 'tacotron2_griffinlim':
            # Tacotron2 artifacts: attention artifacts and slight spectral distortion
            try:
                # Add attention-based artifacts (periodic distortion)
                attention_artifact = 0.02 * np.sin(2 * np.pi * 50 * np.arange(len(synthetic_audio)) / 16000)
                synthetic_audio += attention_artifact
                # Slight spectral tilt
                nyquist = 16000 / 2
                high_freq = min(5000, nyquist - 100)
                b, a = signal.butter(2, high_freq / nyquist, btype='high')
                high_freq_component = signal.filtfilt(b, a, synthetic_audio)
                synthetic_audio = synthetic_audio + high_freq_component * 0.1
            except Exception as e:
                logger.warning(f"Error adding Tacotron2 artifacts: {e}")
            
        elif vocoder_type == 'worldvocoder':
            # WorldVocoder artifacts: slight phase distortion and spectral smoothing
            try:
                nyquist = 16000 / 2
                low_freq = min(7500, nyquist - 100)
                b, a = signal.butter(4, low_freq / nyquist, btype='low')
                smoothed = signal.filtfilt(b, a, synthetic_audio)
                synthetic_audio = smoothed * 0.9 + synthetic_audio * 0.1
                # Add slight phase distortion
                synthetic_audio = synthetic_audio + 0.02 * np.sin(synthetic_audio * 10)
            except Exception as e:
                logger.warning(f"Error adding WorldVocoder artifacts: {e}")
            
        elif vocoder_type == 'vocos':
            # Vocos artifacts: slight temporal jitter and spectral smoothing
            try:
                # Apply temporal jitter
                jitter = np.random.uniform(-0.05, 0.05, len(synthetic_audio))
                synthetic_audio = synthetic_audio * (1 + jitter)
                # Apply spectral smoothing
                nyquist = 16000 / 2
                low_freq = min(6000, nyquist - 100)
                b, a = signal.butter(3, low_freq / nyquist, btype='low')
                synthetic_audio = signal.filtfilt(b, a, synthetic_audio)
            except Exception as e:
                logger.warning(f"Error adding Vocos artifacts: {e}")
            
        elif vocoder_type == 'speechbrain_hifigan':
            # SpeechBrain HiFi-GAN artifacts: similar to HiFi-GAN but with more emphasis
            try:
                nyquist = 16000 / 2
                high_freq = min(7000, nyquist - 100)
                b, a = signal.butter(4, high_freq / nyquist, btype='high')
                high_emphasis = signal.filtfilt(b, a, synthetic_audio)
                synthetic_audio = high_emphasis * 0.2 + synthetic_audio * 0.8
                # Add more pronounced aliasing artifacts
                synthetic_audio = np.clip(synthetic_audio * 1.1, -1.0, 1.0)
            except Exception as e:
                logger.warning(f"Error adding SpeechBrain HiFi-GAN artifacts: {e}")
        
        # Ensure audio is in valid range
        synthetic_audio = np.clip(synthetic_audio, -1.0, 1.0)
        
        return synthetic_audio.astype(np.float32)
        
    except Exception as e:
        logger.warning(f"Artifact addition failed for {vocoder_type}: {e}")
        # Return original audio as fallback
        if isinstance(audio_array, torch.Tensor):
            return audio_array.detach().cpu().numpy().astype(np.float32)
        else:
            return audio_array.astype(np.float32)

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Optimizer with different learning rates for different components
    optimizer = optim.AdamW([
        {'params': model.multi_scale_cnn.parameters(), 'lr': learning_rate},
        {'params': model.cnn_layers.parameters(), 'lr': learning_rate},
        {'params': model.residual_blocks.parameters(), 'lr': learning_rate},
        {'params': model.transformer.parameters(), 'lr': learning_rate * 0.5},  # Lower LR for transformer
        {'params': model.classifier.parameters(), 'lr': learning_rate * 2}  # Higher LR for classifier
    ], weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in pbar:
            inputs = batch['input'].unsqueeze(1).to(device)  # Add channel dimension
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs, labels=labels)
            loss = outputs['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs['logits'], 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch in pbar:
                inputs = batch['input'].unsqueeze(1).to(device)
                labels = batch['label'].to(device)
                
                outputs = model(inputs, labels=labels)
                loss = outputs['loss']
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs['logits'], 1)
                
                val_predictions.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_acc = accuracy_score(val_labels, val_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_predictions, average='binary')
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {100.*train_correct/train_total:.2f}%')
        logger.info(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {100.*val_acc:.2f}%')
        logger.info(f'  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': model.config
            }, 'best_model.pth')
            logger.info(f'New best model saved with validation accuracy: {100.*val_acc:.2f}%')
    
    return model

def save_to_huggingface(model, tokenizer=None, repo_name="enhanced-rawnet2-antispoofing"):
    """Save model to HuggingFace Hub"""
    
    # Save model locally first
    model.save_pretrained("./enhanced_rawnet2_model")
    
    # Upload to HuggingFace Hub
    api = HfApi()
    
    try:
        # Create repository
        api.create_repo(repo_name, exist_ok=True, private=False)
        
        # Upload model files
        api.upload_folder(
            folder_path="./enhanced_rawnet2_model",
            repo_id=repo_name,
            repo_type="model"
        )
        
        logger.info(f"Model successfully uploaded to: https://huggingface.co/{repo_name}")
        
    except Exception as e:
        logger.error(f"Error uploading to HuggingFace: {e}")
        logger.info("Model saved locally in './enhanced_rawnet2_model'")

def main():
    # Configuration
    config = EnhancedRawNet2Config(
        cnn_channels=256,
        num_residual_blocks=4,
        transformer_layers=6,
        transformer_heads=8,
        transformer_dim=256,
        transformer_ff_dim=1024,
        num_classes=2,
        dropout=0.1
    )
    print("CREATING CONFIG")
    
    # Data directories
    data_dir = "./audio_data"
    real_audio_dir = os.path.join(data_dir, "real")
    synthetic_audio_dir = os.path.join(data_dir, "synthetic")
    
    # Create directories
    os.makedirs(real_audio_dir, exist_ok=True)
    os.makedirs(synthetic_audio_dir, exist_ok=True)
    
    # Generate synthetic data using multiple vocoders
    print("CREATING SYNTHETIC DATA")
    login(token="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")

    create_synthetic_data(
        real_audio_dir, 
        synthetic_audio_dir, 
        dataset_name="common_voice",  # Options: "common_voice", "librispeech", "vctk"
        language="en", 
        num_samples=2000  
    )
    print("SYNTHETIC DATA CREATED")
    
    # Create datasets
    dataset = AudioDataset(data_dir, max_length=10000)
    print("DATASET CREATED")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Initialize model
    model = EnhancedRawNet2(config)
    print("MODEL INITIALIZED")
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    print("TRAINING MODEL")
    # Train model
    trained_model = train_model(model, train_loader, val_loader, num_epochs=50)
    print("MODEL TRAINED")
    
    # Save to HuggingFace
    save_to_huggingface(trained_model, repo_name="your-username/enhanced-rawnet2-antispoofing")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()