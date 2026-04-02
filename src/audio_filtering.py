# src/audio_filtering.py
"""
Multi-band audio filtering and feature extraction.
Uses torchaudio for batched mel spectrogram computation with per-band configuration.
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torchaudio
from scipy.signal import butter, filtfilt
import warnings
from nnAudio.features.cqt import CQT
from nnAudio.features.vqt import VQT

# Normalization constants (from HuggingFace ASTFeatureExtractor)
AST_MEAN = -4.2677393
AST_STD = 4.5689974

def normalize_features(
    features: torch.Tensor,
    norm_type: Optional[str] = "ast",
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Applies unified normalization across all feature types.
    Assumes input shape is [batch, n_features, frames].
    """
    if norm_type is None:
        return features

    norm_type = norm_type.lower()
    
    # 1. Determine mean and std
    if norm_type == "ast":
        mean = AST_MEAN
        std = AST_STD * 2.0
    elif norm_type == "sample":
        mean = features.mean(dim=-1, keepdim=True)
        std = features.std(dim=-1, keepdim=True).clamp_min(eps)
    else:
        raise ValueError(f"Unknown norm_type '{norm_type}'. Expected 'ast', 'sample', or None.")
        
    # 2. Apply unified logic
    return (features - mean) / std

def create_fullband_mel_transform(
    sr: int,
    n_fft: int = 4096,
    hop_length: int = 512,
    n_mels: int = 64,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> torchaudio.transforms.MelSpectrogram:
    if fmax is None:
        fmax = sr / 2.0

    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax,
        power=2.0,
    )


def create_fullband_spectrogram_batched(
    waveforms: List[np.ndarray],
    mel_transform: torchaudio.transforms.MelSpectrogram,
    norm_type: Optional[str] = "ast",
) -> torch.Tensor:
    # Stack waveforms into batch
    wav_tensors = [torch.from_numpy(w.copy()).float() for w in waveforms]
    wav_batch = torch.stack(wav_tensors, dim=0)  # [batch, time]

    # Apply mel transform [batch, time] -> [batch, n_mels, frames]
    mel_spec = mel_transform(wav_batch)

    # Convert to dB
    mel_spec = torchaudio.functional.amplitude_to_DB(
        mel_spec, multiplier=10.0, amin=1e-10, db_multiplier=0.0
    )

    # Apply unified normalization
    mel_spec = normalize_features(mel_spec, norm_type=norm_type)

    # Transpose to [batch, frames, n_mels] - AST expects this shape
    mel_spec = mel_spec.transpose(1, 2)

    return mel_spec


def create_fullband_mfcc_transform(
    sr: int,
    n_fft: int = 4096,
    hop_length: int = 512,
    n_mels: int = 64,
    n_mfcc: int = 64,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> torchaudio.transforms.MFCC:
    """
    Create a torchaudio MFCC transform.
    Call once during DataModule init, reuse in collate_fn.
    """
    if fmax is None:
        fmax = sr / 2.0

    return torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": n_mels,
            "f_min": fmin,
            "f_max": fmax,
            "power": 2.0,
        },
    )


def create_fullband_mfcc_batched(
    waveforms: List[np.ndarray],
    mfcc_transform: torchaudio.transforms.MFCC,
    norm_type: Optional[str] = "sample", 
) -> torch.Tensor:
    """
    Batched MFCC extraction:
      - output shape: [batch, frames, n_mfcc] (AST layout)
    """
    # Stack waveforms into batch
    wav_tensors = [torch.from_numpy(w.copy()).float() for w in waveforms]
    wav_batch = torch.stack(wav_tensors, dim=0)  # [batch, time]

    # MFCC: [batch, n_mfcc, frames]
    mfcc = mfcc_transform(wav_batch)

    # Apply unified normalization
    mfcc = normalize_features(mfcc, norm_type=norm_type)

    # Transpose to [batch, frames, n_mfcc] - AST expects this shape
    mfcc = mfcc.transpose(1, 2)
    return mfcc

def create_fullband_cqt_transform(
    sr: int, 
    hop_length: int = 512, 
    n_bins: int = 84,
    fmin: float = 32.70 # Standard C1 frequency, adjust if needed
) -> torch.nn.Module:
    return CQT(sr=sr, hop_length=hop_length, n_bins=n_bins, fmin=fmin, trainable=False)

def create_fullband_vqt_transform(
    sr: int, 
    hop_length: int = 512, 
    n_bins: int = 84,
    fmin: float = 32.70
) -> torch.nn.Module:
    return VQT(sr=sr, hop_length=hop_length, n_bins=n_bins, fmin=fmin, trainable=False)

def create_fullband_cqt_batched(
    waveforms: List[np.ndarray],
    cqt_transform: torch.nn.Module,
    norm_type: Optional[str] = "ast", 
) -> torch.Tensor:
    # Stack waveforms into batch
    wav_tensors = [torch.from_numpy(w.copy()).float() for w in waveforms]
    wav_batch = torch.stack(wav_tensors, dim=0)  # [batch, time]

    # Apply nnAudio CQT transform
    cqt_spec = cqt_transform(wav_batch)

    # Convert to dB (Torchaudio uses multiplier=20 for magnitude)
    cqt_spec = torchaudio.functional.amplitude_to_DB(
        cqt_spec, multiplier=20.0, amin=1e-10, db_multiplier=0.0
    )

    # Apply unified normalization (testing AST stats per your request)
    cqt_spec = normalize_features(cqt_spec, norm_type=norm_type)

    # Transpose to [batch, frames, n_bins]
    return cqt_spec.transpose(1, 2)

def create_fullband_vqt_batched(
    waveforms: List[np.ndarray],
    vqt_transform: torch.nn.Module,
    norm_type: Optional[str] = "ast", 
) -> torch.Tensor:
    # Stack waveforms into batch
    wav_tensors = [torch.from_numpy(w.copy()).float() for w in waveforms]
    wav_batch = torch.stack(wav_tensors, dim=0)  # [batch, time]

    # Apply nnAudio VQT transform
    vqt_spec = vqt_transform(wav_batch)

    # Convert to dB
    vqt_spec = torchaudio.functional.amplitude_to_DB(
        vqt_spec, multiplier=20.0, amin=1e-10, db_multiplier=0.0
    )

    # Apply unified normalization
    vqt_spec = normalize_features(vqt_spec, norm_type=norm_type)

    # Transpose to [batch, frames, n_bins]
    return vqt_spec.transpose(1, 2)
