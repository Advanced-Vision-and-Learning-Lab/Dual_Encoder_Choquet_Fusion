# src/audio_augmentation.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import random

class ShipAudioAugmentation(nn.Module):

    def __init__(
        self,
        noise_prob: float = 0.5,
        noise_snr_db_range: tuple = (15.0, 25.0),
        time_mask_prob: float = 0.5,
        time_mask_max_ratio: float = 0.15,
        freq_mask_prob: float = 0.5,
        freq_mask_max_ratio: float = 0.10,
        training: bool = True,
    ):
        super().__init__()
        self.noise_prob = noise_prob
        self.noise_snr_db_min, self.noise_snr_db_max = noise_snr_db_range
        self.time_mask_prob = time_mask_prob
        self.time_mask_max_ratio = time_mask_max_ratio
        self.freq_mask_prob = freq_mask_prob
        self.freq_mask_max_ratio = freq_mask_max_ratio
        self.training = training
        
        # Validate parameters
        assert 0.0 <= noise_prob <= 1.0
        assert 0.0 <= time_mask_prob <= 1.0
        assert 0.0 <= freq_mask_prob <= 1.0
        assert 0.0 < time_mask_max_ratio < 1.0
        assert 0.0 < freq_mask_max_ratio < 1.0
        assert self.noise_snr_db_min <= self.noise_snr_db_max
    
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return spectrogram
        
        # Handle both batched and single inputs
        input_ndim = spectrogram.ndim
        if input_ndim == 2:
            # [freq, time] -> [1, freq, time]
            spectrogram = spectrogram.unsqueeze(0)
            unbatch = True
        else:
            unbatch = False
        
        # Apply augmentations independently per sample in batch
        batch_size = spectrogram.size(0)
        augmented = spectrogram.clone()
        
        for i in range(batch_size):
            sample = augmented[i]  # [freq_bins, time_frames]
            
            # Background noise injection
            if torch.rand(1).item() < self.noise_prob:
                sample = self._add_background_noise(sample)
            
            # Time masking
            if torch.rand(1).item() < self.time_mask_prob:
                sample = self._apply_time_mask(sample)
            
            # Frequency masking
            if torch.rand(1).item() < self.freq_mask_prob:
                sample = self._apply_freq_mask(sample)
            
            augmented[i] = sample
        
        if unbatch:
            augmented = augmented.squeeze(0)
        
        return augmented
    
    def _add_background_noise(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise with random SNR in dB.
        
        SNR formula: SNR_dB = 10 * log10(P_signal / P_noise)
        => P_noise = P_signal / (10^(SNR_dB / 10))
        
        Args:
            spec: [freq_bins, time_frames]
        
        Returns:
            Noisy spectrogram
        """
        # Random SNR in dB
        snr_db = torch.rand(1).item() * (self.noise_snr_db_max - self.noise_snr_db_min) + self.noise_snr_db_min
        
        # Compute signal power (variance)
        signal_power = torch.var(spec)
        
        # Compute noise power from SNR
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        
        # Generate Gaussian noise
        noise = torch.randn_like(spec) * torch.sqrt(noise_power)
        
        # Add noise to signal
        return spec + noise
    
    def _apply_time_mask(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply time masking (SpecAugment) - mask contiguous time frames.
        
        Args:
            spec: [freq_bins, time_frames]
        
        Returns:
            Time-masked spectrogram
        """
        freq_bins, time_frames = spec.shape
        
        # Random mask width (0 to time_mask_max_ratio of total time)
        max_mask_width = int(time_frames * self.time_mask_max_ratio)
        if max_mask_width == 0:
            return spec
        
        mask_width = torch.randint(1, max_mask_width + 1, (1,)).item()
        
        # Random start position
        mask_start = torch.randint(0, time_frames - mask_width + 1, (1,)).item()
        
        # Apply mask (set to mean value to avoid harsh zeros)
        spec_mean = spec.mean()
        spec_masked = spec.clone()
        spec_masked[:, mask_start:mask_start + mask_width] = spec_mean
        
        return spec_masked
    
    def _apply_freq_mask(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency masking (SpecAugment) - mask contiguous frequency bins.
        
        Args:
            spec: [freq_bins, time_frames]
        
        Returns:
            Frequency-masked spectrogram
        """
        freq_bins, time_frames = spec.shape
        
        # Random mask height (0 to freq_mask_max_ratio of total freq bins)
        max_mask_height = int(freq_bins * self.freq_mask_max_ratio)
        if max_mask_height == 0:
            return spec
        
        mask_height = torch.randint(1, max_mask_height + 1, (1,)).item()
        
        # Random start position
        mask_start = torch.randint(0, freq_bins - mask_height + 1, (1,)).item()
        
        # Apply mask (set to mean value to avoid harsh zeros)
        spec_mean = spec.mean()
        spec_masked = spec.clone()
        spec_masked[mask_start:mask_start + mask_height, :] = spec_mean
        
        return spec_masked
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self.training = mode
        return super().train(mode)
    
    def eval(self):
        """Set evaluation mode (disables augmentations)."""
        self.training = False
        return super().eval()


class WaveformGaussianNoise(nn.Module):
    """Add Gaussian noise to raw waveforms."""
    def __init__(self, snr_db_range: Tuple[float, float] = (15.0, 30.0), p: float = 0.5):
        super().__init__()
        self.snr_db_range = snr_db_range
        self.p = p

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return waveform
        if waveform.dim() == 1:
            # Single sample — same as before
            if random.random() > self.p:
                return waveform
            snr_db = random.uniform(*self.snr_db_range)
            signal_power = waveform.pow(2).mean()
            noise_std = (signal_power / (10 ** (snr_db / 10))).sqrt()
            return waveform + torch.randn_like(waveform) * noise_std
    
        # Batched: [B, T] — per-sample decision
        output = waveform.clone()
        for i in range(waveform.size(0)):
            if random.random() > self.p:
                continue  # this sample skipped
            snr_db = random.uniform(*self.snr_db_range)
            signal_power = waveform[i].pow(2).mean()
            noise_std = (signal_power / (10 ** (snr_db / 10))).sqrt()
            output[i] = waveform[i] + torch.randn_like(waveform[i]) * noise_std
        return output


def get_waveform_augmentation(
    enabled: bool = True,
    noise_prob: float = 0.5,
    noise_snr_db_range: Tuple[float, float] = (15.0, 30.0),
) -> Optional[nn.Module]:
    """Get waveform augmentation for wave encoders."""
    if not enabled:
        return None
    return WaveformGaussianNoise(snr_db_range=noise_snr_db_range, p=noise_prob)

def get_ship_augmentation(
    enabled: bool = True,
    noise_prob: float = 0.5,
    noise_snr_db_range: tuple = (15.0, 25.0),
    time_mask_prob: float = 0.5,
    time_mask_max_ratio: float = 0.15,
    freq_mask_prob: float = 0.5,
    freq_mask_max_ratio: float = 0.10,
) -> Optional[ShipAudioAugmentation]:
    """
    Factory function to create augmentation pipeline.
    
    Args:
        enabled: If False, returns None (no augmentation)
        Other args: See ShipAudioAugmentation
    
    Returns:
        ShipAudioAugmentation instance or None
    """
    if not enabled:
        return None
    
    return ShipAudioAugmentation(
        noise_prob=noise_prob,
        noise_snr_db_range=noise_snr_db_range,
        time_mask_prob=time_mask_prob,
        time_mask_max_ratio=time_mask_max_ratio,
        freq_mask_prob=freq_mask_prob,
        freq_mask_max_ratio=freq_mask_max_ratio,
        training=True,
    )
