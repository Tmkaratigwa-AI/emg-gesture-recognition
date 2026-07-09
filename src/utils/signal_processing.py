import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, filtfilt
from typing import Dict, Any


def apply_emg_filter(
    emg_data: np.ndarray,
    sampling_frequency: int,
    bandpass_lowcut: float = 20.0,
    bandpass_highcut: float = 450.0,
    bandpass_order: int = 4,
    notch_frequency: float = 50.0,
    notch_quality_factor: float = 30.0,
    lowpass_cutoff: float = 10.0,
    lowpass_order: int = 2
) -> np.ndarray:
    """Apply bandpass, notch, rectification, and low-pass filter to EMG."""
    if emg_data.size == 0 or sampling_frequency <= 0:
        raise ValueError("Invalid input.")

    nyquist = 0.5 * sampling_frequency
    processed = np.copy(emg_data)

    # Bandpass
    if bandpass_lowcut is not None and bandpass_highcut is not None:
        low = bandpass_lowcut / nyquist
        high = bandpass_highcut / nyquist
        if low >= high or low >= 1 or high >= 1:
            print("Warning: Bandpass parameters invalid.")
        else:
            b, a = butter(bandpass_order, [low, high], btype='band')
            processed = filtfilt(b, a, processed, axis=0)

    # Notch
    if notch_frequency is not None and notch_frequency <= nyquist:
        b_notch, a_notch = iirnotch(notch_frequency, notch_quality_factor, sampling_frequency)
        processed = filtfilt(b_notch, a_notch, processed, axis=0)

    # Rectification
    processed = np.abs(processed)

    # Low-pass envelope
    if lowpass_cutoff is not None and lowpass_cutoff <= nyquist:
        cutoff_norm = lowpass_cutoff / nyquist
        b_lp, a_lp = butter(lowpass_order, cutoff_norm, btype='low')
        processed = filtfilt(b_lp, a_lp, processed, axis=0)

    return processed


def extract_features(
    processed_emg_data: np.ndarray,
    sampling_frequency: int,
    window_size_ms: int = 200,
    overlap_ms: int = 50,
    zc_threshold_volts: float = 0.01
) -> pd.DataFrame:
    """Extract time-domain features (MAV, RMS, ZC, WL, VAR, STD) per channel."""
    if processed_emg_data.size == 0 or sampling_frequency <= 0:
        raise ValueError("Invalid input.")
    if window_size_ms <= 0 or overlap_ms < 0 or overlap_ms >= window_size_ms:
        raise ValueError("Invalid window/overlap.")

    n_samples, n_channels = processed_emg_data.shape[0], (
        processed_emg_data.shape[1] if processed_emg_data.ndim > 1 else 1
    )
    window_samples = int(sampling_frequency * window_size_ms / 1000)
    overlap_samples = int(sampling_frequency * overlap_ms / 1000)
    step = window_samples - overlap_samples

    if window_samples == 0 or step <= 0:
        raise ValueError("Window size or step invalid.")

    data_2d = processed_emg_data.reshape(-1, n_channels)
    features_list = []

    for i in range(0, n_samples - window_samples + 1, step):
        window = data_2d[i: i + window_samples, :]
        feat = {}
        for ch in range(n_channels):
            sig = window[:, ch]
            feat[f'MAV_channel_{ch}'] = np.mean(np.abs(sig))
            feat[f'RMS_channel_{ch}'] = np.sqrt(np.mean(sig**2))
            zc = np.sum((sig[:-1] * sig[1:] < 0) & (np.abs(sig[:-1] - sig[1:]) > zc_threshold_volts))
            feat[f'ZC_channel_{ch}'] = zc
            feat[f'WL_channel_{ch}'] = np.sum(np.abs(np.diff(sig)))
            feat[f'VAR_channel_{ch}'] = np.var(sig)
            feat[f'STD_channel_{ch}'] = np.std(sig)
        features_list.append(feat)

    return pd.DataFrame(features_list)
