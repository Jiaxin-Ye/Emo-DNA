from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def extract_mfcc(wav_path: str | Path, sample_rate: int | None = None) -> np.ndarray:
    """Extract the 40-D, 508-frame MFCC convention used by this release.

    ``sample_rate=None`` preserves the WAV's native rate. The historical
    IEMOCAP reference features were generated at 48 kHz, while the retained
    WAV copy is 16 kHz, so the release config explicitly requests 48 kHz for
    IEMOCAP.
    """
    signal, sample_rate = librosa.load(str(wav_path), sr=sample_rate, mono=True)
    target_length = 260_000
    if len(signal) < target_length:
        pad = target_length - len(signal)
        signal = np.pad(signal, (pad // 2, pad // 2 + pad % 2))
    else:
        start = (len(signal) - target_length) // 2
        signal = signal[start : start + target_length]
    return librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40).T.astype(np.float32)
