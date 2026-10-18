from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import extract_mfcc


AROUSAL_LOW = {"boredom", "calm", "neutral", "sad"}
AROUSAL_HIGH = {"angry", "disgust", "fear", "happy", "surprise"}
VALENCE_NEGATIVE = {"angry", "boredom", "disgust", "fear", "sad"}
VALENCE_POSITIVE = {"calm", "happy", "neutral", "surprise"}

CORPORA = {
    "CASIA": ("angry", "fear", "happy", "neutral", "sad", "surprise"),
    "EMODB": ("angry", "boredom", "disgust", "fear", "happy", "neutral", "sad"),
    "EMOVO": ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"),
    "IEMOCAP": ("angry", "happy", "neutral", "sad"),
    "RAVDE": ("angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"),
    "SAVEE": ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"),
}


def emotion_to_label(emotion: str, task: str) -> int:
    if task == "arousal":
        groups = (AROUSAL_LOW, AROUSAL_HIGH)
    elif task == "valence":
        groups = (VALENCE_NEGATIVE, VALENCE_POSITIVE)
    else:
        raise ValueError(f"Unsupported task: {task!r}")
    for label, emotions in enumerate(groups):
        if emotion in emotions:
            return label
    raise ValueError(f"Emotion {emotion!r} is not mapped for {task}")


class NpyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load a WAV-derived ``features.npy``/``labels.npy`` corpus bundle."""

    def __init__(self, root: str | Path, corpus: str, task: str):
        root = Path(root)
        features_path = root / "features.npy"
        labels_path = root / "labels.npy"
        if not features_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError(f"Expected features.npy and labels.npy under {root}")
        features = np.load(features_path, mmap_mode="r")
        emotion_ids = np.load(labels_path)
        if features.ndim != 3 or features.shape[2] != 40:
            raise ValueError(f"Expected [N, time, 40] MFCC features, got {features.shape}")
        if emotion_ids.ndim != 1 or len(features) != len(emotion_ids):
            raise ValueError(f"Feature/label length mismatch under {root}")
        emotions = CORPORA[corpus]
        if emotion_ids.size and (emotion_ids.min() < 0 or emotion_ids.max() >= len(emotions)):
            raise ValueError(f"Invalid emotion ids under {root}")
        self.features = features
        self.labels = np.asarray(
            [emotion_to_label(emotions[int(index)], task) for index in emotion_ids],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(np.array(self.features[index], dtype=np.float32, copy=True)),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


class AudioDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Extract paper-aligned MFCCs from WAV once, then train from RAM tensors."""

    def __init__(
        self,
        root: str | Path,
        emotions: Iterable[str],
        task: str,
        sample_rate: int | None = None,
    ):
        self.root = Path(root)
        self.samples: list[tuple[Path, int]] = []
        for emotion in emotions:
            emotion_dir = self.root / emotion
            if not emotion_dir.is_dir():
                raise FileNotFoundError(f"Missing emotion directory: {emotion_dir}")
            label = emotion_to_label(emotion, task)
            self.samples.extend((path, label) for path in sorted(emotion_dir.glob("*.wav")))
        if not self.samples:
            raise RuntimeError(f"No WAV files found under {self.root}")
        # Extraction remains audio-based and happens before training. Keeping
        # one deterministic tensor per WAV avoids decoding/recomputing MFCCs
        # independently in every DataLoader worker and epoch.
        self.features = [
            torch.from_numpy(extract_mfcc(path, sample_rate=sample_rate))
            for path, _ in self.samples
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        _, label = self.samples[index]
        return self.features[index], torch.tensor(label, dtype=torch.long)


def pad_mfcc_batch(
    samples: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(feature.shape[0] for feature, _ in samples)
    batch = torch.zeros(len(samples), max_length, 40, dtype=torch.float32)
    labels = torch.empty(len(samples), dtype=torch.long)
    for index, (feature, label) in enumerate(samples):
        batch[index, : feature.shape[0]] = feature
        labels[index] = label
    return batch, labels
