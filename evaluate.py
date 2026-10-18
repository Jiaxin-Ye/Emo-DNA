"""Evaluate all release checkpoints from packaged NPY features or WAV audio."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score

from emodna.features import extract_mfcc
from emodna.legacy_model import LegacyInferenceModel


CORPORA = {
    "CASIA": ("angry", "fear", "happy", "neutral", "sad", "surprise"),
    "EMODB": ("angry", "boredom", "disgust", "fear", "happy", "neutral", "sad"),
    "EMOVO": ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"),
    "IEMOCAP": ("angry", "happy", "neutral", "sad"),
    "RAVDE": ("angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"),
    "SAVEE": ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"),
}
AROUSAL_LOW = {"boredom", "calm", "neutral", "sad"}
VALENCE_NEGATIVE = {"angry", "boredom", "disgust", "fear", "sad"}
SAMPLE_RATES = {"IEMOCAP": 48000}


def checkpoint_groups(root: Path, task: str) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in sorted((root / task).glob("*.pt")):
        match = re.search(r"-S_(.+)-T_([A-Za-z0-9]+)\.pt$", path.name)
        if match is None:
            raise ValueError(f"Cannot parse checkpoint name: {path.name}")
        groups.setdefault(match.group(2), []).append(path)
    return groups


def load_npy(root: Path, corpus: str, negative: set[str]) -> tuple[np.ndarray, np.ndarray]:
    feature_dir = root / corpus
    features = np.load(feature_dir / "features.npy")
    emotion_ids = np.load(feature_dir / "labels.npy")
    emotion_order = CORPORA[corpus]
    if emotion_ids.ndim != 1 or emotion_ids.max(initial=-1) >= len(emotion_order):
        raise ValueError(f"Invalid emotion labels for {corpus}")
    labels = np.asarray(
        [int(emotion_order[int(index)] not in negative) for index in emotion_ids],
        dtype=np.int64,
    )
    if len(features) != len(labels):
        raise ValueError(f"Feature/label length mismatch for {corpus}")
    return features.astype(np.float32, copy=False), labels


def load_audio(root: Path, corpus: str, negative: set[str]) -> tuple[np.ndarray, np.ndarray]:
    files: list[tuple[Path, str]] = []
    for emotion in CORPORA[corpus]:
        files.extend((path, emotion) for path in sorted((root / corpus / emotion).glob("*.wav")))
    if not files:
        raise FileNotFoundError(f"No WAV files found for {corpus} under {root}")
    features = np.stack([
        extract_mfcc(path, sample_rate=SAMPLE_RATES.get(corpus)) for path, _ in files
    ]).astype(np.float32, copy=False)
    labels = np.asarray([int(emotion not in negative) for _, emotion in files], dtype=np.int64)
    return features, labels


def evaluate(
    model: LegacyInferenceModel,
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    predictions: list[int] = []
    tensor = torch.from_numpy(features)
    with torch.no_grad():
        for batch in tensor.split(batch_size):
            predictions.extend(model.predict(batch.to(device)).argmax(-1).cpu().tolist())
    return {
        "war": float(accuracy_score(labels, predictions)),
        "uar": float(recall_score(labels, predictions, average="macro")),
        "f1": float(f1_score(labels, predictions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate release checkpoints from NPY features or WAV audio."
    )
    parser.add_argument("--input-format", choices=("npy", "audio"), required=True)
    parser.add_argument("--features-root", type=Path, default=Path("reference_features"))
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("evaluation_results.csv"))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.input_format == "audio" and args.audio_root is None:
        parser.error("--audio-root is required when --input-format audio")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []
    for task, negative in (("arousal", AROUSAL_LOW), ("valence", VALENCE_NEGATIVE)):
        for target, checkpoints in sorted(checkpoint_groups(args.checkpoint_root, task).items()):
            if args.input_format == "npy":
                features, labels = load_npy(args.features_root, target, negative)
            else:
                features, labels = load_audio(args.audio_root, target, negative)
            for checkpoint_path in checkpoints:
                state = torch.load(checkpoint_path, map_location=device, weights_only=True)
                model = LegacyInferenceModel(state, device)
                result = evaluate(model, features, labels, device, args.batch_size)
                rows.append({
                    "input_format": args.input_format,
                    "task": task,
                    "source": re.search(r"-S_(.+)-T_", checkpoint_path.name).group(1),
                    "target": target,
                    "samples": len(labels),
                    **result,
                    "checkpoint": checkpoint_path.name,
                })
                print(task, checkpoint_path.name, result, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
