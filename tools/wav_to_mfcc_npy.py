"""Extract corpus WAV files directly into an MFCC NPY reference bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from emodna.features import extract_mfcc


CORPORA = {
    "CASIA": ("angry", "fear", "happy", "neutral", "sad", "surprise"),
    "EMODB": ("angry", "boredom", "disgust", "fear", "happy", "neutral", "sad"),
    "EMOVO": ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"),
    "IEMOCAP": ("angry", "happy", "neutral", "sad"),
    "RAVDE": ("angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"),
    "SAVEE": ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"),
}

# The retained IEMOCAP WAV copy is 16 kHz, but historical MFCC_260 input was
# generated from 48 kHz audio. Other corpora use their native WAV rate.
CORPUS_SAMPLE_RATES: dict[str, int | None] = {"IEMOCAP": 48000}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--corpus", choices=sorted(CORPORA), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=None,
                        help="Override the corpus policy; omit to use the historical policy.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Extract only the first N files per smoke testing.")
    args = parser.parse_args()

    sample_rate = args.sample_rate if args.sample_rate is not None else CORPUS_SAMPLE_RATES.get(args.corpus)
    files: list[tuple[Path, str, int]] = []
    for label, emotion in enumerate(CORPORA[args.corpus]):
        for wav_path in sorted((args.audio_root / args.corpus / emotion).glob("*.wav")):
            files.append((wav_path, emotion, label))
    if args.limit is not None:
        files = files[:args.limit]
    if not files:
        raise FileNotFoundError(f"No WAV files found for {args.corpus}")

    features = np.stack([
        extract_mfcc(path, sample_rate=sample_rate)
        for path, _, _ in files
    ]).astype(np.float32, copy=False)
    if features.shape[1:] != (508, 40):
        raise ValueError(f"Expected [N, 508, 40], got {features.shape}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "features.npy", features)
    np.save(args.out_dir / "labels.npy", np.asarray([label for _, _, label in files], dtype=np.int64))
    metadata = {
        "corpus": args.corpus,
        "source": "WAV files",
        "label_order": list(CORPORA[args.corpus]),
        "sample_rate": sample_rate if sample_rate is not None else "native WAV rate",
        "n_mfcc": 40,
        "target_length_samples": 260000,
        "feature_shape": list(features.shape[1:]),
        "dtype": "float32",
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(files)} WAV features to {args.out_dir} with sample_rate={sample_rate or 'native'}")


if __name__ == "__main__":
    main()
