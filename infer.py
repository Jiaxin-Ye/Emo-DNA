from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from emodna.engine import load_checkpoint
from emodna.features import extract_mfcc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EMODNA inference on NPY features or WAV audio.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=("arousal", "valence"), help="Required for legacy .pt checkpoints.")
    parser.add_argument("--input-format", choices=("npy", "audio"), required=True)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("inputs", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    task = checkpoint.get("config", {}).get("task", args.task)
    if task is None:
        raise ValueError("Pass --task for a legacy checkpoint without embedded config")
    results = []
    for filename in args.inputs:
        if args.input_format == "audio":
            feature = extract_mfcc(filename, sample_rate=args.sample_rate)
            features = feature[None, ...]
            names = [Path(filename).name]
        else:
            features = np.load(filename).astype(np.float32, copy=False)
            if features.ndim == 2:
                features = features[None, ...]
            if features.ndim != 3 or features.shape[2] != 40:
                raise ValueError(f"Expected [time, 40] or [N, time, 40] MFCC, got {features.shape} at {filename}")
            names = [f"{Path(filename).name}[{index}]" for index in range(len(features))]
        inputs = torch.from_numpy(features).to(device)
        with torch.no_grad():
            probabilities = model.predict(inputs).softmax(dim=-1).cpu()
        for name, probability in zip(names, probabilities):
            results.append({
                "file": name,
                "task": task,
                "prediction": int(probability.argmax().item()),
                "probabilities": probability.tolist(),
            })
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
