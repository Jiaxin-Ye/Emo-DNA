from __future__ import annotations

import argparse
import json
from pathlib import Path

from emodna.config import load_config, with_overrides
from emodna.engine import train

RELEASE_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EMODNA on one cross-corpus scenario.")
    parser.add_argument("--config", default=str(RELEASE_ROOT / "configs/emodna.yaml"))
    parser.add_argument("--source")
    parser.add_argument("--target")
    parser.add_argument("--task", choices=("arousal", "valence"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--audio-root")
    parser.add_argument("--features-root")
    parser.add_argument("--input-format", choices=("npy", "audio"))
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--paper-protocol", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = with_overrides(
        load_config(args.config),
        source=args.source,
        target=args.target,
        task=args.task,
        seed=args.seed,
        audio_root=args.audio_root,
        features_root=args.features_root,
        input_format=args.input_format,
        output_dir=args.output_dir,
    )
    if args.paper_protocol:
        config["train"]["selection"] = "paper"
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.steps_per_epoch is not None:
        config["train"]["steps_per_epoch"] = args.steps_per_epoch
    if config["source"] == config["target"]:
        raise ValueError("Source and target corpora must differ")
    print(json.dumps(train(config), indent=2))


if __name__ == "__main__":
    main()
