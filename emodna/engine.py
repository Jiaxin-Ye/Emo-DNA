from __future__ import annotations

import json
import random
from hashlib import sha256
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import trange

from .data import AudioDataset, NpyDataset, pad_mfcc_batch
from .losses import AdaptiveThreshold, emodna_losses
from .metrics import classification_metrics
from .model import EMODNA
from .legacy_model import LegacyInferenceModel


def config_signature(config: dict[str, Any]) -> str:
    public_config = {key: value for key, value in config.items() if not key.startswith("_")}
    serialized = json.dumps(public_config, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()[:16]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infinite_batches(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield from loader


def make_dataset(config: dict[str, Any], corpus: str) -> Dataset:
    input_format = config.get("input_format", "audio")
    if input_format == "audio":
        sample_rate = config.get("audio_sample_rates", {}).get(corpus)
        return AudioDataset(
            Path(config["audio_root"]) / corpus,
            config["corpora"][corpus],
            config["task"],
            sample_rate=sample_rate,
        )
    if input_format == "npy":
        return NpyDataset(Path(config["features_root"]) / corpus, corpus, config["task"])
    raise ValueError("input_format must be 'audio' or 'npy'")


def make_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=drop_last,
        collate_fn=pad_mfcc_batch,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate(model: EMODNA, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    for mfcc, target in loader:
        logits = model.predict(mfcc.to(device))
        labels.extend(target.tolist())
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    return classification_metrics(labels, predictions)


def save_checkpoint(
    path: Path,
    model: EMODNA,
    threshold: AdaptiveThreshold,
    config: dict[str, Any],
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    public_config = {key: value for key, value in config.items() if not key.startswith("_")}
    payload = {
            "model": model.state_dict(),
            "threshold": threshold.state_dict(),
            "config": public_config,
            "epoch": epoch,
            "metrics": metrics,
        }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[EMODNA, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if set(("E", "EMO_E")).issubset(checkpoint):
        return LegacyInferenceModel(checkpoint, device), checkpoint
    model = EMODNA(checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def train(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = config["train"]
    source_dataset = make_dataset(config, config["source"])
    target_dataset = make_dataset(config, config["target"])

    if settings["selection"] == "paper":
        holdout_size = max(1, int(len(target_dataset) * settings["target_holdout"]))
        test_size = len(target_dataset) - holdout_size
        target_test, target_holdout = random_split(
            target_dataset,
            [test_size, holdout_size],
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        target_test = target_dataset
        target_holdout = None
    source_loader = make_loader(
        source_dataset, settings["batch_size"], settings["num_workers"], True, True
    )
    target_loader = make_loader(
        target_dataset, settings["batch_size"], settings["num_workers"], True, True
    )
    source_eval_loader = make_loader(source_dataset, settings["batch_size"], settings["num_workers"], False)
    target_test_loader = make_loader(target_test, settings["batch_size"], settings["num_workers"], False)
    target_holdout_loader = (
        make_loader(target_holdout, settings["batch_size"], settings["num_workers"], False)
        if target_holdout is not None
        else None
    )
    source_batches = infinite_batches(source_loader)
    target_batches = infinite_batches(target_loader)

    model = EMODNA(config["model"]).to(device)
    threshold = AdaptiveThreshold(2, settings["threshold_momentum"]).to(device)
    optimizer = torch.optim.RAdam(
        model.parameters(),
        lr=settings["learning_rate"],
        betas=(settings["beta1"], settings["beta2"]),
    )
    output_dir = Path(config["output_dir"]) / config["task"] / f"{config['source']}_to_{config['target']}" / f"seed_{seed}"
    checkpoint_path = output_dir / "model.pt"
    best_score = -1.0
    last_losses: dict[str, float] = {}

    for epoch in trange(settings["epochs"], desc=f"{config['source']}->{config['target']}"):
        model.train()
        for _ in range(settings["steps_per_epoch"]):
            source_mfcc, source_labels = next(source_batches)
            target_mfcc, _ = next(target_batches)
            source_mfcc = source_mfcc.to(device, non_blocking=True)
            source_labels = source_labels.to(device, non_blocking=True)
            target_mfcc = target_mfcc.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            source_output = model(source_mfcc)
            target_output = model(target_mfcc)
            loss, last_losses = emodna_losses(
                model,
                source_output,
                target_output,
                source_labels,
                threshold,
                settings["prototype_temperature"],
                settings["contrastive_temperature"],
                settings["contrastive_weight"],
            )
            loss.backward()
            optimizer.step()

        selection = settings["selection"]
        if selection == "paper":
            assert target_holdout_loader is not None
            holdout_metrics = evaluate(model, target_holdout_loader, device)
            score = holdout_metrics["war"] + holdout_metrics["f1"]
        elif selection == "last":
            holdout_metrics = {}
            score = float(epoch)
        else:
            raise ValueError("train.selection must be 'last' or 'paper'")
        if score >= best_score:
            best_score = score
            save_checkpoint(checkpoint_path, model, threshold, config, epoch, holdout_metrics)

    model, checkpoint = load_checkpoint(checkpoint_path, device)
    result = {
        "task": config["task"],
        "source": config["source"],
        "target": config["target"],
        "seed": seed,
        "checkpoint": str(checkpoint_path.resolve()),
        "selected_epoch": checkpoint["epoch"],
        "selection": settings["selection"],
        "config_signature": config_signature(config),
        "source_metrics": evaluate(model, source_eval_loader, device),
        "target_metrics": evaluate(model, target_test_loader, device),
        "last_losses": last_losses,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result
