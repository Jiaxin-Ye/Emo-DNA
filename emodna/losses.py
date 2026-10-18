from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveThreshold(nn.Module):
    def __init__(self, num_classes: int, momentum: float):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("value", torch.tensor(1.0 / num_classes))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def update(self, confidence: torch.Tensor) -> torch.Tensor:
        batch_mean = confidence.mean()
        if self.initialized:
            self.value.mul_(self.momentum).add_(batch_mean * (1.0 - self.momentum))
        else:
            self.initialized.fill_(True)
        return self.value.clone()


def supervised_contrastive_loss(
    features: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    features = F.normalize(features, dim=-1)
    logits = features @ features.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    exp_logits = torch.exp(logits) * ~self_mask
    log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return features.sum() * 0.0
    mean_log_probability = (positive_mask * log_probability).sum(dim=1) / positive_count.clamp_min(1)
    return -mean_log_probability[valid].mean()


def prototype_decoupling_loss(
    emotion_source: torch.Tensor,
    emotion_target: torch.Tensor,
    corpus_source: torch.Tensor,
    corpus_target: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    query = torch.stack((emotion_source.mean(0), emotion_target.mean(0)))
    positive = torch.stack((emotion_target.mean(0), emotion_source.mean(0)))
    negatives = torch.stack((corpus_source.mean(0), corpus_target.mean(0)))
    positive_score = (query * positive).sum(dim=1, keepdim=True)
    negative_score = query @ negatives.T
    logits = torch.cat((positive_score, negative_score), dim=1) / temperature
    return F.cross_entropy(logits, torch.zeros(2, dtype=torch.long, device=query.device))


def multi_kernel_mmd(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Linear-time MK-MMD with the five Gaussian kernels used by the paper code."""
    if source.shape != target.shape:
        size = min(source.shape[0], target.shape[0])
        source, target = source[:size], target[:size]
    batch_size = source.shape[0]
    if batch_size < 2:
        return source.sum() * 0.0
    features = torch.cat((source, target), dim=0)
    distances = torch.cdist(features, features).square()
    mean_distance = distances.detach().mean().clamp_min(1e-6)
    kernel = sum(
        torch.exp(-distances / (2.0 * mean_distance * (2.0**power)))
        for power in range(-3, 2)
    )
    indices = torch.arange(batch_size, device=source.device)
    next_indices = (indices + 1) % batch_size
    return (
        kernel[indices, next_indices]
        + kernel[indices + batch_size, next_indices + batch_size]
        - kernel[indices, next_indices + batch_size]
        - kernel[next_indices, indices + batch_size]
    ).mean()


def emodna_losses(
    model: nn.Module,
    source_output: dict[str, torch.Tensor],
    target_output: dict[str, torch.Tensor],
    source_labels: torch.Tensor,
    threshold: AdaptiveThreshold,
    prototype_temperature: float,
    contrastive_temperature: float,
    contrastive_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = source_labels.device
    source_size = source_labels.shape[0]
    target_size = target_output["emotion_logits"].shape[0]
    domain_logits = torch.cat((source_output["corpus_logits"], target_output["corpus_logits"]))
    domain_labels = torch.cat(
        (
            torch.zeros(source_size, dtype=torch.long, device=device),
            torch.ones(target_size, dtype=torch.long, device=device),
        )
    )

    emotion_loss = F.cross_entropy(source_output["emotion_logits"], source_labels)
    corpus_loss = F.cross_entropy(domain_logits, domain_labels)
    mmd_loss = multi_kernel_mmd(
        source_output["emotion_embedding"], target_output["emotion_embedding"]
    )

    probabilities = target_output["emotion_logits"].softmax(dim=-1)
    confidence, pseudo_labels = probabilities.max(dim=-1)
    current_threshold = threshold.update(confidence)
    selected = confidence >= current_threshold
    emotion_features = source_output["emotion_embedding"]
    contrastive_labels = source_labels
    if selected.any():
        emotion_features = torch.cat((emotion_features, target_output["emotion_embedding"][selected]))
        contrastive_labels = torch.cat((contrastive_labels, pseudo_labels[selected]))
    projected_emotion = model.projection(emotion_features)
    contrastive_loss = supervised_contrastive_loss(
        projected_emotion, contrastive_labels, contrastive_temperature
    )

    projected_es = model.projection(source_output["emotion_embedding"])
    projected_et = model.projection(target_output["emotion_embedding"])
    projected_cs = model.projection(source_output["corpus_embedding"])
    projected_ct = model.projection(target_output["corpus_embedding"])
    decoupling_loss = prototype_decoupling_loss(
        projected_es, projected_et, projected_cs, projected_ct, prototype_temperature
    )

    total = emotion_loss + corpus_loss + mmd_loss + contrastive_weight * contrastive_loss + decoupling_loss
    values = {
        "total": total.item(),
        "emotion": emotion_loss.item(),
        "corpus": corpus_loss.item(),
        "mmd": mmd_loss.item(),
        "contrastive": contrastive_loss.item(),
        "decoupling": decoupling_loss.item(),
        "threshold": current_threshold.item(),
        "selected": int(selected.sum().item()),
    }
    return total, values
