from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score


def classification_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    y_true = np.asarray(labels)
    y_pred = np.asarray(predictions)
    return {
        "war": float(accuracy_score(y_true, y_pred)),
        "uar": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
    }
