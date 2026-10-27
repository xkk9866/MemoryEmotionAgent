"""Evaluation metrics shared by strategy, emotion, failure, and memory probes."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def classification_scores(y_true, y_pred, n_labels: int | None = None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(n_labels)) if n_labels is not None else None
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        ),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "n": int(len(y_true)),
    }


def preference_bias(y_true, y_pred, n_labels: int, iters: int = 20) -> float:
    """Preference bias of Wan et al. (NAACL 2025), std of iterated strategy preference.

    w_ij is the count of predicting i when the gold label is j.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    # rows = gold, cols = pred
    w = confusion_matrix(y_true, y_pred, labels=list(range(n_labels))).astype(np.float64)
    p = np.ones(n_labels, dtype=np.float64)
    for _ in range(iters):
        new_p = np.zeros(n_labels, dtype=np.float64)
        for i in range(n_labels):
            num = 0.0
            den = 0.0
            for j in range(n_labels):
                pair = p[i] + p[j] + 1e-8
                wij = w[j, i]
                wji = w[i, j]
                num += wij * p[j] / pair
                den += wji / pair
            new_p[i] = num / (den + 1e-8)
        if new_p.sum() > 0:
            new_p = new_p / new_p.mean()
        p = new_p
    return float(p.std())


def safe_auroc(y_true, scores) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def macro_f1_by_bucket(y_true, y_pred, bucket, n_labels: int) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    bucket = np.asarray(bucket)
    out = {}
    for name in sorted(set(bucket.tolist())):
        m = bucket == name
        if m.sum() == 0:
            continue
        out[str(name)] = classification_scores(y_true[m], y_pred[m], n_labels)
    return out


LENGTH_BINS = [(0, 8, "1-8"), (8, 16, "9-16"), (16, 32, "17-32"), (32, 10_000, "33+")]


def length_bucket(hist_len: int) -> str:
    for lo, hi, name in LENGTH_BINS:
        if lo <= hist_len < hi:
            return name
    return "33+"


def preference_bias_official(y_true, y_pred, n_labels: int, iters: int = 20) -> float:
    """Exact EmoDynamiX metrics.py: in-place updates, rows = gold, cols = pred."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_labels))).astype(np.float64)
    eps = 1e-6
    p = [1.0] * n_labels
    for _ in range(iters):
        for i in range(n_labels):
            num = sum(cm[j, i] * p[j] / (p[i] + p[j] + eps) for j in range(n_labels))
            den = sum(cm[i, j] / (p[i] + p[j] + eps) for j in range(n_labels))
            p[i] = num / den if den > 0 else p[i]
        p = [x * n_labels / sum(p) for x in p]
    return float(np.std(p))
