"""Revealed-Label Adaptation (RLA): test-time Bayesian adaptation to one supporter.

When the supporter acts at turn t, the strategies of all their earlier turns
are already in the dialogue history, and so are the frozen model's logits for
those turns. Each dialogue is therefore a small labeled set. Model the
supporter's policy as an offset b_d on the logits, with prior b_d ~ N(0, s2 I):

    b_d(t) = argmin_b  sum_{p < t} w_p CE(z_p + b, y_p) + |b|^2 / (2 s2),   w_p = 2^{-(t - p) / h}

and predict with z_t + b_d(t). Unlike counting strategies (cate.dpp), the
likelihood only moves b where the model was wrong about this supporter, so
evidence the window already shows is not counted twice. The objective is
strictly convex; a few Newton steps solve it exactly. s2 and h are chosen by
validation negative log-likelihood, never by test scores.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.special import log_softmax, softmax


def dialogue_order(rows) -> list[list[int]]:
    """Row ids grouped by dialogue, sorted by turn."""
    groups: dict = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["dialog"], []).append((r["t"], i))
    return [[i for _, i in sorted(v)] for v in groups.values()]


def style_precision(dialogs, train_rows, k: int, ridge: float = 1e-2) -> np.ndarray:
    """Precision of the supporter-style prior, from training labels only.

    Each training dialogue's policy is its Dirichlet posterior mean (counts + kappa pi);
    the prior covariance of the offset is the covariance of these log-policies across
    supporters (centred per dialogue, since softmax ignores a constant shift), scaled
    to mean variance 1 so that s2 keeps the meaning it has for the isotropic prior.
    """
    from cate.dpp import moment_kappa

    ids = sorted({r["dialog"] for r in train_rows})
    kappa, pi = moment_kappa(dialogs, k, ids)
    logs = []
    for d in ids:
        c = np.bincount([t["strategy"] for t in dialogs[d]["turns"] if t["strategy"] >= 0], minlength=k)
        lp = np.log((c + kappa * pi) / (c.sum() + kappa))
        logs.append(lp - lp.mean())
    cov = np.cov(np.array(logs).T)
    cov = cov / (np.trace(cov) / k) + ridge * np.eye(k)
    return np.linalg.inv(cov)


def adapt(z, y, order, s2: float, half: float = float("inf"), newton: int = 8, prec=None) -> np.ndarray:
    """Offsets b for every row, using only earlier rows of the same dialogue.

    prec is the prior precision (style_precision); None is the isotropic prior.
    """
    k = z.shape[1]
    prior = (np.eye(k) if prec is None else prec) / s2
    b_all = np.zeros_like(z)
    eye = np.eye(k)
    for rows in order:
        b = np.zeros(k)
        for j in range(1, len(rows)):
            past = rows[:j]
            w = np.ones(j) if np.isinf(half) else 0.5 ** ((j - np.arange(j)) / half)
            zp, onehot = z[past], eye[y[past]]
            for _ in range(newton):  # warm start from the previous turn's solution
                p = softmax(zp + b, -1)
                g = (w[:, None] * (p - onehot)).sum(0) + prior @ b
                h = np.einsum("n,ni,ij->ij", w, p, eye) - np.einsum("n,ni,nj->ij", w, p, p) + prior
                b = b - np.linalg.solve(h, g)
            b_all[rows[j]] = b
    return b_all


def nll(z, y) -> float:
    return float(-log_softmax(z, -1)[np.arange(len(y)), y].mean())


GRID_S2 = [0.03, 0.1, 0.2, 0.3, 0.5, 1.0]
GRID_HALF = [float("inf"), 20.0, 8.0]


def select(z_val, y_val, order_val, prec=None):
    """(s2, half) with the lowest validation NLL."""
    best = None
    for s2, half in itertools.product(GRID_S2, GRID_HALF):
        v = nll(z_val + adapt(z_val, y_val, order_val, s2, half, prec=prec), y_val)
        if best is None or v < best[0]:
            best = (v, s2, half)
    return best[1], best[2], best[0]
