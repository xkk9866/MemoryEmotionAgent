"""Dialogue Policy Posterior (DPP).

A text classifier trained on many dialogues learns p(y | x) under the
population strategy prior pi. Inside one dialogue the supporter follows their
own policy theta, so under label shift the Bayes decision is

    log p(y | x, theta) = log p(y | x) + log theta_y - log pi_y + const.

theta is never observed, but the supporter's own earlier strategies are. With
a Dirichlet(kappa * pi) prior and exponentially forgotten counts n_y, the
posterior mean is (n_y + kappa pi_y) / (N + kappa). The shift is therefore

    shift_y = beta * [log(n_y + kappa pi_y) - log(N + kappa) - log pi_y].

kappa starts at its method-of-moments estimate from how much per-dialogue
strategy counts in the training set are overdispersed relative to pi. With no
history the shift is exactly zero, so the model falls back to the text policy.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WINDOW = 5


def history_counts(dialogs, rows, k: int, max_hist: int = 160) -> dict:
    """Past supporter strategies of each sample as padded tensors.

    dist is the number of utterances between the past action and the decision.
    """
    n = len(rows)
    strat = torch.full((n, max_hist), -1, dtype=torch.long)
    dist = torch.zeros((n, max_hist), dtype=torch.float32)
    for i, r in enumerate(rows):
        turns = dialogs[r["dialog"]]["turns"]
        past = [(r["t"] - p, turns[p]["strategy"]) for p in range(r["t"]) if turns[p]["strategy"] >= 0]
        past = past[-max_hist:]
        for j, (d, s) in enumerate(past):
            strat[i, j] = s
            dist[i, j] = d
    return {"strat": strat, "dist": dist}


def moment_kappa(dialogs, k: int, ids) -> tuple[float, np.ndarray]:
    """Dirichlet-multinomial concentration from per-dialogue overdispersion.

    E[(n_y - N pi_y)^2] = N pi_y (1 - pi_y) (N + kappa) / (1 + kappa); pooled over
    dialogues and labels this gives one ratio r = (N + kappa) / (1 + kappa).
    """
    per = []
    for d in ids:
        c = np.bincount([t["strategy"] for t in dialogs[d]["turns"] if t["strategy"] >= 0], minlength=k)
        if c.sum() >= 2:
            per.append(c.astype(np.float64))
    per = np.stack(per)
    pi = per.sum(0) / per.sum()
    num = 0.0
    for c in per:
        nd = c.sum()
        num += ((c - nd * pi) ** 2 / (nd * pi * (1 - pi))).sum() / (k - 1)
    # Solve sum_d r_d = sum_d (N_d + kappa)/(1 + kappa) for kappa with a 1-D search.
    target = num
    sizes = per.sum(1)
    lo, hi = 1e-3, 1e4
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        val = ((sizes + mid) / (1 + mid)).sum()
        if val > target:
            lo = mid
        else:
            hi = mid
    return float(math.sqrt(lo * hi)), pi


class DialoguePolicyPosterior(nn.Module):
    def __init__(self, pi: np.ndarray, kappa: float, half_life: float = 20.0, skip_window: bool = False):
        super().__init__()
        self.register_buffer("log_pi", torch.log(torch.tensor(pi, dtype=torch.float32)))
        self.log_kappa = nn.Parameter(torch.tensor(math.log(kappa), dtype=torch.float32))
        self.log_half = nn.Parameter(torch.tensor(math.log(half_life), dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.skip_window = skip_window

    @classmethod
    def fit(cls, dialogs, train_rows, train_counts, k: int, **kw):
        ids = sorted({r["dialog"] for r in train_rows})
        kappa, _ = moment_kappa(dialogs, k, ids)
        y = np.array([r["label"] for r in train_rows])
        pi = np.bincount(y, minlength=k).astype(np.float64)
        return cls(pi / pi.sum(), kappa, **kw)

    def forward(self, cnt: dict) -> torch.Tensor:
        strat, dist = cnt["strat"], cnt["dist"]
        valid = strat >= 0
        if self.skip_window:
            valid = valid & (dist > WINDOW)
        w = torch.exp(-dist * math.log(2.0) / self.log_half.exp()) * valid.float()
        k = self.log_pi.numel()
        onehot = F.one_hot(strat.clamp(min=0), k).float()
        n = torch.einsum("bl,blk->bk", w, onehot)
        kappa = self.log_kappa.exp()
        pi = self.log_pi.exp()
        post = torch.log(n + kappa * pi) - torch.log(n.sum(-1, keepdim=True) + kappa)
        return self.beta * (post - self.log_pi)

    def describe(self) -> str:
        return (f"kappa {float(self.log_kappa.exp()):.2f} half {float(self.log_half.exp()):.1f} "
                f"beta {float(self.beta):.2f}")
