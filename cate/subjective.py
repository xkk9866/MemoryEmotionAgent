"""Subjective-logic operators used by conflict-preserving fusion.

Binomial opinions (b, d, u) satisfy b + d + u = 1. Cumulative fusion follows
Jøsang's cumulative belief fusion. Conflict-preserving fusion is the operator
introduced in this project: when belief and disbelief collide, fused mass is
scaled by the conflict product and the residual is moved into uncertainty
instead of overwriting either side.
"""

from __future__ import annotations

import math

import torch


def _renorm(b: torch.Tensor, d: torch.Tensor, u: torch.Tensor):
    s = (b + d + u).clamp(min=1e-6)
    return b / s, d / s, u / s


def cumulative_fusion(
    b1: torch.Tensor,
    d1: torch.Tensor,
    u1: torch.Tensor,
    b2: torch.Tensor,
    d2: torch.Tensor,
    u2: torch.Tensor,
    eps: float = 1e-6,
):
    """Cumulative belief fusion for binomial opinions."""
    dogmatic = (u1 < eps) & (u2 < eps)
    k = (u1 + u2 - u1 * u2).clamp(min=eps)
    b = (b1 * u2 + b2 * u1) / k
    d = (d1 * u2 + d2 * u1) / k
    u = (u1 * u2) / k
    b = torch.where(dogmatic, 0.5 * (b1 + b2), b)
    d = torch.where(dogmatic, 0.5 * (d1 + d2), d)
    u = torch.where(dogmatic, torch.zeros_like(u), u)
    return _renorm(b, d, u)


def conflict_degree(b1, d1, b2, d2):
    """Product mass placed on contradictory poles."""
    return b1 * d2 + d1 * b2


def conflict_preserving_fusion(
    b1, d1, u1, b2, d2, u2, tau: torch.Tensor, sharpness: float = 8.0
):
    """Fuse evidence and, under conflict, preserve uncertainty.

    tau is a learned conflict threshold in (0, 1). The gate is soft so the
    threshold receives gradient.
    """
    dc = conflict_degree(b1, d1, b2, d2)
    bf, df, uf = cumulative_fusion(b1, d1, u1, b2, d2, u2)
    gate = torch.sigmoid((dc - tau) * sharpness)
    scale = (1.0 - dc).clamp(0.0, 1.0)
    bc = scale * bf
    dc_ = scale * df
    bf2 = (1.0 - gate) * bf + gate * bc
    df2 = (1.0 - gate) * df + gate * dc_
    uf2 = 1.0 - bf2 - df2
    bf2, df2, uf2 = _renorm(bf2, df2, uf2)
    return bf2, df2, uf2, dc


def decay_opinion(b, d, u, delta, half_life):
    """Move expired belief and disbelief into uncertainty.

    delta is the number of turns since the opinion was last updated.
    half_life is type-specific and positive.
    """
    factor = torch.exp(-delta.clamp(min=0) * math.log(2) / half_life.clamp(min=0.5))
    b2 = b * factor
    d2 = d * factor
    u2 = 1.0 - b2 - d2
    return _renorm(b2, d2, u2)


def evidence_opinion(polarity: torch.Tensor, confidence: torch.Tensor):
    """Map a signed polarity and a confidence into a binomial opinion."""
    c = confidence.clamp(0.0, 0.95)
    b = torch.where(polarity > 0.15, c, torch.zeros_like(c))
    d = torch.where(polarity < -0.15, c, torch.zeros_like(c))
    u = 1.0 - b - d
    return _renorm(b, d, u)


def _self_check() -> None:
    b1 = torch.tensor([0.8])
    d1 = torch.tensor([0.1])
    u1 = torch.tensor([0.1])
    b2 = torch.tensor([0.1])
    d2 = torch.tensor([0.8])
    u2 = torch.tensor([0.1])
    tau = torch.tensor(0.2)
    bf, df, uf, dc = conflict_preserving_fusion(b1, d1, u1, b2, d2, u2, tau)
    assert torch.all(dc > 0.5)
    assert torch.all(uf > 0.2), uf
    b3, d3, u3 = decay_opinion(b1, d1, u1, torch.tensor([8.0]), torch.tensor([4.0]))
    assert torch.all(b3 < b1)
    assert torch.all(u3 > u1)
    assert torch.allclose(b3 + d3 + u3, torch.ones_like(b3), atol=1e-4)
    print("subjective-logic self-check ok", float(uf), float(dc), float(u3))


if __name__ == "__main__":
    _self_check()
