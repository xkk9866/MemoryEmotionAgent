"""Prototype: per-supporter Bayesian head learned by prequential likelihood.

g_d(t) = W_d u_t, vec(W_d) ~ N(0, (P_k kron P_u)^-1). W_d is the MAP fit to the
supporter's revealed turns (exact Newton), and the prior is learned by the
weighted prequential cross-entropy on held-out logits. u_t = [1] is RLA.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from cate.config import RESULT
from cate.metrics import classification_scores, preference_bias_official
from cate.official import load_official
from cate.rla import adapt, dialogue_order, select


def build_targets(order):
    """For each row (in order), indices of its earlier rows in the same dialogue."""
    tgt, past = [], []
    for rows in order:
        for j in range(len(rows)):
            tgt.append(rows[j])
            past.append(rows[:j])
    tmax = max(1, max(len(p) for p in past))
    idx = np.zeros((len(tgt), tmax), dtype=np.int64)
    mask = np.zeros((len(tgt), tmax), dtype=np.float32)
    for n, p in enumerate(past):
        idx[n, : len(p)] = p
        mask[n, : len(p)] = 1
    return torch.tensor(tgt), torch.tensor(idx), torch.tensor(mask)


class Head(torch.nn.Module):
    def __init__(self, k, feat, r):
        super().__init__()
        self.k, self.feat, self.r = k, feat, r
        m = 1 + (0 if feat == "none" else (k if feat == "prob" else r))
        self.m = m
        self.Lk = torch.nn.Parameter(torch.zeros(k, k))
        self.dk = torch.nn.Parameter(torch.zeros(k))
        self.du = torch.nn.Parameter(torch.zeros(m))
        if feat == "proj":
            self.A = torch.nn.Parameter(0.1 * torch.randn(r, k))

    def feats(self, z):
        one = torch.ones(z.shape[:-1] + (1,))
        if self.feat == "none":
            return one
        s = torch.softmax(z, -1)
        f = s if self.feat == "prob" else s @ self.A.T
        return torch.cat([one, f], -1)

    def precision(self):
        Lk = torch.tril(self.Lk, -1) + torch.diag(torch.exp(self.dk))
        Pk = Lk @ Lk.T + 1e-4 * torch.eye(self.k)
        Pu = torch.diag(torch.exp(self.du + torch.tensor([0.0] + [2.0] * (self.m - 1))))
        return torch.kron(Pk, Pu)

    def forward(self, z, y, tgt, idx, mask, newton=6):
        k, m = self.k, self.m
        u = self.feats(z)
        zp, up, yp = z[idx], u[idx], F.one_hot(y[idx], k).float()
        P = self.precision()
        n = len(tgt)
        w = torch.zeros(n, k * m)
        for _ in range(newton):
            W = w.view(n, k, m)
            s = torch.softmax(zp + torch.einsum("nkm,ntm->ntk", W, up), -1)
            r = mask[..., None] * (s - yp)
            g = torch.einsum("ntk,ntm->nkm", r, up).reshape(n, -1) + w @ P
            a = torch.diag_embed(s) - s[..., :, None] * s[..., None, :]
            h = torch.einsum("nt,ntkl,nti,ntj->nkilj", mask, a, up, up).reshape(n, k * m, k * m) + P
            w = w - torch.linalg.solve(h, g)
        W = w.view(n, k, m)
        return z[tgt] + torch.einsum("nkm,nm->nk", W, u[tgt])


def scores(y, z, k):
    p = z.argmax(-1)
    s = classification_scores(y, p, k)
    return [100 * s["macro_f1"], 100 * s["weighted_f1"], preference_bias_official(y, p, k)]


def run(dataset, fam, seed, feat, r, epochs=150):
    dialogs, samples, k = load_official(dataset, check=False)
    res = json.load(open(RESULT / "policy" / dataset / f"{fam}_s{seed}.json", encoding="utf-8"))
    data = {}
    for sp in ("valid", "test"):
        z = np.array(res["splits"][sp]["logits"], dtype=np.float32)
        y = np.array(res["splits"][sp]["y"])
        order = dialogue_order(samples[sp])
        data[sp] = (z, y, order)
    ytr = np.array([s["label"] for s in samples["train"]])
    freq = np.bincount(ytr, minlength=k)
    cw = torch.tensor(freq.sum() / (k * freq), dtype=torch.float32)
    zv, yv, ov = data["valid"]
    zt, yt, ot = data["test"]
    out = {"base": scores(yt, zt, k)}
    s2, half, _ = select(zv, yv, ov)
    out["rla"] = scores(yt, zt + adapt(zt, yt, ot, s2, half), k)
    torch.manual_seed(0)
    head = Head(k, feat, r)
    with torch.no_grad():
        head.dk.fill_(0.5 * np.log(1 / s2))
    opt = torch.optim.Adam(head.parameters(), lr=0.03)
    Zv, Yv = torch.tensor(zv), torch.tensor(yv)
    tv = build_targets(ov)
    for ep in range(epochs):
        zz = head(Zv, Yv, *tv)
        loss = F.cross_entropy(zz, Yv[tv[0]], weight=cw)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        Zt, Yt = torch.tensor(zt), torch.tensor(yt)
        tt = build_targets(ot)
        zz = head(Zt, Yt, *tt).numpy()
        zfull = np.zeros_like(zt)
        zfull[tt[0].numpy()] = zz
    out["spn"] = scores(yt, zfull, k)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esconv")
    ap.add_argument("--fam", default="roberta-base_none")
    ap.add_argument("--feat", default="none")
    ap.add_argument("--r", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    a = ap.parse_args()
    allr = [run(a.dataset, a.fam, s, a.feat, a.r) for s in a.seeds]
    for key in allr[0]:
        v = np.array([x[key] for x in allr])
        print(f"{a.dataset} {a.fam} {a.feat}{a.r} {key:5s} " + " ".join(f"{m:6.2f}" for m in v.mean(0)), flush=True)
