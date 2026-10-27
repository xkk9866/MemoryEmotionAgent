"""SPIRE: Supporter Policy Inference from REvealed actions.

A strategy classifier gives logits z_t. The supporter of dialogue d deviates from
the population policy by a situation-dependent amount, g_d(t) = W_d u_t, where

    u_t = [1, phi_psi(x_t)]              (situation code, learned, r dims)
    vec(W_d) ~ N(0, S_k kron S_u)         (strategy covariance x situation covariance)

At turn t, W_d is the posterior mode given the supporter's revealed turns p < t,
a strictly convex problem solved exactly by Newton's method. The prediction is the
Laplace-moderated posterior predictive. The prior and phi are learned by the
class-weighted prequential log-likelihood, through the solver, on *cross-fitted*
logits: each training decision is scored by a classifier that never saw it, so the
residuals the layer learns from are distributed like test-time residuals.

Special cases: u_t = [1] with S_k = s2 I and no moderation is RLA.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cate.config import RESULT
from cate.crossfit import K_FOLDS, fold_of
from cate.metrics import classification_scores, preference_bias_official
from cate.official import load_official
from cate.rla import adapt, dialogue_order, select

ORACLE = False
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- data


def situation_inputs(dialogs, rows, k):
    """Observable, backbone-independent context of each decision.

    one-hot of the supporter's previous strategy (K+1, last = none) and
    log(1 + number of revealed turns), which lets the prior trust a long history.
    """
    prev = np.zeros((len(rows), k + 1), np.float32)
    pos = np.zeros((len(rows), 1), np.float32)
    for i, r in enumerate(rows):
        past = [x["strategy"] for x in dialogs[r["dialog"]]["turns"][: r["t"]] if x["strategy"] >= 0]
        prev[i, past[-1] if past else k] = 1
        pos[i, 0] = np.log1p(len(past))
    return np.concatenate([prev, pos], 1)




Q_OWN = 16  # last context columns: embedding of the decision's own utterance (evidence only, never a feature)


def semantic_inputs(dataset, split):
    from cate.config import CACHE
    f = np.load(CACHE / f"situation_{dataset}.npz")
    return np.concatenate([f[f"{split}_seeker"], f[f"{split}_self"], f[f"{split}_own"]], 1)


def targets(order):
    """Each row with the indices of its earlier rows in the same dialogue (padded)."""
    tgt, past = [], []
    for rows in order:
        for j in range(len(rows)):
            tgt.append(rows[j])
            past.append(rows[:j])
    tmax = max(1, max(len(p) for p in past))
    idx = np.zeros((len(tgt), tmax), np.int64)
    mask = np.zeros((len(tgt), tmax), np.float32)
    for n, p in enumerate(past):
        idx[n, : len(p)] = p
        mask[n, : len(p)] = 1
    return np.array(tgt), idx, mask


# ----------------------------------------------------------------------------- model


class Spire(nn.Module):
    def __init__(self, k, d_in, r=4, situation=True, structured=True, moderate=True, use_cols=None, text=False):
        super().__init__()
        self.use_cols = use_cols
        d_in = (d_in - Q_OWN) if use_cols is None else use_cols
        self.k, self.r = k, (r if situation else 0)
        self.m = 1 + self.r
        self.structured, self.moderate = structured, moderate
        if self.r:
            # situation code from the backbone's belief and the observable context
            self.phi = nn.Sequential(nn.Linear(k + d_in, 32), nn.Tanh(), nn.Linear(32, self.r))
        self.lk = nn.Parameter(torch.zeros(k, k))       # strategy covariance factor
        self.log_sk = nn.Parameter(torch.zeros(k))
        self.log_su = nn.Parameter(torch.zeros(self.m))  # per-situation-dim scale
        self.log_s2 = nn.Parameter(torch.tensor(0.0))
        self.inner_w = None
        self.text = text
        if text:
            # the supporter also reveals how they write: e_p = M y_p + G vec(W_d) + noise, a
            # linear-Gaussian view of the same latent, so inference stays strictly convex
            self.M = nn.Parameter(torch.zeros(k, Q_OWN))
            self.G = nn.Parameter(0.01 * torch.randn(Q_OWN, k * self.m))
            self.log_tau2 = nn.Parameter(torch.tensor(0.0))

    def init_s2(self, s2):
        with torch.no_grad():
            self.log_s2.fill_(float(np.log(s2)))
            self.log_su[1:] = -1.0

    def covariance(self):
        if self.structured:
            L = torch.tril(self.lk, -1) + torch.diag(torch.exp(self.log_sk))
            Sk = L @ L.T
            Sk = Sk / Sk.diagonal().mean()
        else:
            Sk = torch.eye(self.k, device=self.lk.device)
        Su = torch.diag(torch.exp(self.log_su))
        return torch.exp(self.log_s2) * torch.kron(Sk, Su)

    def code(self, z, ctx):
        one = torch.ones(z.shape[0], 1, device=z.device)
        if not self.r:
            return one
        c = ctx[:, :-Q_OWN] if self.use_cols is None else ctx[:, : self.use_cols]  # never the own utterance
        return torch.cat([one, self.phi(torch.cat([torch.softmax(z, -1), c], -1))], -1)

    def forward(self, z, y, ctx, tgt, idx, mask, newton=8):
        """Adapted logits for the rows tgt, from their revealed rows idx (mask)."""
        k, m = self.k, self.m
        D = k * m
        u = self.code(z, ctx)
        zp, up, yp = z[idx], u[idx], F.one_hot(y[idx], k).float()
        S = self.covariance()
        P = torch.linalg.inv(S + 1e-5 * torch.eye(D, device=z.device))
        n, T = idx.shape
        uu = (up[..., :, None] * up[..., None, :]).reshape(n, T, m * m)
        w = torch.zeros(n, D, device=z.device)
        # balanced geometry: revealed labels weighted like the backbone's training loss, so the
        # person's deviation is inferred in the space the backbone's logits live in
        lmask = mask if self.inner_w is None else mask * self.inner_w[y[idx]]
        if self.text:  # revealed utterances only (rows p < t): residual of their embedding
            prec = torch.exp(-self.log_tau2)
            resid = ctx[idx][..., -Q_OWN:] - yp @ self.M  # n, T, q
            h_text = prec * mask.sum(1)[:, None, None] * (self.G.T @ self.G)
        for _ in range(newton):
            W = w.view(n, k, m)
            s = torch.softmax(zp + torch.einsum("nkm,ntm->ntk", W, up), -1)
            g = torch.einsum("ntk,ntm->nkm", lmask[..., None] * (s - yp), up).reshape(n, D) + w @ P
            a = (torch.diag_embed(s) - s[..., :, None] * s[..., None, :]) * lmask[..., None, None]
            h = torch.bmm(a.reshape(n, T, k * k).transpose(1, 2), uu)  # n, k*k, m*m
            h = h.view(n, k, k, m, m).permute(0, 1, 3, 2, 4).reshape(n, D, D) + P
            if self.text:
                err = mask[..., None] * (resid - (w @ self.G.T)[:, None, :])
                g = g - prec * err.sum(1) @ self.G
                h = h + h_text
            w = w - torch.linalg.solve(h, g.unsqueeze(-1)).squeeze(-1)
        ut = u[tgt]
        mean = z[tgt] + torch.einsum("nkm,nm->nk", w.view(n, k, m), ut)
        if not self.moderate:
            return mean
        # Laplace predictive: logit variance of class j is (e_j kron u)^T H^-1 (e_j kron u)
        hinv = torch.linalg.inv(h)
        hinv = hinv.view(n, k, m, k, m).diagonal(dim1=1, dim2=3)  # n, m, m, k
        var = torch.einsum("nijk,ni,nj->nk", hinv, ut, ut).clamp(min=0)
        return mean / torch.sqrt(1 + np.pi / 8 * var)


class AmortizedNP(nn.Module):
    """Black-box alternative (attentive neural process): the correction is read out by
    cross-attention from the decision to the supporter's revealed turns, each encoded
    from (belief, revealed label, context). Same inputs, data and loss as Spire."""

    def __init__(self, k, d_in, d=64, heads=4):
        super().__init__()
        self.k = k
        self.enc = nn.Sequential(nn.Linear(2 * k + d_in, d), nn.GELU(), nn.Linear(d, d))
        self.qry = nn.Sequential(nn.Linear(k + d_in, d), nn.GELU(), nn.Linear(d, d))
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, k))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def init_s2(self, s2):
        pass

    def forward(self, z, y, ctx, tgt, idx, mask, newton=None):
        ctx = ctx[:, : self.qry[0].in_features - self.k]
        s = torch.softmax(z, -1)
        e = self.enc(torch.cat([s[idx], F.one_hot(y[idx], self.k).float(), ctx[idx]], -1))
        n = len(tgt)
        e = torch.cat([self.null.expand(n, 1, -1), e], 1)
        keep = torch.cat([torch.ones(n, 1, device=z.device), mask], 1) > 0
        q = self.qry(torch.cat([s[tgt], ctx[tgt]], -1)).unsqueeze(1)
        a, _ = self.att(q, e, e, key_padding_mask=~keep)
        return z[tgt] + self.out(torch.cat([a.squeeze(1), q.squeeze(1)], -1))


def run_layer(layer, z, y, ctx, tg, bs=2048):
    out = torch.zeros_like(z)
    t_all, i_all, m_all = tg
    for i in range(0, len(t_all), bs):
        sl = slice(i, i + bs)
        tmax = int(m_all[sl].sum(1).max().item()) or 1
        out[t_all[sl]] = layer(z, y, ctx, t_all[sl], i_all[sl, :tmax], m_all[sl, :tmax])
    return out


# ----------------------------------------------------------------------------- experiment


def metrics(y, z, k):
    p = z.argmax(-1)
    s = classification_scores(y, p, k)
    return [100 * s["macro_f1"], 100 * s["weighted_f1"], preference_bias_official(y, p, k)]


def load_crossfit(dataset, tag, seed):
    runs = [json.load(open(RESULT / "crossfit" / dataset / f"{tag}_s{seed}_f{f}.json", encoding="utf-8"))
            for f in range(K_FOLDS)]
    return runs


def tensors(z, y, ctx):
    return (torch.tensor(np.asarray(z), dtype=torch.float32, device=DEV), torch.tensor(np.asarray(y), device=DEV),
            torch.tensor(ctx, device=DEV))


def loo_tensors(order):
    """Oracle: each row with every OTHER row of its dialogue, future included."""
    tgt, past = [], []
    for rows in order:
        for j in range(len(rows)):
            tgt.append(rows[j])
            past.append(rows[:j] + rows[j + 1 :])
    tmax = max(1, max(len(p) for p in past))
    idx = np.zeros((len(tgt), tmax), np.int64)
    mask = np.zeros((len(tgt), tmax), np.float32)
    for n, p in enumerate(past):
        idx[n, : len(p)] = p
        mask[n, : len(p)] = 1
    return torch.tensor(tgt, device=DEV), torch.tensor(idx, device=DEV), torch.tensor(mask, device=DEV)


def tg_tensors(order):
    t, i, m = targets(order)
    return torch.tensor(t, device=DEV), torch.tensor(i, device=DEV), torch.tensor(m, device=DEV)


def train_layer(cfg, k, oof, val_folds, cw, steps=200, lr=0.02, seed=0, log=None):
    """Fit the layer on cross-fitted training logits by full-batch Adam (deterministic up to
    floating point); keep the step with the best validation NLL, checked every 5 steps."""
    torch.manual_seed(seed)
    if not cfg.get("weighted") and not cfg.get("balanced"):
        # the prequential log-likelihood sum_t log p(y_t | y_<t) is the log evidence of the
        # supporter's revealed sequence: unweighted, the layer's prior is learned by type-II ML
        cw = torch.ones_like(cw)
    if cfg.get("amortized"):
        layer = AmortizedNP(k, k + 2).to(DEV)  # same observable context as Spire
        lr, steps = 3e-3, 400
    else:
        layer = Spire(k, oof[2].shape[1], r=cfg["r"], situation=cfg["situation"], structured=cfg["structured"],
                      moderate=cfg["moderate"], use_cols=None if cfg.get("semantic", True) else k + 2,
                      text=cfg.get("text", False)).to(DEV)
    layer.init_s2(cfg["s2_init"])
    if cfg.get("balanced"):
        layer.inner_w = cw / cw[oof[1][oof[3][0]]].mean()  # mean weight 1 over training decisions
    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    z, y, ctx, tg = oof
    n = len(tg[0])
    total = cw[y[tg[0]]].sum()
    best, state = None, None
    for step in range(steps):
        layer.train()
        opt.zero_grad()
        for i in range(0, n, 2048):
            sl = slice(i, i + 2048)
            tmax = int(tg[2][sl].sum(1).max().item()) or 1
            zz = layer(z, y, ctx, tg[0][sl], tg[1][sl, :tmax], tg[2][sl, :tmax])
            loss = F.cross_entropy(zz, y[tg[0][sl]], weight=cw, reduction="sum") / total
            loss.backward()
        opt.step()
        if step % 5 == 4 or step == steps - 1:
            with torch.no_grad():
                layer.eval()
                v = np.mean([F.cross_entropy(run_layer(layer, *vf[:3], vf[3]), vf[1], weight=cw).item()
                             for vf in val_folds])
            if best is None or v < best:
                best, state = v, {a: b.detach().clone() for a, b in layer.state_dict().items()}
            if log:
                log(f"  step{step} val {v:.4f}")
    layer.load_state_dict(state)
    layer.eval()
    return layer, best


CONFIGS = {
    "spire": dict(r=4, situation=True, structured=True, moderate=True, semantic=False),
    "semantic": dict(r=4, situation=True, structured=True, moderate=True, semantic=True),
    "no_situation": dict(r=0, situation=False, structured=True, moderate=True),
    "isotropic": dict(r=4, situation=True, structured=False, moderate=True, semantic=False),
    "no_moderation": dict(r=4, situation=True, structured=True, moderate=False, semantic=False),
    "learned_rla": dict(r=0, situation=False, structured=False, moderate=False),
    "weighted": dict(r=4, situation=True, structured=True, moderate=True, semantic=False, weighted=True),
    "amortized": dict(amortized=True),
    "balanced": dict(r=4, situation=True, structured=True, moderate=True, semantic=False, balanced=True),
    "balanced_text": dict(r=4, situation=True, structured=True, moderate=True, semantic=False, balanced=True, text=True),
    "balanced_rla": dict(r=0, situation=False, structured=False, moderate=False, mean=False, balanced=True),
    "text": dict(r=4, situation=True, structured=True, moderate=True, semantic=False, text=True),
    "text_iso": dict(r=4, situation=True, structured=False, moderate=True, semantic=False, text=True),
    "no_crossfit": dict(r=4, situation=True, structured=True, moderate=True, semantic=False, insample=True),
}


def experiment(dataset, tag, seed, configs, log=print):
    dialogs, samples, k = load_official(dataset, check=False)
    runs = load_crossfit(dataset, tag, seed)
    ytr = np.array([s["label"] for s in samples["train"]])
    freq = np.bincount(ytr, minlength=k)
    cw = torch.tensor(freq.sum() / (k * freq), dtype=torch.float32, device=DEV)
    ctx = {sp: np.concatenate([situation_inputs(dialogs, samples[sp], k), semantic_inputs(dataset, sp)], 1)
           for sp in samples}
    orders = {sp: dialogue_order(samples[sp]) for sp in samples}
    # cross-fitted training logits: every row scored by the fold model that did not train on it
    z_oof = np.zeros((len(samples["train"]), k), np.float32)
    for r in runs:
        z_oof[r["splits"]["heldout"]["index"]] = np.array(r["splits"]["heldout"]["logits"])
    oof = (*tensors(z_oof, ytr, ctx["train"]), tg_tensors(orders["train"]))
    # the same backbone trained on all training dialogues (results/policy): its in-sample
    # training logits (ablation without cross-fitting) and its test logits (plug-in use)
    fpath = RESULT / "policy" / dataset / f"{tag}_none_s{seed}.json"
    full = json.load(open(fpath, encoding="utf-8")) if fpath.exists() else None
    insample = None
    epath = RESULT / "policy" / f"{dataset}_emodx" / "emodynamix_none_s0.json"
    emodx = json.load(open(epath, encoding="utf-8")) if epath.exists() else None
    emodx_z = {}
    if emodx is not None:  # released checkpoint scores a subset of rows (index); the rest get flat logits
        for sp in ("valid", "test"):
            zz = np.zeros((len(samples[sp]), k), np.float32)
            ix = emodx["splits"][sp].get("index", list(range(len(samples[sp]))))
            zz[ix] = np.array(emodx["splits"][sp]["logits"])
            emodx_z[sp] = (zz, np.array(ix))
    if full is not None:
        insample = (*tensors(full["splits"]["train"]["logits"], ytr, ctx["train"]), tg_tensors(orders["train"]))
    yv = np.array([s["label"] for s in samples["valid"]])
    yt = np.array([s["label"] for s in samples["test"]])
    tgv, tgt = tg_tensors(orders["valid"]), tg_tensors(orders["test"])
    val_folds = [(*tensors(r["splits"]["valid"]["logits"], yv, ctx["valid"]), tgv) for r in runs]
    test_folds = [(*tensors(r["splits"]["test"]["logits"], yt, ctx["test"]), tgt) for r in runs]

    def ens(fn, folds):  # average predictive probabilities over the K cross-fitted models
        p = np.mean([torch.softmax(fn(f), -1).cpu().numpy() for f in folds], 0)
        return np.log(p + 1e-12)

    out = {}
    zv = [f[0].cpu().numpy() for f in val_folds]
    zt = [f[0].cpu().numpy() for f in test_folds]
    out["backbone"] = {"test": metrics(yt, ens(lambda f: f[0], test_folds), k),
                       "single": np.mean([metrics(yt, z, k) for z in zt], 0).tolist()}
    # RLA on the same models, s2 by validation NLL (the published rule)
    rla_t = []
    s2s = []
    for a, b in zip(zv, zt):
        s2, half, _ = select(a, yv, orders["valid"])
        s2s.append(s2)
        rla_t.append(b + adapt(b, yt, orders["test"], s2, half))
    p = np.mean([np.exp(x - x.max(1, keepdims=True)) / np.exp(x - x.max(1, keepdims=True)).sum(1, keepdims=True)
                 for x in rla_t], 0)
    out["rla"] = {"test": metrics(yt, np.log(p), k), "single": np.mean([metrics(yt, x, k) for x in rla_t], 0).tolist()}
    s2_init = float(np.median(s2s))
    preds = {"backbone": ens(lambda f: f[0], test_folds), "rla": np.log(p)}
    for name in configs:
        cfg = dict(CONFIGS[name], s2_init=s2_init)
        if cfg.get("insample") and insample is None:
            continue
        layer, vbest = train_layer(cfg, k, insample if cfg.get("insample") else oof, val_folds, cw)
        with torch.no_grad():
            pt = ens(lambda f: run_layer(layer, *f[:3], f[3]), test_folds)
            single = np.mean([metrics(yt, run_layer(layer, *f[:3], f[3]).cpu().numpy(), k) for f in test_folds], 0)
        out[name] = {"test": metrics(yt, pt, k), "single": single.tolist(), "val_nll": vbest,
                     "s2": float(torch.exp(layer.log_s2).item()) if hasattr(layer, "log_s2") else None}
        preds[name] = pt
        if ORACLE:
            loo = loo_tensors(orders["test"])
            with torch.no_grad():
                po = ens(lambda f: run_layer(layer, *f[:3], loo, bs=256), test_folds)
            out[name]["oracle"] = metrics(yt, po, k) + [float(-po[np.arange(len(yt)), yt].mean())]
            log(f"   oracle {name}: {out[name]['oracle']}")
        if name == "spire":
            out[name]["state"] = {a: b.cpu().numpy().round(5).tolist() for a, b in layer.state_dict().items()}
            out[name]["prior_cov"] = layer.covariance().detach().cpu().numpy().round(5).tolist()
        if full is not None:
            with torch.no_grad():
                zf = run_layer(layer, *tensors(full["splits"]["test"]["logits"], yt, ctx["test"]), tgt)
            out[name]["full"] = metrics(yt, zf.cpu().numpy(), k)
        if emodx is not None:  # plug-in: released EmoDynamiX checkpoint, layer unchanged
            with torch.no_grad():
                ze = run_layer(layer, *tensors(emodx_z["test"][0], yt, ctx["test"]), tgt)
            ix = emodx_z["test"][1]
            out[name]["emodx"] = metrics(yt[ix], ze.cpu().numpy()[ix], k)
        log(f"{dataset} {tag} s{seed} {name:14s} ens {out[name]['test'][0]:6.2f} {out[name]['test'][1]:6.2f} "
            f"{out[name]['test'][2]:5.2f} | single {single[0]:6.2f} | full {out[name].get('full', [0])[0]:6.2f} | emodx {out[name].get('emodx', [0])[0]:6.2f} | val {vbest:.4f}")
    if full is not None:
        zfv, zft = np.array(full["splits"]["valid"]["logits"]), np.array(full["splits"]["test"]["logits"])
        s2, half, _ = select(zfv, yv, orders["valid"])
        out["backbone"]["full"] = metrics(yt, zft, k)
        out["rla"]["full"] = metrics(yt, zft + adapt(zft, yt, orders["test"], s2, half), k)
    if emodx is not None:
        (zev, iv), (zet, ix) = emodx_z["valid"], emodx_z["test"]
        s2, half, _ = select(zev, yv, orders["valid"])
        out["backbone"]["emodx"] = metrics(yt[ix], zet[ix], k)
        out["rla"]["emodx"] = metrics(yt[ix], (zet + adapt(zet, yt, orders["test"], s2, half))[ix], k)
    for key in ("backbone", "rla"):
        log(f"{dataset} {tag} s{seed} {key:14s} ens {out[key]['test'][0]:6.2f} {out[key]['test'][1]:6.2f} "
            f"{out[key]['test'][2]:5.2f} | single {out[key]['single'][0]:6.2f} | full {out[key].get('full', [0])[0]:6.2f} | emodx {out[key].get('emodx', [0])[0]:6.2f}")
    return out, preds, yt, [r["dialog"] for r in samples["test"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="annomi")
    ap.add_argument("--tag", default="roberta-base")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--configs", nargs="+", default=["spire", "no_situation", "learned_rla"])
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--out", default="spire")
    args = ap.parse_args()
    global ORACLE
    ORACLE = args.oracle
    for seed in args.seeds:
        out, preds, yt, dlg = experiment(args.dataset, args.tag, seed, args.configs)
        dst = RESULT / args.out / args.dataset / f"{args.tag}_s{seed}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        probs = {a: np.exp(b).round(4).tolist() for a, b in preds.items() if a in ("backbone", "rla", "spire")}
        dst.write_text(json.dumps({"scores": out, "preds": {a: b.argmax(-1).tolist() for a, b in preds.items()},
                                   "probs": probs,
                                   "y": yt.tolist(), "dialog": dlg}), encoding="utf-8")


if __name__ == "__main__":
    main()
