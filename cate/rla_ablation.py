"""Ablations of revealed-label adaptation on frozen logits, each against full RLA.

  none        the backbone alone
  global      one offset for all dialogues, fit on validation (recalibration, not personal)
  mle         no prior: s2 = 1e3, i.e. the per-dialogue maximum-likelihood offset
  pseudo      the backbone's own earlier predictions replace the revealed labels
  count       the count posterior of cate.dpp instead of the residual offset
  rla         offset prior N(0, s2 I), s2 by validation NLL
  hier        hierarchical prior N(mu, s2 I): mu = population offset, s2 for the
              supporter's deviation, both from validation dialogues only

p: paired dialogue bootstrap of seed-mean M-F1, "variant >= rla".
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict

import numpy as np
import torch
from scipy.special import log_softmax, softmax

from cate.config import RESULT
from cate.dpp import DialoguePolicyPosterior, history_counts
from cate.metrics import classification_scores, preference_bias_official
from cate.official import load_official
from cate.rla import adapt, dialogue_order, select


def global_offset(z, y, s2=10.0, newton=20):
    k = z.shape[1]
    b, eye = np.zeros(k), np.eye(k)
    for _ in range(newton):
        p = softmax(z + b, -1)
        g = (p - eye[y]).sum(0) + b / s2
        h = np.diag(p.sum(0)) - p.T @ p + eye / s2
        b = b - np.linalg.solve(h, g)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esconv")
    ap.add_argument("--prefix", default="roberta-large_none")
    args = ap.parse_args()
    dialogs, samples, k = load_official(args.dataset)
    order = {sp: dialogue_order(samples[sp]) for sp in ("valid", "test")}
    dpp = DialoguePolicyPosterior.fit(dialogs, samples["train"], None, k)
    with torch.no_grad():
        dpp.log_half.fill_(np.log(1e6))
        count_shift = {sp: dpp(history_counts(dialogs, samples[sp], k)).numpy() for sp in ("valid", "test")}
    runs = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(str(RESULT / "policy" / args.dataset / f"{args.prefix}_s*.json")))]
    runs = [r for r in runs if r.get("best_dev", 1.0) >= 0.15][:3]
    preds, mets = defaultdict(list), defaultdict(list)
    for r in runs:
        zv, yv = np.array(r["splits"]["valid"]["logits"]), np.array(r["splits"]["valid"]["y"])
        zt, yt = np.array(r["splits"]["test"]["logits"]), np.array(r["splits"]["test"]["y"])
        s2, half, _ = select(zv, yv, order["valid"])
        pseudo = zt.argmax(-1)
        mu = global_offset(zv, yv)
        s2h, halfh, _ = select(zv + mu, yv, order["valid"])
        variants = {
            "none": zt,
            "global": zt + global_offset(zv, yv),
            "count": zt + count_shift["test"],
            "mle": zt + adapt(zt, yt, order["test"], 1e3, half),
            "pseudo": zt + adapt(zt, pseudo, order["test"], s2, half),
            "rla": zt + adapt(zt, yt, order["test"], s2, half),
            "hier": zt + mu + adapt(zt + mu, yt, order["test"], s2h, halfh),
        }
        for name, z in variants.items():
            p = z.argmax(-1)
            s = classification_scores(yt, p, k)
            mets[name].append((100 * s["macro_f1"], 100 * s["weighted_f1"], preference_bias_official(yt, p, k)))
            preds[name].append(p)
    y = np.array(runs[0]["splits"]["test"]["y"])
    groups = defaultdict(list)
    for i, s in enumerate(samples["test"]):
        groups[s["dialog"]].append(i)
    keys = list(groups)
    rng = np.random.RandomState(0)
    boots = [np.concatenate([groups[keys[j]] for j in rng.randint(0, len(keys), len(keys))]) for _ in range(1000)]
    mf = lambda ps, idx: np.mean([classification_scores(y[idx], p[idx], k)["macro_f1"] for p in ps])
    ref = np.array([mf(preds["rla"], idx) for idx in boots])
    out = {}
    print(f"{args.dataset} {args.prefix} ({len(runs)} seeds)")
    for name in ("none", "global", "count", "mle", "pseudo", "rla", "hier"):
        m = np.mean(mets[name], 0)
        p = "" if name == "rla" else f"{np.mean(np.array([mf(preds[name], idx) for idx in boots]) >= ref):.3f}"
        out[name] = {"mean": m.tolist(), "std": np.std(mets[name], 0).tolist(), "p_ge_rla": p}
        print(f"  {name:7s} M-F1 {m[0]:6.2f}±{np.std(mets[name], 0)[0]:.2f}  W-F1 {m[1]:6.2f}  B {m[2]:.2f}  p(>=RLA) {p}")
    (RESULT / "policy" / f"ablation_{args.dataset}_{args.prefix}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
