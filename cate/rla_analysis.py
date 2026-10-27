"""Where revealed-label adaptation helps.

1. Gain by the number of revealed earlier supporter labels in the dialogue.
2. Oracle: the offset fit on all other labels of the same dialogue, future
   included (leave-one-out), bounds what a per-supporter logit offset can give.
3. Per-dataset policy heterogeneity (moment kappa) next to the gain.
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np
from scipy.special import softmax

from cate.config import RESULT
from cate.dpp import moment_kappa
from cate.metrics import classification_scores
from cate.official import load_official
from cate.rla import adapt, dialogue_order, select

BUCKETS = [(0, 1, "0"), (1, 4, "1-3"), (4, 8, "4-7"), (8, 16, "8-15"), (16, 10_000, "16+")]


def oracle(z, y, order, s2, newton=8):
    """Leave-one-out offset from every other labeled turn of the same dialogue."""
    k = z.shape[1]
    eye = np.eye(k)
    out = np.zeros_like(z)
    for rows in order:
        for i in rows:
            other = [r for r in rows if r != i]
            b = np.zeros(k)
            for _ in range(newton):
                p = softmax(z[other] + b, -1)
                g = (p - eye[y[other]]).sum(0) + b / s2
                h = np.diag(p.sum(0)) - p.T @ p + eye / s2
                b = b - np.linalg.solve(h, g)
            out[i] = b
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esconv")
    ap.add_argument("--prefix", default="roberta-base_resp1")
    args = ap.parse_args()
    dialogs, samples, k = load_official(args.dataset, check=False)
    kappa, _ = moment_kappa(dialogs, k, sorted({r["dialog"] for r in samples["test"]}))
    runs = [json.load(open(f, encoding="utf-8"))
            for f in sorted(glob.glob(str(RESULT / "policy" / args.dataset / f"{args.prefix}_s*.json")))]
    order = {sp: dialogue_order(samples[sp]) for sp in ("valid", "test")}
    pos = np.zeros(len(samples["test"]), dtype=int)
    for rows in order["test"]:
        for j, i in enumerate(rows):
            pos[i] = j
    f1 = lambda y, z, m=None: 100 * classification_scores(y if m is None else y[m], (z if m is None else z[m]).argmax(-1), k)["macro_f1"]
    acc = {"text": [], "rla": [], "oracle": [], "buckets": {b[2]: [] for b in BUCKETS}}
    for run in runs:
        yv, yt = np.array(run["splits"]["valid"]["y"]), np.array(run["splits"]["test"]["y"])
        zv, zt = np.array(run["splits"]["valid"]["logits"]), np.array(run["splits"]["test"]["logits"])
        s2, half, _ = select(zv, yv, order["valid"])
        b = adapt(zt, yt, order["test"], s2, half)
        acc["text"].append(f1(yt, zt))
        acc["rla"].append(f1(yt, zt + b))
        acc["oracle"].append(f1(yt, zt + oracle(zt, yt, order["test"], s2)))
        for lo, hi, name in BUCKETS:
            m = (pos >= lo) & (pos < hi)
            acc["buckets"][name].append((int(m.sum()), f1(yt, zt, m), f1(yt, zt + b, m)))
    out = {"dataset": args.dataset, "prefix": args.prefix, "kappa_test": kappa,
           "text": float(np.mean(acc["text"])), "rla": float(np.mean(acc["rla"])), "oracle": float(np.mean(acc["oracle"]))}
    print(f"{args.dataset} {args.prefix}: kappa(test) {kappa:.1f}  text {out['text']:.2f}  RLA {out['rla']:.2f}  "
          f"oracle {out['oracle']:.2f}  (RLA recovers {100 * (out['rla'] - out['text']) / max(1e-9, out['oracle'] - out['text']):.0f}% of oracle gain)")
    print(f"{'revealed labels':>16s} {'n':>5s} {'text':>6s} {'RLA':>6s} {'gain':>6s}")
    out["buckets"] = {}
    for name, vals in acc["buckets"].items():
        n, a, b = vals[0][0], np.mean([v[1] for v in vals]), np.mean([v[2] for v in vals])
        out["buckets"][name] = {"n": n, "text": a, "rla": b}
        print(f"{name:>16s} {n:5d} {a:6.2f} {b:6.2f} {b - a:+6.2f}")
    (RESULT / "policy" / f"rla_analysis_{args.dataset}_{args.prefix}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
