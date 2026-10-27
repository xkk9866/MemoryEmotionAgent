"""Analyses for the SPIRE paper, from results/spire only.

1. Macro-F1 by number of revealed supporter turns (buckets), backbone / RLA / SPIRE.
2. Calibration: test NLL and ECE (15 bins) of the ensemble predictive.
3. Learned prior: sigma^2 per configuration, and the strategy correlation of S_k.
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np

from cate.config import RESULT
from cate.metrics import classification_scores
from cate.official import load_official

BUCKETS = [(0, 0), (1, 3), (4, 7), (8, 15), (16, 10**6)]


def ece(p, y, bins=15):
    conf, hit, e = p.max(1), p.argmax(1) == y, 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["esconv", "annomi", "p4g"])
    ap.add_argument("--tags", nargs="+", default=["roberta-base", "roberta-large"])
    args = ap.parse_args()
    out = {}
    for ds in args.datasets:
        dialogs, samples, k = load_official(ds, check=False)
        nrev = np.array([sum(1 for x in dialogs[r["dialog"]]["turns"][: r["t"]] if x["strategy"] >= 0)
                         for r in samples["test"]])
        for tag in args.tags:
            files = sorted(glob.glob(str(RESULT / "spire" / ds / f"{tag}_s*.json")))
            if not files:
                continue
            runs = [json.load(open(f, encoding="utf-8")) for f in files]
            y = np.array(runs[0]["y"])
            res = {"buckets": {}, "calibration": {}, "sigma2": {}}
            for lo, hi in BUCKETS:
                m = (nrev >= lo) & (nrev <= hi)
                row = {"n": int(m.sum())}
                for sysname in ("backbone", "rla", "spire"):
                    row[sysname] = float(np.mean([100 * classification_scores(y[m], np.array(r["preds"][sysname])[m], k)
                                                  ["macro_f1"] for r in runs]))
                res["buckets"][f"{lo}-{hi if hi < 10**6 else 'inf'}"] = row
            for sysname in ("backbone", "rla", "spire"):
                nll, ec = [], []
                for r in runs:
                    p = np.array(r["probs"][sysname])
                    p = p / p.sum(1, keepdims=True)
                    nll.append(-np.log(p[np.arange(len(y)), y] + 1e-9).mean())
                    ec.append(ece(p, y))
                res["calibration"][sysname] = {"nll": float(np.mean(nll)), "ece": float(np.mean(ec))}
            for name in runs[0]["scores"]:
                v = [r["scores"][name].get("s2") for r in runs]
                if all(x is not None for x in v):
                    res["sigma2"][name] = float(np.mean(v))
            cov = np.mean([np.array(r["scores"]["spire"]["prior_cov"]) for r in runs], 0)
            m = cov.shape[0] // k
            sk = cov[::m, ::m]  # the constant (base-preference) block: sigma2 * Sk * Su[0,0]
            d = np.sqrt(np.diag(sk))
            res["strategy_corr"] = (sk / d[:, None] / d[None, :]).round(3).tolist()
            out[f"{ds}:{tag}"] = res
            print(f"\n== {ds} {tag}")
            for b, row in res["buckets"].items():
                print(f"  revealed {b:>6s} n={row['n']:4d} backbone {row['backbone']:6.2f} rla {row['rla']:6.2f} "
                      f"spire {row['spire']:6.2f}")
            for s, c in res["calibration"].items():
                print(f"  {s:8s} NLL {c['nll']:.3f} ECE {c['ece']:.3f}")
            print("  sigma2", {a: round(b, 3) for a, b in res["sigma2"].items()})
    (RESULT / "spire" / "analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
