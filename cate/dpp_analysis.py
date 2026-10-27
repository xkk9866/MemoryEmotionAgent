"""Checks of the two predictions DPP makes, on frozen logits of a run family.

1. The gain grows with the number N of earlier supporter actions in the dialogue.
2. The method-of-moments kappa sits near the best kappa (scanned on valid and test).
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import torch

from cate.config import RESULT
from cate.dpp import DialoguePolicyPosterior, history_counts
from cate.metrics import classification_scores
from cate.official import load_official

BUCKETS = [(0, 1, "0"), (1, 5, "1-4"), (5, 15, "5-14"), (15, 30, "15-29"), (30, 10_000, "30+")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="annomi")
    ap.add_argument("--prefix", default="roberta-base_none")
    args = ap.parse_args()
    dialogs, samples, k = load_official(args.dataset, check=False)
    dpp = DialoguePolicyPosterior.fit(dialogs, samples["train"], history_counts(dialogs, samples["train"], k), k)
    kappa0 = float(dpp.log_kappa.exp())
    cnt = {sp: history_counts(dialogs, samples[sp], k) for sp in ("valid", "test")}
    runs = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(str(RESULT / "policy" / args.dataset / f"{args.prefix}_s*.json")))]

    def shift(sp, kappa):
        with torch.no_grad():
            dpp.log_half.fill_(np.log(1e6))
            dpp.log_kappa.fill_(np.log(kappa))
            return dpp(cnt[sp]).numpy()

    def mf1(y, z, mask=None):
        if mask is not None:
            y, z = y[mask], z[mask]
        return 100 * classification_scores(y, z.argmax(-1), k)["macro_f1"]

    out = {"dataset": args.dataset, "kappa_mom": kappa0, "n_seeds": len(runs), "by_history": {}, "kappa_scan": {}}
    n_hist = (cnt["test"]["strat"] >= 0).sum(1).numpy()
    sh = shift("test", kappa0)
    print(f"{args.dataset}: kappa (moments) {kappa0:.2f}, {len(runs)} seeds")
    print(f"{'past actions N':>15s} {'n':>5s} {'text':>7s} {'+DPP':>7s} {'gain':>6s}")
    for lo, hi, name in BUCKETS:
        m = (n_hist >= lo) & (n_hist < hi)
        if m.sum() < 30:
            continue
        a = np.mean([mf1(np.array(r["splits"]["test"]["y"]), np.array(r["splits"]["test"]["logits"]), m) for r in runs])
        b = np.mean([mf1(np.array(r["splits"]["test"]["y"]), np.array(r["splits"]["test"]["logits"]) + sh, m) for r in runs])
        out["by_history"][name] = {"n": int(m.sum()), "text": a, "dpp": b}
        print(f"{name:>15s} {int(m.sum()):5d} {a:7.2f} {b:7.2f} {b - a:+6.2f}")
    print(f"{'kappa':>8s} {'valid':>7s} {'test':>7s}")
    for mult in [1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 64]:
        kap = kappa0 * mult
        row = {}
        for sp in ("valid", "test"):
            s = shift(sp, kap)
            row[sp] = np.mean([mf1(np.array(r["splits"][sp]["y"]), np.array(r["splits"][sp]["logits"]) + s) for r in runs])
        out["kappa_scan"][f"{kap:.2f}"] = row
        print(f"{kap:8.2f} {row['valid']:7.2f} {row['test']:7.2f}{'  <- moments' if mult == 1 else ''}")
    (RESULT / "policy" / f"dpp_analysis_{args.dataset}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
