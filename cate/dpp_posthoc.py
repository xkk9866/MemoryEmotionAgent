"""Apply the dialogue policy posterior to frozen logits of any model.

Only kappa (method of moments, from train) and the half-life / beta (grid on
valid) are chosen; the test split is touched once. An oracle row uses the
leave-one-out strategy distribution of the whole dialogue, future included, to
bound how much of the label entropy is dialogue-specific policy.
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import torch

from cate.dpp import DialoguePolicyPosterior, history_counts
from cate.metrics import classification_scores, preference_bias_official as preference_bias
from cate.official import load_official


def scores(y, z, k):
    p = z.argmax(-1)
    s = classification_scores(y, p, k)
    s["bias"] = preference_bias(y, p, k)
    return s


def oracle_shift(dialogs, rows, k, kappa, log_pi):
    out = np.zeros((len(rows), k))
    pi = np.exp(log_pi)
    for i, r in enumerate(rows):
        c = np.bincount([t["strategy"] for t in dialogs[r["dialog"]]["turns"] if t["strategy"] >= 0], minlength=k)
        c = c.astype(np.float64)
        c[r["label"]] -= 1
        out[i] = np.log(c + kappa * pi) - np.log(c.sum() + kappa) - log_pi
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/policy/<dataset>/<run>.json with prior none")
    ap.add_argument("--dataset", default="esconv")
    ap.add_argument("--skip_window", action="store_true")
    args = ap.parse_args()
    run = json.load(open(args.run, encoding="utf-8"))
    dialogs, samples, k = load_official(args.dataset, check=False)
    dpp = DialoguePolicyPosterior.fit(dialogs, samples["train"], history_counts(dialogs, samples["train"], k), k,
                                      skip_window=args.skip_window)
    z = {sp: np.array(run["splits"][sp]["logits"]) for sp in ("valid", "test")}
    y = {sp: np.array(run["splits"][sp]["y"]) for sp in ("valid", "test")}
    cnt = {sp: history_counts(dialogs, samples[sp], k) for sp in ("valid", "test")}
    log_pi = dpp.log_pi.numpy()

    def shift(sp, half, beta):
        with torch.no_grad():
            dpp.log_half.fill_(np.log(half))
            dpp.beta.fill_(beta)
            return dpp(cnt[sp]).numpy()

    grid = list(itertools.product([5, 10, 20, 40, 1e4], [0.5, 1.0, 1.5, 2.0]))
    best = max(grid, key=lambda g: scores(y["valid"], z["valid"] + shift("valid", *g), k)["macro_f1"])
    rows = {
        "text": scores(y["test"], z["test"], k),
        "text-pi (balanced)": scores(y["test"], z["test"] - log_pi, k),
        "dpp beta=1 half=inf": scores(y["test"], z["test"] + shift("test", 1e4, 1.0), k),
        f"dpp tuned half={best[0]} beta={best[1]}": scores(y["test"], z["test"] + shift("test", *best), k),
    }
    kappa = float(dpp.log_kappa.exp())
    rows["oracle dialogue policy"] = scores(y["test"], z["test"] + oracle_shift(dialogs, samples["test"], k, kappa, log_pi), k)
    print(f"kappa {kappa:.2f}")
    for name, s in rows.items():
        print(f"{name:40s} M-F1 {s['macro_f1']:.4f} W-F1 {s['weighted_f1']:.4f} Acc {s['accuracy']:.4f} B {s['bias']:.3f}")


if __name__ == "__main__":
    main()
