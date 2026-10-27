"""Seed-averaged tables and paired dialogue bootstrap for the official splits.

Readouts are fixed in advance, none is tuned on test:
  cls   classifier logits (plus the trained shift for --prior dpp runs)
  knn   response-memory readout, balanced: log p_knn(y | x) - log pi_y
  mix   geometric pooling of the two balanced experts: log_softmax(z) + knn - log pi
  +dpp  parameter-free dialogue policy posterior added post hoc (beta 1, no forgetting)
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict

import numpy as np
import torch
from scipy.special import log_softmax

from cate.config import RESULT
from cate.dpp import DialoguePolicyPosterior, history_counts
from cate.metrics import classification_scores, preference_bias_official
from cate.official import load_official

PAPER = {
    "esconv": {"EmoDynamiX (NAACL'25)": (27.70, 32.71, 0.45), "TransESC": (26.28, 31.33, 0.73),
               "MultiESC": (25.73, 29.31, 0.61), "LLaMA3-8B FT": (25.91, 29.82, 0.83), "RoBERTa (paper)": (25.04, 27.94, 0.68)},
    "annomi": {"EmoDynamiX (NAACL'25)": (27.92, 35.33, 0.50), "LLaMA3-8B FT": (23.77, 29.98, 0.81),
               "RoBERTa (paper)": (22.26, 27.25, 0.64)},
    "p4g": {},
}


def metrics(y, z, k):
    p = z.argmax(-1)
    s = classification_scores(y, p, k)
    return 100 * s["macro_f1"], 100 * s["weighted_f1"], preference_bias_official(y, p, k)


def readouts(run, sp, log_pi, dpp_shift):
    d = run["splits"][sp]
    z = np.array(d["logits"])
    if run.get("prior") == "dpp":
        z = z + np.array(d["shift"])
    out = {"cls": z}
    if "knn" in d:
        kn = np.array(d["knn"]) - log_pi
        out["knn"] = kn
        out["mix"] = log_softmax(z, -1) + kn
    for name in list(out):
        if run.get("prior") != "dpp":
            out[name + "+dpp"] = out[name] + dpp_shift
    return out


def bootstrap(y, dialog, preds_a, preds_b, k, n=1000, seed=0):
    """P(mean-over-seeds M-F1 of b <= a) under resampling of test dialogues."""
    rng = np.random.RandomState(seed)
    groups = defaultdict(list)
    for i, d in enumerate(dialog):
        groups[d].append(i)
    keys = list(groups)
    worse = 0
    for _ in range(n):
        idx = np.concatenate([groups[keys[j]] for j in rng.randint(0, len(keys), len(keys))])
        da = np.mean([classification_scores(y[idx], p[idx], k)["macro_f1"] for p in preds_a])
        db = np.mean([classification_scores(y[idx], p[idx], k)["macro_f1"] for p in preds_b])
        worse += db <= da
    return worse / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esconv")
    ap.add_argument("--base", default="roberta-base_none", help="run prefix used as the bootstrap reference")
    args = ap.parse_args()
    dialogs, samples, k = load_official(args.dataset, check=False)
    dpp = DialoguePolicyPosterior.fit(dialogs, samples["train"], history_counts(dialogs, samples["train"], k), k)
    log_pi = dpp.log_pi.numpy()
    with torch.no_grad():
        dpp.log_half.fill_(np.log(1e6))
        shift = {sp: dpp(history_counts(dialogs, samples[sp], k)).numpy() for sp in ("valid", "test")}
    y = np.array([s["label"] for s in samples["test"]])
    dialog = [s["dialog"] for s in samples["test"]]

    runs = defaultdict(dict)
    for f in sorted(glob.glob(str(RESULT / "policy" / args.dataset / "*.json"))):
        m = re.match(r"(.+)_s(\d+)\.json$", f.replace("\\", "/").split("/")[-1])
        runs[m.group(1)][int(m.group(2))] = json.load(open(f, encoding="utf-8"))

    table, preds = {}, {}
    for prefix, by_seed in runs.items():
        per = defaultdict(lambda: {"test": [], "valid": [], "pred": []})
        for seed, run in sorted(by_seed.items()):
            for sp in ("valid", "test"):
                for name, z in readouts(run, sp, log_pi, shift[sp]).items():
                    per[name][sp].append(metrics(np.array(run["splits"][sp]["y"]), z, k))
                    if sp == "test":
                        per[name]["pred"].append(z.argmax(-1))
        for name, v in per.items():
            key = f"{prefix}:{name}"
            table[key] = {"n_seeds": len(v["test"]), "test_mean": np.mean(v["test"], 0).tolist(),
                          "test_std": np.std(v["test"], 0).tolist(), "valid_mean": np.mean(v["valid"], 0).tolist()}
            preds[key] = v["pred"]

    ref = f"{args.base}:cls"
    print(f"{'system':42s} {'seeds':>5s} {'M-F1':>13s} {'W-F1':>13s} {'B':>11s} {'dev M-F1':>8s} {'p<base':>7s}")
    for name, (a, b, c) in PAPER[args.dataset].items():
        print(f"{name:42s} {'-':>5s} {a:13.2f} {b:13.2f} {c:11.2f}")
    for key, r in sorted(table.items()):
        m, s = r["test_mean"], r["test_std"]
        p = ""
        if ref in preds and key != ref and len(preds[key]) == len(preds[ref]):
            p = f"{bootstrap(y, dialog, preds[ref], preds[key], k):.3f}"
            r["p_vs_base"] = float(p)
        print(f"{key:42s} {r['n_seeds']:5d} {m[0]:6.2f}±{s[0]:<5.2f} {m[1]:6.2f}±{s[1]:<5.2f} "
              f"{m[2]:5.2f}±{s[2]:<4.2f} {r['valid_mean'][0]:8.2f} {p:>7s}")
    (RESULT / "policy" / f"table_{args.dataset}.json").write_text(json.dumps(table, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
