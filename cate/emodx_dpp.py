"""Parameter-free DPP on top of the released EmoDynamiX logits, with a paired bootstrap."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from cate.config import RESULT
from cate.dpp import DialoguePolicyPosterior, history_counts
from cate.metrics import classification_scores, preference_bias_official
from cate.official import load_official


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="annomi")
    args = ap.parse_args()
    dialogs, samples, k = load_official(args.dataset, check=False)
    dpp = DialoguePolicyPosterior.fit(dialogs, samples["train"], None, k)
    run = json.load(open(RESULT / "policy" / f"{args.dataset}_emodx" / "emodynamix_none_s0.json", encoding="utf-8"))
    out = {}
    for sp in ("valid", "test"):
        d = run["splits"][sp]
        rows = [samples[sp][i] for i in d["index"]]
        with torch.no_grad():
            dpp.log_half.fill_(np.log(1e6))
            sh = dpp(history_counts(dialogs, rows, k)).numpy()
        y, z = np.array(d["y"]), np.array(d["logits"])
        res = {}
        for name, zz in (("EmoDynamiX", z), ("EmoDynamiX + DPP", z + sh)):
            p = zz.argmax(-1)
            s = classification_scores(y, p, k)
            res[name] = (100 * s["macro_f1"], 100 * s["weighted_f1"], preference_bias_official(y, p, k), p)
        if sp == "test":
            rng = np.random.RandomState(0)
            groups = defaultdict(list)
            for n, r in enumerate(rows):
                groups[r["dialog"]].append(n)
            keys = list(groups)
            worse = 0
            pa, pb = res["EmoDynamiX"][3], res["EmoDynamiX + DPP"][3]
            for _ in range(1000):
                idx = np.concatenate([groups[keys[j]] for j in rng.randint(0, len(keys), len(keys))])
                worse += classification_scores(y[idx], pb[idx], k)["macro_f1"] <= classification_scores(y[idx], pa[idx], k)["macro_f1"]
            out["p_bootstrap"] = worse / 1000
        for name, (a, b, c, _) in res.items():
            print(f"{args.dataset} {sp:5s} {name:18s} M-F1 {a:.2f} W-F1 {b:.2f} B {c:.2f}")
            out[f"{sp}:{name}"] = [a, b, c]
    print("paired dialogue bootstrap p =", out["p_bootstrap"])
    (RESULT / "policy" / f"{args.dataset}_emodx" / "emodx_dpp.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
