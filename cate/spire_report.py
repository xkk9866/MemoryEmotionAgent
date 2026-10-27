"""Tables for SPIRE from results/spire: mean ± std over seeds, paired dialogue bootstrap.

p is the share of 1000 dialogue resamples in which the seed-mean macro-F1 of SPIRE
is not above that of system a (one-sided p of SPIRE > a) (cate.rla_report.bootstrap).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict

import numpy as np

from cate.config import RESULT
from cate.rla_report import bootstrap

ORDER = ["backbone", "rla", "learned_rla", "amortized", "no_crossfit", "no_situation", "isotropic",
         "no_moderation", "semantic", "weighted", "spire"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["esconv", "annomi", "p4g"])
    ap.add_argument("--tags", nargs="+", default=["roberta-base", "roberta-large"])
    args = ap.parse_args()
    table = {}
    for ds in args.datasets:
        for tag in args.tags:
            files = sorted(glob.glob(str(RESULT / "spire" / ds / f"{tag}_s*.json")))
            if not files:
                continue
            runs = [json.load(open(f, encoding="utf-8")) for f in files]
            seeds = [int(re.search(r"_s(\d+)\.json", f).group(1)) for f in files]
            y, dlg = np.array(runs[0]["y"]), runs[0]["dialog"]
            k = int(y.max()) + 1
            print(f"\n== {ds} {tag} seeds {seeds}")
            print(f"{'system':14s} {'M-F1':>12s} {'W-F1':>12s} {'B':>10s} | {'single M-F1':>12s} | "
                  f"{'full':>6s} {'emodx':>6s} | p(spire>a)")
            spire_pred = [np.array(r["preds"]["spire"]) for r in runs if "spire" in r["preds"]]
            for name in ORDER:
                if not all(name in r["scores"] for r in runs):
                    continue
                t = np.array([r["scores"][name]["test"] for r in runs])
                sg = np.array([r["scores"][name]["single"] for r in runs])
                fu = [r["scores"][name].get("full") for r in runs]
                em = [r["scores"][name].get("emodx") for r in runs]
                p = ""
                if name != "spire" and len(spire_pred) == len(runs):
                    p = bootstrap(y, dlg, [np.array(r["preds"][name]) for r in runs], spire_pred, k)
                row = {"n": len(runs), "mean": t.mean(0).tolist(), "std": t.std(0).tolist(),
                       "single_mean": sg.mean(0).tolist(), "single_std": sg.std(0).tolist(),
                       "full_mean": np.mean([f for f in fu if f], 0).tolist() if any(fu) else None,
                       "emodx": np.mean([e for e in em if e], 0).tolist() if any(em) else None, "p_vs_spire": p}
                table[f"{ds}:{tag}:{name}"] = row
                fm = f"{row['full_mean'][0]:6.2f}" if row["full_mean"] else "     -"
                ex = f"{row['emodx'][0]:6.2f}" if row["emodx"] else "     -"
                print(f"{name:14s} {t.mean(0)[0]:6.2f}±{t.std(0)[0]:<5.2f} {t.mean(0)[1]:6.2f}±{t.std(0)[1]:<5.2f} "
                      f"{t.mean(0)[2]:4.2f}±{t.std(0)[2]:<4.2f} | {sg.mean(0)[0]:6.2f}±{sg.std(0)[0]:<4.2f} | "
                      f"{fm} {ex} | {p if p == '' else f'{p:.3f}'}")
    (RESULT / "spire" / "tables.json").write_text(json.dumps(table, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
