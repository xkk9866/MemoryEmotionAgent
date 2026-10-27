"""Hierarchical revealed-policy inference (HRPI) and every comparison system.

Two levels of session evidence, both from the supporter's revealed strategies only:
  1. Strategies revealed OUTSIDE the backbone's window update a conjugate Dirichlet
     posterior over the session's strategy prior, theta_d ~ Dir(kappa pi), kappa by
     empirical Bayes (moment estimate on training sessions). Its log ratio to pi shifts
     the logits; strategies inside the window are excluded because the backbone has
     already read them (no double counting).
  2. A residual random effect, inferred by Laplace-approximate MAP from the errors that
     remain on all revealed turns, with prior variance chosen by validation NLL.
No recency decay and no tuned constant beyond the validation-chosen variance.
Systems share the same five cross-fitted fold classifiers; predictives are averaged.
"""

from __future__ import annotations

import glob
import json
import re
from collections import defaultdict

import numpy as np

from cate.calib_baselines import fit_map, softmax
from cate.config import RESULT
from cate.dpp import DialoguePolicyPosterior, history_counts
from cate.external_baselines import m3
from cate.official import load_official
from cate.rla import adapt, dialogue_order, select
from cate.rla_report import bootstrap
from cate.spire import load_crossfit

SETTINGS = [("esconv", "roberta-base"), ("esconv", "roberta-large"), ("annomi", "roberta-base"),
            ("annomi", "roberta-large"), ("p4g", "roberta-base")]


def count_shift(dialogs, samples, k, out_of_window):
    dpp = DialoguePolicyPosterior.fit(dialogs, samples["train"], None, k, half_life=1e9, skip_window=out_of_window)
    return {sp: dpp(history_counts(dialogs, samples[sp], k)).detach().numpy() for sp in ("valid", "test")}


def residual(zv, zt, yv, yt, ov, ot):
    s2, half, _ = select(zv, yv, ov)
    return zt + adapt(zt, yt, ot, s2, half)


def main():
    out = {}
    for ds, tag in SETTINGS:
        dialogs, samples, k = load_official(ds, check=False)
        yv = np.array([s["label"] for s in samples["valid"]])
        yt = np.array([s["label"] for s in samples["test"]])
        ov, ot = dialogue_order(samples["valid"]), dialogue_order(samples["test"])
        dlg = [s["dialog"] for s in samples["test"]]
        c_all, c_out = count_shift(dialogs, samples, k, False), count_shift(dialogs, samples, k, True)
        preds = defaultdict(list)
        for f in sorted(glob.glob(str(RESULT / "spire" / ds / f"{tag}_s*.json"))):
            seed = int(re.search(r"_s(\d+)\.json", f).group(1))
            saved = json.load(open(f, encoding="utf-8"))["preds"]
            probs = defaultdict(list)
            for r in load_crossfit(ds, tag, seed):
                zv, zt = np.array(r["splits"]["valid"]["logits"]), np.array(r["splits"]["test"]["logits"])
                probs["ensemble"].append(softmax(zt))
                probs["+TS"].append(softmax(fit_map(zv, yv, False)(zt)))
                probs["+label shift"].append(softmax(zt + c_all["test"]))
                probs["+residual (RPI)"].append(softmax(residual(zv, zt, yv, yt, ov, ot)))
                probs["HRPI (ours)"].append(softmax(residual(zv + c_out["valid"], zt + c_out["test"], yv, yt, ov, ot)))
                probs["HRPI w/o window exclusion"].append(
                    softmax(residual(zv + c_all["valid"], zt + c_all["test"], yv, yt, ov, ot)))
            for a, v in probs.items():
                preds[a].append(np.mean(v, 0).argmax(1))
            preds["+neural process"].append(np.array(saved["amortized"]))
            preds["+SPIRE"].append(np.array(saved["spire"]))
        res = {}
        for name, ps in preds.items():
            v = np.array([m3(yt, p, k) for p in ps])
            p = "" if name == "HRPI (ours)" else bootstrap(yt, dlg, ps, preds["HRPI (ours)"], k)
            res[name] = {"mean": v.mean(0).tolist(), "std": v.std(0).tolist(), "p": p}
            print(f"{ds:7s} {tag:13s} {name:27s} {v.mean(0)[0]:6.2f}±{v.std(0)[0]:.2f} {v.mean(0)[1]:6.2f} "
                  f"{v.mean(0)[2]:5.2f} p={p if p == '' else f'{p:.3f}'}", flush=True)
        out[f"{ds}:{tag}"] = res
        (RESULT / "hrpi.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
