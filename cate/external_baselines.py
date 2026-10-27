"""Our method against standard alternatives on the same cross-fitted fold models.

Ours (RPI): revealed-policy inference with the prior learned by cross-fitted prequential
evidence (config learned_rla in cate.spire; its decisions are in results/spire/*.json).
Alternatives, all on the same five fold classifiers with averaged predictive:
  ensemble; + temperature / vector scaling (fitted on validation);
  + label-shift count posterior (Saerens-style Dirichlet prior over session counts);
  + attentive neural process (config amortized); + SPIRE (structured prior).
p: one-sided paired dialogue bootstrap of the seed-mean macro-F1 of RPI vs. each.
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
from cate.metrics import classification_scores, preference_bias_official
from cate.official import load_official
from cate.rla_report import bootstrap
from cate.spire import load_crossfit


def m3(y, pred, k):
    s = classification_scores(y, pred, k)
    return [100 * s["macro_f1"], 100 * s["weighted_f1"], preference_bias_official(y, pred, k)]


def main():
    out = {}
    for ds in ("esconv", "annomi", "p4g"):
        dialogs, samples, k = load_official(ds, check=False)
        dpp = DialoguePolicyPosterior.fit(dialogs, samples["train"], None, k)
        shift = dpp(history_counts(dialogs, samples["test"], k)).detach().numpy()
        yv = np.array([s["label"] for s in samples["valid"]])
        yt = np.array([s["label"] for s in samples["test"]])
        dlg = [s["dialog"] for s in samples["test"]]
        for tag in ("roberta-base", "roberta-large"):
            files = sorted(glob.glob(str(RESULT / "spire" / ds / f"{tag}_s*.json")))
            if not files:
                continue
            preds = defaultdict(list)
            for f in files:
                seed = int(re.search(r"_s(\d+)\.json", f).group(1))
                saved = json.load(open(f, encoding="utf-8"))["preds"]
                probs = defaultdict(list)
                for r in load_crossfit(ds, tag, seed):
                    zv, zt = np.array(r["splits"]["valid"]["logits"]), np.array(r["splits"]["test"]["logits"])
                    probs["ensemble"].append(softmax(zt))
                    probs["+TS"].append(softmax(fit_map(zv, yv, False)(zt)))
                    probs["+VS"].append(softmax(fit_map(zv, yv, True)(zt)))
                    probs["+label shift"].append(softmax(zt + shift))
                for a, v in probs.items():
                    preds[a].append(np.mean(v, 0).argmax(1))
                for a, name in (("amortized", "+neural process"), ("spire", "+SPIRE"), ("learned_rla", "RPI (ours)")):
                    preds[name].append(np.array(saved[a]))
            res = {}
            for name, ps in preds.items():
                v = np.array([m3(yt, p, k) for p in ps])
                p = "" if name == "RPI (ours)" else bootstrap(yt, dlg, ps, preds["RPI (ours)"], k)
                res[name] = {"mean": v.mean(0).tolist(), "std": v.std(0).tolist(), "p": p}
                print(f"{ds:7s} {tag:13s} {name:16s} {v.mean(0)[0]:6.2f}±{v.std(0)[0]:.2f} {v.mean(0)[1]:6.2f} "
                      f"{v.mean(0)[2]:5.2f}  p={p if p == '' else f'{p:.3f}'}", flush=True)
            out[f"{ds}:{tag}"] = res
    (RESULT / "spire" / "external_baselines.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
