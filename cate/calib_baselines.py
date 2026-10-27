"""Generic calibration baselines on the same cross-fitted models.

For every fold model: temperature scaling (TS) and vector scaling (VS, per-class
scale and bias) are fitted on its validation logits by NLL, then applied to its
test logits; predictive distributions are averaged over the folds, as for SPIRE.
RLA+TS / RLA+VS fit the same maps on RLA-adapted validation logits.
Compared against SPIRE's saved predictive (results/spire/*.json, "probs").
"""

from __future__ import annotations

import glob
import json
import re

import numpy as np
import torch

from cate.config import RESULT
from cate.metrics import classification_scores, preference_bias_official
from cate.official import load_official
from cate.rla import adapt, dialogue_order, select
from cate.spire import load_crossfit
from cate.spire_analysis import ece


def fit_map(z, y, vector):
    z, y = torch.tensor(z, dtype=torch.float64), torch.tensor(y)
    k = z.shape[1]
    a = torch.zeros(k if vector else 1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(k, dtype=torch.float64, requires_grad=vector)
    opt = torch.optim.LBFGS([a, b] if vector else [a], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z * torch.exp(a) + b, y)
        loss.backward()
        return loss

    opt.step(closure)
    a, b = torch.exp(a).detach().numpy(), b.detach().numpy()
    return lambda x: x * a + b


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def scores(y, p, k):
    pred = p.argmax(1)
    s = classification_scores(y, pred, k)
    return {"mf1": 100 * s["macro_f1"], "wf1": 100 * s["weighted_f1"], "bias": preference_bias_official(y, pred, k),
            "nll": float(-np.log(p[np.arange(len(y)), y] + 1e-9).mean()), "ece": float(ece(p, y))}


def main():
    out = {}
    for ds in ("esconv", "annomi", "p4g"):
        _, samples, k = load_official(ds, check=False)
        yv = np.array([s["label"] for s in samples["valid"]])
        yt = np.array([s["label"] for s in samples["test"]])
        ov, ot = dialogue_order(samples["valid"]), dialogue_order(samples["test"])
        for tag in ("roberta-base", "roberta-large"):
            files = sorted(glob.glob(str(RESULT / "spire" / ds / f"{tag}_s*.json")))
            if not files:
                continue
            acc = {}
            for f in files:
                seed = int(re.search(r"_s(\d+)\.json", f).group(1))
                runs = load_crossfit(ds, tag, seed)
                probs = {}
                for r in runs:
                    zv, zt = np.array(r["splits"]["valid"]["logits"]), np.array(r["splits"]["test"]["logits"])
                    s2, half, _ = select(zv, yv, ov)
                    rv, rt = zv + adapt(zv, yv, ov, s2, half), zt + adapt(zt, yt, ot, s2, half)
                    cand = {"backbone": zt, "backbone+TS": fit_map(zv, yv, False)(zt),
                            "backbone+VS": fit_map(zv, yv, True)(zt), "rla": rt,
                            "rla+TS": fit_map(rv, yv, False)(rt), "rla+VS": fit_map(rv, yv, True)(rt)}
                    for name, z in cand.items():
                        probs.setdefault(name, []).append(softmax(z))
                spire = json.load(open(f, encoding="utf-8"))["probs"]["spire"]
                probs = {a: np.mean(b, 0) for a, b in probs.items()}
                probs["spire"] = np.array(spire) / np.array(spire).sum(1, keepdims=True)
                for name, p in probs.items():
                    acc.setdefault(name, []).append(scores(yt, p, k))
            res = {a: {m: float(np.mean([x[m] for x in v])) for m in v[0]} for a, v in acc.items()}
            out[f"{ds}:{tag}"] = res
            print(f"== {ds} {tag}")
            for a, v in res.items():
                print(f"  {a:12s} M-F1 {v['mf1']:6.2f} W-F1 {v['wf1']:6.2f} B {v['bias']:.2f} "
                      f"NLL {v['nll']:.3f} ECE {v['ece']:.3f}", flush=True)
    (RESULT / "spire" / "calib_baselines.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
