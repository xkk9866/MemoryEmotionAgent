"""Reviewer analyses for HRPI, on the saved cross-fitted logits (no retraining).

  variants   full HRPI; level-1 only; level-2 only (RPI); disjoint HRPI (the residual level
             sees only revealed turns whose labels are NOT in the count, i.e. in-window
             turns, so no observation enters both levels)
  history    gold revealed labels vs. labels predicted by the ensemble vs. gold with
             10/20/30% of revealed labels replaced uniformly at random (both levels see
             the same corrupted history; evaluation always against gold)
  quality    M-F1, W-F1, B, NLL, ECE; per-class F1
  stats      paired dialogue bootstrap of the seed-mean M-F1 difference HRPI - other:
             95% CI, two-sided p, Holm-adjusted p within each setting
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict

import numpy as np

from cate.calib_baselines import softmax
from cate.config import RESULT
from cate.dpp import DialoguePolicyPosterior, history_counts
from cate.hrpi import SETTINGS
from cate.metrics import classification_scores, preference_bias_official
from cate.official import WINDOW, load_official
from cate.rla import adapt, dialogue_order, select
from cate.spire import load_crossfit
from cate.spire_analysis import ece

NOISE = (0.1, 0.2, 0.3)


def shift_of(dialogs, samples, k, sp):
    dpp = DialoguePolicyPosterior.fit(dialogs, samples["train"], None, k, half_life=1e9, skip_window=True)
    return dpp(history_counts(dialogs, samples[sp], k)).detach().numpy()


def corrupt(dialogs, samples, k, labels, rng=None, q=0.0):
    """Copy of dialogs whose test-split supporter strategies are replaced by `labels`
    (predicted history) or corrupted with rate q; returns (dialogs, revealed y)."""
    d2 = copy.deepcopy(dialogs)
    y = np.array([s["label"] for s in samples["test"]])
    new = labels.copy() if labels is not None else y.copy()
    if q > 0:
        flip = rng.rand(len(y)) < q
        new[flip] = (y[flip] + rng.randint(1, k, flip.sum())) % k
    for s, v in zip(samples["test"], new):
        d2[s["dialog"]]["turns"][s["t"]]["strategy"] = int(v)
    return d2, new


def disjoint_adapt(z, y, rows_t, order, s2):
    """Residual offset from in-window revealed turns only (those the count excludes)."""
    out = np.zeros_like(z)
    for rows in order:
        for j in range(1, len(rows)):
            past = [p for p in rows[:j] if rows_t[rows[j]] - rows_t[p] <= WINDOW]
            if not past:
                continue
            zz = np.vstack([z[past], np.zeros((1, z.shape[1]))])
            out[rows[j]] = adapt(zz, np.append(y[past], 0), [list(range(len(past) + 1))], s2)[-1]
    return out


def scores(y, p, k):
    pred = p.argmax(1)
    s = classification_scores(y, pred, k)
    per = [100 * classification_scores((y == c).astype(int), (pred == c).astype(int), 2)["macro_f1"] for c in range(k)]
    return {"mf1": 100 * s["macro_f1"], "wf1": 100 * s["weighted_f1"], "bias": preference_bias_official(y, pred, k),
            "nll": float(-np.log(p[np.arange(len(y)), y] + 1e-9).mean()), "ece": float(ece(p, y)), "pred": pred,
            "per_class_f1": per}


def boot(y, dlg, a, b, k, n=1000, seed=0):
    """Seed-mean M-F1 difference a - b over dialogue resamples."""
    rng = np.random.RandomState(seed)
    groups = defaultdict(list)
    for i, d in enumerate(dlg):
        groups[d].append(i)
    keys = list(groups)
    diffs = []
    for _ in range(n):
        idx = np.concatenate([groups[keys[j]] for j in rng.randint(0, len(keys), len(keys))])
        fa = np.mean([classification_scores(y[idx], p[idx], k)["macro_f1"] for p in a])
        fb = np.mean([classification_scores(y[idx], p[idx], k)["macro_f1"] for p in b])
        diffs.append(100 * (fa - fb))
    diffs = np.array(diffs)
    p_two = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), p_two


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.zeros(len(ps)), 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def main():
    out = {}
    hr = json.load(open(RESULT / "hrpi.json", encoding="utf-8"))
    for ds, tag in SETTINGS:
        dialogs, samples, k = load_official(ds, check=False)
        yv = np.array([s["label"] for s in samples["valid"]])
        yt = np.array([s["label"] for s in samples["test"]])
        tt = np.array([s["t"] for s in samples["test"]])
        ov, ot = dialogue_order(samples["valid"]), dialogue_order(samples["test"])
        dlg = [s["dialog"] for s in samples["test"]]
        lv = shift_of(dialogs, samples, k, "valid")
        res, preds = defaultdict(list), defaultdict(list)
        for seed in (42, 43, 44):
            runs = load_crossfit(ds, tag, seed)
            zt_ens = np.mean([softmax(np.array(r["splits"]["test"]["logits"])) for r in runs], 0)
            histories = {"gold": (dialogs, yt), "predicted": corrupt(dialogs, samples, k, zt_ens.argmax(1))}
            rng = np.random.RandomState(seed)
            for q in NOISE:
                histories[f"noise{int(q * 100)}"] = corrupt(dialogs, samples, k, None, rng, q)
            probs = defaultdict(list)
            for r in runs:
                zv, zt = np.array(r["splits"]["valid"]["logits"]), np.array(r["splits"]["test"]["logits"])
                s2, half, _ = select(zv + lv, yv, ov)
                s2r, halfr, _ = select(zv, yv, ov)
                for hname, (dh, yh) in histories.items():
                    lt = shift_of(dh, samples, k, "test")
                    full = zt + lt + adapt(zt + lt, yh, ot, s2, half)
                    probs[f"HRPI|{hname}"].append(softmax(full))
                    probs[f"RPI|{hname}"].append(softmax(zt + adapt(zt, yh, ot, s2r, halfr)))
                    if hname == "gold":
                        probs["level-1 only"].append(softmax(zt + lt))
                        probs["disjoint HRPI"].append(softmax(zt + lt + disjoint_adapt(zt + lt, yh, tt, ot, s2)))
                        probs["ensemble"].append(softmax(zt))
            for name, v in probs.items():
                s = scores(yt, np.mean(v, 0), k)
                preds[name].append(s.pop("pred"))
                res[name].append(s)
        summ = {}
        for name, v in res.items():
            summ[name] = {m: (np.mean([x[m] for x in v], 0).tolist()) for m in v[0]}
        # statistics: HRPI (gold) vs every other system of the main comparison, plus variants
        others = {"ensemble": preds["ensemble"], "RPI": preds["RPI|gold"], "level-1 only": preds["level-1 only"],
                  "disjoint HRPI": preds["disjoint HRPI"]}
        saved = {}
        for f in sorted((RESULT / "spire" / ds).glob(f"{tag}_s*.json")):
            d = json.load(open(f, encoding="utf-8"))["preds"]
            for a, nm in (("spire", "SPIRE"), ("amortized", "neural process")):
                saved.setdefault(nm, []).append(np.array(d[a]))
        others.update(saved)
        stats = {}
        for nm, ps in others.items():
            lo, hi, p = boot(yt, dlg, preds["HRPI|gold"], ps, k)
            stats[nm] = {"ci95": [lo, hi], "p_two": p}
        adj = holm([v["p_two"] for v in stats.values()])
        for (nm, v), a in zip(stats.items(), adj):
            v["p_holm"] = a
        out[f"{ds}:{tag}"] = {"summary": summ, "stats": stats}
        print(f"== {ds} {tag}", flush=True)
        for nm, v in summ.items():
            print(f"  {nm:22s} M-F1 {v['mf1']:6.2f} W-F1 {v['wf1']:6.2f} B {v['bias']:.2f} NLL {v['nll']:.3f} "
                  f"ECE {v['ece']:.3f}", flush=True)
        for nm, v in stats.items():
            print(f"  HRPI - {nm:16s} CI [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] p2 {v['p_two']:.3f} "
                  f"holm {v['p_holm']:.3f}", flush=True)
        (RESULT / "hrpi_analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
