"""Pre-registered evaluation of HRPI on independent re-splits (cate.resplit, cate.crossfit --split_seed).

Primary endpoint, fixed before looking at re-split results: the macro-F1 difference
HRPI - RPI (residual correction), per corpus, over split seeds 0 (official) .. 3.
Each split: RoBERTa-base, 5-fold cross-fitting, seed 42. Reported per split with a paired
dialogue bootstrap 95% CI, and pooled per corpus as the mean over splits with a bootstrap
CI that resamples dialogues within every split. Secondary: ensemble, label shift.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from cate.calib_baselines import softmax
from cate.config import RESULT
from cate.dpp import DialoguePolicyPosterior, history_counts
from cate.external_baselines import m3
from cate.metrics import classification_scores
from cate.resplit import load_split
from cate.rla import adapt, dialogue_order, select

SPLITS = (0, 1, 2, 3)


def runs_of(ds, rs):
    sub = "crossfit" if rs == 0 else f"crossfit_rs{rs}"
    return [json.load(open(RESULT / sub / ds / f"roberta-base_s42_f{f}.json", encoding="utf-8")) for f in range(5)]


def shift(dialogs, samples, k, sp, out_of_window):
    dpp = DialoguePolicyPosterior.fit(dialogs, samples["train"], None, k, half_life=1e9, skip_window=out_of_window)
    return dpp(history_counts(dialogs, samples[sp], k)).detach().numpy()


def systems(ds, rs):
    dialogs, samples, k = load_split(ds, rs)
    yv = np.array([s["label"] for s in samples["valid"]])
    yt = np.array([s["label"] for s in samples["test"]])
    ov, ot = dialogue_order(samples["valid"]), dialogue_order(samples["test"])
    lv, lt = shift(dialogs, samples, k, "valid", True), shift(dialogs, samples, k, "test", True)
    la = shift(dialogs, samples, k, "test", False)
    P = defaultdict(list)
    for r in runs_of(ds, rs):
        zv, zt = np.array(r["splits"]["valid"]["logits"]), np.array(r["splits"]["test"]["logits"])
        s2r, hr, _ = select(zv, yv, ov)
        s2, h, _ = select(zv + lv, yv, ov)
        P["ensemble"].append(softmax(zt))
        P["label shift"].append(softmax(zt + la))
        P["RPI"].append(softmax(zt + adapt(zt, yt, ot, s2r, hr)))
        P["HRPI"].append(softmax(zt + lt + adapt(zt + lt, yt, ot, s2, h)))
    preds = {a: np.mean(v, 0).argmax(1) for a, v in P.items()}
    return yt, [s["dialog"] for s in samples["test"]], k, preds


def main(n_boot=1000):
    import sys
    path = RESULT / "resplit_eval.json"
    out = json.load(open(path, encoding="utf-8")) if path.exists() else {}
    for ds in (sys.argv[1:] or ("annomi", "p4g", "esconv")):
        per, data = {}, []
        for rs in SPLITS:
            try:
                yt, dlg, k, preds = systems(ds, rs)
            except FileNotFoundError:
                print(ds, rs, "missing")
                continue
            data.append((yt, dlg, k, preds))
            per[rs] = {a: m3(yt, p, k) for a, p in preds.items()}
            print(ds, rs, {a: round(v[0], 2) for a, v in per[rs].items()}, flush=True)
        rng = np.random.RandomState(0)
        diffs = defaultdict(list)  # pooled mean over splits of HRPI - other, per bootstrap replicate
        split_diffs = defaultdict(lambda: defaultdict(list))
        for _ in range(n_boot):
            rep = defaultdict(list)
            for i, (yt, dlg, k, preds) in enumerate(data):
                groups = defaultdict(list)
                for j, d in enumerate(dlg):
                    groups[d].append(j)
                keys = list(groups)
                idx = np.concatenate([groups[keys[j]] for j in rng.randint(0, len(keys), len(keys))])
                f = {a: 100 * classification_scores(yt[idx], p[idx], k)["macro_f1"] for a, p in preds.items()}
                for a in ("RPI", "ensemble", "label shift"):
                    rep[a].append(f["HRPI"] - f[a])
                    split_diffs[i][a].append(f["HRPI"] - f[a])
            for a, v in rep.items():
                diffs[a].append(np.mean(v))
        res = {"per_split": per, "pooled": {}, "split_ci": {}}
        for a, v in diffs.items():
            v = np.array(v)
            point = float(np.mean([per[rs]["HRPI"][0] - per[rs][a][0] for rs in per]))
            res["pooled"][a] = {"diff": point, "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                                "p_two": float(min(1, 2 * min((v <= 0).mean(), (v >= 0).mean())))}
            print(f"{ds} pooled HRPI-{a}: {point:+.2f} CI [{res['pooled'][a]['ci95'][0]:+.2f}, "
                  f"{res['pooled'][a]['ci95'][1]:+.2f}] p2 {res['pooled'][a]['p_two']:.3f}", flush=True)
        for i, rs in enumerate(per):
            res["split_ci"][rs] = {a: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                                   for a, v in split_diffs[i].items()}
        out[ds] = res
        (RESULT / "resplit_eval.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
