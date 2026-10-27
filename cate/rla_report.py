"""Revealed-label adaptation on every run family of a dataset, released EmoDynamiX included.

For each run, (s2, half) are selected by validation NLL (cate.rla.select) and
applied once to test. p is a paired dialogue bootstrap of the seed-mean M-F1:
the same model with vs. without adaptation.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict

import numpy as np
from scipy.special import log_softmax

from cate.config import RESULT
from cate.dpp import DialoguePolicyPosterior
from cate.metrics import classification_scores, preference_bias_official
from cate.official import load_official
from cate.policy_report import PAPER
from cate.rla import adapt, dialogue_order, select


def metrics(y, z, k):
    p = z.argmax(-1)
    s = classification_scores(y, p, k)
    return [100 * s["macro_f1"], 100 * s["weighted_f1"], preference_bias_official(y, p, k)]


def base_readouts(run, sp, log_pi):
    d = run["splits"][sp]
    z = np.array(d["logits"])
    if run.get("prior") == "dpp":
        z = z + np.array(d["shift"])
    out = {"cls": z}
    if "knn" in d:
        out["mix"] = log_softmax(z, -1) + np.array(d["knn"]) - log_pi
    return out


def bootstrap(y, dialog, pa, pb, k, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    groups = defaultdict(list)
    for i, d in enumerate(dialog):
        groups[d].append(i)
    keys = list(groups)
    worse = 0
    for _ in range(n):
        idx = np.concatenate([groups[keys[j]] for j in rng.randint(0, len(keys), len(keys))])
        a = np.mean([classification_scores(y[idx], p[idx], k)["macro_f1"] for p in pa])
        b = np.mean([classification_scores(y[idx], p[idx], k)["macro_f1"] for p in pb])
        worse += b <= a
    return worse / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esconv")
    args = ap.parse_args()
    dialogs, samples, k = load_official(args.dataset, check=False)
    log_pi = DialoguePolicyPosterior.fit(dialogs, samples["train"], None, k).log_pi.numpy()

    families = defaultdict(dict)
    for sub in (args.dataset, f"{args.dataset}_emodx"):
        for f in sorted(glob.glob(str(RESULT / "policy" / sub / "*_s*.json"))):
            m = re.match(r"(.+)_s(\d+)\.json$", f.replace("\\", "/").split("/")[-1])
            families[m.group(1)][int(m.group(2))] = json.load(open(f, encoding="utf-8"))

    # Failed fine-tuning runs (Mosbach et al., 2021) are decided on dev only: best dev
    # macro-F1 below half the family median. The first three surviving seeds are kept.
    for fam, by_seed in families.items():
        devs = [r.get("best_dev") for r in by_seed.values() if r.get("best_dev") is not None]
        if not devs:
            continue
        cut = 0.5 * float(np.median(devs))
        failed = [s for s, r in by_seed.items() if r.get("best_dev", 1.0) < cut]
        for s in failed:
            print(f"excluded failed run {fam} s{s}: dev {by_seed[s]['best_dev']:.4f} < {cut:.4f}")
            del by_seed[s]
        for s in sorted(by_seed)[3:]:
            del by_seed[s]

    table = {}
    print(f"{'system':36s} {'n':>2s} {'M-F1':>12s} {'W-F1':>12s} {'B':>10s} {'dev':>6s} {'s2/half':>10s} {'p':>6s}")
    for name, (a, b, c) in PAPER.get(args.dataset, {}).items():
        print(f"{name:36s} {'-':>2s} {a:12.2f} {b:12.2f} {c:10.2f}")
    for fam, by_seed in sorted(families.items()):
        acc = defaultdict(lambda: {"test": [], "dev": [], "pred": [], "hp": []})
        y_ref, dialog_ref = None, None
        for seed, run in sorted(by_seed.items()):
            rows, ys, orders = {}, {}, {}
            for sp in ("valid", "test"):
                idx = run["splits"][sp].get("index", list(range(len(samples[sp]))))
                rows[sp] = [samples[sp][i] for i in idx]
                ys[sp] = np.array(run["splits"][sp]["y"])
                orders[sp] = dialogue_order(rows[sp])
            y_ref, dialog_ref = ys["test"], [r["dialog"] for r in rows["test"]]
            zv, zt = base_readouts(run, "valid", log_pi), base_readouts(run, "test", log_pi)
            for ro in zv:
                # Every system, MERLA included, gets s2 by the same held-out evidence rule:
                # on training dialogues the revealed turns are partly memorised, so a
                # variance learned there is too small.
                s2, half, _ = select(zv[ro], ys["valid"], orders["valid"])
                for tag, fv, ft in (
                    (ro, zv[ro], zt[ro]),
                    (ro + "+rla", zv[ro] + adapt(zv[ro], ys["valid"], orders["valid"], s2, half),
                     zt[ro] + adapt(zt[ro], ys["test"], orders["test"], s2, half)),
                ):
                    acc[tag]["test"].append(metrics(ys["test"], ft, k))
                    acc[tag]["dev"].append(metrics(ys["valid"], fv, k)[0])
                    acc[tag]["pred"].append(ft.argmax(-1))
                    acc[tag]["hp"].append(f"{s2:g}/{half:g}" if tag.endswith("rla") else "")
        for tag, v in acc.items():
            key = f"{fam}:{tag}"
            m, s = np.mean(v["test"], 0), np.std(v["test"], 0)
            p = ""
            if tag.endswith("+rla"):
                p = bootstrap(y_ref, dialog_ref, acc[tag[:-4]]["pred"], v["pred"], k)
            table[key] = {"n": len(v["test"]), "test_mean": m.tolist(), "test_std": s.tolist(),
                          "dev": float(np.mean(v["dev"])), "hp": v["hp"], "p_vs_no_rla": p}
            print(f"{key:36s} {len(v['test']):2d} {m[0]:6.2f}±{s[0]:<4.2f} {m[1]:6.2f}±{s[1]:<4.2f} "
                  f"{m[2]:4.2f}±{s[2]:<4.2f} {np.mean(v['dev']):6.2f} {','.join(sorted(set(v['hp']))):>10s} "
                  f"{p if p == '' else f'{p:.3f}':>6s}", flush=True)
    (RESULT / "policy" / f"rla_{args.dataset}.json").write_text(json.dumps(table, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
