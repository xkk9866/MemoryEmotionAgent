"""Aggregate multi-seed strategy-prediction runs into the main table with paired bootstrap tests."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score

from cate.config import RESULT

BUCKETS = ["1-8", "9-16", "17-32", "33+"]


def load_runs():
    runs = defaultdict(list)
    for sub in sorted(p for p in RESULT.iterdir() if p.is_dir() and (p.name.startswith("eam") or p.name == "llm")):
        for f in sorted(sub.glob("*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            if sub.name == "llm":
                stem = f.stem.rsplit("_s", 1)[0]
                key = ("llm", stem)
            else:
                enc = "roberta-base" if sub.name == "eam" else sub.name[4:]
                key = (enc, d["variant"])
            runs[key].append(d)
    return runs


def summarize(rs):
    out = {"n_seeds": len(rs)}
    for m in ("macro_f1", "weighted_f1", "accuracy", "bias"):
        v = np.array([r["test"][m] for r in rs], dtype=float)
        out[m] = [float(v.mean()), float(v.std())]
    for b in BUCKETS:
        v = []
        for r in rs:
            cell = r["test"]["by_length"].get(b)
            if isinstance(cell, dict):
                cell = cell.get("macro_f1")
            if cell is not None:
                v.append(cell)
        v = np.array(v, dtype=float)
        if len(v):
            out[f"f1@{b}"] = [float(v.mean()), float(v.std())]
    return out


def aligned(rs):
    """Average-of-seeds is not defined for labels, so each seed's predictions are stacked by (dialog, turn)."""
    per = []
    for r in rs:
        p = r["preds"]
        per.append({tuple(m): (y, q) for m, y, q in zip(p["meta"], p["y"], p["pred"])})
    keys = sorted(set.intersection(*[set(d) for d in per]))
    y = np.array([per[0][k][0] for k in keys])
    preds = np.stack([[d[k][1] for k in keys] for d in per])
    groups = np.array([hash(k[0]) for k in keys])
    return keys, y, preds, groups


def paired_bootstrap(a_rs, b_rs, n=2000, seed=0):
    ka, ya, pa, ga = aligned(a_rs)
    kb, yb, pb, _ = aligned(b_rs)
    common = sorted(set(ka) & set(kb))
    ia = {k: i for i, k in enumerate(ka)}
    ib = {k: i for i, k in enumerate(kb)}
    sa = [ia[k] for k in common]
    sb = [ib[k] for k in common]
    y = ya[sa]
    pa, pb, g = pa[:, sa], pb[:, sb], ga[sa]
    uniq = np.unique(g)
    idx_by = {u: np.where(g == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)

    def f1(p, idx):
        return np.mean([f1_score(y[idx], row[idx], average="macro", labels=range(8), zero_division=0) for row in p])

    full = np.arange(len(y))
    delta = f1(pa, full) - f1(pb, full)
    wins = 0
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[u] for u in pick])
        wins += f1(pa, idx) - f1(pb, idx) > 0
    return {"delta": float(delta), "p_value": float(1 - wins / n)}


def main():
    runs = load_runs()
    table = {f"{k[0]}|{k[1]}": summarize(v) for k, v in runs.items()}
    tests = {}
    ours = runs.get(("roberta-base", "eam"))
    if ours:
        for k, v in runs.items():
            if k != ("roberta-base", "eam"):
                tests[f"eam vs {k[0]}|{k[1]}"] = paired_bootstrap(ours, v, n=1000)
    for enc in {k[0] for k in runs}:
        if (enc, "eam") in runs and (enc, "base") in runs and enc != "roberta-base":
            tests[f"{enc}: eam vs base"] = paired_bootstrap(runs[(enc, "eam")], runs[(enc, "base")], n=1000)
    out = {"table": table, "tests": tests}
    (RESULT / "main_table.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in sorted(table.items(), key=lambda kv: -kv[1]["macro_f1"][0]):
        print(f"{k:40s} n={v['n_seeds']} F1 {v['macro_f1'][0]:.4f}±{v['macro_f1'][1]:.4f} "
              f"wF1 {v['weighted_f1'][0]:.4f} B {v['bias'][0]:.3f}")
    for k, v in tests.items():
        print(k, v)


if __name__ == "__main__":
    main()
