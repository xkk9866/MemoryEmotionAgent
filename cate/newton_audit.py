"""Convergence and cost of HRPI's level-2 solver, exactly as run (8 undamped, warm-started
Newton steps per turn), against a reference solve (damped Newton to ||g|| < 1e-10).

Per setting (official split, seed 42, all 5 fold models, test sessions): gradient norm
after the 8 steps, objective gap to the reference, share of turns with ||g|| > 1e-6,
and wall-clock time per decision.
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.special import logsumexp, softmax

from cate.config import RESULT
from cate.hrpi import SETTINGS
from cate.hrpi_analysis import shift_of
from cate.official import load_official
from cate.rla import dialogue_order, select
from cate.spire import load_crossfit


def objective(b, z, y, s2):
    return float((logsumexp(z + b, 1) - (z + b)[np.arange(len(y)), y]).sum() + b @ b / (2 * s2))


def grad_hess(b, z, y, s2, k):
    p = softmax(z + b, 1)
    g = (p - np.eye(k)[y]).sum(0) + b / s2
    h = np.diag(p.sum(0)) - p.T @ p + np.eye(k) / s2
    return g, h


def reference(z, y, s2, k):
    b = np.zeros(k)
    for _ in range(100):
        g, h = grad_hess(b, z, y, s2, k)
        if np.linalg.norm(g) < 1e-10:
            break
        step, f0 = np.linalg.solve(h, g), objective(b, z, y, s2)
        t = 1.0
        while objective(b - t * step, z, y, s2) > f0 - 0.25 * t * g @ step and t > 1e-8:
            t /= 2
        b = b - t * step
    return b


def main():
    out = {}
    for ds, tag in SETTINGS:
        dialogs, samples, k = load_official(ds, check=False)
        yv = np.array([s["label"] for s in samples["valid"]])
        yt = np.array([s["label"] for s in samples["test"]])
        ov, ot = dialogue_order(samples["valid"]), dialogue_order(samples["test"])
        lv, lt = shift_of(dialogs, samples, k, "valid"), shift_of(dialogs, samples, k, "test")
        gn, gap, sec, n = [], [], 0.0, 0
        for r in load_crossfit(ds, tag, 42):
            zv = np.array(r["splits"]["valid"]["logits"]) + lv
            zt = np.array(r["splits"]["test"]["logits"]) + lt
            s2, _, _ = select(zv, yv, ov)
            for rows in ot:
                b = np.zeros(k)
                for j in range(1, len(rows)):  # the production solver (cate.rla.adapt, half = inf)
                    past = rows[:j]
                    t0 = time.perf_counter()
                    for _ in range(8):
                        g, h = grad_hess(b, zt[past], yt[past], s2, k)
                        b = b - np.linalg.solve(h, g)
                    sec += time.perf_counter() - t0
                    gn.append(np.linalg.norm(grad_hess(b, zt[past], yt[past], s2, k)[0]))
                    ref = reference(zt[past], yt[past], s2, k)
                    gap.append(objective(b, zt[past], yt[past], s2) - objective(ref, zt[past], yt[past], s2))
                n += len(rows)
        gn, gap = np.array(gn), np.array(gap)
        out[f"{ds}:{tag}"] = {"turns": int(len(gn)), "grad_median": float(np.median(gn)), "grad_max": float(gn.max()),
                              "share_grad_gt_1e-6": float((gn > 1e-6).mean()), "gap_max": float(gap.max()),
                              "ms_per_decision": 1000 * sec / n}
        print(ds, tag, out[f"{ds}:{tag}"], flush=True)
    (RESULT / "newton_audit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
