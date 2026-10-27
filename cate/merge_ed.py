"""Rebuild results/ed_memory_s42.json from prediction files after a partial retrain."""

from __future__ import annotations

import json

from cate.config import RESULT
from cate.metrics import classification_scores

NAMES = [
    "meanpool",
    "recency",
    "rag",
    "forget",
    "seqtrans",
    "window_graph",
    "overwrite",
    "deg",
    "deg_no_decay",
    "deg_no_conflict",
    "deg_no_retrieval",
    "deg_no_reject",
    "deg_scalar",
]


def main() -> None:
    existing = {}
    path = RESULT / "ed_memory_s42.json"
    if path.exists():
        for row in json.loads(path.read_text(encoding="utf-8")):
            existing[row["variant"]] = row
    rows = []
    for name in NAMES:
        pred_path = RESULT / f"preds_ed_{name}_s42.json"
        if not pred_path.exists():
            print(name, "missing preds", flush=True)
            continue
        preds = json.loads(pred_path.read_text(encoding="utf-8"))
        emo = classification_scores(preds["emo_y"], preds["emo_p"])
        prev = existing.get(name, {})
        row = {
            "variant": name,
            "dataset": "ed",
            "seed": 42,
            "best_dev": prev.get("best_dev"),
            "test": {"loss": (prev.get("test") or {}).get("loss"), "strategy": None, "emotion": emo},
        }
        rows.append(row)
        print(f"{name:18} macro {emo['macro_f1']:.4f} n {emo['n']}", flush=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
