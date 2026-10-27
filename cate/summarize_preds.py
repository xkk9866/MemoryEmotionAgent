"""Print strategy scores already saved under results/. Does not train."""

from __future__ import annotations

import json

from cate.config import RESULT
from cate.metrics import classification_scores, length_bucket, macro_f1_by_bucket

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
    "deg_shared_hl",
    "deg_scalar",
    "deg_no_retloss",
]


def main() -> None:
    for name in NAMES:
        path = RESULT / f"preds_esconv_{name}_s42.json"
        if not path.exists():
            print(name, "missing", flush=True)
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        scores = classification_scores(data["y"], data["pred"], 8)
        buckets = macro_f1_by_bucket(
            data["y"],
            data["pred"],
            [length_bucket(h) for h in data["hist"]],
            8,
        )
        brief = {key: round(value["macro_f1"], 4) for key, value in buckets.items()}
        print(
            f"{name:16} macro {scores['macro_f1']:.4f} acc {scores['accuracy']:.4f} n {scores['n']} {brief}",
            flush=True,
        )


if __name__ == "__main__":
    main()
