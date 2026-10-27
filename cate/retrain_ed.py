"""Retrain EmpatheticDialogues DEG variants that collapsed, then rebuild the summary."""

from __future__ import annotations

import json
import os
import shutil
import sys

from cate.config import RESULT


def main() -> None:
    os.environ.setdefault("HF_HOME", r"D:\hf_cache")
    backup = RESULT / "ed_memory_s42.backup.json"
    src = RESULT / "ed_memory_s42.json"
    if src.exists() and not backup.exists():
        shutil.copy(src, backup)
        print("backed up", backup, flush=True)
    sys.argv = [
        "train_memory.py",
        "--dataset",
        "ed",
        "--epochs",
        "4",
        "--batch-size",
        "16",
        "--variants",
        "overwrite",
        "deg",
        "deg_no_decay",
        "deg_no_conflict",
        "deg_no_reject",
        "deg_scalar",
    ]
    from cate.train_memory import main as train_main

    train_main()
    from cate.merge_ed import main as merge_main

    # Restore non-retrained rows from backup before merging scores from preds.
    if backup.exists():
        old = {r["variant"]: r for r in json.loads(backup.read_text(encoding="utf-8"))}
        new = {r["variant"]: r for r in json.loads(src.read_text(encoding="utf-8"))}
        old.update(new)
        order = [
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
        rows = [old[n] for n in order if n in old]
        src.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    merge_main()
    print("ED RETRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
