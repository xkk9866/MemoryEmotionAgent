"""Run the DEG-Mem pipeline. Set SMOKE=1 for a short crash test."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", r"D:\hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")


def step(name: str) -> None:
    print(f"\n===== {name} =====", flush=True)
    runpy.run_module(name, run_name="__main__")


if __name__ == "__main__":
    step("cate.subjective")
    step("cate.train_emotion")
    step("cate.encode_and_build")
    sys.argv = ["train_memory.py", "--dataset", "esconv", "--epochs", "8", "--batch-size", "4"]
    step("cate.train_memory")
    sys.argv = ["train_memory.py", "--dataset", "ed", "--epochs", "4", "--batch-size", "8"]
    step("cate.train_memory")
    step("cate.train_roberta")
    if os.environ.get("SMOKE", "0") != "1":
        step("cate.evaluate")
