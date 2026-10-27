"""Load ESConv, FailedESConv, and EmpatheticDialogues, and assign splits."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict

import numpy as np

from cate.config import ED_DIR, ESCONV_JSON, FAILED_JSON, STRATEGY2ID

_COMMA = re.compile(r"_comma_")


def _clean(text: str) -> str:
    text = text or ""
    text = _COMMA.sub(",", text)
    return re.sub(r"\s+", " ", text).strip()


def _feedback(ann: dict):
    if not isinstance(ann, dict) or "feedback" not in ann:
        return None
    try:
        return int(float(ann["feedback"]))
    except (TypeError, ValueError):
        return None


def _load_esconv_file(path, failed: int) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for i, item in enumerate(raw):
        turns = []
        for turn in item.get("dialog") or []:
            speaker = (turn.get("speaker") or "").lower()
            if speaker in {"seeker", "speaker"}:
                role = "seeker"
            elif speaker in {"supporter", "listener"}:
                role = "supporter"
            else:
                role = "seeker"
            ann = turn.get("annotation") or {}
            strategy = ann.get("strategy") if isinstance(ann, dict) else None
            if strategy not in STRATEGY2ID:
                strategy = None
            turns.append(
                {
                    "role": role,
                    "text": _clean(turn.get("content") or ""),
                    "strategy": strategy,
                    "feedback": _feedback(ann if isinstance(ann, dict) else {}),
                }
            )
        rows.append(
            {
                "source_index": i,
                "failed": failed,
                "emotion": item.get("emotion_type") or "unknown",
                "problem": item.get("problem_type") or "unknown",
                "situation": _clean(item.get("situation") or ""),
                "turns": turns,
            }
        )
    return rows


def load_esconv() -> list[dict]:
    rows = _load_esconv_file(ESCONV_JSON, 0)
    rows += _load_esconv_file(FAILED_JSON, 1)
    for i, row in enumerate(rows):
        row["dialog_id"] = i
    return rows


def stratified_split(labels, seed: int = 42, ratios=(0.8, 0.1, 0.1)) -> list[str]:
    rng = np.random.RandomState(seed)
    buckets = defaultdict(list)
    for i, lab in enumerate(labels):
        buckets[lab].append(i)
    split = [""] * len(labels)
    for _, idxs in buckets.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(n * ratios[0]))
        n_dev = int(round(n * ratios[1]))
        if n >= 3:
            n_train = min(max(n_train, 1), n - 2)
            n_dev = min(max(n_dev, 1), n - n_train - 1)
        elif n == 2:
            n_train, n_dev = 1, 0
        else:
            n_train, n_dev = n, 0
        for j, idx in enumerate(idxs):
            if j < n_train:
                split[idx] = "train"
            elif j < n_train + n_dev:
                split[idx] = "dev"
            else:
                split[idx] = "test"
    return split


def assign_esconv_splits(rows: list[dict], seed: int = 42) -> None:
    success_ids = [r["dialog_id"] for r in rows if r["failed"] == 0]
    success_labels = [r["emotion"] for r in rows if r["failed"] == 0]
    success_split = stratified_split(success_labels, seed=seed)
    strategy_split = {i: s for i, s in zip(success_ids, success_split)}
    fail_labels = [str(r["failed"]) for r in rows]
    fail_split = stratified_split(fail_labels, seed=seed + 1, ratios=(0.7, 0.15, 0.15))
    for r, fs in zip(rows, fail_split):
        r["fail_split"] = fs
        r["strategy_split"] = strategy_split.get(r["dialog_id"], "ignore")


def load_empathetic() -> list[dict]:
    split_file = {"train": "train.csv", "dev": "valid.csv", "test": "test.csv"}
    grouped = []
    for split, name in split_file.items():
        path = ED_DIR / name
        buckets = defaultdict(list)
        with path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                buckets[row["conv_id"]].append(row)
        for conv_id, utts in buckets.items():
            utts = sorted(utts, key=lambda r: int(r["utterance_idx"]))
            first_speaker = utts[0]["speaker_idx"]
            turns = []
            for row in utts:
                role = "seeker" if row["speaker_idx"] == first_speaker else "supporter"
                turns.append(
                    {
                        "role": role,
                        "text": _clean(row["utterance"]),
                        "strategy": None,
                        "feedback": None,
                    }
                )
            grouped.append(
                {
                    "dialog_id": conv_id,
                    "split": split,
                    "emotion": utts[0]["context"],
                    "situation": _clean(utts[0]["prompt"]),
                    "turns": turns,
                    "failed": 0,
                }
            )
    return grouped


def emotion_vocab(labels: list[str]) -> dict[str, int]:
    names = sorted(set(labels))
    return {n: i for i, n in enumerate(names)}
