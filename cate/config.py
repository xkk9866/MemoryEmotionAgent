"""Paths and label inventories for DEG-Mem experiments."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("CATE_DATA", r"D:\cate_data"))
CACHE = DATA / "cache"
CKPT = DATA / "checkpoints"
RESULT = ROOT / "results"
HF_HOME = Path(os.environ.get("HF_HOME", r"D:\hf_cache"))

for _p in (DATA, CACHE, CKPT, RESULT, HF_HOME):
    _p.mkdir(parents=True, exist_ok=True)

ESCONV_JSON = DATA / "ESConv.json"
FAILED_JSON = DATA / "FailedESConv.json"
ED_DIR = DATA / "empatheticdialogues"

ENCODER_NAME = "roberta-base"
GEN_NAME = "t5-small"
MAX_UTT_LEN = 96
BUDGET = 8
MAX_FEEDBACK = 4
MAX_CONFLICT = 3

STRATEGIES = [
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others",
]
STRATEGY2ID = {s: i for i, s in enumerate(STRATEGIES)}

# GoEmotions simplified label order used by the Hugging Face dataset.
GOEMO_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]
NEG_EMO = {
    "anger",
    "annoyance",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "fear",
    "grief",
    "nervousness",
    "remorse",
    "sadness",
}
POS_EMO = {
    "admiration",
    "amusement",
    "approval",
    "caring",
    "excitement",
    "gratitude",
    "joy",
    "love",
    "optimism",
    "pride",
    "relief",
}
NEG_IDX = [i for i, n in enumerate(GOEMO_LABELS) if n in NEG_EMO]
POS_IDX = [i for i, n in enumerate(GOEMO_LABELS) if n in POS_EMO]

# Node types in the evidence graph.
TYPES = ["emotion", "event", "need", "strategy", "feedback", "risk"]
TYPE2ID = {n: i for i, n in enumerate(TYPES)}

# Edge relations stored in the adjacency tensor.
REL_TEMPORAL = 0
REL_SUPPORT = 1
REL_CONTRADICT = 2
REL_FEEDBACK = 3
N_REL = 4
