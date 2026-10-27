"""Official EmoDynamiX splits (NAACL 2025) rebuilt with full dialogue history.

The released CSVs keep only a 5-utterance window. We replay the exact
split procedure (same shuffle seed, same dialogue order) so every sample is
matched to its whole dialogue, and check the window text against the CSV.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from cate.config import DATA

EXT = DATA / "ext" / "EmoDynamiX-v2" / "data"
WINDOW = 5


def _norm(x: str) -> str:
    return " ".join(x.strip().split())


def _esconv_dialogs():
    strat = json.load(open(EXT / "esconv" / "strategies.json", encoding="utf-8"))
    raw = json.load(open(EXT / "esconv" / "ESConv.json", encoding="utf-8"))
    dialogs = []
    for i, d in enumerate(raw):
        turns = []
        for t in d["dialog"]:
            ann = t.get("annotation") or {}
            s = strat[ann["strategy"]] if "strategy" in ann else -1
            fb = int(float(ann["feedback"])) if "feedback" in ann else None
            turns.append({"role": t["speaker"], "text": _norm(t["content"]), "strategy": s, "feedback": fb})
        dialogs.append({"id": i, "turns": turns, "emotion": d.get("emotion_type"), "problem": d.get("problem_type"),
                        "situation": _norm(d.get("situation") or "")})
    rng = random.Random(13)
    order = list(range(len(dialogs)))
    rng.shuffle(order)
    n = len(order)
    dev, test = int(0.15 * n), int(0.15 * n)
    split = {}
    for j, k in enumerate(order):
        split[k] = "valid" if j < dev else ("test" if j < dev + test else "train")
    return dialogs, split, len(strat)


def _annomi_dialogs():
    strat = json.load(open(EXT / "annomi" / "strategies.json", encoding="utf-8"))
    raw = [d for d in json.load(open(EXT / "annomi" / "AnnoMI-clean.json", encoding="utf-8")) if d["quality"] == "high"]
    dialogs = []
    for i, d in enumerate(raw):
        turns = []
        for t in d["dialog"]:
            role = "seeker" if t["interlocutor"] == "client" else "supporter"
            s = strat[t["annotation"]] if t["annotation"] else -1
            turns.append({"role": role, "text": _norm(t["content"]), "strategy": s, "feedback": None})
        dialogs.append({"id": i, "turns": turns, "emotion": d.get("topic"), "problem": d.get("topic"), "situation": ""})
    rng = random.Random(77)
    order = list(range(len(dialogs)))
    rng.shuffle(order)
    n = len(order)
    dev, test = int(0.1 * n), int(0.1 * n)
    split = {}
    for j, k in enumerate(order):
        split[k] = "test" if j < test else ("valid" if j < test + dev else "train")
    return dialogs, split, len(strat)


P4G_XLSX = DATA / "ext" / "p4g" / "300_dialog.xlsx"
P4G_STRATEGIES = [
    "logical-appeal", "emotion-appeal", "credibility-appeal", "foot-in-the-door", "self-modeling",
    "personal-story", "donation-information", "source-related-inquiry", "task-related-inquiry",
    "personal-related-inquiry", "non-strategy",
]


def _p4g_dialogs():
    """PersuasionForGood annotated set (Wang et al., ACL 2019), persuader units as samples.

    The ten persuasion strategies of the paper keep their names; every other
    persuader dialogue act is "non-strategy". Split 8:1:1 by dialogue, seed 2019.
    """
    import pandas as pd

    df = pd.read_excel(P4G_XLSX)
    s2i = {s: i for i, s in enumerate(P4G_STRATEGIES)}
    dialogs = []
    for i, (_, g) in enumerate(df.groupby("B2", sort=True)):
        turns = []
        for _, r in g.sort_values("Unnamed: 0").iterrows():
            role = "supporter" if r["B4"] == 0 else "seeker"
            s = -1
            if role == "supporter":
                lab = r["er_label_1"] if isinstance(r["er_label_1"], str) else "non-strategy"
                s = s2i.get(lab, s2i["non-strategy"])
            turns.append({"role": role, "text": _norm(str(r["Unit"])), "strategy": s, "feedback": None})
        dialogs.append({"id": i, "turns": turns, "emotion": None, "problem": None, "situation": ""})
    rng = random.Random(2019)
    order = list(range(len(dialogs)))
    rng.shuffle(order)
    n = len(order)
    split = {}
    for j, k in enumerate(order):
        split[k] = "test" if j < n // 10 else ("valid" if j < 2 * (n // 10) else "train")
    return dialogs, split, len(P4G_STRATEGIES)


def load_official(name: str, check: bool = True):
    cached = DATA / "ext" / f"official_{name}.json"
    if cached.exists():
        blob = json.load(open(cached, encoding="utf-8"))
        return blob["dialogs"], blob["samples"], blob["n_labels"]
    return _build_official(name, check)


def export_official(name: str) -> Path:
    """Write the rebuilt split to JSON so machines without the raw files or pandas can load it."""
    dialogs, samples, k = _build_official(name, check=name != "p4g")
    out = DATA / "ext" / f"official_{name}.json"
    out.write_text(json.dumps({"dialogs": dialogs, "samples": samples, "n_labels": k}), encoding="utf-8")
    return out


def _build_official(name: str, check: bool = True):
    """Return dialogs, samples by split, n_labels.

    A sample is one annotated supporter utterance at position t (0-based in
    turns); its official input is the 5 preceding utterances (with a <START>
    pad), the label is its strategy.
    """
    loader = {"esconv": _esconv_dialogs, "annomi": _annomi_dialogs, "p4g": _p4g_dialogs}[name]
    dialogs, split, n_labels = loader()
    check = check and name != "p4g"
    samples = {"train": [], "valid": [], "test": []}
    # replay CSV order: dialogs in shuffled order within split
    rng = random.Random({"esconv": 13, "annomi": 77, "p4g": 2019}[name])
    order = list(range(len(dialogs)))
    rng.shuffle(order)
    for k in order:
        d = dialogs[k]
        utts = ["<START>"] + [t["text"] for t in d["turns"]]
        for t, turn in enumerate(d["turns"]):
            if turn["strategy"] < 0:
                continue
            idx = t + 1
            window = " </s> ".join(utts[max(0, idx - WINDOW): idx])
            samples[split[k]].append({"dialog": k, "t": t, "label": turn["strategy"], "window": window})
    if check:
        for sp, rows in samples.items():
            with open(EXT / name / f"{sp}.csv", encoding="utf-8", newline="") as f:
                ref = list(csv.DictReader(f))
            assert len(ref) == len(rows), (name, sp, len(ref), len(rows))
            bad = sum(1 for a, b in zip(ref, rows) if int(a["Next_Strategy"]) != b["label"] or _norm(a["Dialogue_History"]) != _norm(b["window"]))
            assert bad == 0, (name, sp, bad)
    return dialogs, samples, n_labels
