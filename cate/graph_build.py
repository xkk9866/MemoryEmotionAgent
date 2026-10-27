"""Build typed evidence graphs from encoded utterances.

Each utterance becomes one evidence node. Explicit seeker feedback does not
overwrite the earlier strategy node; it is stored as a later evidence event
on that node. Lexical overlap plus negation or polarity disagreement creates
a CONTRADICTS edge and leaves both nodes in the graph.
"""

from __future__ import annotations

import re

import numpy as np

from cate.config import (
    MAX_CONFLICT,
    MAX_FEEDBACK,
    NEG_IDX,
    POS_IDX,
    REL_CONTRADICT,
    REL_FEEDBACK,
    REL_SUPPORT,
    REL_TEMPORAL,
    STRATEGY2ID,
    TYPE2ID,
)

STOP = {
    "a", "an", "the", "to", "of", "and", "or", "in", "on", "for", "with", "my",
    "i", "im", "i'm", "me", "you", "your", "is", "was", "were", "be", "been",
    "it", "this", "that", "so", "just", "but", "if", "at", "from", "about", "as",
    "not", "no", "do", "did", "have", "has", "had", "we", "they", "them", "he",
    "she", "his", "her", "our", "are", "am", "very", "really", "too",
}
NEG_RE = re.compile(
    r"\b(not|no|never|cannot|can't|don't|didn't|doesn't|isn't|wasn't|weren't|without|no longer)\b",
    re.I,
)
NEED_RE = re.compile(
    r"\b(i need|i want|i wish|help me|what should i|don't know what to do|i dont know what to do)\b",
    re.I,
)
TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> set[str]:
    return {t for t in TOKEN_RE.findall(text.lower()) if t not in STOP and len(t) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _polarity_from_probs(probs: np.ndarray) -> tuple[float, float]:
    neg = float(probs[NEG_IDX].sum()) if len(probs) else 0.0
    pos = float(probs[POS_IDX].sum()) if len(probs) else 0.0
    return pos - neg, max(pos, neg)


def _init_opinion(polarity: float, confidence: float, reliability: float) -> tuple[float, float, float]:
    c = float(np.clip(confidence * reliability, 0.0, 0.95))
    if polarity > 0.15:
        b, d = c, 0.0
    elif polarity < -0.15:
        b, d = 0.0, c
    else:
        b, d = 0.0, 0.0
    return b, d, 1.0 - b - d


def _primary_type(role: str, text: str, turn: int, polarity: float) -> str:
    if role == "supporter":
        return "strategy"
    if NEED_RE.search(text):
        return "need"
    if turn <= 2:
        return "event"
    if abs(polarity) >= 0.25:
        return "emotion"
    return "event"


def build_graph(
    dialog: dict,
    utt_emb: np.ndarray,
    emo_probs: np.ndarray,
    situation_emb: np.ndarray,
    emotion_id: int,
) -> dict:
    turns = dialog["turns"]
    n_utt = len(turns)
    assert utt_emb.shape[0] == n_utt

    nodes = []
    for t, turn in enumerate(turns):
        pol, conf = _polarity_from_probs(emo_probs[t])
        ntype = _primary_type(turn["role"], turn["text"], t, pol)
        if turn["role"] == "supporter":
            # The node records that this strategy was used. Acceptance stays
            # uncertain until seeker feedback arrives.
            b0, d0, u0 = 0.0, 0.0, 1.0
            reliability = 0.9
            polarity = 0.0
            confidence = 1.0
        else:
            reliability = 0.55
            polarity = pol
            confidence = conf
            b0, d0, u0 = _init_opinion(polarity, confidence, reliability)
        sid = STRATEGY2ID.get(turn["strategy"], -1) if turn["strategy"] else -1
        nodes.append(
            {
                "emb": utt_emb[t].astype(np.float32),
                "type": TYPE2ID[ntype],
                "turn": t,
                "strategy": sid,
                "pol": polarity,
                "conf": confidence,
                "rel": reliability,
                "b0": b0,
                "d0": d0,
                "u0": u0,
                "text": turn["text"][:240],
                "role": 0 if turn["role"] == "seeker" else 1,
                "is_utt": 1,
                "toks": _tokens(turn["text"]),
                "neg": int(bool(NEG_RE.search(turn["text"]))),
                "fb": [],
            }
        )

    # Feedback is evidence about the previous supporter strategy, not a rewrite.
    for t, turn in enumerate(turns):
        if turn["feedback"] is None:
            continue
        prev = t - 1
        if prev < 0 or turns[prev]["role"] != "supporter":
            continue
        score = turn["feedback"]
        if score <= 2:
            pol, conf = -1.0, 0.9
        elif score >= 4:
            pol, conf = 1.0, 0.75
        else:
            pol, conf = 0.0, 0.35
        nodes[prev]["fb"].append((t, pol, conf, 0.95))
        if score <= 2 and len(nodes) < n_utt + 8:
            # A risk node accumulates repeated rejection without deleting the suggestion.
            if not any(nd["type"] == TYPE2ID["risk"] for nd in nodes):
                nodes.append(
                    {
                        "emb": utt_emb[t].astype(np.float32),
                        "type": TYPE2ID["risk"],
                        "turn": t,
                        "strategy": -1,
                        "pol": -1.0,
                        "conf": 0.8,
                        "rel": 0.9,
                        "b0": 0.0,
                        "d0": 0.0,
                        "u0": 1.0,
                        "text": "[risk] repeated or strong rejection",
                        "role": 0,
                        "is_utt": 0,
                        "toks": set(),
                        "neg": 1,
                        "fb": [(t, -1.0, 0.9, 0.95)],
                    }
                )
            else:
                for nd in nodes:
                    if nd["type"] == TYPE2ID["risk"]:
                        nd["fb"].append((t, -1.0, 0.9, 0.95))

    edges = []
    for i in range(n_utt - 1):
        edges.append((i, i + 1, REL_TEMPORAL))
        edges.append((i + 1, i, REL_TEMPORAL))
    for i in range(len(nodes)):
        for j in range(i):
            if nodes[i]["type"] != nodes[j]["type"]:
                continue
            if nodes[i]["type"] == TYPE2ID["strategy"]:
                continue
            jac = _jaccard(nodes[i]["toks"], nodes[j]["toks"])
            neg_xor = nodes[i]["neg"] != nodes[j]["neg"]
            pol_flip = nodes[i]["pol"] * nodes[j]["pol"] < -0.02
            if jac >= 0.18 and (neg_xor or pol_flip):
                edges.append((i, j, REL_CONTRADICT))
                edges.append((j, i, REL_CONTRADICT))
            elif jac >= 0.25:
                edges.append((i, j, REL_SUPPORT))
                edges.append((j, i, REL_SUPPORT))
    for i, nd in enumerate(nodes):
        for (ft, _pol, _conf, _rel) in nd["fb"]:
            if 0 <= ft < len(nodes):
                edges.append((i, ft, REL_FEEDBACK))
                edges.append((ft, i, REL_FEEDBACK))

    # Situation is a probe query, never a graph node.
    sit_toks = _tokens(dialog.get("situation") or "")
    anchor = -1
    best = 0.0
    for i, nd in enumerate(nodes):
        if not nd["is_utt"] or nd["role"] != 0:
            continue
        jac = _jaccard(sit_toks, nd["toks"])
        if jac > best:
            best = jac
            anchor = i

    seeker_idx = [i for i, nd in enumerate(nodes) if nd["is_utt"] and nd["role"] == 0]
    late_src = seeker_idx[max(0, (2 * len(seeker_idx)) // 3) :] or seeker_idx
    if late_src:
        late_query = utt_emb[late_src].mean(axis=0).astype(np.float32)
    else:
        late_query = utt_emb.mean(axis=0).astype(np.float32)

    decisions_turn = []
    decisions_label = []
    decisions_query = []
    decisions_phase = []
    decisions_response = []
    last_seeker = None
    for t, turn in enumerate(turns):
        if turn["role"] == "seeker":
            last_seeker = t
        elif turn["strategy"] and last_seeker is not None:
            decisions_turn.append(t)
            decisions_label.append(STRATEGY2ID[turn["strategy"]])
            decisions_query.append(utt_emb[last_seeker].astype(np.float32))
            decisions_phase.append(t / max(n_utt, 1))
            decisions_response.append(turn["text"][:240])

    n = len(nodes)
    k = MAX_FEEDBACK
    fb_turn = np.full((n, k), -1, dtype=np.int64)
    fb_pol = np.zeros((n, k), dtype=np.float32)
    fb_conf = np.zeros((n, k), dtype=np.float32)
    fb_rel = np.zeros((n, k), dtype=np.float32)
    for i, nd in enumerate(nodes):
        for j, (ft, pol, conf, rel) in enumerate(nd["fb"][:k]):
            fb_turn[i, j] = ft
            fb_pol[i, j] = pol
            fb_conf[i, j] = conf
            fb_rel[i, j] = rel

    contradict = np.full((n, MAX_CONFLICT), -1, dtype=np.int64)
    buckets = {i: [] for i in range(n)}
    for a, b, rel in edges:
        if rel == REL_CONTRADICT and b not in buckets[a] and len(buckets[a]) < MAX_CONFLICT:
            buckets[a].append(b)
    for i, partners in buckets.items():
        for j, p in enumerate(partners):
            contradict[i, j] = p

    edge_arr = np.array(edges, dtype=np.int64) if edges else np.zeros((0, 3), dtype=np.int64)
    graph = {
        "dialog_id": dialog["dialog_id"],
        "dataset": dialog.get("dataset", "esconv"),
        "strategy_split": dialog.get("strategy_split", dialog.get("split", "train")),
        "fail_split": dialog.get("fail_split", "ignore"),
        "split": dialog.get("split", dialog.get("strategy_split", "train")),
        "failed": int(dialog.get("failed", 0)),
        "emotion": dialog.get("emotion", ""),
        "emotion_id": int(emotion_id),
        "n_turns": n_utt,
        "anchor": int(anchor),
        "anchor_jaccard": float(best),
        "situation_emb": situation_emb.astype(np.float32),
        "late_query": late_query,
        "emb": np.stack([nd["emb"] for nd in nodes]).astype(np.float32),
        "type": np.array([nd["type"] for nd in nodes], dtype=np.int64),
        "turn": np.array([nd["turn"] for nd in nodes], dtype=np.int64),
        "strategy": np.array([nd["strategy"] for nd in nodes], dtype=np.int64),
        "pol": np.array([nd["pol"] for nd in nodes], dtype=np.float32),
        "conf": np.array([nd["conf"] for nd in nodes], dtype=np.float32),
        "rel": np.array([nd["rel"] for nd in nodes], dtype=np.float32),
        "b0": np.array([nd["b0"] for nd in nodes], dtype=np.float32),
        "d0": np.array([nd["d0"] for nd in nodes], dtype=np.float32),
        "u0": np.array([nd["u0"] for nd in nodes], dtype=np.float32),
        "is_utt": np.array([nd["is_utt"] for nd in nodes], dtype=np.int64),
        "role": np.array([nd["role"] for nd in nodes], dtype=np.int64),
        "fb_turn": fb_turn,
        "fb_pol": fb_pol,
        "fb_conf": fb_conf,
        "fb_rel": fb_rel,
        "contradict": contradict,
        "edges": edge_arr,
        "node_text": [nd["text"] for nd in nodes],
        "dec_turn": np.array(decisions_turn, dtype=np.int64),
        "dec_label": np.array(decisions_label, dtype=np.int64),
        "dec_query": np.stack(decisions_query).astype(np.float32) if decisions_query else np.zeros((0, utt_emb.shape[1]), np.float32),
        "dec_phase": np.array(decisions_phase, dtype=np.float32),
        "dec_response": decisions_response,
        "feedbacks": [turn["feedback"] if turn["feedback"] is not None else 0 for turn in turns],
    }
    return graph
