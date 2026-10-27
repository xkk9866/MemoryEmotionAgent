"""Memory probes: retention, rejection, conflict, and failure detection."""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from cate.config import CACHE, CKPT, REL_CONTRADICT, RESULT, STRATEGIES, TYPE2ID
from cate.metrics import safe_auroc
from cate.models import DEGNet, collate_graphs, variant_flags


def _load_graphs():
    return torch.load(CACHE / "graphs.pt", map_location="cpu", weights_only=False)


def _load_model(dataset: str, name: str, n_emotion: int, seed: int = 42):
    flags = variant_flags(name)
    model = DEGNet(n_strategy=len(STRATEGIES), n_emotion=n_emotion)
    ckpt = torch.load(CKPT / f"{dataset}_{name}_s{seed}.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state"])
    model.cuda().eval()
    return model, flags


def _one(graph, query, turn, device):
    g = dict(graph)
    g["dec_turn"] = np.array([turn], dtype=np.int64)
    g["dec_query"] = np.asarray(query, dtype=np.float32)[None, :]
    g["dec_label"] = np.array([-100], dtype=np.int64)
    g["dec_phase"] = np.array([float(turn) / max(graph["n_turns"], 1)], dtype=np.float32)
    g["dec_response"] = [""]
    return collate_graphs([g], device)


@torch.no_grad()
def _forward(model, flags, graph, query, turn):
    batch = _one(graph, query, turn, torch.device("cuda"))
    return model(batch, flags)


def retention_at_k(dataset_graphs, model, flags, k=5) -> dict:
    hits, lengths = [], []
    for graph in dataset_graphs:
        if graph["anchor"] < 0 or graph["anchor_jaccard"] < 0.08:
            continue
        out = _forward(model, flags, graph, graph["situation_emb"], graph["n_turns"])
        scores = out["scores"][0, 0].detach().float().cpu()
        visible = out["visible"][0, 0].detach().cpu()
        scores = scores.masked_fill(~visible, -1e9)
        top = torch.topk(scores, k=min(k, scores.numel())).indices.tolist()
        hits.append(int(graph["anchor"] in top))
        lengths.append(int(graph["n_turns"]))
    hits = np.asarray(hits)
    lengths = np.asarray(lengths)
    def rate(mask):
        if mask.sum() == 0:
            return None
        return float(hits[mask].mean())
    return {
        "recall@5": float(hits.mean()) if len(hits) else None,
        "n": int(len(hits)),
        "len_1_16": rate(lengths <= 16),
        "len_17_32": rate((lengths >= 17) & (lengths <= 32)),
        "len_33+": rate(lengths >= 33),
    }


def numpy_retention(graphs, mode: str, k=5) -> dict:
    hits, lengths = [], []
    for graph in graphs:
        if graph["anchor"] < 0 or graph["anchor_jaccard"] < 0.08:
            continue
        emb = graph["emb"]
        utt = graph["is_utt"].astype(bool)
        turn = graph["turn"]
        q = graph["situation_emb"]
        if mode == "recency":
            idx = [i for i in range(len(turn)) if utt[i]]
            idx = sorted(idx, key=lambda i: turn[i])[-k:]
        elif mode == "rag":
            kn = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-6, None)
            qn = q / max(np.linalg.norm(q), 1e-6)
            sim = kn @ qn
            sim = np.where(utt, sim, -1e9)
            idx = np.argsort(-sim)[:k].tolist()
        elif mode == "forget":
            delta = graph["n_turns"] - turn
            strength = graph["conf"] * np.exp(-delta / 8.0)
            strength = np.where(utt, strength, -1e9)
            idx = np.argsort(-strength)[:k].tolist()
        else:
            raise KeyError(mode)
        hits.append(int(graph["anchor"] in idx))
        lengths.append(int(graph["n_turns"]))
    hits = np.asarray(hits)
    lengths = np.asarray(lengths)
    def rate(mask):
        return float(hits[mask].mean()) if mask.sum() else None
    return {
        "recall@5": float(hits.mean()) if len(hits) else None,
        "n": int(len(hits)),
        "len_1_16": rate(lengths <= 16),
        "len_17_32": rate((lengths >= 17) & (lengths <= 32)),
        "len_33+": rate(lengths >= 33),
    }


def rejection_persistence(graphs, model, flags) -> dict:
    """Whether a rejected strategy is still marked or retrieved after Δ turns."""
    buckets = {"1-4": [], "5-12": [], "13+": [], "window4": [], "deg_state": []}
    # Reuse buckets more carefully below.
    records = {name: [] for name in ("1-4", "5-12", "13+")}
    for graph in graphs:
        fb_events = []
        # Utterance i is node i. Feedback lives on the previous supporter node.
        for i, nd_fb in enumerate(graph["fb_turn"]):
            for slot, ft in enumerate(nd_fb):
                if ft >= 0 and graph["fb_pol"][i, slot] < -0.5 and graph["strategy"][i] >= 0:
                    fb_events.append((i, int(ft), int(graph["strategy"][i])))
        if not fb_events or len(graph["dec_turn"]) == 0:
            continue
        # One forward at every later decision.
        g = dict(graph)
        batch = collate_graphs([g], torch.device("cuda"))
        with torch.no_grad():
            out = model(batch, flags)
        scores = out["scores"][0].detach().float().cpu()
        visible = out["visible"][0].detach().cpu()
        disbelief = out["d"][0].detach().float().cpu()
        belief = out["b"][0].detach().float().cpu()
        for node_i, ft, _sid in fb_events:
            for j, t in enumerate(graph["dec_turn"].tolist()):
                if t <= ft:
                    continue
                delta = t - ft
                name = "1-4" if delta <= 4 else "5-12" if delta <= 12 else "13+"
                sc = scores[j].masked_fill(~visible[j], -1e9)
                top = set(torch.topk(sc, k=min(8, sc.numel())).indices.tolist())
                state = bool(disbelief[j, node_i] > belief[j, node_i] and disbelief[j, node_i] > 0.25)
                retrieved = node_i in top or ft in top
                if flags.get("local_channel") and delta <= 4:
                    # The local channel always keeps the last four turns in the readout.
                    retrieved = True
                records[name].append({"state": state, "retrieved": retrieved, "in_window4": delta <= 4})
    summary = {}
    for name, rows in records.items():
        if not rows:
            summary[name] = None
            continue
        summary[name] = {
            "n": len(rows),
            "rejection_state_rate": float(np.mean([r["state"] for r in rows])),
            "retrieved_rate": float(np.mean([r["retrieved"] for r in rows])),
            "recency4_covers_feedback": float(np.mean([r["in_window4"] for r in rows])),
        }
    return summary


def type_decay_gap(graphs, model, flags) -> dict:
    gaps = []
    for graph in graphs:
        if graph["n_turns"] < 16:
            continue
        out = _forward(model, flags, graph, graph["late_query"], graph["n_turns"])
        u = out["u"][0, 0].detach().float().cpu().numpy()
        visible = out["visible"][0, 0].detach().cpu().numpy()
        typ = graph["type"]
        turn = graph["turn"]
        early = turn <= max(graph["n_turns"] // 5, 1)
        emo = (typ == TYPE2ID["emotion"]) & early & visible
        event = (typ == TYPE2ID["event"]) & early & visible
        if emo.sum() == 0 or event.sum() == 0:
            continue
        gaps.append(float(u[emo].mean() - u[event].mean()))
    return {"uncertainty_gap_emotion_minus_event": float(np.mean(gaps)) if gaps else None, "n": len(gaps)}


def conflict_probe(graphs, deg_model, deg_flags, ow_model, ow_flags) -> dict:
    """Append an explicit contradiction and test whether the old node survives."""
    both_deg, both_ow, u_deg, u_ow, n = [], [], [], [], 0
    for graph in graphs:
        if graph["anchor"] < 0 or graph["anchor_jaccard"] < 0.05:
            continue
        if n >= 200:
            break
        g = _append_contradiction(graph)
        q = graph["situation_emb"]
        turn = g["n_turns"]
        for store, model, flags, bucket_b, bucket_u in (
            ("deg", deg_model, deg_flags, both_deg, u_deg),
            ("ow", ow_model, ow_flags, both_ow, u_ow),
        ):
            out = _forward(model, flags, g, q, turn)
            vis = out["visible"][0, 0].detach().cpu().numpy()
            u = out["u"][0, 0].detach().float().cpu().numpy()
            old_i = graph["anchor"]
            new_i = g["emb"].shape[0] - 1
            bucket_b.append(int(bool(vis[old_i]) and bool(vis[new_i])))
            if vis[old_i] or vis[new_i]:
                bucket_u.append(float(u[new_i] if vis[new_i] else 0.0))
        n += 1
    return {
        "n": n,
        "deg_both_retained": float(np.mean(both_deg)) if both_deg else None,
        "overwrite_both_retained": float(np.mean(both_ow)) if both_ow else None,
        "deg_new_uncertainty": float(np.mean(u_deg)) if u_deg else None,
        "overwrite_new_uncertainty": float(np.mean(u_ow)) if u_ow else None,
        "natural_contradict_edges": int(
            sum(int((g["edges"][:, 2] == REL_CONTRADICT).sum()) if len(g["edges"]) else 0 for g in graphs)
        ),
    }


def _append_contradiction(graph):
    g = {k: (v.copy() if isinstance(v, np.ndarray) else (list(v) if isinstance(v, list) else v)) for k, v in graph.items()}
    anchor = int(graph["anchor"])
    emb = graph["emb"][anchor].copy()
    # Flip the direction of the embedding so the new evidence is not a copy.
    emb = (-emb).astype(np.float32)
    n = graph["emb"].shape[0]
    def cat1(arr, value, dtype=None):
        value = np.asarray(value, dtype=dtype or arr.dtype)
        return np.concatenate([arr, value[None]], axis=0)
    g["emb"] = cat1(graph["emb"], emb, np.float32)
    g["type"] = cat1(graph["type"], TYPE2ID["event"], np.int64)
    g["turn"] = cat1(graph["turn"], graph["n_turns"], np.int64)
    g["strategy"] = cat1(graph["strategy"], -1, np.int64)
    g["pol"] = cat1(graph["pol"], -1.0, np.float32)
    g["conf"] = cat1(graph["conf"], 0.9, np.float32)
    g["rel"] = cat1(graph["rel"], 0.9, np.float32)
    g["b0"] = cat1(graph["b0"], 0.0, np.float32)
    g["d0"] = cat1(graph["d0"], 0.8, np.float32)
    g["u0"] = cat1(graph["u0"], 0.2, np.float32)
    g["is_utt"] = cat1(graph["is_utt"], 1, np.int64)
    g["role"] = cat1(graph["role"], 0, np.int64)
    k = graph["fb_turn"].shape[1]
    g["fb_turn"] = np.concatenate([graph["fb_turn"], np.full((1, k), -1, np.int64)], 0)
    g["fb_pol"] = np.concatenate([graph["fb_pol"], np.zeros((1, k), np.float32)], 0)
    g["fb_conf"] = np.concatenate([graph["fb_conf"], np.zeros((1, k), np.float32)], 0)
    g["fb_rel"] = np.concatenate([graph["fb_rel"], np.zeros((1, k), np.float32)], 0)
    contradict = np.concatenate([graph["contradict"], np.full((1, graph["contradict"].shape[1]), -1, np.int64)], 0)
    # Link both directions in the first free slot.
    for src, dst in ((anchor, n), (n, anchor)):
        slot = int(np.argmax(contradict[src] < 0)) if np.any(contradict[src] < 0) else 0
        contradict[src, slot] = dst
    g["contradict"] = contradict
    extra_edges = np.array([[anchor, n, REL_CONTRADICT], [n, anchor, REL_CONTRADICT]], dtype=np.int64)
    g["edges"] = np.concatenate([graph["edges"], extra_edges], 0) if len(graph["edges"]) else extra_edges
    g["node_text"] = list(graph["node_text"]) + ["[conflict] It is not true that the earlier situation still holds."]
    g["n_turns"] = int(graph["n_turns"]) + 1
    return g


def _dialogue_text(graph) -> str:
    return " ".join(graph["node_text"])


def failure_auroc(graphs) -> dict:
    rows = [g for g in graphs if g["fail_split"] in {"train", "dev", "test"}]
    def pack(split):
        return [g for g in rows if g["fail_split"] == split]

    def feats(g):
        fb = [x for x in g["feedbacks"] if x > 0]
        low = float(np.mean([x <= 2 for x in fb])) if fb else 0.0
        mean_fb = float(np.mean(fb)) if fb else 3.0
        last = float(fb[-1]) if fb else 3.0
        n_con = 0
        if len(g["edges"]):
            n_con = int((g["edges"][:, 2] == REL_CONTRADICT).sum() // 2)
        pol = g["pol"]
        turn = g["turn"]
        early = pol[turn <= max(g["n_turns"] // 3, 1)]
        late = pol[turn >= (2 * g["n_turns"]) // 3]
        return np.array(
            [
                g["n_turns"],
                np.mean([len(t) for t in g["node_text"]]),
                mean_fb,
                low,
                last,
                len(fb),
                n_con,
                float(early.mean()) if len(early) else 0.0,
                float(late.mean()) if len(late) else 0.0,
                float(late.mean() - early.mean()) if len(early) and len(late) else 0.0,
                float((g["type"] == TYPE2ID["risk"]).any()),
            ],
            dtype=np.float32,
        )

    feature_sets = {
        "length": [0, 1],
        "feedback": [2, 3, 4, 5],
        "graph_no_feedback": [0, 6, 7, 8, 9, 10],
        "graph_plus_feedback": list(range(11)),
    }
    y_train = np.array([g["failed"] for g in pack("train")])
    y_test = np.array([g["failed"] for g in pack("test")])
    x_train = np.stack([feats(g) for g in pack("train")])
    x_test = np.stack([feats(g) for g in pack("test")])
    out = {"n_test": int(len(y_test)), "n_test_failed": int(y_test.sum())}
    for name, cols in feature_sets.items():
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(x_train[:, cols], y_train)
        score = clf.predict_proba(x_test[:, cols])[:, 1]
        out[name] = safe_auroc(y_test, score)
    texts_train = [_dialogue_text(g) for g in pack("train")]
    texts_test = [_dialogue_text(g) for g in pack("test")]
    pipe = make_pipeline(TfidfVectorizer(max_features=20000, ngram_range=(1, 2)), LogisticRegression(max_iter=500, class_weight="balanced"))
    pipe.fit(texts_train, y_train)
    out["tfidf_text"] = safe_auroc(y_test, pipe.predict_proba(texts_test)[:, 1])
    return out


def main():
    payload = _load_graphs()
    es = [g for g in payload["esconv"] if g["strategy_split"] == "test"]
    n_emo = len(payload["esconv_emotions"])
    report = {"retention": {}, "rejection": {}, "type_decay": {}, "failure": failure_auroc(payload["esconv"])}
    for mode in ("recency", "rag", "forget"):
        report["retention"][mode] = numpy_retention(es, mode)
    deg, deg_flags = _load_model("esconv", "deg", n_emo)
    ow, ow_flags = _load_model("esconv", "overwrite", n_emo)
    shared, shared_flags = _load_model("esconv", "deg_shared_hl", n_emo)
    nodecay, nodecay_flags = _load_model("esconv", "deg_no_decay", n_emo)
    report["retention"]["deg"] = retention_at_k(es, deg, deg_flags)
    report["retention"]["overwrite"] = retention_at_k(es, ow, ow_flags)
    report["retention"]["deg_no_decay"] = retention_at_k(es, nodecay, nodecay_flags)
    report["rejection"]["deg"] = rejection_persistence(es, deg, deg_flags)
    report["rejection"]["overwrite"] = rejection_persistence(es, ow, ow_flags)
    report["rejection"]["deg_no_decay"] = rejection_persistence(es, nodecay, nodecay_flags)
    report["type_decay"]["deg"] = type_decay_gap(es, deg, deg_flags)
    report["type_decay"]["deg_shared_hl"] = type_decay_gap(es, shared, shared_flags)
    report["type_decay"]["deg_no_decay"] = type_decay_gap(es, nodecay, nodecay_flags)
    report["conflict"] = conflict_probe(es, deg, deg_flags, ow, ow_flags)
    path = RESULT / "memory_probes.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
