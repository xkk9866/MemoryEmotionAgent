"""Retrieval-based responses conditioned on each memory, scored with BERTScore.

Nearest-neighbour reply lookup uses the seeker query AND the mean embedding
of the nodes that memory actually retrieved. Query-only lookup would make
every memory system return the same hypothesis.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from cate.config import RESULT
from cate.evaluate import _forward, _load_graphs, _load_model


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-6, None)


def _memory_text(graph, indices) -> str:
    lines = []
    for i in indices:
        if i < 0 or i >= len(graph["node_text"]):
            continue
        kind = graph["type"][i]
        lines.append(f"[t{int(graph['turn'][i])}/type{int(kind)}] {graph['node_text'][i]}")
    return " \n ".join(lines)


def _memory_vec(graph, indices) -> np.ndarray:
    if not indices:
        return np.zeros(graph["emb"].shape[1], dtype=np.float32)
    return graph["emb"][np.asarray(indices, dtype=np.int64)].mean(axis=0).astype(np.float32)


def _state(query: np.ndarray, memory: np.ndarray) -> np.ndarray:
    return _unit(np.concatenate([_unit(query), _unit(memory)], axis=0))


def collect(graphs, mode: str, model=None, flags=None, k=4):
    rows = []
    for graph in graphs:
        if len(graph["dec_turn"]) == 0:
            continue
        emb = graph["emb"]
        for j, turn in enumerate(graph["dec_turn"].tolist()):
            query = graph["dec_query"][j]
            if mode == "recency":
                idx = [i for i in range(len(graph["turn"])) if graph["turn"][i] < turn and graph["is_utt"][i]]
                idx = idx[-k:]
            elif mode == "rag":
                past = graph["turn"] < turn
                sim = _unit(emb) @ _unit(query)
                sim = np.where(past & graph["is_utt"].astype(bool), sim, -1e9)
                idx = np.argsort(-sim)[:k].tolist()
            elif mode == "deg":
                out = _forward(model, flags, graph, query, int(turn))
                scores = out["scores"][0, 0].detach().float().cpu()
                visible = out["visible"][0, 0].detach().cpu()
                scores = scores.masked_fill(~visible, -1e9)
                idx = torch.topk(scores, k=min(8, scores.numel())).indices.tolist()
                local = [
                    i
                    for i in range(len(graph["turn"]))
                    if graph["is_utt"][i] and turn - 4 <= graph["turn"][i] < turn
                ]
                idx = list(dict.fromkeys(local + idx))
            else:
                raise KeyError(mode)
            mem = _memory_vec(graph, idx)
            rows.append(
                {
                    "state": _state(query, mem),
                    "response": graph["dec_response"][j],
                    "memory": _memory_text(graph, idx),
                    "split": graph["strategy_split"],
                }
            )
    return rows


def retrieve(train_rows, test_rows):
    bank = np.stack([r["state"] for r in train_rows])
    preds = []
    for row in test_rows:
        sim = bank @ row["state"]
        preds.append(train_rows[int(np.argmax(sim))]["response"])
    return preds


def main():
    payload = _load_graphs()
    graphs = [g for g in payload["esconv"] if g["failed"] == 0]
    train = [g for g in graphs if g["strategy_split"] == "train"]
    test = [g for g in graphs if g["strategy_split"] == "test"]
    n_emo = len(payload["esconv_emotions"])
    deg, flags = _load_model("esconv", "deg", n_emo)
    report = {}
    for mode, model, fl in (
        ("recency", None, None),
        ("rag", None, None),
        ("deg", deg, flags),
    ):
        print("collect", mode, flush=True)
        tr = collect(train, mode, model, fl)
        te = collect(test, mode, model, fl)
        hyps = retrieve(tr, te)
        gold = [r["response"] for r in te]
        import bert_score

        _, _, f1 = bert_score.score(hyps, gold, lang="en", verbose=False, batch_size=32)
        report[mode] = {
            "bertscore_f1": float(f1.mean()),
            "n": len(gold),
            "examples": [
                {"memory": te[i]["memory"][:400], "hyp": hyps[i][:240], "gold": gold[i][:240]}
                for i in range(3)
            ],
        }
        print(mode, report[mode]["bertscore_f1"], flush=True)
    bleurt = None
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        name = "lucadiliello/BLEURT-20-D3"
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForSequenceClassification.from_pretrained(name).cuda().eval()
        te = collect(test, "deg", deg, flags)
        tr = collect(train, "deg", deg, flags)
        hyps = retrieve(tr, te)
        gold = [r["response"] for r in te]
        scores = []
        for i in range(0, len(hyps), 16):
            batch = tok(
                hyps[i : i + 16],
                gold[i : i + 16],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.no_grad():
                scores.append(mdl(**batch).logits.squeeze(-1).float().cpu())
        bleurt = float(torch.cat(scores).mean())
    except Exception as exc:  # noqa: BLE001
        bleurt = f"unavailable: {type(exc).__name__}: {exc}"
    report["deg_bleurt"] = bleurt
    path = RESULT / "generation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: (
                    v
                    if not isinstance(v, dict)
                    else {kk: vv for kk, vv in v.items() if kk != "examples"}
                )
                for k, v in report.items()
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
