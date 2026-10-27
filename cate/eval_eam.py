"""Probes for acceptance memory: calibration, persistence, and re-suggestion.

Labels come from seeker feedback that the model does not receive as a number.
The reader only sees utterance text. Metrics are rates and gaps, computed on
the ESConv strategy test split.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from cate.config import CACHE, CKPT, ENCODER_NAME, RESULT
from cate.data import assign_esconv_splits, load_esconv
from cate.eam import EAMNet, eam_flags
from cate.train_eam import build_examples, collate, dialog_streams


def _load(name, seed, n_labels, device):
    model = EAMNet(ENCODER_NAME, n_labels=n_labels).to(device)
    ckpt = torch.load(CKPT / f"eam_{name}_s{seed}.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model


@torch.no_grad()
def acceptance_at(model, tokenizer, streams, examples, flags, device):
    """Return per-example acceptance opinions and the feedback events before the decision."""
    rows = []
    bs = 16
    for start in range(0, len(examples), bs):
        batch = examples[start : start + bs]
        tok, stream, labels = collate(batch, streams, tokenizer, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(tok, stream, flags)
        acc = out["acc"].float().cpu().numpy()
        e_pos = out["e_pos"].float().cpu().numpy()
        e_neg = out["e_neg"].float().cpu().numpy()
        for i, ex in enumerate(batch):
            s = streams[ex["dialog_id"]]
            t = ex["t"]
            events = []
            for u in range(t):
                if s["fb"][u] < 0 or s["eval"][u] < 0:
                    continue
                events.append(
                    {
                        "turn": u,
                        "strategy": int(s["eval"][u]),
                        "accepted": int(s["fb"][u] == 1),
                        "e_pos": float(e_pos[i, u]),
                        "e_neg": float(e_neg[i, u]),
                    }
                )
            rows.append(
                {
                    "dialog_id": ex["dialog_id"],
                    "t": t,
                    "label": ex["label"],
                    "acc": acc[i],
                    "events": events,
                }
            )
    return rows


def calibration(rows):
    """Does the reaction evidence separate accepted and rejected turns?"""
    pos, neg, y = [], [], []
    for row in rows:
        for ev in row["events"]:
            pos.append(ev["e_pos"])
            neg.append(ev["e_neg"])
            y.append(ev["accepted"])
    y = np.asarray(y)
    score = np.asarray(pos) - np.asarray(neg)
    if len(np.unique(y)) < 2:
        return None
    # Binned empirical rate, five bins.
    edges = np.quantile(score, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    bins = []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (score >= lo) & (score < hi)
        if m.sum() == 0:
            continue
        pred = float(1 / (1 + np.exp(-score[m].mean())))
        emp = float(y[m].mean())
        ece += float(m.mean()) * abs(pred - emp)
        bins.append({"n": int(m.sum()), "mean_evidence_gap": float(score[m].mean()), "accept_rate": emp})
    return {"auroc": float(roc_auc_score(y, score)), "ece": float(ece), "n": int(len(y)), "bins": bins}


def persistence(rows):
    """At late decisions, are previously rejected strategies still less accepted?"""
    gaps = []
    for row in rows:
        if row["t"] < 16 or len(row["events"]) < 2:
            continue
        acc = row["acc"]
        rej = [e["strategy"] for e in row["events"] if e["accepted"] == 0 and row["t"] - e["turn"] >= 8]
        accp = [e["strategy"] for e in row["events"] if e["accepted"] == 1 and row["t"] - e["turn"] >= 8]
        rej, accp = sorted(set(rej)), sorted(set(accp))
        if not rej or not accp:
            continue
        # acceptance margin b - d
        margin = acc[:, 0] - acc[:, 1]
        gaps.append(float(margin[accp].mean() - margin[rej].mean()))
    if not gaps:
        return None
    gaps = np.asarray(gaps)
    return {
        "n_decisions": int(len(gaps)),
        "mean_margin_gap": float(gaps.mean()),
        "share_positive_gap": float((gaps > 0).mean()),
    }


def resuggestion(rows, preds_by_key):
    """After a rejection, how often the next three supporter turns repeat that strategy."""
    # group decisions by dialog
    by = {}
    for row in rows:
        by.setdefault(row["dialog_id"], []).append(row)
    rates_model, rates_gold = [], []
    for decisions in by.values():
        decisions = sorted(decisions, key=lambda r: r["t"])
        events = decisions[-1]["events"] if decisions else []
        for ev in events:
            if ev["accepted"] != 0:
                continue
            later = [d for d in decisions if d["t"] > ev["turn"]][:3]
            if not later:
                continue
            gold_hit = any(d["label"] == ev["strategy"] for d in later)
            pred_hit = any(preds_by_key[(d["dialog_id"], d["t"])] == ev["strategy"] for d in later)
            rates_gold.append(float(gold_hit))
            rates_model.append(float(pred_hit))
    if not rates_model:
        return None
    return {
        "n": int(len(rates_model)),
        "model_repeat_rate": float(np.mean(rates_model)),
        "gold_repeat_rate": float(np.mean(rates_gold)),
    }


def dissonance(rows):
    """Strategies that received both ratings should carry more dissonance than one-sided ones."""
    both, single = [], []
    for row in rows:
        if row["t"] < 12:
            continue
        acc = row["acc"]
        by_s = {}
        for ev in row["events"]:
            if row["t"] - ev["turn"] < 4:
                continue
            by_s.setdefault(ev["strategy"], set()).add(ev["accepted"])
        for sid, kinds in by_s.items():
            (both if len(kinds) == 2 else single).append(float(acc[sid, 3]))
    if len(both) < 20 or len(single) < 20:
        return None
    return {
        "n_conflict": int(len(both)),
        "n_consistent": int(len(single)),
        "dissonance_conflict": float(np.mean(both)),
        "dissonance_consistent": float(np.mean(single)),
    }


def main():
    device = torch.device("cuda")
    rows = load_esconv()
    assign_esconv_splits(rows)
    payload = torch.load(CACHE / "graphs.pt", map_location="cpu", weights_only=False)
    graphs_by_id = {g["dialog_id"]: g for g in payload["esconv"]}
    reactions = torch.load(CACHE / "reactions.pt", map_location="cpu", weights_only=False)
    rows = [r for r in rows if r["failed"] == 0]
    streams = dialog_streams(rows, graphs_by_id, reactions)
    examples = [e for e in build_examples(rows) if e["split"] == "test"]
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    report = {}
    for name in ("eam", "eam_net", "eam_no_decay", "base"):
        flags = eam_flags(name)
        if not flags["memory"]:
            continue
        path = RESULT / "eam" / f"{name}_s42.json"
        if not path.exists() or not (CKPT / f"eam_{name}_s42.pt").exists():
            print("missing", name, flush=True)
            continue
        preds = json.loads(path.read_text(encoding="utf-8"))["preds"]
        pred_map = {(a, b): p for (a, b), p in zip(preds["meta"], preds["pred"])}
        model = _load(name, 42, 8, device)
        got = acceptance_at(model, tokenizer, streams, examples, flags, device)
        report[name] = {
            "calibration": calibration(got),
            "persistence": persistence(got),
            "resuggestion": resuggestion(got, pred_map),
            "dissonance": dissonance(got),
        }
        del model
        torch.cuda.empty_cache()
        print(name, json.dumps(report[name]), flush=True)
    out = RESULT / "eam_probes.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
