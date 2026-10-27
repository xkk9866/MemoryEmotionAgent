"""Train flat memories, window graphs, overwrite graphs, and DEG-Mem."""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cate.config import CACHE, CKPT, RESULT, STRATEGIES
from cate.metrics import classification_scores, length_bucket, macro_f1_by_bucket, preference_bias
from cate.models import DEGNet, collate_graphs, variant_flags


ESCONV_VARIANTS = [
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
ED_VARIANTS = [
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_ce(logits, labels, weight=None):
    if logits.dim() == 3:
        logits = logits.reshape(-1, logits.size(-1))
        labels = labels.reshape(-1)
    keep = labels != -100
    if not bool(keep.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[keep], labels[keep], weight=weight)


def retention_loss(scores, anchor):
    valid = anchor >= 0
    if not bool(valid.any()):
        return scores.sum() * 0.0
    target = anchor.clamp(min=0)
    per = F.cross_entropy(scores, target, reduction="none")
    return (per * valid.float()).sum() / valid.float().sum().clamp(min=1.0)


def run_split(model, graphs, flags, class_weight, train, batch_size, optimizer=None):
    if train:
        model.train()
        order = list(graphs)
        random.shuffle(order)
    else:
        model.eval()
        order = list(graphs)
    total = 0.0
    steps = 0
    ys, preds, hists = [], [], []
    emo_y, emo_p = [], []
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        batch = collate_graphs(chunk, next(model.parameters()).device)
        with torch.set_grad_enabled(train):
            out = model(batch, flags)
            loss_s = masked_ce(out["strategy_logits"], batch["labels"], class_weight)
            emo_logits, anchor_scores = model.emotion_forward(batch, flags)
            loss_e = masked_ce(emo_logits, batch["emotion_id"])
            loss_r = retention_loss(anchor_scores, batch["anchor"]) if flags["ret_loss"] else loss_s * 0.0
            loss = loss_s + 0.3 * loss_e + 0.2 * loss_r
            if train:
                optimizer.zero_grad(set_to_none=True)
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
        if torch.isfinite(loss):
            total += float(loss.detach())
            steps += 1
        if not train:
            pred = out["strategy_logits"].argmax(-1).detach().cpu().numpy()
            labels = batch["labels"].detach().cpu().numpy()
            hist = batch["hist"].detach().cpu().numpy()
            for i, graph in enumerate(chunk):
                m = len(graph["dec_label"])
                for j in range(m):
                    if labels[i, j] < 0:
                        continue
                    ys.append(int(labels[i, j]))
                    preds.append(int(pred[i, j]))
                    hists.append(int(hist[i, j]))
            emo_pred = emo_logits.argmax(-1).detach().cpu().numpy()
            for i, graph in enumerate(chunk):
                emo_y.append(int(graph["emotion_id"]))
                emo_p.append(int(emo_pred[i]))
    stats = {
        "loss": total / max(steps, 1),
        "strategy": classification_scores(ys, preds, len(STRATEGIES)) if ys else None,
        "emotion": classification_scores(emo_y, emo_p) if emo_y else None,
    }
    if ys:
        stats["strategy"]["bias"] = preference_bias(ys, preds, len(STRATEGIES))
        stats["by_length"] = macro_f1_by_bucket(ys, preds, [length_bucket(h) for h in hists], len(STRATEGIES))
    return stats, {"y": ys, "pred": preds, "hist": hists, "emo_y": emo_y, "emo_p": emo_p}


def class_weights(graphs, n_labels: int, device):
    counts = np.zeros(n_labels, dtype=np.float64)
    for graph in graphs:
        for lab in graph["dec_label"].tolist():
            if 0 <= int(lab) < n_labels:
                counts[int(lab)] += 1
    if counts.sum() == 0:
        return None
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["esconv", "ed"], default="esconv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--variants", nargs="*", default=None)
    args = parser.parse_args()
    set_seed(args.seed)
    smoke = os.environ.get("SMOKE", "0") == "1"
    path = CACHE / ("graphs_smoke.pt" if smoke else "graphs.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    graphs = payload[args.dataset]
    if args.dataset == "esconv":
        split_key = "strategy_split"
        n_emotion = len(payload["esconv_emotions"])
        variants = args.variants or ESCONV_VARIANTS
    else:
        split_key = "split"
        n_emotion = len(payload["ed_emotions"])
        variants = args.variants or ED_VARIANTS
    groups = {"train": [], "dev": [], "test": []}
    for graph in graphs:
        sp = graph.get(split_key, "train")
        if sp in groups:
            groups[sp].append(graph)
    print({k: len(v) for k, v in groups.items()}, "emotions", n_emotion, flush=True)
    device = torch.device("cuda")
    epochs = 1 if smoke else args.epochs
    summary = []
    for name in variants:
        set_seed(args.seed)
        flags = variant_flags(name)
        # EmpatheticDialogues dialogues are short. A 4-turn local channel plus
        # a separate old-evidence budget leaves almost no persistent nodes and
        # collapses the emotion head. Keep retrieval over the full visible graph.
        if args.dataset == "ed":
            flags["local_channel"] = False
        model = DEGNet(n_strategy=len(STRATEGIES), n_emotion=n_emotion).to(device)
        weight = class_weights(groups["train"], len(STRATEGIES), device)
        lr = 3e-4 if args.dataset == "ed" else 8e-4
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        best = {"f1": -1.0, "state": None, "epoch": -1}
        patience = 0
        for epoch in range(epochs):
            train_stats, _ = run_split(model, groups["train"], flags, weight, True, args.batch_size, optim)
            dev_stats, _ = run_split(model, groups["dev"], flags, weight, False, args.batch_size)
            score = 0.0
            if args.dataset == "esconv" and dev_stats["strategy"]:
                score = dev_stats["strategy"]["macro_f1"]
            elif dev_stats["emotion"]:
                score = dev_stats["emotion"]["macro_f1"]
            print(
                f"{name} epoch {epoch} train_loss {train_stats['loss']:.4f} dev {score:.4f}",
                flush=True,
            )
            if score > best["f1"]:
                best = {
                    "f1": score,
                    "epoch": epoch,
                    "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                }
                patience = 0
            else:
                patience += 1
                if patience >= 3 and not smoke:
                    break
        model.load_state_dict(best["state"])
        test_stats, preds = run_split(model, groups["test"], flags, weight, False, args.batch_size)
        row = {"variant": name, "dataset": args.dataset, "seed": args.seed, "best_dev": best["f1"], "test": test_stats}
        summary.append(row)
        ckpt = CKPT / f"{args.dataset}_{name}_s{args.seed}.pt"
        torch.save({"state": best["state"], "flags": flags, "n_emotion": n_emotion}, ckpt)
        pred_path = RESULT / f"preds_{args.dataset}_{name}_s{args.seed}.json"
        pred_path.write_text(json.dumps(preds), encoding="utf-8")
        print(name, "TEST", json.dumps(test_stats)[:500], flush=True)
    out = RESULT / f"{args.dataset}_memory_s{args.seed}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
