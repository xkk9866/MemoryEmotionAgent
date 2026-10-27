"""Train EAM and its ablations for next-strategy prediction on ESConv.

Every variant shares the same RoBERTa window encoder, split, schedule and
seeds, so differences come from the acceptance memory alone.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from cate.config import CACHE, CKPT, ENCODER_NAME, HF_HOME, RESULT, STRATEGIES, STRATEGY2ID
from cate.data import assign_esconv_splits, load_esconv
from cate.eam import EAMNet, eam_flags, evidential_feedback_loss
from cate.metrics import classification_scores, length_bucket, macro_f1_by_bucket, preference_bias

WINDOW = 4
MAX_LEN = 256
FB_WEIGHT = 0.5


def _cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cpu(v) for v in obj]
    return obj


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dialog_streams(rows, graphs_by_id, reactions):
    """Per dialogue arrays aligned with utterance index."""
    streams = {}
    for row in rows:
        g = graphs_by_id[row["dialog_id"]]
        turns = row["turns"]
        n = len(turns)
        role = np.array([0 if t["role"] == "seeker" else 1 for t in turns], dtype=np.int64)
        sup = np.array(
            [STRATEGY2ID[t["strategy"]] if (t["role"] == "supporter" and t["strategy"]) else -1 for t in turns],
            dtype=np.int64,
        )
        eval_strat = np.full(n, -1, dtype=np.int64)
        last = -1
        for i, t in enumerate(turns):
            if t["role"] == "supporter" and t["strategy"]:
                last = STRATEGY2ID[t["strategy"]]
            elif t["role"] == "seeker":
                eval_strat[i] = last
        fb = np.full(n, -1, dtype=np.int64)
        for i, t in enumerate(turns):
            if t["role"] == "seeker" and t["feedback"] is not None:
                if t["feedback"] <= 2:
                    fb[i] = 0
                elif t["feedback"] >= 4:
                    fb[i] = 1
        react = np.zeros((n, 768), dtype=np.float32)
        cached = reactions.get(row["dialog_id"])
        if cached is not None:
            m = min(n, len(cached))
            react[:m] = np.asarray(cached[:m], dtype=np.float32)
        streams[row["dialog_id"]] = {
            "emb": g["emb"][:n].astype(np.float32),
            "react": react,
            "pol": g["pol"][:n].astype(np.float32),
            "role": role,
            "sup": sup,
            "eval": eval_strat,
            "fb": fb,
            "n": n,
        }
    return streams


def build_examples(rows):
    out = []
    for row in rows:
        turns = row["turns"]
        seen_seeker = False
        for t, turn in enumerate(turns):
            if turn["role"] == "seeker":
                seen_seeker = True
                continue
            if not turn["strategy"] or not seen_seeker:
                continue
            hist = turns[max(0, t - WINDOW) : t]
            parts = []
            for prev in hist:
                tag = prev["role"]
                if prev["strategy"]:
                    tag += f" [{prev['strategy']}]"
                parts.append(f"{tag}: {prev['text']}")
            out.append(
                {
                    "text": " </s> ".join(parts),
                    "label": STRATEGY2ID[turn["strategy"]],
                    "t": t,
                    "dialog_id": row["dialog_id"],
                    "split": row["strategy_split"],
                }
            )
    return out


def collate(batch, streams, tokenizer, device):
    tok = tokenizer(
        [b["text"] for b in batch], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt"
    )
    tok = {k: v.to(device) for k, v in tok.items()}
    L = max(max(b["t"], 1) for b in batch)
    B = len(batch)
    emb = torch.zeros(B, L, 768)
    react = torch.zeros(B, L, 768)
    role = torch.ones(B, L, dtype=torch.long)
    sup = torch.full((B, L), -1, dtype=torch.long)
    ev = torch.full((B, L), -1, dtype=torch.long)
    fb = torch.full((B, L), -1, dtype=torch.long)
    turn = torch.zeros(B, L, dtype=torch.long)
    mask = torch.zeros(B, L, dtype=torch.bool)
    qt = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        s = streams[b["dialog_id"]]
        t = b["t"]
        if t > 0:
            emb[i, :t] = torch.from_numpy(s["emb"][:t])
            react[i, :t] = torch.from_numpy(s["react"][:t])
            role[i, :t] = torch.from_numpy(s["role"][:t])
            sup[i, :t] = torch.from_numpy(s["sup"][:t])
            ev[i, :t] = torch.from_numpy(s["eval"][:t])
            fb[i, :t] = torch.from_numpy(s["fb"][:t])
            turn[i, :t] = torch.arange(t)
            mask[i, :t] = True
        qt[i] = t
    stream = {
        "emb": emb.to(device),
        "react": react.to(device),
        "role": role.to(device),
        "sup_strat": sup.to(device),
        "eval_strat": ev.to(device),
        "fb": fb.to(device),
        "turn": turn.to(device),
        "mask": mask.to(device),
        "query_turn": qt.to(device),
    }
    labels = torch.tensor([b["label"] for b in batch], device=device)
    return tok, stream, labels


def run(model, examples, streams, tokenizer, device, flags, train, optim=None, weights=None, bs=16):
    model.train(train)
    order = list(range(len(examples)))
    if train:
        random.shuffle(order)
    ys, ps, hs, meta = [], [], [], []
    total = 0.0
    steps = 0
    for start in range(0, len(order), bs):
        batch = [examples[i] for i in order[start : start + bs]]
        tok, stream, labels = collate(batch, streams, tokenizer, device)
        with torch.set_grad_enabled(train), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(tok, stream, flags)
            logits = out["logits"].float()
            loss = F.cross_entropy(logits, labels, weight=weights)
            if train and flags["fb_sup"] and "e_pos" in out:
                valid = (stream["fb"] >= 0) & stream["mask"] & (stream["eval_strat"] >= 0)
                loss = loss + FB_WEIGHT * evidential_feedback_loss(
                    out["e_pos"].float(), out["e_neg"].float(), stream["fb"], valid
                )
        if train:
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        total += float(loss.detach())
        steps += 1
        if not train:
            pred = logits.argmax(-1).cpu().tolist()
            ys.extend(labels.cpu().tolist())
            ps.extend(pred)
            hs.extend([b["t"] for b in batch])
            meta.extend([(b["dialog_id"], b["t"]) for b in batch])
    if train:
        return {"loss": total / max(steps, 1)}, None
    stats = classification_scores(ys, ps, len(STRATEGIES))
    stats["bias"] = preference_bias(ys, ps, len(STRATEGIES))
    stats["by_length"] = macro_f1_by_bucket(ys, ps, [length_bucket(h) for h in hs], len(STRATEGIES))
    return stats, {"y": ys, "pred": ps, "hist": hs, "meta": meta}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--encoder", default=ENCODER_NAME)
    parser.add_argument("--tag", default="roberta-base")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--bs", type=int, default=16)
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    device = torch.device("cuda")
    rows = load_esconv()
    assign_esconv_splits(rows)
    payload = torch.load(CACHE / "graphs.pt", map_location="cpu", weights_only=False)
    graphs_by_id = {g["dialog_id"]: g for g in payload["esconv"]}
    reactions = torch.load(CACHE / "reactions.pt", map_location="cpu", weights_only=False)
    rows = [r for r in rows if r["failed"] == 0]
    streams = dialog_streams(rows, graphs_by_id, reactions)
    examples = build_examples(rows)
    splits = {sp: [e for e in examples if e["split"] == sp] for sp in ("train", "dev", "test")}
    print({k: len(v) for k, v in splits.items()}, flush=True)
    counts = np.bincount([e["label"] for e in splits["train"]], minlength=len(STRATEGIES)).astype(np.float32)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1.0), device=device)
    weights = weights / weights.mean()
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    sub = "eam" if args.tag == "roberta-base" else f"eam_{args.tag}"
    for name in args.variants:
        flags = eam_flags(name)
        for seed in args.seeds:
            out_path = RESULT / sub / f"{name}_s{seed}.json"
            if out_path.exists():
                print("skip", out_path, flush=True)
                continue
            set_seed(seed)
            model = EAMNet(args.encoder).to(device)
            enc_params = list(model.encoder.parameters())
            enc_ids = {id(p) for p in enc_params}
            other = [p for p in model.parameters() if id(p) not in enc_ids]
            optim = torch.optim.AdamW(
                [{"params": enc_params, "lr": args.lr}, {"params": other, "lr": 2e-4}], weight_decay=0.01
            )
            best = {"f1": -1.0, "state": None, "epoch": -1}
            partial = CKPT / f"partial_{args.tag}_{name}_s{seed}.pt"
            start_epoch = 0
            if partial.exists():
                saved = torch.load(partial, map_location="cpu", weights_only=False)
                model.load_state_dict(saved["model"])
                if "best" in saved:
                    best = saved["best"]
                else:
                    best = {"f1": saved["best_f1"], "epoch": saved["best_epoch"], "state": saved["best_state"]}
                start_epoch = int(saved["epoch"]) + 1
                print(f"resume {name} s{seed} from epoch {start_epoch}", flush=True)
            for epoch in range(start_epoch, args.epochs):
                random.seed(seed * 100 + epoch)
                tr, _ = run(model, splits["train"], streams, tokenizer, device, flags, True, optim, weights, bs=args.bs)
                dv, _ = run(model, splits["dev"], streams, tokenizer, device, flags, False, weights=weights)
                scale = float(torch.nn.functional.softplus(model.alpha).detach()) if flags.get("closed", True) else float(torch.sigmoid(model.gate).detach())
                print(
                    f"{name} s{seed} ep{epoch} loss {tr['loss']:.4f} dev {dv['macro_f1']:.4f} scale {scale:.3f}",
                    flush=True,
                )
                if dv["macro_f1"] > best["f1"]:
                    best = {
                        "f1": dv["macro_f1"],
                        "epoch": epoch,
                        "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    }
                blob = {
                    "epoch": epoch,
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "best_f1": best["f1"],
                    "best_epoch": best["epoch"],
                    "best_state": best["state"],
                }
                tmp = partial.with_suffix(".tmp")
                torch.save(blob, tmp)
                del blob
                os.replace(tmp, partial)
            model.load_state_dict(best["state"])
            te, preds = run(model, splits["test"], streams, tokenizer, device, flags, False, weights=weights)
            print(f"{name} s{seed} TEST macro {te['macro_f1']:.4f} w {te['weighted_f1']:.4f}", flush=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps({"variant": name, "seed": seed, "best_dev": best["f1"], "best_epoch": best["epoch"], "test": te, "preds": preds}),
                encoding="utf-8",
            )
            if sub == "eam":
                torch.save(
                    {"state": best["state"], "variant": name, "seed": seed},
                    CKPT / f"eam_{name}_s{seed}.pt",
                )
            if partial.exists():
                partial.unlink()
            del model, optim, best
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
