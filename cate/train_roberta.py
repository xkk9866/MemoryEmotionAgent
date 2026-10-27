"""RoBERTa window baselines for next-strategy prediction on ESConv."""

from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from cate.config import CKPT, ENCODER_NAME, HF_HOME, RESULT, STRATEGIES, STRATEGY2ID
from cate.data import assign_esconv_splits, load_esconv
from cate.metrics import classification_scores, length_bucket, macro_f1_by_bucket, preference_bias


class StrategySet(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def build_examples(dialogs, window: int):
    examples = []
    for dialog in dialogs:
        if dialog["strategy_split"] not in {"train", "dev", "test"}:
            continue
        turns = dialog["turns"]
        last_seeker = None
        for t, turn in enumerate(turns):
            if turn["role"] == "seeker":
                last_seeker = t
                continue
            if not turn["strategy"] or last_seeker is None:
                continue
            hist = turns[max(0, t - window) : t] if window > 0 else turns[:t]
            parts = []
            for prev in hist:
                tag = prev["role"]
                if prev["strategy"]:
                    tag += f" [{prev['strategy']}]"
                parts.append(f"{tag}: {prev['text']}")
            examples.append(
                {
                    "text": " </s> ".join(parts) if parts else "seeker: hello",
                    "label": STRATEGY2ID[turn["strategy"]],
                    "split": dialog["strategy_split"],
                    "hist": t,
                    "dialog_id": dialog["dialog_id"],
                }
            )
    return examples


def run_loader(model, tokenizer, loader, device, train, optimizer=None, scaler=None, weights=None):
    model.train(train)
    total = 0.0
    ys, preds, hists = [], [], []
    for batch in tqdm(loader, leave=False):
        tok = tokenizer(
            batch["text"], padding=True, truncation=True, max_length=384, return_tensors="pt"
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        labels = torch.tensor(batch["label"], device=device)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda"):
                logits = model(**tok).logits
                loss = F.cross_entropy(logits, labels, weight=weights)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        total += float(loss.item())
        pred = logits.argmax(-1).detach().cpu().tolist()
        ys.extend(batch["label"])
        preds.extend(pred)
        hists.extend(batch["hist"])
    stats = classification_scores(ys, preds, len(STRATEGIES))
    stats["bias"] = preference_bias(ys, preds, len(STRATEGIES))
    stats["loss"] = total / max(len(loader), 1)
    stats["by_length"] = macro_f1_by_bucket(ys, preds, [length_bucket(h) for h in hists], len(STRATEGIES))
    return stats, {"y": ys, "pred": preds, "hist": hists}


def collate(rows):
    return {
        "text": [r["text"] for r in rows],
        "label": [r["label"] for r in rows],
        "hist": [r["hist"] for r in rows],
    }


def main():
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    smoke = os.environ.get("SMOKE", "0") == "1"
    dialogs = load_esconv()
    assign_esconv_splits(dialogs)
    dialogs = [d for d in dialogs if d["failed"] == 0]
    if smoke:
        dialogs = dialogs[:60]
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    windows = [4] if smoke else [4, 12, 0]
    summary = []
    for window in windows:
        name = f"roberta_w{window if window else 'full'}"
        examples = build_examples(dialogs, window if window else 10_000)
        splits = {sp: [e for e in examples if e["split"] == sp] for sp in ("train", "dev", "test")}
        counts = np.bincount([e["label"] for e in splits["train"]], minlength=len(STRATEGIES)).astype(np.float32)
        weights = torch.tensor(counts.sum() / np.maximum(counts, 1.0), device=device)
        weights = weights / weights.mean()
        model = AutoModelForSequenceClassification.from_pretrained(
            ENCODER_NAME, num_labels=len(STRATEGIES)
        ).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda")
        train_loader = DataLoader(StrategySet(splits["train"]), batch_size=16, shuffle=True, collate_fn=collate)
        dev_loader = DataLoader(StrategySet(splits["dev"]), batch_size=16, shuffle=False, collate_fn=collate)
        test_loader = DataLoader(StrategySet(splits["test"]), batch_size=16, shuffle=False, collate_fn=collate)
        best = {"f1": -1.0, "state": None}
        epochs = 1 if smoke else 3
        for epoch in range(epochs):
            random.seed(42 + epoch)
            tr, _ = run_loader(model, tokenizer, train_loader, device, True, optim, scaler, weights)
            dv, _ = run_loader(model, tokenizer, dev_loader, device, False, weights=weights)
            print(f"{name} epoch {epoch} train {tr['macro_f1']:.4f} dev {dv['macro_f1']:.4f}", flush=True)
            if dv["macro_f1"] > best["f1"]:
                best = {
                    "f1": dv["macro_f1"],
                    "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                }
        model.load_state_dict(best["state"])
        te, preds = run_loader(model, tokenizer, test_loader, device, False, weights=weights)
        print(name, "TEST", te["macro_f1"], te["weighted_f1"], "bias", te["bias"], flush=True)
        summary.append({"variant": name, "best_dev": best["f1"], "test": te})
        (RESULT / f"preds_{name}.json").write_text(json.dumps(preds), encoding="utf-8")
        torch.save(best["state"], CKPT / f"{name}.pt")
    (RESULT / "roberta_windows.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
