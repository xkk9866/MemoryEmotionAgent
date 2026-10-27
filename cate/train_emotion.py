"""Fine-tune a RoBERTa emotion tagger on GoEmotions (multi-label)."""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from cate.config import CKPT, ENCODER_NAME, GOEMO_LABELS, HF_HOME, RESULT


class GoEmoSet(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        y = torch.zeros(len(GOEMO_LABELS))
        for lab in self.labels[idx]:
            y[int(lab)] = 1.0
        return self.texts[idx], y


def _f1(probs: np.ndarray, gold: np.ndarray, threshold: float) -> float:
    pred = (probs >= threshold).astype(int)
    return float(f1_score(gold, pred, average="macro", zero_division=0))


def best_threshold(probs, gold) -> tuple[float, float]:
    best_t, best_s = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.6, 12):
        score = _f1(probs, gold, float(threshold))
        if score > best_s:
            best_t, best_s = float(threshold), score
    return best_t, best_s


def _collect(model, head, loader, device):
    model.eval()
    head.eval()
    probs, golds = [], []
    with torch.no_grad():
        for texts, y in loader:
            tok = tokenizer_batch(texts, device)
            hidden = model(**tok).last_hidden_state[:, 0, :]
            logit = head(hidden)
            probs.append(torch.sigmoid(logit).float().cpu().numpy())
            golds.append(y.numpy())
    return np.concatenate(probs), np.concatenate(golds)


def tokenizer_batch(texts, device, tokenizer=None, max_len=128):
    raise RuntimeError("tokenizer_batch must be bound in main")


def main():
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    smoke = os.environ.get("SMOKE", "0") == "1"
    from datasets import load_dataset

    raw = load_dataset("google-research-datasets/go_emotions", "simplified")
    if smoke:
        raw = {sp: raw[sp].select(range(min(512 if sp == "train" else 128, len(raw[sp])))) for sp in raw}

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    encoder = AutoModel.from_pretrained(ENCODER_NAME).to(device)
    head = nn.Linear(encoder.config.hidden_size, len(GOEMO_LABELS)).to(device)

    def make_loader(split, shuffle):
        ds = GoEmoSet(split["text"], split["labels"])
        return DataLoader(ds, batch_size=32, shuffle=shuffle, collate_fn=lambda xs: (
            [x[0] for x in xs],
            torch.stack([x[1] for x in xs]),
        ))

    global tokenizer_batch

    def tokenizer_batch(texts, device, max_len=128):  # noqa: F811
        tok = tokenizer(list(texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return {k: v.to(device) for k, v in tok.items()}

    train_loader = make_loader(raw["train"], True)
    dev_loader = make_loader(raw["validation"], False)
    test_loader = make_loader(raw["test"], False)
    optim = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=2e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = nn.BCEWithLogitsLoss()
    best = {"f1": -1.0, "threshold": 0.5, "state": None}
    epochs = 1 if smoke else 2
    for epoch in range(epochs):
        encoder.train()
        head.train()
        total = 0.0
        for texts, y in tqdm(train_loader, desc=f"goemotions {epoch}"):
            y = y.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                tok = tokenizer_batch(texts, device)
                hidden = encoder(**tok).last_hidden_state[:, 0, :]
                loss = loss_fn(head(hidden), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            total += float(loss.item())
        dev_probs, dev_gold = _collect(encoder, head, dev_loader, device)
        threshold, dev_f1 = best_threshold(dev_probs, dev_gold)
        print(f"epoch {epoch} loss {total / max(len(train_loader), 1):.4f} dev_macro_f1 {dev_f1:.4f} t {threshold:.2f}", flush=True)
        if dev_f1 > best["f1"]:
            best = {
                "f1": dev_f1,
                "threshold": threshold,
                "encoder": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
                "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
            }
    encoder.load_state_dict(best["encoder"])
    head.load_state_dict(best["head"])
    test_probs, test_gold = _collect(encoder, head, test_loader, device)
    test_f1 = _f1(test_probs, test_gold, best["threshold"])
    test_f1_05 = _f1(test_probs, test_gold, 0.5)
    out = {
        "dev_macro_f1": best["f1"],
        "test_macro_f1": test_f1,
        "test_macro_f1_threshold_0.5": test_f1_05,
        "threshold": best["threshold"],
        "n_test": int(len(test_gold)),
        "n_labels": len(GOEMO_LABELS),
        "smoke": smoke,
    }
    RESULT.joinpath("goemotions.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    path = CKPT / "goemotions.pt"
    torch.save({"encoder": best["encoder"], "head": best["head"], "threshold": best["threshold"]}, path)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
