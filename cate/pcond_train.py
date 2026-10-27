"""Person-conditioned encoder: the supporter's inferred policy enters the encoder as one token.

For decision t of dialogue d, b_d(t) in R^K is the posterior-mode offset of the supporter's
policy inferred from their revealed turns p < t (cate.rla.adapt). It is projected to one
embedding and prepended to the token sequence, so self-attention can read the window
*given who is acting*. A post-hoc correction of frozen logits cannot do this, which is
what bounds it (leave-one-out oracle, results/spire_oracle).

Leakage control: b for training rows is inferred from cross-fitted logits (models that
never saw the dialogue) on past turns only; for validation/test rows from the fold
ensemble's logits on past turns only. The prior variance s2 is selected on validation.
--token zero gives the identical model with the token fixed to zero (control).
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from cate.config import HF_HOME, RESULT
from cate.official import load_official
from cate.policy_train import flatten, score, set_seed, strategy_names
from cate.rla import adapt, dialogue_order, select
from cate.spire import load_crossfit


class PersonEncoder(nn.Module):
    def __init__(self, encoder, k):
        super().__init__()
        self.enc = AutoModel.from_pretrained(encoder)
        hid = self.enc.config.hidden_size
        self.person = nn.Sequential(nn.Linear(k, hid), nn.Tanh(), nn.Linear(hid, hid))
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(hid, k)

    def forward(self, tok, b):
        emb = self.enc.embeddings.word_embeddings(tok["input_ids"])
        p = self.person(b).unsqueeze(1).to(emb.dtype)
        # [CLS] person tokens...: the person token sits right after [CLS]
        x = torch.cat([emb[:, :1], p, emb[:, 1:]], 1)
        m = torch.cat([tok["attention_mask"][:, :1], torch.ones_like(tok["attention_mask"][:, :1]),
                       tok["attention_mask"][:, 1:]], 1)
        h = self.enc(inputs_embeds=x, attention_mask=m).last_hidden_state[:, 0]
        return self.head(self.drop(h))


def offsets(dataset, tag, seed, samples, k):
    """b_d(t) for every row of every split, from past turns only, never from a model that saw the dialogue."""
    runs = load_crossfit(dataset, tag, seed)
    yv = np.array([s["label"] for s in samples["valid"]])
    z = {"train": np.zeros((len(samples["train"]), k))}
    for r in runs:
        z["train"][r["splits"]["heldout"]["index"]] = np.array(r["splits"]["heldout"]["logits"])
    for sp in ("valid", "test"):
        z[sp] = np.mean([np.array(r["splits"][sp]["logits"]) for r in runs], 0)
    s2, half, _ = select(z["valid"], yv, dialogue_order(samples["valid"]))
    out = {}
    for sp in ("train", "valid", "test"):
        y = np.array([s["label"] for s in samples[sp]])
        out[sp] = adapt(z[sp], y, dialogue_order(samples[sp]), s2, half).astype(np.float32)
    return out, s2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="annomi")
    ap.add_argument("--encoder", default="roberta-base")
    ap.add_argument("--tag", default="roberta-base")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--token", default="posterior", choices=["posterior", "zero"])
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    out = RESULT / "pcond" / args.dataset / f"{args.tag}_{args.token}_s{args.seed}.json"
    if out.exists():
        return
    dev = torch.device("cuda")
    dialogs, samples, k = load_official(args.dataset, check=False)
    names = strategy_names(args.dataset)
    for sp in samples.values():
        for s in sp:
            s["text"] = flatten(dialogs[s["dialog"]], s["t"], names)
    b, s2 = offsets(args.dataset, args.tag, args.seed, samples, k)
    if args.token == "zero":
        b = {a: np.zeros_like(v) for a, v in b.items()}
    ytr = np.array([s["label"] for s in samples["train"]])
    freq = np.bincount(ytr, minlength=k)
    cw = torch.tensor(freq.sum() / (k * freq), dtype=torch.float32, device=dev)
    tok = AutoTokenizer.from_pretrained(args.encoder)
    tok.truncation_side = "left"
    set_seed(args.seed)
    model = PersonEncoder(args.encoder, k).to(dev)
    enc_ids = {id(p) for p in model.enc.parameters()}
    optim = torch.optim.AdamW([{"params": list(model.enc.parameters()), "lr": args.lr},
                               {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": 1e-4}],
                              weight_decay=0.01)
    n = len(ytr)
    steps = args.epochs * math.ceil(n / args.bs)
    sched = get_linear_schedule_with_warmup(optim, int(0.06 * steps), steps)
    every = max(1, math.ceil(n / args.bs) // 2)

    def run(sp, idx):
        rows = [samples[sp][i] for i in idx]
        t = tok([r["text"] for r in rows], padding=True, truncation=True, max_length=511, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(t, torch.tensor(b[sp][idx], device=dev)).float()

    @torch.no_grad()
    def predict(sp):
        model.eval()
        return torch.cat([run(sp, np.arange(i, min(i + 64, len(samples[sp])))) for i in range(0, len(samples[sp]), 64)]).cpu().numpy()

    yv = np.array([s["label"] for s in samples["valid"]])
    best, step = {"f1": -1}, 0
    for ep in range(args.epochs):
        order = np.random.RandomState(args.seed * 100 + ep).permutation(n)
        for i in range(0, n, args.bs):
            model.train()
            idx = order[i : i + args.bs]
            loss = F.cross_entropy(run("train", idx), torch.tensor(ytr[idx], device=dev), weight=cw)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            step += 1
            if step % every == 0 or step == steps:
                f1 = score(yv, predict("valid"), k)["macro_f1"]
                print(f"{args.dataset} {args.token} s{args.seed} ep{ep} step{step} dev {f1:.4f}", flush=True)
                if f1 > best["f1"]:
                    best = {"f1": f1, "state": {a: v.detach().cpu().clone() for a, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    res = {"dataset": args.dataset, "token": args.token, "seed": args.seed, "s2": s2, "best_dev": best["f1"], "splits": {}}
    for sp in ("valid", "test"):
        z = predict(sp)
        y = np.array([s["label"] for s in samples[sp]])
        res["splits"][sp] = {"y": y.tolist(), "logits": np.round(z, 4).tolist(), "scores": score(y, z, k)}
    te = res["splits"]["test"]["scores"]
    print(f"{args.dataset} {args.token} s{args.seed} TEST M-F1 {te['macro_f1']:.4f} W-F1 {te['weighted_f1']:.4f} B {te['bias']:.3f}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res), encoding="utf-8")


if __name__ == "__main__":
    main()
