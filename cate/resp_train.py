"""Response-supervised strategy prediction.

A strategy label is a lossy name for what the supporter actually said. Besides
the 8-way cross-entropy, the context encoder is trained to pick the gold
response out of the batch (InfoNCE). At test time the strategy can be read
from the response memory as well: p(y | x) = sum_r p(r | x) 1[y_r = y], with
p(r | x) a softmax over all training responses.
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
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from cate.config import HF_HOME, RESULT
from cate.official import load_official
from cate.policy_train import Classifier, flatten, score, set_seed, strategy_names


class DualClassifier(Classifier):
    def __init__(self, encoder: str, n_labels: int, lora_r: int = 0, dim: int = 256):
        super().__init__(encoder, n_labels, lora_r)
        hid = self.head.in_features
        self.proj_c = nn.Linear(hid, dim)
        self.proj_r = nn.Linear(hid, dim)

    def pooled(self, tok):
        h = self.encoder(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]).last_hidden_state
        if self.decoder_only:
            return h[:, -1].float()
        if self.enc_dec:
            m = tok["attention_mask"].unsqueeze(-1).to(h.dtype)
            return (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return h[:, 0]

    def context(self, tok):
        h = self.drop(self.pooled(tok))
        return self.head(h).float(), F.normalize(self.proj_c(h).float(), dim=-1)

    def response(self, tok):
        return F.normalize(self.proj_r(self.pooled(tok)).float(), dim=-1)


def knn_logits(q, bank, bank_y, k, tau):
    """log sum_r softmax(q.r / tau) 1[y_r = y], computed in chunks."""
    out = []
    onehot = F.one_hot(bank_y, k).float()
    for i in range(0, q.shape[0], 256):
        sim = q[i : i + 256] @ bank.T / tau
        p = torch.softmax(sim, dim=-1) @ onehot
        out.append(torch.log(p.clamp(min=1e-8)))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esconv", choices=["esconv", "annomi", "p4g"])
    ap.add_argument("--encoder", default="roberta-base")
    ap.add_argument("--tag", default="roberta-base")
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the response InfoNCE")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--resp_len", type=int, default=96)
    ap.add_argument("--evals_per_epoch", type=int, default=2)
    ap.add_argument("--lora_r", type=int, default=0)
    ap.add_argument("--job_tag", default="", help="ignored; lets the dispatcher find the process")
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    device = torch.device("cuda")

    dialogs, samples, k = load_official(args.dataset)
    names = strategy_names(args.dataset)
    for sp in samples.values():
        for s in sp:
            s["text"] = flatten(dialogs[s["dialog"]], s["t"], names)
            s["resp"] = dialogs[s["dialog"]]["turns"][s["t"]]["text"]
            s["resp_key"] = " ".join(s["resp"].lower().split())
    y_tr = np.array([s["label"] for s in samples["train"]])
    freq = np.bincount(y_tr, minlength=k).astype(np.float64)
    weights = torch.tensor(freq.sum() / (k * freq), dtype=torch.float32, device=device)
    tok_c = AutoTokenizer.from_pretrained(args.encoder)
    tok_c.truncation_side = "left"
    tok_r = AutoTokenizer.from_pretrained(args.encoder)
    if args.lora_r:
        for t in (tok_c, tok_r):
            t.padding_side = "left"
            if t.pad_token is None:
                t.pad_token = t.eos_token

    def enc(tok, texts, n):
        return tok(texts, padding=True, truncation=True, max_length=n, return_tensors="pt").to(device)

    for seed in args.seeds:
        out = RESULT / "policy" / args.dataset / f"{args.tag}_resp{args.lam:g}_s{seed}.json"
        if out.exists():
            print("skip", out, flush=True)
            continue
        set_seed(seed)
        model = DualClassifier(args.encoder, k, args.lora_r).to(device)
        enc_ids = {id(p) for p in model.encoder.parameters()}
        optim = torch.optim.AdamW(
            [
                {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": args.lr},
                {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": 1e-4},
            ],
            weight_decay=0.01,
        )
        n_train = len(samples["train"])
        steps = args.epochs * math.ceil(n_train / args.bs)
        sched = get_linear_schedule_with_warmup(optim, int(0.06 * steps), steps)
        eval_every = max(1, math.ceil(n_train / args.bs) // args.evals_per_epoch)

        @torch.no_grad()
        def embed_bank():
            model.eval()
            rows = samples["train"]
            vecs = []
            for i in range(0, len(rows), 128):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    vecs.append(model.response(enc(tok_r, [r["resp"] for r in rows[i : i + 128]], args.resp_len)))
            return torch.cat(vecs), torch.tensor(y_tr, device=device)

        @torch.no_grad()
        def predict(sp, bank=None):
            model.eval()
            rows = samples[sp]
            zs, qs = [], []
            for i in range(0, len(rows), 64):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z, q = model.context(enc(tok_c, [r["text"] for r in rows[i : i + 64]], args.max_len))
                zs.append(z)
                qs.append(q)
            z, q = torch.cat(zs), torch.cat(qs)
            kn = knn_logits(q, *bank, k, args.tau) if bank is not None else None
            return z, kn

        best = {"f1": -1.0}
        step = 0
        for epoch in range(args.epochs):
            order = np.random.RandomState(seed * 100 + epoch).permutation(n_train)
            for i in range(0, n_train, args.bs):
                model.train()
                batch = [samples["train"][j] for j in order[i : i + args.bs]]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z, q = model.context(enc(tok_c, [b["text"] for b in batch], args.max_len))
                    r = model.response(enc(tok_r, [b["resp"] for b in batch], args.resp_len))
                y = torch.tensor([b["label"] for b in batch], device=device)
                loss = F.cross_entropy(z, y, weight=weights)
                if args.lam > 0:
                    sim = q @ r.T / args.tau
                    keys = [b["resp_key"] for b in batch]
                    same = torch.tensor([[a == c for c in keys] for a in keys], device=device)
                    # Identical responses ("Hello") in other rows are positives, not negatives.
                    target = same.float() / same.float().sum(-1, keepdim=True)
                    nce = -(target * F.log_softmax(sim, -1)).sum(-1).mean()
                    nce = 0.5 * (nce + -(target.T * F.log_softmax(sim.T, -1)).sum(-1).mean())
                    loss = loss + args.lam * nce
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                sched.step()
                step += 1
                if step % eval_every == 0 or step == steps:
                    yv = np.array([s["label"] for s in samples["valid"]])
                    bank = embed_bank() if args.lam > 0 else None
                    z, kn = predict("valid", bank)
                    f1 = score(yv, z.cpu().numpy(), k)["macro_f1"]
                    f1k = score(yv, kn.cpu().numpy(), k)["macro_f1"] if kn is not None else float("nan")
                    print(f"{args.tag} resp{args.lam:g} s{seed} ep{epoch} step{step} loss {loss.item():.3f} "
                          f"dev cls {f1:.4f} knn {f1k:.4f}", flush=True)
                    if f1 > best["f1"]:
                        best = {"f1": f1, "f1_knn": f1k, "step": step,
                                "model": {n: v.detach().cpu().clone() for n, v in model.state_dict().items()
                                          if not args.lora_r or "lora" in n or n.startswith(("head", "proj"))}}
        model.load_state_dict(best["model"], strict=not args.lora_r)
        bank = embed_bank() if args.lam > 0 else None
        res = {"dataset": args.dataset, "encoder": args.encoder, "lam": args.lam, "tau": args.tau, "seed": seed,
               "best_dev": best["f1"], "best_dev_knn": best["f1_knn"], "best_step": best["step"], "splits": {}}
        for sp in ("valid", "test"):
            z, kn = predict(sp, bank)
            y = np.array([s["label"] for s in samples[sp]])
            entry = {"y": y.tolist(), "logits": np.round(z.cpu().numpy(), 4).tolist(),
                     "scores": score(y, z.cpu().numpy(), k)}
            if kn is not None:
                entry["knn"] = np.round(kn.cpu().numpy(), 4).tolist()
                entry["scores_knn"] = score(y, kn.cpu().numpy(), k)
            res["splits"][sp] = entry
        te = res["splits"]["test"]
        msg = f"cls M-F1 {te['scores']['macro_f1']:.4f} W-F1 {te['scores']['weighted_f1']:.4f}"
        if "scores_knn" in te:
            msg += f" | knn M-F1 {te['scores_knn']['macro_f1']:.4f} W-F1 {te['scores_knn']['weighted_f1']:.4f}"
        print(f"{args.tag} resp{args.lam:g} s{seed} TEST {msg}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res), encoding="utf-8")
        del model, optim, best
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
