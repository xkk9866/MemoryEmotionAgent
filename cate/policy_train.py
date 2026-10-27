"""Next-strategy prediction on the official EmoDynamiX splits.

The encoder reads the official 5-utterance window, flattened with role and
strategy tags. With --prior dpp the logits are shifted, during training and at
test time, by the dialogue policy posterior of cate.dpp, so the encoder learns
what the window says beyond the supporter's own habits. Logits for every split
are saved so the posterior can also be applied post hoc.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from cate.config import CKPT, HF_HOME, RESULT
from cate.dpp import DialoguePolicyPosterior, history_counts
from cate.metrics import classification_scores, preference_bias_official as preference_bias
from cate.official import EXT, P4G_STRATEGIES, load_official


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strategy_names(name: str) -> list[str]:
    if name == "p4g":
        return list(P4G_STRATEGIES)
    s2i = json.load(open(EXT / name / "strategies.json", encoding="utf-8"))
    return [k for k, _ in sorted(s2i.items(), key=lambda kv: kv[1])]


def flatten(dialog, t, names, window=5, hist=False) -> str:
    """Same flattening as EmoDynamiX make_datasets.py, most recent turn last.

    hist=True appends every earlier strategy of the dialogue (the information
    revealed-label adaptation uses), so a trained model can exploit it too.
    """
    parts = []
    for p in range(max(0, t - window), t):
        turn = dialog["turns"][p]
        if turn["role"] == "seeker":
            parts.append(f"[seeker] {turn['text']}")
        else:
            tag = f" [{names[turn['strategy']]}]" if turn["strategy"] >= 0 else ""
            parts.append(f"[supporter]{tag} {turn['text']}")
    if t - window < 0:
        parts.insert(0, "<START>")
    if hist:
        past = [names[x["strategy"]] for x in dialog["turns"][:t] if x["strategy"] >= 0]
        parts.append("[supporter history] " + (", ".join(past) if past else "none"))
    return " ".join(parts)


class Classifier(nn.Module):
    """Encoder [CLS] / encoder-decoder mean pool / decoder-only last token (LoRA)."""

    def __init__(self, encoder: str, n_labels: int, lora_r: int = 0):
        super().__init__()
        dtype = torch.bfloat16 if lora_r else None
        full = AutoModel.from_pretrained(encoder, torch_dtype=dtype)
        self.enc_dec = bool(getattr(full.config, "is_encoder_decoder", False))
        self.decoder_only = lora_r > 0
        self.encoder = full.get_encoder() if self.enc_dec else full
        if self.decoder_only:
            from peft import LoraConfig, get_peft_model

            self.encoder.gradient_checkpointing_enable()
            self.encoder.enable_input_require_grads()
            cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            self.encoder = get_peft_model(self.encoder, cfg)
        hid = getattr(full.config, "hidden_size", None) or full.config.d_model
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(hid, n_labels)

    def forward(self, tok):
        h = self.encoder(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]).last_hidden_state
        if self.decoder_only:
            h = h[:, -1].float()  # left padding: last position is the final token
        elif self.enc_dec:
            m = tok["attention_mask"].unsqueeze(-1).to(h.dtype)
            h = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        else:
            h = h[:, 0]
        return self.head(self.drop(h))


def score(y, logits, k) -> dict:
    p = logits.argmax(-1)
    out = classification_scores(y, p, k)
    out["bias"] = preference_bias(y, p, k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esconv", choices=["esconv", "annomi", "p4g"])
    ap.add_argument("--encoder", default="roberta-base")
    ap.add_argument("--tag", default="roberta-base")
    ap.add_argument("--prior", default="none", choices=["none", "dpp"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--evals_per_epoch", type=int, default=2)
    ap.add_argument("--lora_r", type=int, default=0, help=">0: decoder-only LLM with LoRA")
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--hist", action="store_true", help="append the full past strategy sequence to the input")
    ap.add_argument("--job_tag", default="", help="ignored; lets the dispatcher find the process")
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    device = torch.device("cuda")

    dialogs, samples, k = load_official(args.dataset)
    names = strategy_names(args.dataset)
    for sp in samples.values():
        for s in sp:
            s["text"] = flatten(dialogs[s["dialog"]], s["t"], names, hist=args.hist)
    counts = {sp: history_counts(dialogs, rows, k) for sp, rows in samples.items()}
    y_tr = np.array([s["label"] for s in samples["train"]])
    freq = np.bincount(y_tr, minlength=k).astype(np.float64)
    weights = torch.tensor(freq.sum() / (k * freq), dtype=torch.float32, device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    tokenizer.truncation_side = "left"
    if args.lora_r:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    for seed in args.seeds:
        out = RESULT / "policy" / args.dataset / f"{args.tag}_{args.prior}_s{seed}.json"
        if out.exists():
            print("skip", out, flush=True)
            continue
        set_seed(seed)
        model = Classifier(args.encoder, k, args.lora_r).to(device)
        dpp = DialoguePolicyPosterior.fit(dialogs, samples["train"], counts["train"], k).to(device)
        use_prior = args.prior == "dpp"
        enc_ids = {id(p) for p in model.encoder.parameters()}
        optim = torch.optim.AdamW(
            [
                {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": args.lr},
                {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": args.head_lr},
                {"params": list(dpp.parameters()), "lr": 1e-2, "weight_decay": 0.0},
            ],
            weight_decay=0.01,
        )
        n_train = len(samples["train"])
        steps = args.epochs * math.ceil(n_train / args.bs)
        sched = get_linear_schedule_with_warmup(optim, int(0.06 * steps), steps)
        eval_every = max(1, math.ceil(n_train / args.bs) // args.evals_per_epoch)

        def logits_of(rows, cnt, idx):
            batch = [rows[i] for i in idx]
            tok = tokenizer([b["text"] for b in batch], padding=True, truncation=True,
                            max_length=args.max_len, return_tensors="pt").to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(tok).float()
            c = {key: v[idx].to(device) for key, v in cnt.items()}
            return z, dpp(c)

        @torch.no_grad()
        def predict(sp):
            model.eval()
            rows, cnt = samples[sp], counts[sp]
            zs, shifts = [], []
            for i in range(0, len(rows), 64):
                z, sh = logits_of(rows, cnt, list(range(i, min(i + 64, len(rows)))))
                zs.append(z.cpu())
                shifts.append(sh.cpu())
            return torch.cat(zs).numpy(), torch.cat(shifts).numpy()

        best = {"f1": -1.0}
        step = 0
        for epoch in range(args.epochs):
            order = np.random.RandomState(seed * 100 + epoch).permutation(n_train)
            for i in range(0, n_train, args.bs):
                model.train()
                idx = order[i : i + args.bs].tolist()
                z, sh = logits_of(samples["train"], counts["train"], idx)
                y = torch.tensor([samples["train"][j]["label"] for j in idx], device=device)
                loss = F.cross_entropy(z + sh if use_prior else z, y, weight=weights)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad] + list(dpp.parameters()), 1.0)
                optim.step()
                sched.step()
                step += 1
                if step % eval_every == 0 or step == steps:
                    z, sh = predict("valid")
                    yv = np.array([s["label"] for s in samples["valid"]])
                    f1 = score(yv, z + sh if use_prior else z, k)["macro_f1"]
                    print(f"{args.tag} {args.prior} s{seed} ep{epoch} step{step} loss {loss.item():.3f} "
                          f"dev {f1:.4f} {dpp.describe()}", flush=True)
                    if f1 > best["f1"]:
                        best = {"f1": f1, "step": step,
                                "model": {n: v.detach().cpu().clone() for n, v in model.state_dict().items()
                                          if not args.lora_r or "lora" in n or n.startswith("head")},
                                "dpp": {n: v.detach().cpu().clone() for n, v in dpp.state_dict().items()}}
        model.load_state_dict(best["model"], strict=not args.lora_r)
        dpp.load_state_dict(best["dpp"])
        res = {"dataset": args.dataset, "encoder": args.encoder, "prior": args.prior, "seed": seed,
               "best_dev": best["f1"], "best_step": best["step"], "dpp": dpp.describe(), "splits": {}}
        for sp in ("train", "valid", "test"):
            z, sh = predict(sp)
            y = np.array([s["label"] for s in samples[sp]])
            res["splits"][sp] = {
                "y": y.tolist(),
                "logits": np.round(z, 4).tolist(),
                "shift": np.round(sh, 4).tolist(),
                "scores": score(y, z + sh if use_prior else z, k),
            }
        te = res["splits"]["test"]["scores"]
        print(f"{args.tag} {args.prior} s{seed} TEST M-F1 {te['macro_f1']:.4f} W-F1 {te['weighted_f1']:.4f} "
              f"B {te['bias']:.3f}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res), encoding="utf-8")
        del model, optim, best
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
