"""Cross-fitted strategy classifiers.

The training dialogues are split into K folds. The model of fold f is trained on
the other K-1 folds, selected on the official validation split, and then scores
(i) the training dialogues of fold f, which it has never seen, (ii) validation and
(iii) test. Every training decision therefore gets logits from a model that did not
fit its label: the error distribution the supporter layer is trained on is the one
it meets at test time. In-sample logits are not (the revealed turns are memorised,
see Appendix A of the paper).
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from cate.config import CACHE, HF_HOME, RESULT
from cate.resplit import load_split
from cate.policy_train import Classifier, flatten, score, set_seed, strategy_names

K_FOLDS = 5


def fold_of(train_rows, k_folds=K_FOLDS, seed=0):
    ids = sorted({r["dialog"] for r in train_rows})
    perm = np.random.RandomState(seed).permutation(len(ids))
    fold = {ids[j]: n % k_folds for n, j in enumerate(perm)}
    return np.array([fold[r["dialog"]] for r in train_rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esconv", choices=["esconv", "annomi", "p4g"])
    ap.add_argument("--encoder", default="roberta-base")
    ap.add_argument("--tag", default="roberta-base")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--evals_per_epoch", type=int, default=2)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--split_seed", type=int, default=0, help="0 = official split; >0 = independent re-split")
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    sub = "crossfit" if args.split_seed == 0 else f"crossfit_rs{args.split_seed}"
    out = RESULT / sub / args.dataset / f"{args.tag}_s{args.seed}_f{args.fold}.json"
    if out.exists():
        print("skip", out)
        return
    device = torch.device("cuda")
    dialogs, samples, k = load_split(args.dataset, args.split_seed)
    names = strategy_names(args.dataset)
    for sp in samples.values():
        for s in sp:
            s["text"] = flatten(dialogs[s["dialog"]], s["t"], names)
    folds = fold_of(samples["train"])
    fit_idx = np.where(folds != args.fold)[0]
    held_idx = np.where(folds == args.fold)[0]
    y_fit = np.array([samples["train"][i]["label"] for i in fit_idx])
    freq = np.bincount(y_fit, minlength=k).astype(np.float64)
    weights = torch.tensor(freq.sum() / (k * np.maximum(freq, 1)), dtype=torch.float32, device=device)
    tok = AutoTokenizer.from_pretrained(args.encoder)
    tok.truncation_side = "left"

    set_seed(args.seed * 10 + args.fold)
    model = Classifier(args.encoder, k).to(device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    optim = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": args.lr},
        {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": args.head_lr},
    ], weight_decay=0.01)
    n = len(fit_idx)
    steps = args.epochs * math.ceil(n / args.bs)
    sched = get_linear_schedule_with_warmup(optim, int(0.06 * steps), steps)
    eval_every = max(1, math.ceil(n / args.bs) // args.evals_per_epoch)

    def run(rows, feats=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t = tok([r["text"] for r in rows], padding=True, truncation=True, max_length=args.max_len,
                    return_tensors="pt").to(device)
            h = model.encoder(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).last_hidden_state[:, 0]
            z = model.head(model.drop(h))
        return (z.float(), h.float()) if feats else z.float()

    @torch.no_grad()
    def predict(rows, feats=False):
        model.eval()
        zs, hs = [], []
        for i in range(0, len(rows), 64):
            z, h = run(rows[i : i + 64], feats=True)
            zs.append(z.cpu())
            hs.append(h.cpu())
        z = torch.cat(zs).numpy()
        return (z, torch.cat(hs).numpy()) if feats else z

    yv = np.array([s["label"] for s in samples["valid"]])
    best, step = {"f1": -1.0}, 0
    for epoch in range(args.epochs):
        order = np.random.RandomState(args.seed * 100 + epoch + 7 * args.fold).permutation(n)
        for i in range(0, n, args.bs):
            model.train()
            idx = fit_idx[order[i : i + args.bs]]
            rows = [samples["train"][j] for j in idx]
            z = run(rows)
            y = torch.tensor([r["label"] for r in rows], device=device)
            loss = F.cross_entropy(z, y, weight=weights)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            step += 1
            if step % eval_every == 0 or step == steps:
                f1 = score(yv, predict(samples["valid"]), k)["macro_f1"]
                print(f"{args.dataset} {args.tag} s{args.seed} f{args.fold} ep{epoch} step{step} "
                      f"loss {loss.item():.3f} dev {f1:.4f}", flush=True)
                if f1 > best["f1"]:
                    best = {"f1": f1, "step": step,
                            "model": {a: v.detach().cpu().clone() for a, v in model.state_dict().items()}}
    model.load_state_dict(best["model"])
    res = {"dataset": args.dataset, "encoder": args.encoder, "seed": args.seed, "fold": args.fold,
           "best_dev": best["f1"], "best_step": best["step"], "splits": {}}
    feats = {}
    for sp, rows, idx in (("heldout", [samples["train"][i] for i in held_idx], held_idx.tolist()),
                          ("valid", samples["valid"], list(range(len(samples["valid"])))),
                          ("test", samples["test"], list(range(len(samples["test"]))))):
        z, h = predict(rows, feats=True)
        y = np.array([r["label"] for r in rows])
        res["splits"][sp] = {"index": idx, "y": y.tolist(), "logits": np.round(z, 4).tolist(),
                             "scores": score(y, z, k)}
        feats[sp] = h.astype(np.float16)
    te = res["splits"]["test"]["scores"]
    print(f"{args.dataset} {args.tag} s{args.seed} f{args.fold} TEST M-F1 {te['macro_f1']:.4f} "
          f"W-F1 {te['weighted_f1']:.4f} B {te['bias']:.3f}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fdir = CACHE / "crossfit_feats"
    fdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fdir / f"{args.dataset}_{args.tag}_rs{args.split_seed}_s{args.seed}_f{args.fold}.npz", **feats)
    out.write_text(json.dumps(res), encoding="utf-8")


if __name__ == "__main__":
    main()
