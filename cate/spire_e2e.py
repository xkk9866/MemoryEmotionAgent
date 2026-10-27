"""SPIRE trained end to end: the encoder is fine-tuned through the supporter-inference layer.

Training decision t of dialogue d is predicted as softmax(f_theta(x_t) + W_d(t) u_t), where
W_d(t) is inferred by the SPIRE layer from the supporter's revealed turns p < t. The logits
of those revealed turns are the cross-fitted ones (cate.crossfit): a model that never saw
dialogue d scored them, so their residuals are honest. The encoder therefore learns what
the layer cannot infer about the person, the population part of the policy, and the
layer (initialised from cate.spire on the same cross-fitted logits) keeps training jointly.
At test time the trained encoder scores every turn of the unseen dialogue, as in cross-fitting.
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

from cate.config import HF_HOME, RESULT
from cate.official import load_official
from cate.policy_train import Classifier, flatten, set_seed, strategy_names
from cate.rla import dialogue_order
from cate.spire import (CONFIGS, DEV, load_crossfit, metrics, run_layer, semantic_inputs, situation_inputs,
                        tensors, tg_tensors, train_layer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="annomi")
    ap.add_argument("--encoder", default="roberta-base")
    ap.add_argument("--tag", default="roberta-base")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--evals_per_epoch", type=int, default=2)
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    out = RESULT / "spire_e2e" / args.dataset / f"{args.tag}_s{args.seed}.json"
    dialogs, samples, k = load_official(args.dataset, check=False)
    names = strategy_names(args.dataset)
    for sp in samples.values():
        for s in sp:
            s["text"] = flatten(dialogs[s["dialog"]], s["t"], names)
    ytr = np.array([s["label"] for s in samples["train"]])
    freq = np.bincount(ytr, minlength=k)
    cw = torch.tensor(freq.sum() / (k * freq), dtype=torch.float32, device=DEV)
    ctx = {sp: np.concatenate([situation_inputs(dialogs, samples[sp], k), semantic_inputs(args.dataset, sp)], 1)
           for sp in samples}
    orders = {sp: dialogue_order(samples[sp]) for sp in samples}
    runs = load_crossfit(args.dataset, args.tag, args.seed)
    z_oof = np.zeros((len(ytr), k), np.float32)
    for r in runs:
        z_oof[r["splits"]["heldout"]["index"]] = np.array(r["splits"]["heldout"]["logits"])
    Z, Y, C = tensors(z_oof, ytr, ctx["train"])
    tg = tg_tensors(orders["train"])
    yv = np.array([s["label"] for s in samples["valid"]])
    yt = np.array([s["label"] for s in samples["test"]])
    tgv, tgt = tg_tensors(orders["valid"]), tg_tensors(orders["test"])
    val_folds = [(*tensors(r["splits"]["valid"]["logits"], yv, ctx["valid"]), tgv) for r in runs]

    # stage 1: the layer on cross-fitted logits (identical to cate.spire)
    layer, _ = train_layer(dict(CONFIGS["spire"], s2_init=0.3), k, (Z, Y, C, tg), val_folds, cw)
    # row -> (position in the target list) so a batch of rows finds its revealed rows
    pos = torch.empty(len(ytr), dtype=torch.long, device=DEV)
    pos[tg[0]] = torch.arange(len(tg[0]), device=DEV)

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.encoder)
    tok.truncation_side = "left"
    model = Classifier(args.encoder, k).to(DEV)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    optim = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": args.lr},
        {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": 1e-4},
        {"params": list(layer.parameters()), "lr": 1e-3, "weight_decay": 0.0},
    ], weight_decay=0.01)
    n = len(ytr)
    steps = args.epochs * math.ceil(n / args.bs)
    sched = get_linear_schedule_with_warmup(optim, int(0.06 * steps), steps)
    eval_every = max(1, math.ceil(n / args.bs) // args.evals_per_epoch)

    def encode(rows):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t = tok([r["text"] for r in rows], padding=True, truncation=True, max_length=args.max_len,
                    return_tensors="pt").to(DEV)
            return model(t).float()

    @torch.no_grad()
    def predict(sp):
        model.eval()
        rows = samples[sp]
        return torch.cat([encode(rows[i : i + 64]) for i in range(0, len(rows), 64)])

    @torch.no_grad()
    def adapted(sp, y, tgs):
        layer.eval()
        z = predict(sp)
        return z, run_layer(layer, z, torch.tensor(y, device=DEV), torch.tensor(ctx[sp], device=DEV), tgs)

    best, step = {"f1": -1.0}, 0
    for epoch in range(args.epochs):
        order = np.random.RandomState(args.seed * 100 + epoch).permutation(n)
        for i in range(0, n, args.bs):
            model.train()
            layer.train()
            rows_i = torch.tensor(order[i : i + args.bs], device=DEV)
            z_cur = encode([samples["train"][j] for j in rows_i.tolist()])
            zall = Z.index_copy(0, rows_i, z_cur)  # revealed turns keep their cross-fitted logits
            p = pos[rows_i]
            tmax = int(tg[2][p].sum(1).max().item()) or 1
            zz = layer(zall, Y, C, tg[0][p], tg[1][p, :tmax], tg[2][p, :tmax])
            loss = F.cross_entropy(zz, Y[rows_i], weight=cw)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(layer.parameters()), 1.0)
            optim.step()
            sched.step()
            step += 1
            if step % eval_every == 0 or step == steps:
                _, za = adapted("valid", yv, tgv)
                f1 = metrics(yv, za.cpu().numpy(), k)[0]
                print(f"{args.dataset} e2e s{args.seed} ep{epoch} step{step} loss {loss.item():.3f} dev {f1:.2f}",
                      flush=True)
                if f1 > best["f1"]:
                    best = {"f1": f1, "step": step,
                            "model": {a: v.detach().cpu().clone() for a, v in model.state_dict().items()},
                            "layer": {a: v.detach().clone() for a, v in layer.state_dict().items()}}
    model.load_state_dict(best["model"])
    layer.load_state_dict(best["layer"])
    zb, za = adapted("test", yt, tgt)
    res = {"dataset": args.dataset, "seed": args.seed, "best_dev": best["f1"], "best_step": best["step"],
           "test": {"backbone_logits": zb.cpu().numpy().round(4).tolist(), "y": yt.tolist(),
                    "e2e": metrics(yt, za.cpu().numpy(), k), "encoder_only": metrics(yt, zb.cpu().numpy(), k)},
           "preds": za.argmax(-1).cpu().tolist(), "dialog": [r["dialog"] for r in samples["test"]]}
    print(f"{args.dataset} e2e s{args.seed} TEST e2e {res['test']['e2e']} encoder-only {res['test']['encoder_only']}",
          flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res), encoding="utf-8")


if __name__ == "__main__":
    main()
