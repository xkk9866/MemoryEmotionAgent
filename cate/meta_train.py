"""Meta-RLA: train the strategy classifier through revealed-label adaptation.

Test time is cate.rla: each dialogue's earlier supporter turns are labelled,
and the per-supporter logit offset b_d(t) is their MAP fit under N(0, s2 I).
Here the same inner problem is solved during training, by unrolled Newton
steps that gradients flow through, and the outer loss is the cross-entropy of
the adapted prediction z_t + b_d(t). The encoder therefore learns what is
predictable from the window across supporters, and leaves what is specific to
one supporter to the offset. The prior variance s2 is a parameter of the
outer objective, so it is learned rather than tuned.

A step samples dialogues and, in each, a window of consecutive supporter turns
that receive gradients; all earlier turns of that dialogue enter the inner
problem through no-grad logits, so the offset sees the full history.
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
from cate.policy_train import Classifier, flatten, score, set_seed, strategy_names
from cate.rla import adapt, dialogue_order


def inner_offsets(z_mem, y_mem, z_win, y_win, log_s2, newton: int = 5):
    """Offsets for each window turn j from every earlier turn (memory, then window < j).

    z_mem [M, K] no-grad, z_win [W, K] with grad. Returns [W, K], differentiable
    in z_win and log_s2.
    """
    k = z_win.shape[1]
    z_all = torch.cat([z_mem, z_win])
    y_all = torch.cat([y_mem, y_win]).long()
    m, w = z_mem.shape[0], z_win.shape[0]
    n = m + w
    # mask[j, p] = 1 if row p precedes window turn j
    mask = (torch.arange(n, device=z_win.device)[None, :] < (m + torch.arange(w, device=z_win.device))[:, None]).float()
    onehot = F.one_hot(y_all, k).float()
    inv = torch.exp(-log_s2)
    eye = torch.eye(k, device=z_win.device)
    b = z_win.new_zeros(w, k)
    for _ in range(newton):
        p = torch.softmax(z_all[None, :, :] + b[:, None, :], -1)  # [W, N, K]
        g = torch.einsum("jn,jnk->jk", mask, p - onehot[None]) + inv * b
        h = torch.einsum("jn,jnk->jk", mask, p).diag_embed() - torch.einsum("jn,jni,jnk->jik", mask, p, p) + inv * eye
        b = b - torch.linalg.solve(h, g.unsqueeze(-1)).squeeze(-1)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esconv", choices=["esconv", "annomi", "p4g"])
    ap.add_argument("--encoder", default="roberta-base")
    ap.add_argument("--tag", default="roberta-base")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16, help="turns with gradients per step")
    ap.add_argument("--dialogs_per_step", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--evals_per_epoch", type=int, default=2)
    ap.add_argument("--s2_init", type=float, default=0.3)
    ap.add_argument("--no_inner", action="store_true", help="ablation: train without the inner adaptation")
    ap.add_argument("--job_tag", default="", help="ignored; lets the dispatcher find the process")
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    device = torch.device("cuda")

    dialogs, samples, k = load_official(args.dataset)
    names = strategy_names(args.dataset)
    for sp in samples.values():
        for s in sp:
            s["text"] = flatten(dialogs[s["dialog"]], s["t"], names)
    y_tr = np.array([s["label"] for s in samples["train"]])
    freq = np.bincount(y_tr, minlength=k).astype(np.float64)
    weights = torch.tensor(freq.sum() / (k * freq), dtype=torch.float32, device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    tokenizer.truncation_side = "left"
    train_dialogs = dialogue_order(samples["train"])
    lengths = np.array([len(d) for d in train_dialogs], dtype=np.float64)
    order = {sp: dialogue_order(samples[sp]) for sp in ("valid", "test")}
    variant = "metanoinner" if args.no_inner else "meta"

    def logits(rows, grad):
        tok = tokenizer([r["text"] for r in rows], padding=True, truncation=True,
                        max_length=args.max_len, return_tensors="pt").to(device)
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
            return model(tok).float()

    for seed in args.seeds:
        out = RESULT / "policy" / args.dataset / f"{args.tag}_{variant}_s{seed}.json"
        if out.exists():
            print("skip", out, flush=True)
            continue
        set_seed(seed)
        rng = np.random.RandomState(seed)
        model = Classifier(args.encoder, k).to(device)
        log_s2 = torch.nn.Parameter(torch.tensor(math.log(args.s2_init), device=device))
        enc_ids = {id(p) for p in model.encoder.parameters()}
        optim = torch.optim.AdamW(
            [
                {"params": list(model.encoder.parameters()), "lr": args.lr},
                {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": 1e-4},
                {"params": [log_s2], "lr": 1e-2, "weight_decay": 0.0},
            ],
            weight_decay=0.01,
        )
        steps_per_epoch = math.ceil(len(samples["train"]) / args.bs)
        total = args.epochs * steps_per_epoch
        sched = get_linear_schedule_with_warmup(optim, int(0.06 * total), total)
        eval_every = max(1, steps_per_epoch // args.evals_per_epoch)
        window = max(1, args.bs // args.dialogs_per_step)

        @torch.no_grad()
        def predict(sp):
            model.eval()
            rows = samples[sp]
            z = torch.cat([logits(rows[i : i + 64], False) for i in range(0, len(rows), 64)]).cpu().numpy()
            y = np.array([r["label"] for r in rows])
            s2 = float(log_s2.exp())
            return z, z + adapt(z, y, order[sp], s2), y

        best = {"f1": -1.0}
        step = 0
        for epoch in range(args.epochs):
            for _ in range(steps_per_epoch):
                model.train()
                loss = 0.0
                picks = rng.choice(len(train_dialogs), args.dialogs_per_step, replace=False, p=lengths / lengths.sum())
                for d in picks:
                    ids = train_dialogs[d]
                    a = rng.randint(0, max(1, len(ids) - window + 1))
                    win, mem = ids[a : a + window], ids[:a]
                    rows_w = [samples["train"][i] for i in win]
                    z_w = logits(rows_w, True)
                    y_w = torch.tensor([r["label"] for r in rows_w], device=device)
                    if args.no_inner:
                        loss = loss + F.cross_entropy(z_w, y_w, weight=weights)
                        continue
                    if mem:
                        model.eval()  # memory logits as at test time
                        z_m = torch.cat([logits([samples["train"][i] for i in mem[j : j + 64]], False)
                                         for j in range(0, len(mem), 64)])
                        model.train()
                        y_m = torch.tensor([samples["train"][i]["label"] for i in mem], device=device)
                    else:
                        z_m, y_m = z_w.new_zeros(0, k), y_w.new_zeros(0)
                    b = inner_offsets(z_m.detach(), y_m, z_w, y_w, log_s2)
                    loss = loss + F.cross_entropy(z_w + b, y_w, weight=weights)
                loss = loss / args.dialogs_per_step
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [log_s2], 1.0)
                optim.step()
                sched.step()
                step += 1
                if step % eval_every == 0 or step == total:
                    z, za, yv = predict("valid")
                    f_raw, f_ad = score(yv, z, k)["macro_f1"], score(yv, za, k)["macro_f1"]
                    sel = f_raw if args.no_inner else f_ad
                    print(f"{args.tag} {variant} s{seed} ep{epoch} step{step} loss {loss.item():.3f} dev raw {f_raw:.4f} "
                          f"adapted {f_ad:.4f} s2 {float(log_s2.exp()):.3f}", flush=True)
                    if sel > best["f1"]:
                        best = {"f1": sel, "step": step, "s2": float(log_s2.exp()),
                                "model": {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}}
        model.load_state_dict(best["model"])
        with torch.no_grad():
            log_s2.fill_(math.log(best["s2"]))
        res = {"dataset": args.dataset, "encoder": args.encoder, "prior": variant, "seed": seed, "s2": best["s2"],
               "best_dev": best["f1"], "best_step": best["step"], "splits": {}}
        for sp in ("valid", "test"):
            z, za, y = predict(sp)
            res["splits"][sp] = {"y": y.tolist(), "logits": np.round(z, 4).tolist(),
                                 "scores": score(y, z, k), "scores_adapted": score(y, za, k)}
        te = res["splits"]["test"]
        print(f"{args.tag} {variant} s{seed} TEST raw M-F1 {te['scores']['macro_f1']:.4f} | adapted M-F1 "
              f"{te['scores_adapted']['macro_f1']:.4f} W-F1 {te['scores_adapted']['weighted_f1']:.4f} "
              f"B {te['scores_adapted']['bias']:.3f} s2 {best['s2']:.3f}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res), encoding="utf-8")
        del model, optim, best
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
