"""LoRA-fine-tuned decoder LLM as a next-strategy classifier on the shared 4-turn windows."""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from cate.config import HF_HOME, RESULT, STRATEGIES
from cate.data import assign_esconv_splits, load_esconv
from cate.metrics import classification_scores, length_bucket, macro_f1_by_bucket, preference_bias
from cate.train_eam import build_examples, set_seed


def evaluate(model, tok, data, bs):
    model.eval()
    ys, ps, hs, meta = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(data), bs):
            batch = data[i : i + bs]
            enc = tok([b["text"] for b in batch], truncation=True, max_length=320, padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**enc).logits
            ps.extend(logits.argmax(-1).cpu().tolist())
            ys.extend([b["label"] for b in batch])
            hs.extend([b["t"] for b in batch])
            meta.extend([(b["dialog_id"], b["t"]) for b in batch])
    stats = classification_scores(ys, ps, len(STRATEGIES))
    stats["bias"] = preference_bias(ys, ps, len(STRATEGIES))
    stats["by_length"] = macro_f1_by_bucket(ys, ps, [length_bucket(h) for h in hs], len(STRATEGIES))
    return stats, {"y": ys, "pred": ps, "hist": hs, "meta": meta}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--tag", default="qwen2.5-1.5b-lora")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--bs", type=int, default=8)
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    rows = load_esconv()
    assign_esconv_splits(rows)
    rows = [r for r in rows if r["failed"] == 0]
    examples = build_examples(rows)
    splits = {sp: [e for e in examples if e["split"] == sp] for sp in ("train", "dev", "test")}
    counts = np.bincount([e["label"] for e in splits["train"]], minlength=len(STRATEGIES)).astype(np.float32)
    weights = torch.tensor(counts.sum() / (len(STRATEGIES) * counts), device="cuda")
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for seed in args.seeds:
        out = RESULT / "llm" / f"{args.tag}_s{seed}.json"
        if out.exists():
            print("skip", out, flush=True)
            continue
        set_seed(seed)
        base = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=len(STRATEGIES), torch_dtype=torch.bfloat16
        )
        base.config.pad_token_id = tok.pad_token_id
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model = get_peft_model(base, cfg).cuda()
        for n, p in model.named_parameters():
            if "score" in n:
                p.data = p.data.float()
                p.requires_grad = True
        optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0)
        best_f1, best_state = -1.0, None
        for epoch in range(args.epochs):
            model.train()
            order = list(range(len(splits["train"])))
            random.Random(seed * 100 + epoch).shuffle(order)
            for step, i in enumerate(range(0, len(order), args.bs)):
                batch = [splits["train"][j] for j in order[i : i + args.bs]]
                enc = tok([b["text"] for b in batch], truncation=True, max_length=320, padding=True, return_tensors="pt").to("cuda")
                labels = torch.tensor([b["label"] for b in batch], device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**enc).logits
                loss = F.cross_entropy(logits.float(), labels, weight=weights)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                if step % 400 == 0:
                    print(args.tag, seed, epoch, step, round(float(loss), 4), flush=True)
            dv, _ = evaluate(model, tok, splits["dev"], args.bs * 2)
            print(args.tag, seed, epoch, "dev", round(dv["macro_f1"], 4), flush=True)
            if dv["macro_f1"] > best_f1:
                best_f1 = dv["macro_f1"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora" in k or "score" in k}
        model.load_state_dict(best_state, strict=False)
        te, preds = evaluate(model, tok, splits["test"], args.bs * 2)
        print(args.tag, seed, "TEST macro", round(te["macro_f1"], 4), flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"model": args.model, "seed": seed, "best_dev": best_f1, "test": te, "preds": preds}), encoding="utf-8")
        del model, base, optim
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
