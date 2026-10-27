"""Zero-/few-shot instruction-tuned LLM baseline for next-strategy prediction.

The model sees the same 4-turn window as the encoder baselines and is scored by the
next-token probability of the option letter of each of the eight ESConv strategies.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cate.config import CACHE, HF_HOME, RESULT, STRATEGIES
from cate.data import assign_esconv_splits, load_esconv
from cate.metrics import classification_scores, length_bucket, macro_f1_by_bucket, preference_bias
from cate.train_eam import build_examples

DESCRIPTIONS = {
    "Question": "ask for information related to the problem",
    "Restatement or Paraphrasing": "restate what the seeker said more concisely",
    "Reflection of feelings": "articulate and describe the seeker's feelings",
    "Self-disclosure": "share similar experiences or emotions of the supporter",
    "Affirmation and Reassurance": "affirm the seeker's strengths, motivation and capabilities",
    "Providing Suggestions": "provide suggestions about how to change",
    "Information": "provide useful information, e.g. data, facts, opinions, resources",
    "Others": "exchange pleasantries or use other strategies",
}
LETTERS = "ABCDEFGH"


def option_block() -> str:
    return "\n".join(f"{LETTERS[i]}. {s}: {DESCRIPTIONS.get(s, s)}" for i, s in enumerate(STRATEGIES))


def render(example) -> str:
    return example["text"].replace(" </s> ", "\n")


def build_prompt(example, shots):
    head = (
        "You are an expert in emotional-support conversation. Given the recent dialogue, choose the "
        "support strategy the supporter should use in the next turn.\nStrategies:\n" + option_block()
    )
    msgs = [{"role": "system", "content": head}]
    for s in shots:
        msgs.append({"role": "user", "content": f"Dialogue:\n{render(s)}\nAnswer with one letter."})
        msgs.append({"role": "assistant", "content": LETTERS[s["label"]]})
    msgs.append({"role": "user", "content": f"Dialogue:\n{render(example)}\nAnswer with one letter."})
    return msgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tag", default="qwen2.5-7b")
    parser.add_argument("--shots", type=int, nargs="+", default=[0, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--load_4bit", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    rows = load_esconv()
    assign_esconv_splits(rows)
    rows = [r for r in rows if r["failed"] == 0]
    examples = build_examples(rows)
    train = [e for e in examples if e["split"] == "train"]
    test = [e for e in examples if e["split"] == "test"]

    tok = AutoTokenizer.from_pretrained(args.model)
    kw = {"torch_dtype": torch.bfloat16, "device_map": "cuda"}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()
    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]

    for k in args.shots:
        for seed in args.seeds if k > 0 else [42]:
            out = RESULT / "llm" / f"{args.tag}_k{k}_s{seed}.json"
            if out.exists():
                print("skip", out, flush=True)
                continue
            rng = random.Random(seed)
            by_label = {c: [e for e in train if e["label"] == c] for c in range(len(STRATEGIES))}
            ys, ps, hs, meta = [], [], [], []
            for i, ex in enumerate(test):
                shots = []
                if k > 0:
                    pool = [rng.choice(by_label[c % len(STRATEGIES)]) for c in range(k)]
                    rng.shuffle(pool)
                    shots = pool
                text = tok.apply_chat_template(build_prompt(ex, shots), tokenize=False, add_generation_prompt=True)
                ids = tok(text, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    logits = model(**ids).logits[0, -1, letter_ids].float()
                ys.append(ex["label"])
                ps.append(int(logits.argmax()))
                hs.append(ex["t"])
                meta.append((ex["dialog_id"], ex["t"]))
                if i % 200 == 0:
                    print(args.tag, k, seed, i, flush=True)
            stats = classification_scores(ys, ps, len(STRATEGIES))
            stats["bias"] = preference_bias(ys, ps, len(STRATEGIES))
            stats["by_length"] = macro_f1_by_bucket(ys, ps, [length_bucket(h) for h in hs], len(STRATEGIES))
            print(args.tag, k, seed, "TEST macro", round(stats["macro_f1"], 4), flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps({"model": args.model, "shots": k, "seed": seed, "test": stats,
                            "preds": {"y": ys, "pred": ps, "hist": hs, "meta": meta}}),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
