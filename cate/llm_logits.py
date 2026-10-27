"""Zero-shot int4 (AWQ) LLM strategy predictor; strategy-letter logprobs are its logits.

Same 5-utterance window as all other systems. Output follows the
policy_train result format, so cate.rla / rla_report apply unchanged.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from cate.config import RESULT
from cate.official import load_official
from cate.policy_train import flatten, strategy_names

LETTERS = "ABCDEFGHIJK"
DESC = {
    "Question": "ask for information related to the problem",
    "Restatement or Paraphrasing": "restate what the seeker said more concisely",
    "Reflection of feelings": "articulate and describe the seeker's feelings",
    "Self-disclosure": "share similar experiences or emotions of the supporter",
    "Affirmation and Reassurance": "affirm the seeker's strengths, motivation and capabilities",
    "Providing Suggestions": "provide suggestions about how to change",
    "Information": "provide useful information, e.g. data, facts, opinions, resources",
    "Others": "exchange pleasantries or use other strategies",
}
ROLE = {"esconv": "an emotional-support supporter", "annomi": "a motivational-interviewing therapist",
        "p4g": "a persuader asking for donations to Save the Children"}


def prompt(tok, ds, names, text):
    opts = "\n".join(f"{LETTERS[i]}. {n}" + (f": {DESC[n]}" if n in DESC else "") for i, n in enumerate(names))
    msg = (f"You are an expert annotator of dialogue strategies. The speaker tagged [supporter] is {ROLE[ds]}.\n"
           f"Strategies:\n{opts}\n\nDialogue so far:\n{text}\n\n"
           f"Which strategy will the supporter use in the next utterance? Answer with one letter.")
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--datasets", nargs="+", default=["esconv", "annomi", "p4g"])
    args = ap.parse_args()
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, quantization="awq_marlin", max_model_len=4096, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    for ds in args.datasets:
        dialogs, samples, k = load_official(ds)
        names = strategy_names(ds)
        res = {"dataset": ds, "encoder": args.model, "prior": "none", "seed": 0, "best_dev": 1.0, "splits": {}}
        for split in ("valid", "test"):
            rows = samples[split]
            prompts = [prompt(tok, ds, names, flatten(dialogs[r["dialog"]], r["t"], names)) for r in rows]
            outs = llm.generate(prompts, sp)
            z = np.full((len(rows), k), -30.0)
            for i, o in enumerate(outs):
                lp = o.outputs[0].logprobs[0]
                for j in range(k):
                    if letter_ids[j] in lp:
                        z[i, j] = lp[letter_ids[j]].logprob
            y = [r["label"] for r in rows]
            acc = float((z.argmax(-1) == np.array(y)).mean())
            print(ds, split, "acc", round(acc, 4), flush=True)
            res["splits"][split] = {"y": y, "logits": np.round(z, 4).tolist()}
        out = RESULT / "policy" / ds / f"{args.tag}_none_s0.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res), encoding="utf-8")


if __name__ == "__main__":
    main()
