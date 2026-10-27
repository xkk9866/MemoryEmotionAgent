"""Frozen RoBERTa encodings of each seeker reply conditioned on the strategy it answers.

These vectors are the observations of Evidential Acceptance Memory. They are not
emotion labels and they do not contain feedback scores.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from cate.config import CACHE, ENCODER_NAME, HF_HOME
from cate.data import load_esconv


def main():
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    rows = load_esconv()
    tok = AutoTokenizer.from_pretrained(ENCODER_NAME)
    enc = AutoModel.from_pretrained(ENCODER_NAME).cuda().eval()
    texts, where = [], []
    for row in rows:
        if row["failed"]:
            continue
        last = None
        for i, turn in enumerate(row["turns"]):
            if turn["role"] == "supporter" and turn["strategy"]:
                last = turn["strategy"]
            elif turn["role"] == "seeker" and last:
                texts.append(f"strategy: {last} </s> seeker: {turn['text']}")
                where.append((row["dialog_id"], i, len(row["turns"])))
    out = {}
    bs = 64
    with torch.no_grad():
        for start in range(0, len(texts), bs):
            batch = texts[start : start + bs]
            ids = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
            cls = enc(**ids).last_hidden_state[:, 0].float().cpu().numpy()
            for j, vec in enumerate(cls):
                did, i, n = where[start + j]
                buf = out.get(did)
                if buf is None:
                    buf = np.zeros((n, 768), dtype=np.float32)
                    out[did] = buf
                buf[i] = vec
            if start % (bs * 20) == 0:
                print(start, len(texts), flush=True)
    path = CACHE / "reactions.pt"
    torch.save(out, path)
    print("wrote", path, "dialogs", len(out), "turns", len(texts), flush=True)


if __name__ == "__main__":
    main()
