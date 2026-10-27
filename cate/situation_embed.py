"""Frozen semantic code of each decision's situation, shared by every backbone.

For decision t: the seeker's latest utterance and the supporter's own latest
utterance before t, each embedded by a frozen sentence encoder (BGE-base, CLS,
L2-normalised), then reduced by a PCA fitted on training rows only.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from cate.config import CACHE
from cate.official import load_official

ENC = "BAAI/bge-base-en-v1.5"


def last_texts(dialogs, rows):
    seek, sup = [], []
    for r in rows:
        turns = dialogs[r["dialog"]]["turns"][: r["t"]]
        s = [x["text"] for x in turns if x["role"] == "seeker"]
        u = [x["text"] for x in turns if x["role"] != "seeker"]
        seek.append(s[-1] if s else "<none>")
        sup.append(u[-1] if u else "<none>")
    return seek, sup


@torch.no_grad()
def embed(texts, tok, model, bs=128):
    out = []
    for i in range(0, len(texts), bs):
        t = tok(texts[i : i + bs], padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
        h = model(**t).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(h, dim=-1).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["esconv", "annomi", "p4g"])
    ap.add_argument("--dim", type=int, default=16)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(ENC)
    model = AutoModel.from_pretrained(ENC, torch_dtype=torch.float16).cuda().eval()
    for ds in args.datasets:
        dialogs, samples, _ = load_official(ds, check=False)
        raw = {}
        for sp, rows in samples.items():
            a, b = last_texts(dialogs, rows)
            own = [dialogs[r["dialog"]]["turns"][r["t"]]["text"] for r in rows]  # the decision's own utterance
            raw[sp] = (embed(a, tok, model), embed(b, tok, model), embed(own, tok, model))
        out = {}
        for j, name in enumerate(("seeker", "self", "own")):
            x = raw["train"][j]
            mu = x.mean(0)
            _, _, vt = np.linalg.svd(x - mu, full_matrices=False)
            proj = vt[: args.dim].T
            sd = ((x - mu) @ proj).std(0)
            for sp in samples:
                out[f"{sp}_{name}"] = (((raw[sp][j] - mu) @ proj) / sd).astype(np.float32)
        np.savez(CACHE / f"situation_{ds}.npz", **out)
        print(ds, {k: v.shape for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    main()
