"""Logits of the released EmoDynamiX checkpoints on the official valid/test pickles.

The released preprocessing drops a few samples whose ERC output did not match
the window length, so every pickle row is aligned to our full-history sample
by its window text and gold response. Output has the same layout as
policy_train results, restricted to the aligned rows ("index" lists them).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

from cate.config import DATA, RESULT
from cate.metrics import classification_scores, preference_bias_official
from cate.official import _norm, load_official

REPO = DATA / "ext" / "EmoDynamiX-v2"
CKPT = {"esconv": DATA / "ext" / "emodx_ckpt" / "roberta-hg-esconv-preprocessed-checkpoints" / "checkpoint-2600.pth",
        "annomi": DATA / "ext" / "emodx_ckpt" / "roberta-hg-annomi-preprocessed-checkpoints" / "checkpoint-1800.pth"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esconv", choices=["esconv", "annomi"])
    args = ap.parse_args()
    dialogs, samples, k = load_official(args.dataset, check=False)
    cwd = os.getcwd()
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from dataloaders import AnnoMIPreProcessed, ESConvPreProcessed
    from modules.roberta import RobertaHeterogeneousGraph

    margs = SimpleNamespace(dataset=f"{args.dataset}-preprocessed", hg_dim=512, erc_temperature=0.5, erc_mixed=1,
                            exclude_others=0, feedback_threshold=0)
    model = RobertaHeterogeneousGraph(margs).cuda()
    model.load(str(CKPT[args.dataset]))
    model.eval()
    ds_cls = ESConvPreProcessed if args.dataset == "esconv" else AnnoMIPreProcessed
    res = {"dataset": args.dataset, "encoder": "EmoDynamiX released checkpoint", "prior": "none", "seed": 0, "splits": {}}
    for sp in ("valid", "test"):
        data = ds_cls(sp, margs)
        key = {}
        for i, s in enumerate(samples[sp]):
            gold = dialogs[s["dialog"]]["turns"][s["t"]]["text"]
            key.setdefault((_norm(s["window"]), _norm(gold)), []).append(i)
        index, logits, ys = [], [], []
        with torch.no_grad():
            for j in range(len(data)):
                b = data.collate_fn([data[j]])
                out = model(b)["logits"].float().cpu().numpy()[0]
                cand = key.get((_norm(str(data[j]["dialogue_history"])), _norm(str(data[j]["gold_standard"]))), [])
                cand = [c for c in cand if c not in index]
                if not cand:
                    continue
                index.append(cand[0])
                logits.append(out)
                ys.append(int(data[j]["label"]))
        y = np.array(ys)
        z = np.array(logits)
        assert all(samples[sp][i]["label"] == ys[n] for n, i in enumerate(index))
        sc = classification_scores(y, z.argmax(-1), k)
        sc["bias"] = preference_bias_official(y, z.argmax(-1), k)
        print(f"{args.dataset} {sp}: aligned {len(index)}/{len(data)} of {len(samples[sp])} official, "
              f"M-F1 {100 * sc['macro_f1']:.2f} W-F1 {100 * sc['weighted_f1']:.2f} B {sc['bias']:.2f}", flush=True)
        res["splits"][sp] = {"index": index, "y": ys, "logits": np.round(z, 4).tolist(), "scores": sc}
    os.chdir(cwd)
    out = RESULT / "policy" / f"{args.dataset}_emodx" / "emodynamix_none_s0.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res), encoding="utf-8")


if __name__ == "__main__":
    main()
