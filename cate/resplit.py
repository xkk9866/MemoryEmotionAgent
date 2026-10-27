"""Independent dialogue-level re-splits of a corpus, same split proportions as the official one.

split_seed 0 is the official split. For split_seed > 0 every dialogue is reassigned to
train/valid/test at random (numpy RandomState(split_seed)), preserving the official number
of dialogues per split, so each re-split is a fresh, independent test of the protocol.
"""

from __future__ import annotations

import numpy as np

from cate.official import load_official


def load_split(name: str, split_seed: int = 0):
    dialogs, samples, k = load_official(name, check=False)
    if split_seed == 0:
        return dialogs, samples, k
    rows = [r for sp in ("train", "valid", "test") for r in samples[sp]]
    sizes = {sp: len({r["dialog"] for r in samples[sp]}) for sp in ("train", "valid", "test")}
    ids = sorted({r["dialog"] for r in rows})
    perm = np.random.RandomState(split_seed).permutation(len(ids))
    assign, j = {}, 0
    for sp in ("test", "valid", "train"):
        for i in perm[j: j + sizes[sp]]:
            assign[ids[i]] = sp
        j += sizes[sp]
    out = {"train": [], "valid": [], "test": []}
    for r in sorted(rows, key=lambda r: (r["dialog"], r["t"])):
        out[assign[r["dialog"]]].append(r)
    return dialogs, out, k
