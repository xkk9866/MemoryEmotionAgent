"""Is the inferred correction specific to this supporter-in-session, or a generic stage effect?

For every test dialogue: fit the offset (RLA form, s2 selected on validation) on the
first half of its supporter turns, and predict the second half with
  own      the dialogue's own offset,
  stranger the offset fitted on the first half of a different, randomly paired dialogue,
  none     no offset.
If the gain came from dialogue stage (later turns differ from earlier ones in every
conversation), a stranger's offset would help as much as the own one.
"""

from __future__ import annotations

import json

import numpy as np

from cate.config import RESULT
from cate.metrics import classification_scores
from cate.official import load_official
from cate.rla import adapt, dialogue_order, nll, select
from cate.spire import load_crossfit


def fit_offset(z, y, s2):
    order = [list(range(len(y))) + [len(y)]]
    zz = np.vstack([z, np.zeros((1, z.shape[1]))])
    yy = np.append(y, 0)
    return adapt(zz, yy, order, s2)[-1]  # offset after all rows are revealed


def main():
    out = {}
    rng = np.random.RandomState(0)
    for ds, tags in (("esconv", ("roberta-base", "roberta-large")), ("annomi", ("roberta-base", "roberta-large")),
                     ("p4g", ("roberta-base",))):
        _, samples, k = load_official(ds, check=False)
        yv = np.array([s["label"] for s in samples["valid"]])
        yt = np.array([s["label"] for s in samples["test"]])
        ov, ot = dialogue_order(samples["valid"]), dialogue_order(samples["test"])
        ot = [d for d in ot if len(d) >= 4]
        for tag in tags:
            res = {"own": [], "stranger": [], "none": []}
            for seed in (42, 43, 44):
                for r in load_crossfit(ds, tag, seed):
                    zv, zt = np.array(r["splits"]["valid"]["logits"]), np.array(r["splits"]["test"]["logits"])
                    s2, _, _ = select(zv, yv, ov)
                    halves = [(d[: len(d) // 2], d[len(d) // 2 :]) for d in ot]
                    offs = [fit_offset(zt[a], yt[a], s2) for a, _ in halves]
                    perm = rng.permutation(len(halves))
                    perm = np.where(perm == np.arange(len(perm)), np.roll(perm, 1), perm)
                    rows = np.concatenate([b for _, b in halves])
                    for name, shift in (("own", offs), ("stranger", [offs[j] for j in perm]),
                                        ("none", [np.zeros(k)] * len(offs))):
                        z = np.concatenate([zt[b] + sh for (_, b), sh in zip(halves, shift)])
                        f1 = 100 * classification_scores(yt[rows], z.argmax(1), k)["macro_f1"]
                        res[name].append((f1, nll(z, yt[rows])))
            out[f"{ds}:{tag}"] = {a: np.mean(v, 0).tolist() for a, v in res.items()}
            print(ds, tag, {a: [round(x, 3) for x in v] for a, v in out[f"{ds}:{tag}"].items()}, flush=True)
    (RESULT / "spire" / "session_specificity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
