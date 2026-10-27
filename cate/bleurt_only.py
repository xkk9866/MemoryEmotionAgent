"""Optional BLEURT on DEG hypotheses already implied by generation.json examples.

Recomputes memory-conditioned DEG retrieval and scores BLEURT-20-D3 if available.
"""

from __future__ import annotations

import json

import torch

from cate.config import RESULT
from cate.evaluate import _load_graphs, _load_model
from cate.generate import collect, retrieve


def main() -> None:
    payload = _load_graphs()
    graphs = [g for g in payload["esconv"] if g["failed"] == 0]
    train = [g for g in graphs if g["strategy_split"] == "train"]
    test = [g for g in graphs if g["strategy_split"] == "test"]
    n_emo = len(payload["esconv_emotions"])
    deg, flags = _load_model("esconv", "deg", n_emo)
    te = collect(test, "deg", deg, flags)
    tr = collect(train, "deg", deg, flags)
    hyps = retrieve(tr, te)
    gold = [r["response"] for r in te]
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        name = "lucadiliello/BLEURT-20-D3"
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForSequenceClassification.from_pretrained(name).cuda().eval()
        scores = []
        for i in range(0, len(hyps), 16):
            batch = tok(
                hyps[i : i + 16],
                gold[i : i + 16],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.no_grad():
                scores.append(mdl(**batch).logits.squeeze(-1).float().cpu())
        bleurt = float(torch.cat(scores).mean())
    except Exception as exc:  # noqa: BLE001
        bleurt = f"unavailable: {type(exc).__name__}: {exc}"
    path = RESULT / "generation.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    report["deg_bleurt"] = bleurt
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("deg_bleurt", bleurt, flush=True)


if __name__ == "__main__":
    main()
