"""Encode dialogues with RoBERTa and materialise evidence graphs."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from cate.config import CACHE, CKPT, ENCODER_NAME, HF_HOME, MAX_UTT_LEN
from cate.data import assign_esconv_splits, emotion_vocab, load_empathetic, load_esconv
from cate.graph_build import build_graph


def _bind_tokenizer(tokenizer, device, max_len):
    def encode(texts, batch_size=64):
        vectors = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            tok = tokenizer(
                chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
            )
            tok = {k: v.to(device) for k, v in tok.items()}
            with torch.no_grad(), torch.amp.autocast("cuda"):
                hidden = encoder(**tok).last_hidden_state[:, 0, :]
                logit = head(hidden.float())
            vectors.append((hidden.float().cpu().numpy(), torch.sigmoid(logit).float().cpu().numpy()))
        emb = np.concatenate([v[0] for v in vectors], axis=0).astype(np.float32)
        prob = np.concatenate([v[1] for v in vectors], axis=0).astype(np.float32)
        return emb, prob

    return encode


def _flatten(dialogs):
    texts, spans, situations = [], [], []
    cursor = 0
    for dialog in dialogs:
        n = len(dialog["turns"])
        texts.extend(turn["text"] if turn["text"] else "." for turn in dialog["turns"])
        spans.append((cursor, cursor + n))
        cursor += n
        situations.append(dialog.get("situation") or ".")
    return texts, spans, situations


def main():
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    smoke = os.environ.get("SMOKE", "0") == "1"
    device = torch.device("cuda")
    global encoder, head
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    encoder = AutoModel.from_pretrained(ENCODER_NAME).to(device).eval()
    head = nn.Linear(encoder.config.hidden_size, 28).to(device).eval()
    saved = torch.load(CKPT / "goemotions.pt", map_location="cpu", weights_only=False)
    encoder.load_state_dict(saved["encoder"])
    head.load_state_dict(saved["head"])
    encode = _bind_tokenizer(tokenizer, device, MAX_UTT_LEN)

    esconv = load_esconv()
    assign_esconv_splits(esconv)
    if smoke:
        esconv = esconv[:80]
    for row in esconv:
        row["dataset"] = "esconv"
    emo_map = emotion_vocab([r["emotion"] for r in esconv if r["failed"] == 0])
    # Failed dialogues reuse the same label set; unseen names get their own tail ids.
    for name in sorted({r["emotion"] for r in esconv}):
        emo_map.setdefault(name, len(emo_map))

    ed = load_empathetic()
    if smoke:
        ed = ed[:80]
    for row in ed:
        row["dataset"] = "ed"
        row["strategy_split"] = row["split"]
        row["fail_split"] = "ignore"
    ed_map = emotion_vocab([r["emotion"] for r in ed])

    def materialise(rows, label_map, desc):
        texts, spans, situations = _flatten(rows)
        print(f"encoding {desc}: {len(texts)} utterances", flush=True)
        utt_emb, utt_prob = encode(texts)
        sit_emb, _ = encode(situations)
        graphs = []
        for row, (a, b), s_emb in tqdm(list(zip(rows, spans, sit_emb)), desc=f"graph {desc}"):
            graphs.append(
                build_graph(row, utt_emb[a:b], utt_prob[a:b], s_emb, label_map[row["emotion"]])
            )
        return graphs

    es_graphs = materialise(esconv, emo_map, "esconv")
    ed_graphs = materialise(ed, ed_map, "ed")
    payload = {
        "esconv": es_graphs,
        "ed": ed_graphs,
        "esconv_emotions": emo_map,
        "ed_emotions": ed_map,
    }
    path = CACHE / ("graphs_smoke.pt" if smoke else "graphs.pt")
    torch.save(payload, path)
    print("saved", path, "esconv", len(es_graphs), "ed", len(ed_graphs), flush=True)
    del encoder, head
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
