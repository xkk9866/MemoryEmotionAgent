"""Evidential Acceptance Memory (EAM).

A supporter does not store the dialogue. For each support strategy they keep a
binomial opinion about whether the seeker accepts it. The opinion is not read
from survey scores. An evidence reader looks at the seeker's reply to that
strategy and emits non-negative support and opposition. The two channels never
cancel, so a contradictory history becomes dissonance instead of a false
neutral. Evidence fades with a content-conditioned time constant.

The next-strategy distribution is the local text policy shifted by the
opinion's signed acceptance b - d, which is also the distance of the expected
acceptance probability from one half. One non-negative scale learns how far to
shift. At initialization the reader emits almost no evidence, so the shift is
zero and the policy equals the text policy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

N_STRAT = 8
PRIOR_W = 2.0


def eam_flags(name: str) -> dict:
    base = {
        "name": name,
        "memory": True,
        "decay": True,
        "dual": True,
        "fb_sup": True,
        "acceptance": True,
        "prior": True,
        "reaction": True,
        "closed": True,
    }
    spec = {
        "base": {"memory": False, "acceptance": False, "prior": False, "fb_sup": False},
        "eam": {},
        "eam_no_decay": {"decay": False},
        "eam_net": {"dual": False},
        "eam_no_fbsup": {"fb_sup": False},
        "eam_emotion": {"reaction": False},
        "eam_free": {"closed": False},
    }
    if name not in spec:
        raise KeyError(name)
    base.update(spec[name])
    return base


class EvidenceReader(nn.Module):
    """Seeker turn plus the strategy it reacts to -> evidence and time constant."""

    def __init__(self, dim: int = 768, hidden: int = 256):
        super().__init__()
        self.strat = nn.Embedding(N_STRAT + 1, 64)
        self.net = nn.Sequential(
            nn.Linear(dim + 64, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 3),
        )
        # Evidence starts near zero so the policy equals the text policy.
        # The time constant starts near 10 turns, long enough to cross a dialogue.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 0.0)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([-4.0, -4.0, 8.0]))

    def forward(self, emb: torch.Tensor, eval_strat: torch.Tensor):
        s = torch.where(eval_strat < 0, torch.full_like(eval_strat, N_STRAT), eval_strat)
        out = self.net(torch.cat([emb, self.strat(s)], dim=-1))
        e_pos = F.softplus(out[..., 0])
        e_neg = F.softplus(out[..., 1])
        tau = F.softplus(out[..., 2]) + 2.0
        return e_pos, e_neg, tau


def opinions_from_stream(
    e_pos, e_neg, tau, eval_strat, sup_strat, turn, query_turn, mask, flags
):
    """Accumulate per-strategy evidence at the query turn.

    Shapes: stream tensors [B, L], query_turn [B]. Returns acceptance
    features [B, 8, 4] = (b, d, u, dissonance) and usage features [B, 8, 2].
    """
    delta = (query_turn[:, None] - turn).float().clamp(min=0)
    if flags["decay"]:
        keep = torch.exp(-delta / tau)
    else:
        keep = torch.ones_like(delta)
    keep = keep * mask.float()
    onehot = F.one_hot(eval_strat.clamp(min=0), N_STRAT).float() * (eval_strat >= 0).float()[..., None]
    if flags["dual"]:
        pos = torch.einsum("bl,blk->bk", e_pos * keep, onehot)
        neg = torch.einsum("bl,blk->bk", e_neg * keep, onehot)
    else:
        net = torch.einsum("bl,blk->bk", (e_pos - e_neg) * keep, onehot)
        pos = F.relu(net)
        neg = F.relu(-net)
    total = pos + neg + PRIOR_W
    b = pos / total
    d = neg / total
    u = PRIOR_W / total
    diss = 2.0 * torch.minimum(b, d)
    acc = torch.stack([b, d, u, diss], dim=-1)

    sup_onehot = F.one_hot(sup_strat.clamp(min=0), N_STRAT).float() * (sup_strat >= 0).float()[..., None]
    use_decay = torch.exp(-delta / 6.0) * mask.float()
    usage = torch.einsum("bl,blk->bk", use_decay, sup_onehot)
    count = torch.einsum("bl,blk->bk", mask.float(), sup_onehot)
    usage_feat = torch.stack([usage, torch.log1p(count)], dim=-1)
    return acc, usage_feat


class EAMNet(nn.Module):
    def __init__(self, encoder_name: str, n_labels: int = N_STRAT):
        super().__init__()
        full = AutoModel.from_pretrained(encoder_name)
        self.enc_dec = bool(getattr(full.config, "is_encoder_decoder", False))
        # Encoder-decoder baselines (BlenderBot) are read through their encoder.
        self.encoder = full.get_encoder() if self.enc_dec else full
        hid = full.config.hidden_size if hasattr(full.config, "hidden_size") else full.config.d_model
        self.reader = EvidenceReader(768)
        self.drop = nn.Dropout(0.1)
        self.head_base = nn.Linear(hid, n_labels)
        # Closed-form scale. softplus(-4) is near zero, so the policy starts as the text policy.
        self.alpha = nn.Parameter(torch.tensor(-4.0))
        # Unconstrained mixer, used only by the eam_free ablation.
        self.delta = nn.Linear(N_STRAT * 6, n_labels)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, tok, stream, flags):
        tok = {k: v for k, v in tok.items() if k in ("input_ids", "attention_mask")}
        hidden = self.encoder(**tok).last_hidden_state
        if self.enc_dec:
            m = tok["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            h = (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)
        else:
            h = hidden[:, 0]
        h = self.drop(h)
        out = {}
        text_logits = self.head_base(h)
        if not flags["memory"]:
            out["logits"] = text_logits
            return out
        observed = stream["react"] if flags.get("reaction", True) else stream["emb"]
        e_pos, e_neg, tau = self.reader(observed, stream["eval_strat"])
        seeker = (stream["role"] == 0) & stream["mask"]
        e_pos = e_pos * seeker.float()
        e_neg = e_neg * seeker.float()
        acc, usage = opinions_from_stream(
            e_pos,
            e_neg,
            tau,
            torch.where(seeker, stream["eval_strat"], torch.full_like(stream["eval_strat"], -1)),
            stream["sup_strat"],
            stream["turn"],
            stream["query_turn"],
            stream["mask"],
            flags,
        )
        if not flags["acceptance"]:
            acc = torch.zeros_like(acc)
        b, d = acc[..., 0], acc[..., 1]
        # Expected acceptance is b + u/2, so b - d is twice the distance from 1/2.
        signed = b - d
        if flags.get("closed", True):
            shift = F.softplus(self.alpha) * signed
        else:
            feat = torch.cat([acc, usage], dim=-1).flatten(1)
            shift = torch.sigmoid(self.gate) * self.delta(feat)
            if flags["prior"]:
                shift = shift + torch.sigmoid(self.gate) * signed
        logits = text_logits + shift
        out.update({"logits": logits, "e_pos": e_pos, "e_neg": e_neg, "acc": acc})
        return out


def evidential_feedback_loss(e_pos, e_neg, fb_label, valid):
    """Expected cross-entropy under Beta(e_pos+1, e_neg+1), plus a KL shrink.

    fb_label is 1 for accepted (feedback >= 4), 0 for rejected (<= 2).
    """
    if not bool(valid.any()):
        return e_pos.sum() * 0.0
    a = e_pos[valid] + 1.0
    b = e_neg[valid] + 1.0
    y = fb_label[valid].float()
    s = a + b
    ece = y * (torch.digamma(s) - torch.digamma(a)) + (1 - y) * (torch.digamma(s) - torch.digamma(b))
    wrong_a = y * 1.0 + (1 - y) * a
    wrong_b = (1 - y) * 1.0 + y * b
    kl = (
        torch.lgamma(wrong_a + wrong_b)
        - torch.lgamma(wrong_a)
        - torch.lgamma(wrong_b)
        - torch.lgamma(torch.tensor(2.0, device=a.device))
        + (wrong_a - 1) * (torch.digamma(wrong_a) - torch.digamma(wrong_a + wrong_b))
        + (wrong_b - 1) * (torch.digamma(wrong_b) - torch.digamma(wrong_a + wrong_b))
    )
    return (ece + 0.05 * kl).mean()
