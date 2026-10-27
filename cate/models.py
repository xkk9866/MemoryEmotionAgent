"""DEG-Mem network and the memory ablations that share its readout."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cate.config import BUDGET, MAX_CONFLICT, N_REL, TYPE2ID
from cate.subjective import conflict_preserving_fusion, cumulative_fusion, decay_opinion, evidence_opinion


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax on the last dimension. A row with an empty mask becomes zeros.

    Invalid keys are dropped after a finite shift. Filling them with the dtype
    minimum or with negative infinity makes the backward pass return NaN.
    """
    mask_f = mask.to(dtype=logits.dtype)
    invalid = ~mask.any(dim=-1, keepdim=True)
    finite_floor = torch.tensor(-1e4, device=logits.device, dtype=logits.dtype)
    filled = torch.where(mask, logits, finite_floor)
    filled = torch.where(invalid, torch.zeros_like(filled), filled)
    shifted = filled - filled.amax(dim=-1, keepdim=True)
    exp = torch.exp(shifted) * mask_f
    attn = exp / exp.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return torch.where(invalid, torch.zeros_like(attn), attn)


def _finite_opinion(b: torch.Tensor, d: torch.Tensor, u: torch.Tensor):
    """Keep a binomial opinion inside the simplex after a fused update."""
    b = torch.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    u = torch.nan_to_num(u, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    total = (b + d + u).clamp(min=1e-6)
    return b / total, d / total, u / total


def variant_flags(name: str) -> dict:
    base = {
        "name": name,
        "use_edges": False,
        "use_opinions": False,
        "use_decay": False,
        "use_conflict": False,
        "use_retrieval": False,
        "use_reject": False,
        "typed_halflife": False,
        "scalar": False,
        "overwrite": False,
        "window": 0,
        "agg": "mean",
        "ret_loss": False,
        "closure": False,
        "local_channel": False,
    }
    spec = {
        "meanpool": {},
        "recency": {"agg": "recency", "window": 4, "use_retrieval": True},
        "rag": {"agg": "rag", "use_retrieval": True},
        "forget": {"agg": "forget", "use_retrieval": True},
        "seqtrans": {"use_edges": False, "use_retrieval": True, "agg": "attention"},
        "window_graph": {
            "use_edges": True,
            "use_retrieval": True,
            "agg": "attention",
            "window": 5,
        },
        "overwrite": {
            "use_edges": True,
            "use_opinions": True,
            "use_decay": True,
            "use_conflict": True,
            "use_retrieval": True,
            "use_reject": True,
            "typed_halflife": True,
            "overwrite": True,
            "agg": "attention",
            "ret_loss": True,
        },
        "deg": {
            "use_edges": True,
            "use_opinions": True,
            "use_decay": True,
            "use_conflict": True,
            "use_retrieval": True,
            "use_reject": True,
            "typed_halflife": True,
            "agg": "attention",
            "ret_loss": True,
            "closure": True,
        },
        "deg_no_decay": {
            "use_edges": True,
            "use_opinions": True,
            "use_conflict": True,
            "use_retrieval": True,
            "use_reject": True,
            "typed_halflife": True,
            "agg": "attention",
            "ret_loss": True,
            "closure": True,
        },
        "deg_no_conflict": {
            "use_edges": True,
            "use_opinions": True,
            "use_decay": True,
            "use_retrieval": True,
            "use_reject": True,
            "typed_halflife": True,
            "agg": "attention",
            "ret_loss": True,
        },
        "deg_no_retrieval": {
            "use_edges": True,
            "use_opinions": True,
            "use_decay": True,
            "use_conflict": True,
            "use_reject": True,
            "typed_halflife": True,
            "agg": "mean",
            "ret_loss": False,
        },
        "deg_no_reject": {
            "use_edges": True,
            "use_opinions": True,
            "use_decay": True,
            "use_conflict": True,
            "use_retrieval": True,
            "typed_halflife": True,
            "agg": "attention",
            "ret_loss": True,
            "closure": True,
        },
        "deg_shared_hl": {
            "use_edges": True,
            "use_opinions": True,
            "use_decay": True,
            "use_conflict": True,
            "use_retrieval": True,
            "use_reject": True,
            "typed_halflife": False,
            "agg": "attention",
            "ret_loss": True,
            "closure": True,
        },
        "deg_scalar": {
            "use_edges": True,
            "use_opinions": True,
            "use_decay": True,
            "use_conflict": True,
            "use_retrieval": True,
            "use_reject": True,
            "typed_halflife": True,
            "scalar": True,
            "agg": "attention",
            "ret_loss": True,
            "closure": True,
        },
        "deg_no_retloss": {
            "use_edges": True,
            "use_opinions": True,
            "use_decay": True,
            "use_conflict": True,
            "use_retrieval": True,
            "use_reject": True,
            "typed_halflife": True,
            "agg": "attention",
            "ret_loss": False,
            "closure": True,
        },
    }
    if name not in spec:
        raise KeyError(name)
    base.update(spec[name])
    # The next strategy depends on the last few turns and on older evidence that
    # those turns would drop. Budgeted retrieval is reserved for the older nodes.
    if base["use_retrieval"] and base["agg"] == "attention" and (name == "overwrite" or name.startswith("deg")):
        base["local_channel"] = True
    return base


class DEGNet(nn.Module):
    def __init__(self, n_strategy: int, n_emotion: int, dim: int = 256, dropout: float = 0.2, budget: int = BUDGET):
        super().__init__()
        self.dim = dim
        self.budget = budget
        self.proj = nn.Linear(768, dim)
        self.type_emb = nn.Embedding(6, dim)
        self.strat_emb = nn.Embedding(9, dim)
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.edge_bias = nn.Parameter(torch.tensor([0.2, 0.4, 0.6, 0.5]))
        self.half_life = nn.Parameter(torch.tensor([4.0, 40.0, 20.0, 12.0, 30.0, 16.0]))
        self.tau_raw = nn.Parameter(torch.tensor(-1.386))
        self.gamma_raw = nn.Parameter(torch.tensor(0.5))
        self.temp_raw = nn.Parameter(torch.tensor(0.0))
        score_in = dim + dim + 4 + 16
        self.scorer = nn.Sequential(
            nn.Linear(score_in, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        head_in = dim * 3 + 8 + 4 + 1
        self.strategy_head = nn.Sequential(
            nn.Linear(head_in, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, n_strategy),
        )
        self.emotion_head = nn.Sequential(
            nn.Linear(head_in, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, n_emotion),
        )
        self.drop = nn.Dropout(dropout)

    def half_lives(self, flags: dict) -> torch.Tensor:
        hl = F.softplus(self.half_life) + 0.5
        if not flags["typed_halflife"]:
            hl = hl.mean().expand_as(hl)
        return hl

    def encode(self, batch, flags: dict) -> torch.Tensor:
        emb = batch["emb"]
        typ = batch["type"]
        strat = batch["strategy"]
        strat_idx = torch.where(strat < 0, torch.full_like(strat, 8), strat)
        h = self.proj(emb) + self.type_emb(typ) + self.strat_emb(strat_idx)
        h = self.drop(h)
        # Flat memories do not run graph attention. A short window graph does,
        # but only inside encode() with a local turn mask below.
        if flags["name"] in {"meanpool", "recency", "rag", "forget"}:
            return h
        if not flags["use_edges"] and flags["agg"] != "attention":
            return h
        q, k, v = self.Wq(h), self.Wk(h), self.Wv(h)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dim)
        turn = batch["turn"]
        causal = turn[:, None, :] <= turn[:, :, None]
        pad = batch["node_mask"]
        pair = pad[:, :, None] & pad[:, None, :] & causal
        if flags["name"] == "window_graph" and flags["window"] > 0:
            local = (turn[:, None, :] - turn[:, :, None]).abs() <= flags["window"]
            pair = pair & local
        if flags["use_edges"]:
            bias = (batch["adj"] * self.edge_bias.view(1, 1, 1, N_REL)).sum(-1)
            logits = logits + bias
        attn = _masked_softmax(logits, pair)
        h = h + self.drop(torch.matmul(attn, v))
        return h

    def opinions(self, batch, query_turn: torch.Tensor, flags: dict):
        """Return b, d, u at each query time. Shapes are [B, M, N]."""
        b0 = batch["b0"]
        d0 = batch["d0"]
        u0 = batch["u0"]
        turn = batch["turn"].float()
        b = b0[:, None, :].expand(-1, query_turn.size(1), -1).clone()
        d = d0[:, None, :].expand_as(b).clone()
        u = u0[:, None, :].expand_as(b).clone()
        last = turn[:, None, :].expand_as(b).clone()
        node_hl = self.half_lives(flags)[batch["type"]]
        node_hl = node_hl[:, None, :].expand_as(b)
        tau = torch.sigmoid(self.tau_raw)
        visible = (batch["turn"][:, None, :] < query_turn[:, :, None]) & batch["node_mask"][:, None, :]
        if flags["window"] > 0:
            visible = visible & (
                batch["turn"][:, None, :] >= (query_turn[:, :, None] - flags["window"])
            )
        if not flags["use_opinions"]:
            z = torch.zeros_like(b)
            return z, z, torch.ones_like(b), visible
        for step in range(batch["fb_turn"].size(-1)):
            ft = batch["fb_turn"][:, :, step]
            valid = (ft >= 0)[:, None, :] & (ft[:, None, :] <= query_turn[:, :, None]) & visible
            delta = (ft[:, None, :].float() - last).clamp(min=0)
            if flags["use_decay"]:
                b_before, d_before, u_before = decay_opinion(b, d, u, delta, node_hl)
            else:
                b_before, d_before, u_before = b, d, u
            pol = batch["fb_pol"][:, None, :, step].expand_as(b)
            conf = (batch["fb_conf"][:, :, step] * batch["fb_rel"][:, :, step]).clamp(0, 0.95)
            conf = conf[:, None, :].expand_as(b)
            be, de, ue = evidence_opinion(pol, conf)
            if flags["use_conflict"] and not flags["scalar"]:
                bf, df, uf, _dc = conflict_preserving_fusion(
                    b_before, d_before, u_before, be, de, ue, tau
                )
            else:
                bf, df, uf = cumulative_fusion(b_before, d_before, u_before, be, de, ue)
            b = torch.where(valid, bf, b)
            d = torch.where(valid, df, d)
            u = torch.where(valid, uf, u)
            last = torch.where(valid, ft[:, None, :].float().expand_as(last), last)
        if flags["use_decay"]:
            delta = (query_turn[:, :, None].float() - last).clamp(min=0)
            b, d, u = decay_opinion(b, d, u, delta, node_hl)
        if flags["scalar"]:
            c = (b - d + 1) * 0.5
            b, d, u = c, 1 - c, torch.zeros_like(c)
        if flags["overwrite"]:
            killed = self._overwrite_mask(batch, query_turn)
            visible = visible & ~killed
        b, d, u = _finite_opinion(b, d, u)
        b = b * visible.float()
        d = d * visible.float()
        u = torch.where(visible, u, torch.zeros_like(u))
        return b, d, u, visible

    def _overwrite_mask(self, batch, query_turn: torch.Tensor) -> torch.Tensor:
        contradict = batch["contradict"]
        turn = batch["turn"]
        killed = torch.zeros(
            query_turn.size(0), query_turn.size(1), turn.size(1), dtype=torch.bool, device=turn.device
        )
        for c in range(contradict.size(-1)):
            idx = contradict[:, :, c].clamp(min=0)
            valid = contradict[:, :, c] >= 0
            partner_turn = torch.gather(turn, 1, idx)
            later = valid[:, None, :] & (partner_turn[:, None, :] > turn[:, None, :])
            later = later & (partner_turn[:, None, :] < query_turn[:, :, None])
            killed = killed | later
        return killed

    def _alpha(self, scores: torch.Tensor, visible: torch.Tensor, batch, flags: dict, query_emb_raw, h):
        neg = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~visible, neg)
        if flags["agg"] == "mean" or not flags["use_retrieval"]:
            alpha = visible.float()
            alpha = alpha / alpha.sum(-1, keepdim=True).clamp(min=1.0)
            return alpha, scores
        if flags["agg"] == "recency":
            scores = -batch["turn"][:, None, :].float().expand_as(scores)
            scores = scores.masked_fill(~visible, neg)
        elif flags["agg"] == "rag":
            qn = F.normalize(query_emb_raw, dim=-1)
            kn = F.normalize(batch["emb"], dim=-1)
            scores = torch.einsum("bnd,bmd->bmn", kn, qn)
            utt = batch["is_utt"][:, None, :].bool()
            scores = scores.masked_fill(~(visible & utt), neg)
        elif flags["agg"] == "forget":
            delta = (batch["query_turn"][:, :, None] - batch["turn"][:, None, :]).float().clamp(min=0)
            strength = batch["conf"][:, None, :] * torch.exp(-delta / 8.0)
            scores = strength.masked_fill(~visible, neg)
        temp = F.softplus(self.temp_raw) + 0.15
        # Do not divide the dtype-minimum filler. That division overflows, and
        # torch.where does not stop the overflow from entering the backward.
        safe_scores = torch.where(visible, scores, torch.zeros_like(scores))
        alpha_soft = _masked_softmax(safe_scores / temp, visible)
        if flags["agg"] != "attention":
            k = min(4, scores.size(-1))
            idx = torch.topk(scores, k=k, dim=-1).indices
            hard = torch.zeros_like(alpha_soft).scatter(-1, idx, 1.0) * visible.float()
            alpha = hard / hard.sum(-1, keepdim=True).clamp(min=1.0)
            return alpha, scores
        k = min(self.budget, scores.size(-1))
        idx = torch.topk(scores, k=k, dim=-1).indices
        hard = torch.zeros_like(alpha_soft).scatter(-1, idx, 1.0)
        if flags["closure"] and flags["use_conflict"]:
            hard = self._close_conflict(hard, batch, visible)
        hard = hard * visible.float()
        alpha_hard = hard / hard.sum(-1, keepdim=True).clamp(min=1.0)
        alpha = alpha_hard + alpha_soft - alpha_soft.detach()
        return alpha, scores

    def _close_conflict(self, hard: torch.Tensor, batch, visible: torch.Tensor) -> torch.Tensor:
        contradict = batch["contradict"]
        for c in range(MAX_CONFLICT):
            partner = contradict[:, :, c].clamp(min=0)
            valid = (contradict[:, :, c] >= 0)[:, None, :].expand_as(hard)
            add = (hard > 0) & valid
            partner_exp = partner[:, None, :].expand_as(hard)
            extra = torch.zeros_like(hard).scatter(-1, partner_exp, add.float())
            hard = hard + extra
        return (hard > 0).float() * visible.float()

    def _features(self, h, alpha, b, d, u, visible, batch, query_h, query_turn, flags):
        readout = torch.einsum("bmn,bnd->bmd", alpha, h)
        emo = batch["type"][:, None, :] == TYPE2ID["emotion"]
        turn = batch["turn"][:, None, :].float()
        qt = query_turn[:, :, None].float().clamp(min=1.0)
        early = visible & emo & (turn <= qt / 3.0)
        late = visible & emo & (turn >= 2.0 * qt / 3.0)
        early_h = torch.einsum("bmn,bnd->bmd", early.float(), h)
        late_h = torch.einsum("bmn,bnd->bmd", late.float(), h)
        early_h = early_h / early.float().sum(-1, keepdim=True).clamp(min=1.0)
        late_h = late_h / late.float().sum(-1, keepdim=True).clamp(min=1.0)
        traj = late_h - early_h
        if not flags["use_opinions"]:
            traj = torch.zeros_like(traj)
        d_max = []
        strat = batch["strategy"]
        for sid in range(8):
            m = visible & (strat[:, None, :] == sid) & (batch["type"][:, None, :] == TYPE2ID["strategy"])
            masked = d.masked_fill(~m, 0.0)
            d_max.append(masked.max(dim=-1).values)
        d_max = torch.stack(d_max, dim=-1)
        if not flags["use_reject"]:
            d_max = torch.zeros_like(d_max)
        conf_mass = (batch["contradict"] >= 0).float().sum(-1)
        conf_mass = conf_mass[:, None, :].expand_as(b)
        stats = torch.stack(
            [
                (b * visible.float()).sum(-1) / visible.float().sum(-1).clamp(min=1),
                (d * visible.float()).sum(-1) / visible.float().sum(-1).clamp(min=1),
                (u * visible.float()).sum(-1) / visible.float().sum(-1).clamp(min=1),
                (conf_mass * visible.float()).sum(-1) / visible.float().sum(-1).clamp(min=1) / 3.0,
            ],
            dim=-1,
        )
        if not flags["use_opinions"]:
            stats = torch.zeros_like(stats)
        return readout, traj, d_max, stats

    def _window_readout(self, h, visible, batch, query_turn, width: int):
        turn = batch["turn"][:, None, :].float()
        qt = query_turn[:, :, None].float()
        local = visible & (turn >= qt - width)
        alpha = local.float()
        alpha = alpha / alpha.sum(-1, keepdim=True).clamp(min=1.0)
        return torch.einsum("bmn,bnd->bmd", alpha, h)

    def _compose_readout(self, h, scores, visible, b, d, u, batch, query_h, query_turn, query_emb_raw, flags):
        """Local discourse plus a budgeted subgraph of older evidence.

        Short dialogues have no older nodes, so retrieval covers the whole
        visible graph and the local window is not added a second time.
        """
        persist = visible
        has_old = None
        if flags["local_channel"]:
            turn = batch["turn"][:, None, :].float()
            qt = query_turn[:, :, None].float()
            old = visible & (turn < qt - 4)
            has_old = old.any(dim=-1, keepdim=True)
            persist = torch.where(has_old, old, visible)
        alpha, raw_scores = self._alpha(scores, persist, batch, flags, query_emb_raw, h)
        readout, traj, d_max, stats = self._features(
            h, alpha, b, d, u, visible, batch, query_h, query_turn, flags
        )
        if has_old is not None:
            local = self._window_readout(h, visible, batch, query_turn, 4)
            readout = torch.where(has_old, readout + local, readout)
        return readout, traj, d_max, stats, alpha, raw_scores

    def forward(self, batch, flags: dict):
        h = self.encode(batch, flags)
        q_turn = batch["query_turn"]
        b, d, u, visible = self.opinions(batch, q_turn, flags)
        q = self.proj(batch["query_emb"])
        rec = torch.exp(-(q_turn[:, :, None] - batch["turn"][:, None, :]).float().clamp(min=0) / 8.0)
        typ_e = self.type_emb(batch["type"])[:, None, :, :].expand(-1, q_turn.size(1), -1, -1)
        feat_s = torch.cat(
            [
                h[:, None, :, :].expand(-1, q_turn.size(1), -1, -1),
                q[:, :, None, :].expand(-1, -1, h.size(1), -1),
                b.unsqueeze(-1),
                d.unsqueeze(-1),
                u.unsqueeze(-1),
                rec.unsqueeze(-1),
                typ_e[..., :16],
            ],
            dim=-1,
        )
        scores = self.scorer(feat_s).squeeze(-1)
        readout, traj, d_max, stats, alpha, raw_scores = self._compose_readout(
            h, scores, visible, b, d, u, batch, q, q_turn, batch["query_emb"], flags
        )
        phase = batch["phase"].unsqueeze(-1)
        head_in = torch.cat([readout, q, traj, d_max, stats, phase], dim=-1)
        strategy_logits = self.strategy_head(head_in)
        if flags["use_reject"]:
            gamma = F.softplus(self.gamma_raw)
            strategy_logits = strategy_logits - gamma * d_max
        return {
            "strategy_logits": strategy_logits,
            "alpha": alpha,
            "scores": raw_scores,
            "visible": visible,
            "b": b,
            "d": d,
            "u": u,
            "h": h,
        }

    def emotion_forward(self, batch, flags: dict):
        """Conversation-level emotion readout at the end of the dialogue."""
        h = self.encode(batch, flags)
        q_turn = batch["full_turn"].unsqueeze(1)
        saved_qt = batch["query_turn"]
        batch["query_turn"] = q_turn
        b, d, u, visible = self.opinions(batch, q_turn, flags)
        q = self.proj(batch["full_query"]).unsqueeze(1)
        rec = torch.exp(-(q_turn[:, :, None] - batch["turn"][:, None, :]).float().clamp(min=0) / 8.0)
        typ_e = self.type_emb(batch["type"])[:, None, :, :16]
        feat_s = torch.cat(
            [
                h[:, None],
                q[:, :, None, :].expand(-1, 1, h.size(1), -1),
                b.unsqueeze(-1),
                d.unsqueeze(-1),
                u.unsqueeze(-1),
                rec.unsqueeze(-1),
                typ_e,
            ],
            dim=-1,
        )
        scores = self.scorer(feat_s).squeeze(-1)
        readout, traj, d_max, stats, _alpha, raw_scores = self._compose_readout(
            h, scores, visible, b, d, u, batch, q, q_turn, batch["full_query"].unsqueeze(1), flags
        )
        phase = torch.ones(batch["emb"].size(0), 1, 1, device=h.device)
        head_in = torch.cat([readout, q, traj, d_max, stats, phase], dim=-1)
        logits = self.emotion_head(head_in).squeeze(1)
        batch["query_turn"] = saved_qt
        anchor_scores = raw_scores.squeeze(1)
        return logits, anchor_scores


def collate_graphs(graphs: list[dict], device: torch.device) -> dict:
    bsz = len(graphs)
    n_max = max(g["emb"].shape[0] for g in graphs)
    m_max = max(max(int(g["dec_turn"].shape[0]), 1) for g in graphs)
    dim = graphs[0]["emb"].shape[1]
    k = graphs[0]["fb_turn"].shape[1]
    emb = torch.zeros(bsz, n_max, dim)
    typ = torch.zeros(bsz, n_max, dtype=torch.long)
    turn = torch.zeros(bsz, n_max, dtype=torch.long)
    strategy = torch.full((bsz, n_max), -1, dtype=torch.long)
    pol = torch.zeros(bsz, n_max)
    conf = torch.zeros(bsz, n_max)
    b0 = torch.zeros(bsz, n_max)
    d0 = torch.zeros(bsz, n_max)
    u0 = torch.ones(bsz, n_max)
    is_utt = torch.zeros(bsz, n_max, dtype=torch.long)
    node_mask = torch.zeros(bsz, n_max, dtype=torch.bool)
    fb_turn = torch.full((bsz, n_max, k), -1, dtype=torch.long)
    fb_pol = torch.zeros(bsz, n_max, k)
    fb_conf = torch.zeros(bsz, n_max, k)
    fb_rel = torch.zeros(bsz, n_max, k)
    contradict = torch.full((bsz, n_max, MAX_CONFLICT), -1, dtype=torch.long)
    adj = torch.zeros(bsz, n_max, n_max, N_REL)
    query = torch.zeros(bsz, m_max, dim)
    query_turn = torch.zeros(bsz, m_max, dtype=torch.long)
    phase = torch.zeros(bsz, m_max)
    labels = torch.full((bsz, m_max), -100, dtype=torch.long)
    hist = torch.zeros(bsz, m_max, dtype=torch.long)
    full_query = torch.zeros(bsz, dim)
    full_turn = torch.zeros(bsz, dtype=torch.long)
    emo = torch.full((bsz,), -100, dtype=torch.long)
    anchor = torch.full((bsz,), -1, dtype=torch.long)
    late_query = torch.zeros(bsz, dim)
    for i, g in enumerate(graphs):
        n = g["emb"].shape[0]
        m = int(g["dec_turn"].shape[0])
        emb[i, :n] = torch.from_numpy(g["emb"])
        typ[i, :n] = torch.from_numpy(g["type"])
        turn[i, :n] = torch.from_numpy(g["turn"])
        strategy[i, :n] = torch.from_numpy(g["strategy"])
        pol[i, :n] = torch.from_numpy(g["pol"])
        conf[i, :n] = torch.from_numpy(g["conf"])
        b0[i, :n] = torch.from_numpy(g["b0"])
        d0[i, :n] = torch.from_numpy(g["d0"])
        u0[i, :n] = torch.from_numpy(g["u0"])
        is_utt[i, :n] = torch.from_numpy(g["is_utt"])
        node_mask[i, :n] = True
        fb_turn[i, :n] = torch.from_numpy(g["fb_turn"])
        fb_pol[i, :n] = torch.from_numpy(g["fb_pol"])
        fb_conf[i, :n] = torch.from_numpy(g["fb_conf"])
        fb_rel[i, :n] = torch.from_numpy(g["fb_rel"])
        contradict[i, :n] = torch.from_numpy(g["contradict"])
        if len(g["edges"]):
            ee = torch.from_numpy(g["edges"])
            adj[i, ee[:, 0], ee[:, 1], ee[:, 2]] = 1.0
        if m:
            query[i, :m] = torch.from_numpy(g["dec_query"])
            query_turn[i, :m] = torch.from_numpy(g["dec_turn"])
            phase[i, :m] = torch.from_numpy(g["dec_phase"])
            labels[i, :m] = torch.from_numpy(g["dec_label"])
            hist[i, :m] = torch.from_numpy(g["dec_turn"])
        else:
            query_turn[i, 0] = int(g["n_turns"])
        seeker = (g["role"] == 0) & (g["is_utt"] == 1)
        if seeker.any():
            full_query[i] = torch.from_numpy(g["emb"][seeker].mean(axis=0))
        else:
            full_query[i] = torch.from_numpy(g["emb"].mean(axis=0))
        full_turn[i] = int(g["n_turns"])
        emo[i] = int(g["emotion_id"])
        anchor[i] = int(g["anchor"])
        late_query[i] = torch.from_numpy(g["late_query"])
    batch = {
        "emb": emb.to(device),
        "type": typ.to(device),
        "turn": turn.to(device),
        "strategy": strategy.to(device),
        "pol": pol.to(device),
        "conf": conf.to(device),
        "b0": b0.to(device),
        "d0": d0.to(device),
        "u0": u0.to(device),
        "is_utt": is_utt.to(device),
        "node_mask": node_mask.to(device),
        "fb_turn": fb_turn.to(device),
        "fb_pol": fb_pol.to(device),
        "fb_conf": fb_conf.to(device),
        "fb_rel": fb_rel.to(device),
        "contradict": contradict.to(device),
        "adj": adj.to(device),
        "query_emb": query.to(device),
        "query_turn": query_turn.to(device),
        "phase": phase.to(device),
        "labels": labels.to(device),
        "hist": hist.to(device),
        "full_query": full_query.to(device),
        "full_turn": full_turn.to(device),
        "emotion_id": emo.to(device),
        "anchor": anchor.to(device),
        "late_query": late_query.to(device),
    }
    return batch
