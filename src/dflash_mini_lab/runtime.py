from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * weight + bias


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    y = x @ weight.T
    return y if bias is None else y + bias


def _attention(x: np.ndarray, in_proj_weight: np.ndarray, in_proj_bias: np.ndarray, out_weight: np.ndarray, out_bias: np.ndarray, nhead: int, causal: bool) -> np.ndarray:
    seq_len, d_model = x.shape
    qkv = _linear(x, in_proj_weight, in_proj_bias)
    q, k, v = np.split(qkv, 3, axis=-1)
    head_dim = d_model // nhead
    q = q.reshape(seq_len, nhead, head_dim).transpose(1, 0, 2)
    k = k.reshape(seq_len, nhead, head_dim).transpose(1, 0, 2)
    v = v.reshape(seq_len, nhead, head_dim).transpose(1, 0, 2)
    scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(head_dim)
    if causal:
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        scores = np.where(mask[None, :, :], -1.0e30, scores)
    probs = _softmax(scores, axis=-1)
    out = probs @ v
    out = out.transpose(1, 0, 2).reshape(seq_len, d_model)
    return _linear(out, out_weight, out_bias)


def _encoder_layer(x: np.ndarray, p: dict[str, np.ndarray], prefix: str, nhead: int, causal: bool) -> np.ndarray:
    n1 = _layer_norm(x, p[f"{prefix}.norm1.weight"], p[f"{prefix}.norm1.bias"])
    attn = _attention(n1, p[f"{prefix}.self_attn.in_proj_weight"], p[f"{prefix}.self_attn.in_proj_bias"], p[f"{prefix}.self_attn.out_proj.weight"], p[f"{prefix}.self_attn.out_proj.bias"], nhead=nhead, causal=causal)
    x = x + attn
    n2 = _layer_norm(x, p[f"{prefix}.norm2.weight"], p[f"{prefix}.norm2.bias"])
    ff = _linear(n2, p[f"{prefix}.linear1.weight"], p[f"{prefix}.linear1.bias"])
    ff = np.maximum(ff, 0.0)
    ff = _linear(ff, p[f"{prefix}.linear2.weight"], p[f"{prefix}.linear2.bias"])
    return x + ff


class CpuReferenceRuntime:
    """Dependency-light NumPy/BLAS CPU runtime for the bundled tiny target and speculators."""

    def __init__(self, weights_path: str | Path):
        raw = np.load(weights_path)
        self.p: dict[str, np.ndarray] = {}
        for key in raw.files:
            dtype = np.int64 if key == "jump_offsets" else np.float32
            self.p[key] = np.asarray(raw[key], dtype=dtype)
        self.block_size = int(self.p["drafter.slot_emb.weight"].shape[0])
        self.vocab_size = int(self.p["target.token_emb.weight"].shape[0])
        self.target_dim = int(self.p["target.token_emb.weight"].shape[1])
        self.selector_scale = float(self.p["selector.a.weight"].shape[1] ** -0.5)
        self.jump_offsets = np.asarray(self.p.get("jump_offsets", np.asarray([], dtype=np.int64)), dtype=np.int64)

    def context_features(self, input_ids: np.ndarray) -> np.ndarray:
        emb = self.p["target.token_emb.weight"][input_ids]
        return np.concatenate([emb[-1], emb.mean(axis=0)], axis=0).astype(np.float32, copy=False)

    def target_logits(self, input_ids: np.ndarray) -> np.ndarray:
        input_ids = np.asarray(input_ids, dtype=np.int64)
        seq_len = int(input_ids.size)
        if seq_len > self.p["target.pos_emb.weight"].shape[0]:
            raise ValueError("sequence exceeds bundled model maximum length")
        x = self.p["target.token_emb.weight"][input_ids] + self.p["target.pos_emb.weight"][:seq_len]
        for i in range(2):
            x = _encoder_layer(x, self.p, f"target.transformer.layers.{i}", nhead=4, causal=True)
        x = _layer_norm(x, self.p["target.norm.weight"], self.p["target.norm.bias"])
        return _linear(x, self.p["target.lm_head.weight"]).astype(np.float32, copy=False)

    def draft_logits(self, context: np.ndarray) -> np.ndarray:
        context = np.asarray(context, dtype=np.float32)
        slots = self.p["drafter.slot_emb.weight"]
        base = _linear(context[None, :], self.p["drafter.context_proj.weight"], self.p["drafter.context_proj.bias"])[0]
        x = slots + base[None, :]
        x = _encoder_layer(x, self.p, "drafter.block_net.layers.0", nhead=4, causal=False)
        x = _layer_norm(x, self.p["drafter.norm.weight"], self.p["drafter.norm.bias"])
        return _linear(x, self.p["drafter.head.weight"], self.p["drafter.head.bias"]).astype(np.float32, copy=False)

    def jump_logits(self, context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict sparse future tokens directly at indexed offsets such as +2 and +4."""
        if self.jump_offsets.size == 0:
            return self.jump_offsets.copy(), np.empty((0, self.vocab_size), dtype=np.float32)
        context = np.asarray(context, dtype=np.float32)
        base = _linear(context[None, :], self.p["jump.context_proj.weight"], self.p["jump.context_proj.bias"])[0]
        offset_emb = self.p["jump.offset_emb.weight"][self.jump_offsets]
        x = np.tanh(base[None, :] + offset_emb)
        x = _layer_norm(x, self.p["jump.norm.weight"], self.p["jump.norm.bias"])
        logits = _linear(x, self.p["jump.head.weight"], self.p["jump.head.bias"])
        return self.jump_offsets.copy(), logits.astype(np.float32, copy=False)

    def _selector_state(self, context: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gate = np.tanh(_linear(context[None, :], self.p["selector.gate.0.weight"], self.p["selector.gate.0.bias"])[0])
        return gate, self.p["selector.a.weight"], self.p["selector.b.weight"]

    def dflash2_select_path(self, draft_logits: np.ndarray, context: np.ndarray, prev_token: int, top_k: int = 4) -> np.ndarray:
        k = min(int(top_k), int(draft_logits.shape[-1]))
        top_ids = np.argsort(draft_logits, axis=-1)[:, -k:][:, ::-1]
        top_vals = np.take_along_axis(draft_logits, top_ids, axis=-1)
        gate, a, b = self._selector_state(context)

        def pair(prev_ids: np.ndarray, cur_ids: np.ndarray) -> np.ndarray:
            pa = a[np.asarray(prev_ids, dtype=np.int64)] * gate[None, :]
            cb = b[np.asarray(cur_ids, dtype=np.int64)]
            return (pa[:, None, :] * cb[None, :, :]).sum(axis=-1) * self.selector_scale

        dp = top_vals[0] + pair(np.asarray([prev_token]), top_ids[0])[0]
        backptrs: list[np.ndarray] = []
        for pos in range(1, top_ids.shape[0]):
            scores = dp[:, None] + pair(top_ids[pos - 1], top_ids[pos]) + top_vals[pos][None, :]
            best_prev = np.argmax(scores, axis=0)
            dp = scores[best_prev, np.arange(scores.shape[1])]
            backptrs.append(best_prev)
        idx = int(np.argmax(dp))
        chosen = [idx]
        for bp in reversed(backptrs):
            idx = int(bp[idx])
            chosen.append(idx)
        chosen.reverse()
        return np.asarray([top_ids[pos, chosen[pos]] for pos in range(top_ids.shape[0])], dtype=np.int64)

    def dflash3_mobs_select_path(
        self,
        draft_logits: np.ndarray,
        context: np.ndarray,
        prev_token: int,
        top_k: int = 4,
        refine_passes: int = 1,
    ) -> tuple[np.ndarray, int]:
        """Middle-Out Bidirectional Selection (MOBS), with O(B*K) selector work."""
        k = min(int(top_k), int(draft_logits.shape[-1]))
        top_ids = np.argsort(draft_logits, axis=-1)[:, -k:][:, ::-1]
        top_vals = np.take_along_axis(draft_logits, top_ids, axis=-1)
        block = int(top_ids.shape[0])
        if block == 0:
            return np.empty(0, dtype=np.int64), 0

        gate, a, b = self._selector_state(context)
        pair_scores = 0

        def forward(prev_id: int, cur_ids: np.ndarray) -> np.ndarray:
            nonlocal pair_scores
            ids = np.asarray(cur_ids, dtype=np.int64); pair_scores += int(ids.size)
            pa = a[int(prev_id)] * gate
            return (b[ids] * pa[None, :]).sum(axis=-1) * self.selector_scale

        def backward(prev_ids: np.ndarray, next_id: int) -> np.ndarray:
            nonlocal pair_scores
            ids = np.asarray(prev_ids, dtype=np.int64); pair_scores += int(ids.size)
            cb = b[int(next_id)]
            return ((a[ids] * gate[None, :]) * cb[None, :]).sum(axis=-1) * self.selector_scale

        def forward_many(prev_ids: np.ndarray, cur_ids: np.ndarray) -> np.ndarray:
            nonlocal pair_scores
            prev = np.asarray(prev_ids, dtype=np.int64); cur = np.asarray(cur_ids, dtype=np.int64); pair_scores += int(cur.size)
            pa = a[prev] * gate[None, :]
            return (b[cur] * pa[:, None, :]).sum(axis=-1) * self.selector_scale

        def backward_many(prev_ids: np.ndarray, next_ids: np.ndarray) -> np.ndarray:
            nonlocal pair_scores
            prev = np.asarray(prev_ids, dtype=np.int64); nxt = np.asarray(next_ids, dtype=np.int64); pair_scores += int(prev.size)
            cb = b[nxt]
            return ((a[prev] * gate[None, None, :]) * cb[:, None, :]).sum(axis=-1) * self.selector_scale

        if block % 2:
            anchor = block // 2
        else:
            centers = (block // 2 - 1, block // 2)
            context_fingerprint = int(float(np.abs(context[: min(8, context.size)]).sum()) * 1_000_000.0)
            anchor = centers[(context_fingerprint ^ (int(prev_token) * 0x9E3779B1)) & 1]

        chosen = np.full(block, -1, dtype=np.int64)
        anchor_candidates = top_ids[anchor]
        anchor_scores = top_vals[anchor] + forward(prev_token, anchor_candidates)
        chosen[anchor] = int(anchor_candidates[int(np.argmax(anchor_scores))])

        for distance in range(1, block + 1):
            left = anchor - distance
            if left >= 0:
                candidates = top_ids[left]
                scores = top_vals[left] + backward(candidates, int(chosen[left + 1]))
                if left == 0: scores = scores + forward(prev_token, candidates)
                chosen[left] = int(candidates[int(np.argmax(scores))])
            right = anchor + distance
            if right < block:
                candidates = top_ids[right]
                scores = top_vals[right] + forward(int(chosen[right - 1]), candidates)
                chosen[right] = int(candidates[int(np.argmax(scores))])

        for _ in range(max(0, int(refine_passes))):
            for parity in (0, 1):
                positions = np.arange(parity, block, 2, dtype=np.int64)
                if positions.size == 0: continue
                snapshot = chosen.copy(); candidates = top_ids[positions]; scores = top_vals[positions].copy()
                prev_positions = np.maximum(positions - 1, 0); prev_ids = snapshot[prev_positions]
                prev_ids = np.where(positions == 0, int(prev_token), prev_ids)
                scores += forward_many(prev_ids, candidates)
                has_right = positions + 1 < block
                if np.any(has_right):
                    right_positions = positions[has_right] + 1
                    scores[has_right] += backward_many(candidates[has_right], snapshot[right_positions])
                best = np.argmax(scores, axis=1)
                chosen[positions] = candidates[np.arange(positions.size), best]
        return chosen, pair_scores

    def dflash4_jump_mobs_select_path(
        self,
        draft_logits: np.ndarray,
        jump_offsets: np.ndarray,
        jump_logits: np.ndarray,
        context: np.ndarray,
        prev_token: int,
        top_k: int = 4,
        jump_weight: float = 0.5,
    ) -> tuple[np.ndarray, int, int]:
        """Sparse indexed future anchors + O(B*K) gap filling.

        The jump head selects approximate tokens at sparse future offsets (+2/+4 in
        this tiny lab). Those anchors are selected from the drafter's top-k set and
        remaining positions are filled in parallel wavefronts using only adjacent
        selected neighbors. Target verification still decides every emitted token.
        """
        k = min(int(top_k), int(draft_logits.shape[-1]))
        top_ids = np.argsort(draft_logits, axis=-1)[:, -k:][:, ::-1]
        top_vals = np.take_along_axis(draft_logits, top_ids, axis=-1)
        block = int(top_ids.shape[0])
        if block == 0:
            return np.empty(0, dtype=np.int64), 0, 0

        gate, a, b = self._selector_state(context)
        pair_scores = 0
        jump_candidate_scores = 0

        def forward(prev_id: int, cur_ids: np.ndarray) -> np.ndarray:
            nonlocal pair_scores
            ids = np.asarray(cur_ids, dtype=np.int64); pair_scores += int(ids.size)
            pa = a[int(prev_id)] * gate
            return (b[ids] * pa[None, :]).sum(axis=-1) * self.selector_scale

        def backward(prev_ids: np.ndarray, next_id: int) -> np.ndarray:
            nonlocal pair_scores
            ids = np.asarray(prev_ids, dtype=np.int64); pair_scores += int(ids.size)
            cb = b[int(next_id)]
            return ((a[ids] * gate[None, :]) * cb[None, :]).sum(axis=-1) * self.selector_scale

        chosen = np.full(block, -1, dtype=np.int64)
        offsets = np.asarray(jump_offsets, dtype=np.int64)
        for row, offset in enumerate(offsets.tolist()):
            pos = int(offset) - 1
            if pos < 0 or pos >= block or row >= int(jump_logits.shape[0]):
                continue
            candidates = top_ids[pos]
            jump_candidate_scores += int(candidates.size)
            bonus = np.asarray(jump_logits[row, candidates], dtype=np.float32)
            bonus = bonus - float(np.max(bonus))
            scores = top_vals[pos] + float(jump_weight) * bonus
            chosen[pos] = int(candidates[int(np.argmax(scores))])

        if not np.any(chosen >= 0):
            fallback, work = self.dflash3_mobs_select_path(draft_logits, context, prev_token, top_k=top_k, refine_passes=0)
            return fallback, int(work), 0

        # Fill unanchored gaps in wavefronts. Every unanchored position is evaluated
        # once, against at most two already selected adjacent neighbors: O(B*K).
        while np.any(chosen < 0):
            snapshot = chosen.copy()
            frontier: list[int] = []
            for pos in range(block):
                if snapshot[pos] >= 0:
                    continue
                left_ready = pos == 0 or snapshot[pos - 1] >= 0
                right_ready = pos + 1 < block and snapshot[pos + 1] >= 0
                if left_ready or right_ready:
                    frontier.append(pos)
            if not frontier:
                raise RuntimeError("JUMP-MOBS could not advance gap-filling frontier")
            for pos in frontier:
                candidates = top_ids[pos]
                scores = top_vals[pos].copy()
                if pos == 0:
                    scores += forward(prev_token, candidates)
                elif snapshot[pos - 1] >= 0:
                    scores += forward(int(snapshot[pos - 1]), candidates)
                if pos + 1 < block and snapshot[pos + 1] >= 0:
                    scores += backward(candidates, int(snapshot[pos + 1]))
                chosen[pos] = int(candidates[int(np.argmax(scores))])

        return chosen, pair_scores, jump_candidate_scores
