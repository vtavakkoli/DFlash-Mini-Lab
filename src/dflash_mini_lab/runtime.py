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
        self.p = {k: np.asarray(raw[k], dtype=np.float32) for k in raw.files}
        self.block_size = int(self.p["drafter.slot_emb.weight"].shape[0])
        self.vocab_size = int(self.p["target.token_emb.weight"].shape[0])
        self.target_dim = int(self.p["target.token_emb.weight"].shape[1])
        self.selector_scale = float(self.p["selector.a.weight"].shape[1] ** -0.5)

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

    def dflash2_select_path(self, draft_logits: np.ndarray, context: np.ndarray, prev_token: int, top_k: int = 4) -> np.ndarray:
        k = min(int(top_k), int(draft_logits.shape[-1]))
        top_ids = np.argsort(draft_logits, axis=-1)[:, -k:][:, ::-1]
        top_vals = np.take_along_axis(draft_logits, top_ids, axis=-1)
        gate = np.tanh(_linear(context[None, :], self.p["selector.gate.0.weight"], self.p["selector.gate.0.bias"])[0])
        a = self.p["selector.a.weight"]
        b = self.p["selector.b.weight"]

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
