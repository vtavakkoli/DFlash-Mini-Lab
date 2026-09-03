from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True); var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * weight + bias


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True); e = np.exp(x); return e / np.sum(e, axis=axis, keepdims=True)


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    y = x @ weight.T; return y if bias is None else y + bias


def _attention(x: np.ndarray, in_proj_weight: np.ndarray, in_proj_bias: np.ndarray, out_weight: np.ndarray, out_bias: np.ndarray, nhead: int, causal: bool) -> np.ndarray:
    seq_len, d_model = x.shape; qkv = _linear(x, in_proj_weight, in_proj_bias); q, k, v = np.split(qkv, 3, axis=-1); head_dim = d_model // nhead
    q = q.reshape(seq_len, nhead, head_dim).transpose(1, 0, 2); k = k.reshape(seq_len, nhead, head_dim).transpose(1, 0, 2); v = v.reshape(seq_len, nhead, head_dim).transpose(1, 0, 2)
    scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(head_dim)
    if causal:
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1); scores = np.where(mask[None, :, :], -1.0e30, scores)
    probs = _softmax(scores, axis=-1); out = probs @ v; out = out.transpose(1, 0, 2).reshape(seq_len, d_model)
    return _linear(out, out_weight, out_bias)


def _encoder_layer(x: np.ndarray, p: dict[str, np.ndarray], prefix: str, nhead: int, causal: bool) -> np.ndarray:
    n1 = _layer_norm(x, p[f"{prefix}.norm1.weight"], p[f"{prefix}.norm1.bias"])
    x = x + _attention(n1, p[f"{prefix}.self_attn.in_proj_weight"], p[f"{prefix}.self_attn.in_proj_bias"], p[f"{prefix}.self_attn.out_proj.weight"], p[f"{prefix}.self_attn.out_proj.bias"], nhead=nhead, causal=causal)
    n2 = _layer_norm(x, p[f"{prefix}.norm2.weight"], p[f"{prefix}.norm2.bias"]); ff = _linear(n2, p[f"{prefix}.linear1.weight"], p[f"{prefix}.linear1.bias"]); ff = np.maximum(ff, 0.0)
    return x + _linear(ff, p[f"{prefix}.linear2.weight"], p[f"{prefix}.linear2.bias"])


def _top_k_ids_values(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k with a tiny-vocabulary fast path and partial selection for large V."""
    k = min(max(1, int(k)), int(logits.shape[-1])); vocab = int(logits.shape[-1])
    if vocab <= 256 or k * 2 >= vocab:
        ids = np.argsort(logits, axis=-1)[:, -k:][:, ::-1]
    else:
        ids = np.argpartition(logits, -k, axis=-1)[:, -k:]; vals = np.take_along_axis(logits, ids, axis=-1); order = np.argsort(vals, axis=-1)[:, ::-1]; ids = np.take_along_axis(ids, order, axis=-1)
    vals = np.take_along_axis(logits, ids, axis=-1)
    return ids.astype(np.int64, copy=False), vals.astype(np.float32, copy=False)


class CpuReferenceRuntime:
    def __init__(self, weights_path: str | Path):
        raw = np.load(weights_path); self.p: dict[str, np.ndarray] = {}
        for key in raw.files: self.p[key] = np.asarray(raw[key], dtype=np.int64 if key == "jump_offsets" else np.float32)
        self.block_size = int(self.p["drafter.slot_emb.weight"].shape[0]); self.vocab_size = int(self.p["target.token_emb.weight"].shape[0]); self.target_dim = int(self.p["target.token_emb.weight"].shape[1])
        self.selector_scale = float(self.p["selector.a.weight"].shape[1] ** -0.5); self.jump_offsets = np.asarray(self.p.get("jump_offsets", np.asarray([], dtype=np.int64)), dtype=np.int64)
        self.fused_jump_scale = float(self.p.get("fused_jump.codebook.weight", np.empty((1, 16))).shape[1] ** -0.5)

    def context_features(self, input_ids: np.ndarray) -> np.ndarray:
        emb = self.p["target.token_emb.weight"][input_ids]; return np.concatenate([emb[-1], emb.mean(axis=0)], axis=0).astype(np.float32, copy=False)

    def target_logits(self, input_ids: np.ndarray) -> np.ndarray:
        input_ids = np.asarray(input_ids, dtype=np.int64); seq_len = int(input_ids.size)
        if seq_len > self.p["target.pos_emb.weight"].shape[0]: raise ValueError("sequence exceeds bundled model maximum length")
        x = self.p["target.token_emb.weight"][input_ids] + self.p["target.pos_emb.weight"][:seq_len]
        for i in range(2): x = _encoder_layer(x, self.p, f"target.transformer.layers.{i}", nhead=4, causal=True)
        x = _layer_norm(x, self.p["target.norm.weight"], self.p["target.norm.bias"]); return _linear(x, self.p["target.lm_head.weight"]).astype(np.float32, copy=False)

    def draft_hidden_and_logits(self, context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        context = np.asarray(context, dtype=np.float32); slots = self.p["drafter.slot_emb.weight"]
        base = _linear(context[None, :], self.p["drafter.context_proj.weight"], self.p["drafter.context_proj.bias"])[0]; x = slots + base[None, :]
        x = _encoder_layer(x, self.p, "drafter.block_net.layers.0", nhead=4, causal=False); hidden = _layer_norm(x, self.p["drafter.norm.weight"], self.p["drafter.norm.bias"]).astype(np.float32, copy=False)
        logits = _linear(hidden, self.p["drafter.head.weight"], self.p["drafter.head.bias"]).astype(np.float32, copy=False); return hidden, logits

    def draft_logits(self, context: np.ndarray) -> np.ndarray:
        return self.draft_hidden_and_logits(context)[1]

    def jump_logits(self, context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.jump_offsets.size == 0: return self.jump_offsets.copy(), np.empty((0, self.vocab_size), dtype=np.float32)
        context = np.asarray(context, dtype=np.float32); base = _linear(context[None, :], self.p["jump.context_proj.weight"], self.p["jump.context_proj.bias"])[0]
        x = np.tanh(base[None, :] + self.p["jump.offset_emb.weight"][self.jump_offsets]); x = _layer_norm(x, self.p["jump.norm.weight"], self.p["jump.norm.bias"])
        return self.jump_offsets.copy(), _linear(x, self.p["jump.head.weight"], self.p["jump.head.bias"]).astype(np.float32, copy=False)

    def _selector_state(self, context: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gate = np.tanh(_linear(context[None, :], self.p["selector.gate.0.weight"], self.p["selector.gate.0.bias"])[0]); return gate, self.p["selector.a.weight"], self.p["selector.b.weight"]

    def dflash2_select_path(self, draft_logits: np.ndarray, context: np.ndarray, prev_token: int, top_k: int = 4) -> np.ndarray:
        k = min(int(top_k), int(draft_logits.shape[-1])); top_ids = np.argsort(draft_logits, axis=-1)[:, -k:][:, ::-1]; top_vals = np.take_along_axis(draft_logits, top_ids, axis=-1); gate, a, b = self._selector_state(context)
        def pair(prev_ids, cur_ids):
            pa = a[np.asarray(prev_ids,dtype=np.int64)] * gate[None,:]; cb = b[np.asarray(cur_ids,dtype=np.int64)]; return (pa[:,None,:]*cb[None,:,:]).sum(axis=-1)*self.selector_scale
        dp = top_vals[0] + pair(np.asarray([prev_token]), top_ids[0])[0]; backptrs=[]
        for pos in range(1,top_ids.shape[0]):
            scores=dp[:,None]+pair(top_ids[pos-1],top_ids[pos])+top_vals[pos][None,:]; best=np.argmax(scores,axis=0); dp=scores[best,np.arange(scores.shape[1])]; backptrs.append(best)
        idx=int(np.argmax(dp)); chosen=[idx]
        for bp in reversed(backptrs): idx=int(bp[idx]); chosen.append(idx)
        chosen.reverse(); return np.asarray([top_ids[pos,chosen[pos]] for pos in range(top_ids.shape[0])],dtype=np.int64)

    def dflash3_mobs_select_path(self, draft_logits: np.ndarray, context: np.ndarray, prev_token: int, top_k: int = 4, refine_passes: int = 1) -> tuple[np.ndarray,int]:
        k=min(int(top_k),int(draft_logits.shape[-1])); top_ids=np.argsort(draft_logits,axis=-1)[:,-k:][:,::-1]; top_vals=np.take_along_axis(draft_logits,top_ids,axis=-1); block=int(top_ids.shape[0])
        if block==0:return np.empty(0,dtype=np.int64),0
        gate,a,b=self._selector_state(context); pair_scores=0
        def forward(prev_id,cur_ids):
            nonlocal pair_scores
            ids=np.asarray(cur_ids,dtype=np.int64);pair_scores+=int(ids.size);pa=a[int(prev_id)]*gate;return (b[ids]*pa[None,:]).sum(axis=-1)*self.selector_scale
        def backward(prev_ids,next_id):
            nonlocal pair_scores
            ids=np.asarray(prev_ids,dtype=np.int64);pair_scores+=int(ids.size);cb=b[int(next_id)];return ((a[ids]*gate[None,:])*cb[None,:]).sum(axis=-1)*self.selector_scale
        def forward_many(prev_ids,cur_ids):
            nonlocal pair_scores
            prev=np.asarray(prev_ids,dtype=np.int64);cur=np.asarray(cur_ids,dtype=np.int64);pair_scores+=int(cur.size);pa=a[prev]*gate[None,:];return (b[cur]*pa[:,None,:]).sum(axis=-1)*self.selector_scale
        def backward_many(prev_ids,next_ids):
            nonlocal pair_scores
            prev=np.asarray(prev_ids,dtype=np.int64);nxt=np.asarray(next_ids,dtype=np.int64);pair_scores+=int(prev.size);cb=b[nxt];return ((a[prev]*gate[None,None,:])*cb[:,None,:]).sum(axis=-1)*self.selector_scale
        if block%2:anchor=block//2
        else:
            centers=(block//2-1,block//2);finger=int(float(np.abs(context[:min(8,context.size)]).sum())*1_000_000.0);anchor=centers[(finger^(int(prev_token)*0x9E3779B1))&1]
        chosen=np.full(block,-1,dtype=np.int64);c=top_ids[anchor];s=top_vals[anchor]+forward(prev_token,c);chosen[anchor]=int(c[int(np.argmax(s))])
        for dist in range(1,block+1):
            left=anchor-dist
            if left>=0:
                c=top_ids[left];s=top_vals[left]+backward(c,int(chosen[left+1]));
                if left==0:s+=forward(prev_token,c)
                chosen[left]=int(c[int(np.argmax(s))])
            right=anchor+dist
            if right<block:
                c=top_ids[right];s=top_vals[right]+forward(int(chosen[right-1]),c);chosen[right]=int(c[int(np.argmax(s))])
        for _ in range(max(0,int(refine_passes))):
            for parity in (0,1):
                positions=np.arange(parity,block,2,dtype=np.int64)
                if positions.size==0:continue
                snapshot=chosen.copy();c=top_ids[positions];s=top_vals[positions].copy();prev_pos=np.maximum(positions-1,0);prev=snapshot[prev_pos];prev=np.where(positions==0,int(prev_token),prev);s+=forward_many(prev,c);right=positions+1<block
                if np.any(right):s[right]+=backward_many(c[right],snapshot[positions[right]+1])
                best=np.argmax(s,axis=1);chosen[positions]=c[np.arange(positions.size),best]
        return chosen,pair_scores

    def _anchored_gap_fill(self, top_ids: np.ndarray, top_vals: np.ndarray, chosen: np.ndarray, context: np.ndarray, prev_token: int) -> tuple[np.ndarray,int]:
        gate,a,b=self._selector_state(context);pair_scores=0;block=int(top_ids.shape[0])
        def forward(prev_id,cur_ids):
            nonlocal pair_scores
            ids=np.asarray(cur_ids,dtype=np.int64);pair_scores+=int(ids.size);pa=a[int(prev_id)]*gate;return (b[ids]*pa[None,:]).sum(axis=-1)*self.selector_scale
        def backward(prev_ids,next_id):
            nonlocal pair_scores
            ids=np.asarray(prev_ids,dtype=np.int64);pair_scores+=int(ids.size);cb=b[int(next_id)];return ((a[ids]*gate[None,:])*cb[None,:]).sum(axis=-1)*self.selector_scale
        while np.any(chosen<0):
            snapshot=chosen.copy();frontier=[]
            for pos in range(block):
                if snapshot[pos]>=0:continue
                if pos==0 or snapshot[pos-1]>=0 or (pos+1<block and snapshot[pos+1]>=0):frontier.append(pos)
            if not frontier:raise RuntimeError("anchored gap filling could not advance")
            for pos in frontier:
                c=top_ids[pos];s=top_vals[pos].copy()
                if pos==0:s+=forward(prev_token,c)
                elif snapshot[pos-1]>=0:s+=forward(int(snapshot[pos-1]),c)
                if pos+1<block and snapshot[pos+1]>=0:s+=backward(c,int(snapshot[pos+1]))
                chosen[pos]=int(c[int(np.argmax(s))])
        return chosen,pair_scores

    def dflash4_jump_mobs_select_path(self,draft_logits,jump_offsets,jump_logits,context,prev_token,top_k=4,jump_weight=0.5):
        top_ids,top_vals=_top_k_ids_values(draft_logits,top_k);block=int(top_ids.shape[0])
        if block==0:return np.empty(0,dtype=np.int64),0,0
        chosen=np.full(block,-1,dtype=np.int64);count=0
        for row,offset in enumerate(np.asarray(jump_offsets,dtype=np.int64).tolist()):
            pos=int(offset)-1
            if pos<0 or pos>=block or row>=int(jump_logits.shape[0]):continue
            c=top_ids[pos];count+=int(c.size);bonus=np.asarray(jump_logits[row,c],dtype=np.float32);bonus-=float(np.max(bonus));s=top_vals[pos]+float(jump_weight)*bonus;chosen[pos]=int(c[int(np.argmax(s))])
        if not np.any(chosen>=0):
            fallback,work=self.dflash3_mobs_select_path(draft_logits,context,prev_token,top_k=top_k,refine_passes=0);return fallback,int(work),0
        chosen,pairs=self._anchored_gap_fill(top_ids,top_vals,chosen,context,prev_token);return chosen,int(pairs),count

    def dflash5_fused_jump_mobs_select_path(self,draft_hidden,draft_logits,context,prev_token,top_k=4,fused_weight=1.0,min_margin=0.0):
        """DFlash5 hot path: one top-k pass, fused candidate scoring, O(BK) fill."""
        top_ids,top_vals=_top_k_ids_values(draft_logits,top_k);block=int(top_ids.shape[0])
        if block==0:return np.empty(0,dtype=np.int64),0,0,0
        offsets=self.jump_offsets[(self.jump_offsets>=1)&(self.jump_offsets<=block)]
        if offsets.size==0 or "fused_jump.query.weight" not in self.p:
            fallback,work=self.dflash3_mobs_select_path(draft_logits,context,prev_token,top_k=top_k,refine_passes=0);return fallback,int(work),0,0
        positions=offsets-1;h=np.asarray(draft_hidden[positions],dtype=np.float32);q=_linear(h,self.p["fused_jump.query.weight"],self.p["fused_jump.query.bias"]);q=np.tanh(q+self.p["fused_jump.offset_emb.weight"][offsets]);candidate_ids=top_ids[positions];codebook=self.p["fused_jump.codebook.weight"][candidate_ids];residual=(codebook*q[:,None,:]).sum(axis=-1)*self.fused_jump_scale
        candidate_count=int(residual.size);chosen=np.full(block,-1,dtype=np.int64);anchors=0
        for row,pos in enumerate(positions.tolist()):
            scores=top_vals[pos]+float(fused_weight)*residual[row]
            best=int(np.argmax(scores))
            if float(min_margin)>0.0 and scores.size>1:
                second=float(np.partition(scores,-2)[-2]);margin=float(scores[best]-second)
                if margin<float(min_margin):continue
            chosen[pos]=int(top_ids[pos,best]);anchors+=1
        if anchors==0:
            fallback,work=self.dflash3_mobs_select_path(draft_logits,context,prev_token,top_k=top_k,refine_passes=0);return fallback,int(work),candidate_count,0
        chosen,pairs=self._anchored_gap_fill(top_ids,top_vals,chosen,context,prev_token);return chosen,int(pairs),candidate_count,anchors
