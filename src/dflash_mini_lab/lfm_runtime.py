from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .lfm_aux import JUMP_OFFSETS, load_aux_bundle


_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_C1 = np.uint64(0x9E3779B97F4A7C15)
_C2 = np.uint64(0xBF58476D1CE4E5B9)
_C3 = np.uint64(0x94D049BB133111EB)


def _top_k_local(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(max(1, int(k)), int(logits.shape[-1])); vocab = int(logits.shape[-1])
    if vocab <= 256 or 2 * k >= vocab:
        ids = np.argsort(logits, axis=-1)[:, -k:][:, ::-1]
    else:
        ids = np.argpartition(logits, -k, axis=-1)[:, -k:]
        vals = np.take_along_axis(logits, ids, axis=-1); order = np.argsort(vals, axis=-1)[:, ::-1]; ids = np.take_along_axis(ids, order, axis=-1)
    vals = np.take_along_axis(logits, ids, axis=-1)
    return ids.astype(np.int64, copy=False), vals.astype(np.float32, copy=False)


def _context_seed(context: np.ndarray, prev_token: int, salt: int = 0) -> np.uint64:
    x = int(float(np.abs(np.asarray(context[: min(12, context.size)], dtype=np.float64)).sum()) * 1_000_000.0)
    return np.uint64((x ^ (int(prev_token) * 0x9E3779B1) ^ int(salt)) & 0xFFFFFFFFFFFFFFFF)


def _gumbel_for_ids(ids: np.ndarray, seed: np.uint64, position: int) -> np.ndarray:
    x = np.asarray(ids, dtype=np.uint64) + seed + np.uint64(position + 1) * _C1
    x = (x ^ (x >> np.uint64(30))) * _C2 & _MASK64; x = (x ^ (x >> np.uint64(27))) * _C3 & _MASK64; x = x ^ (x >> np.uint64(31))
    u = ((x >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / (1 << 53)); u = np.clip(u, 1e-12, 1.0 - 1e-12)
    return -np.log(-np.log(u))


def _adaptive_temperature(top_vals: np.ndarray, base_temperature: float) -> np.ndarray:
    base = max(float(base_temperature), 1e-6)
    if top_vals.shape[-1] < 2: return np.full(top_vals.shape[0], 1e-6, dtype=np.float64)
    margin = np.maximum(0.0, top_vals[:, 0].astype(np.float64) - top_vals[:, 1].astype(np.float64))
    return np.maximum(base * np.exp(-margin), base * 0.02)


class LfmReferenceRuntime:
    """Frozen real LFM verifier plus compact experimental DFlash auxiliaries."""

    def __init__(self, aux_path: str | Path, *, model_id: str | None = None, cpu_threads: int | None = None, dtype: str = "float32"):
        threads = int(cpu_threads or os.getenv("CPU_THREADS", "2")); torch.set_num_threads(max(1, threads))
        try: torch.set_num_interop_threads(1)
        except RuntimeError: pass
        config, candidate_ids, drafter, selector, jump, fused, metadata = load_aux_bundle(aux_path)
        self.config = config; self.metadata = metadata; self.candidate_ids_t = candidate_ids.long().cpu(); self.candidate_ids = self.candidate_ids_t.numpy().astype(np.int64, copy=False); self.candidate_size = int(self.candidate_ids.size); self.block_size = int(config.block_size); self.jump_offsets = np.asarray(JUMP_OFFSETS, dtype=np.int64)
        self.drafter = drafter.eval(); self.selector = selector.eval(); self.jump = jump.eval(); self.fused = fused.eval()
        self.selector_a = self.selector.a.weight.detach().float().cpu().numpy(); self.selector_b = self.selector.b.weight.detach().float().cpu().numpy(); self.selector_scale = float(self.selector.scale)
        self.model_id = model_id or config.model_id; torch_dtype = torch.float32 if dtype == "float32" else torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id); self.target = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch_dtype); self.target.eval(); self.target.to("cpu"); self.embedding = self.target.get_input_embeddings().weight.detach()
        self.target_parameter_count = sum(p.numel() for p in self.target.parameters()); self.aux_parameter_count = sum(p.numel() for model in (self.drafter, self.selector, self.jump, self.fused) for p in model.parameters())

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(self.tokenizer.encode(text, add_special_tokens=True), dtype=np.int64)

    def decode(self, ids: np.ndarray | list[int]) -> str:
        return self.tokenizer.decode([int(x) for x in ids], skip_special_tokens=True)

    @torch.inference_mode()
    def target_logits(self, input_ids: np.ndarray) -> np.ndarray:
        ids = torch.from_numpy(np.asarray(input_ids, dtype=np.int64)).long().unsqueeze(0); out = self.target(input_ids=ids, use_cache=False, return_dict=True)
        return out.logits[0].float().cpu().numpy()

    @torch.inference_mode()
    def context_features(self, input_ids: np.ndarray) -> np.ndarray:
        ids = torch.from_numpy(np.asarray(input_ids, dtype=np.int64)).long(); emb = self.embedding[ids].float(); n = int(self.config.context_tokens)
        recent = emb[max(0, int(emb.shape[0]) - n):]
        if int(recent.shape[0]) < n:
            pad = torch.zeros(n - int(recent.shape[0]), int(emb.shape[1]), dtype=emb.dtype); recent = torch.cat([pad, recent], dim=0)
        context = torch.cat([recent.reshape(-1), emb.mean(dim=0)], dim=-1)
        return context.cpu().numpy().astype(np.float32, copy=False)

    @torch.inference_mode()
    def draft_hidden_and_logits(self, context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = torch.from_numpy(np.asarray(context, dtype=np.float32)).unsqueeze(0); hidden = self.drafter.encode(x); logits = self.drafter.head(hidden)
        return hidden[0].float().cpu().numpy(), logits[0].float().cpu().numpy()

    def draft_logits(self, context: np.ndarray) -> np.ndarray: return self.draft_hidden_and_logits(context)[1]

    @torch.inference_mode()
    def jump_logits(self, context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = torch.from_numpy(np.asarray(context, dtype=np.float32)).unsqueeze(0); return self.jump_offsets.copy(), self.jump(x)[0].float().cpu().numpy()

    def proposal_argmax(self, draft_logits: np.ndarray) -> np.ndarray:
        return self.candidate_ids[np.argmax(draft_logits, axis=-1).astype(np.int64)]

    def _top_k(self, draft_logits: np.ndarray, top_k: int):
        local, vals = _top_k_local(draft_logits, top_k); return local, self.candidate_ids[local], vals

    @torch.inference_mode()
    def _selector_state(self, context: np.ndarray):
        x = torch.from_numpy(np.asarray(context, dtype=np.float32)).unsqueeze(0); gate = self.selector.gate(x)[0].float().cpu().numpy(); return gate, self.selector_a, self.selector_b

    def dflash2_select_path(self, draft_logits: np.ndarray, context: np.ndarray, prev_token: int, top_k: int = 8) -> np.ndarray:
        _, top_ids, top_vals = self._top_k(draft_logits, top_k); gate, a, b = self._selector_state(context)
        def pair(prev_ids, cur_ids):
            pa = a[np.asarray(prev_ids, dtype=np.int64)] * gate[None, :]; cb = b[np.asarray(cur_ids, dtype=np.int64)]; return (pa[:, None, :] * cb[None, :, :]).sum(axis=-1) * self.selector_scale
        dp = top_vals[0] + pair(np.asarray([prev_token]), top_ids[0])[0]; backptrs=[]
        for pos in range(1, top_ids.shape[0]):
            scores=dp[:,None]+pair(top_ids[pos-1],top_ids[pos])+top_vals[pos][None,:]; best=np.argmax(scores,axis=0); dp=scores[best,np.arange(scores.shape[1])]; backptrs.append(best)
        idx=int(np.argmax(dp)); chosen=[idx]
        for bp in reversed(backptrs): idx=int(bp[idx]); chosen.append(idx)
        chosen.reverse(); return np.asarray([top_ids[pos,chosen[pos]] for pos in range(top_ids.shape[0])],dtype=np.int64)

    def dflash3_mobs_select_path(self, draft_logits: np.ndarray, context: np.ndarray, prev_token: int, top_k: int = 8, refine_passes: int = 0):
        _, top_ids, top_vals = self._top_k(draft_logits, top_k); block=int(top_ids.shape[0])
        if block==0: return np.empty(0,dtype=np.int64),0
        gate,a,b=self._selector_state(context); pair_scores=0
        def forward(prev_id,cur_ids):
            nonlocal pair_scores
            ids=np.asarray(cur_ids,dtype=np.int64); pair_scores+=int(ids.size); pa=a[int(prev_id)]*gate; return (b[ids]*pa[None,:]).sum(axis=-1)*self.selector_scale
        def backward(prev_ids,next_id):
            nonlocal pair_scores
            ids=np.asarray(prev_ids,dtype=np.int64); pair_scores+=int(ids.size); cb=b[int(next_id)]; return ((a[ids]*gate[None,:])*cb[None,:]).sum(axis=-1)*self.selector_scale
        if block%2: anchor=block//2
        else:
            centers=(block//2-1,block//2); finger=int(float(np.abs(context[:min(8,context.size)]).sum())*1_000_000.0); anchor=centers[(finger^(int(prev_token)*0x9E3779B1))&1]
        chosen=np.full(block,-1,dtype=np.int64); c=top_ids[anchor]; s=top_vals[anchor]+forward(prev_token,c); chosen[anchor]=int(c[int(np.argmax(s))])
        for dist in range(1,block+1):
            left=anchor-dist
            if left>=0:
                c=top_ids[left]; s=top_vals[left]+backward(c,int(chosen[left+1]));
                if left==0: s+=forward(prev_token,c)
                chosen[left]=int(c[int(np.argmax(s))])
            right=anchor+dist
            if right<block:
                c=top_ids[right]; s=top_vals[right]+forward(int(chosen[right-1]),c); chosen[right]=int(c[int(np.argmax(s))])
        if refine_passes: raise ValueError("LFM real benchmark currently supports refine_passes=0 only")
        return chosen,int(pair_scores)

    def _anchored_gap_fill(self, top_ids: np.ndarray, top_vals: np.ndarray, chosen: np.ndarray, context: np.ndarray, prev_token: int):
        gate,a,b=self._selector_state(context); pair_scores=0; block=int(top_ids.shape[0])
        def forward(prev_id,cur_ids):
            nonlocal pair_scores
            ids=np.asarray(cur_ids,dtype=np.int64); pair_scores+=int(ids.size); pa=a[int(prev_id)]*gate; return (b[ids]*pa[None,:]).sum(axis=-1)*self.selector_scale
        def backward(prev_ids,next_id):
            nonlocal pair_scores
            ids=np.asarray(prev_ids,dtype=np.int64); pair_scores+=int(ids.size); cb=b[int(next_id)]; return ((a[ids]*gate[None,:])*cb[None,:]).sum(axis=-1)*self.selector_scale
        while np.any(chosen<0):
            snapshot=chosen.copy(); frontier=[]
            for pos in range(block):
                if snapshot[pos]>=0: continue
                if pos==0 or snapshot[pos-1]>=0 or (pos+1<block and snapshot[pos+1]>=0): frontier.append(pos)
            if not frontier: raise RuntimeError("anchored gap filling could not advance")
            for pos in frontier:
                c=top_ids[pos]; s=top_vals[pos].copy()
                if pos==0: s+=forward(prev_token,c)
                elif snapshot[pos-1]>=0: s+=forward(int(snapshot[pos-1]),c)
                if pos+1<block and snapshot[pos+1]>=0: s+=backward(c,int(snapshot[pos+1]))
                chosen[pos]=int(c[int(np.argmax(s))])
        return chosen,int(pair_scores)

    def dflash4_jump_mobs_select_path(self,draft_logits,jump_offsets,jump_logits,context,prev_token,top_k=8,jump_weight=0.5):
        top_local,top_ids,top_vals=self._top_k(draft_logits,top_k); block=int(top_ids.shape[0])
        if block==0:return np.empty(0,dtype=np.int64),0,0
        chosen=np.full(block,-1,dtype=np.int64); count=0
        for row,offset in enumerate(np.asarray(jump_offsets,dtype=np.int64).tolist()):
            pos=int(offset)-1
            if pos<0 or pos>=block or row>=int(jump_logits.shape[0]):continue
            local=top_local[pos]; c=top_ids[pos]; count+=int(c.size); bonus=np.asarray(jump_logits[row,local],dtype=np.float32); bonus-=float(np.max(bonus)); scores=top_vals[pos]+float(jump_weight)*bonus; chosen[pos]=int(c[int(np.argmax(scores))])
        if not np.any(chosen>=0):
            fallback,work=self.dflash3_mobs_select_path(draft_logits,context,prev_token,top_k=top_k,refine_passes=0); return fallback,int(work),0
        chosen,pairs=self._anchored_gap_fill(top_ids,top_vals,chosen,context,prev_token); return chosen,int(pairs),count

    @torch.inference_mode()
    def dflash5_fused_jump_mobs_select_path(self,draft_hidden,draft_logits,context,prev_token,top_k=8,fused_weight=1.0,min_margin=0.0):
        top_local,top_ids,top_vals=self._top_k(draft_logits,top_k); block=int(top_ids.shape[0])
        if block==0:return np.empty(0,dtype=np.int64),0,0,0
        hidden=torch.from_numpy(np.asarray(draft_hidden,dtype=np.float32)).unsqueeze(0); residual=self.fused.residual_logits(hidden)[0].float().cpu().numpy(); positions=self.jump_offsets-1; chosen=np.full(block,-1,dtype=np.int64); anchors=0; count=0
        for row,pos in enumerate(positions.tolist()):
            if pos<0 or pos>=block: continue
            local=top_local[pos]; c=top_ids[pos]; extra=residual[row,local]; count+=int(local.size); scores=top_vals[pos]+float(fused_weight)*extra; best=int(np.argmax(scores))
            if float(min_margin)>0 and scores.size>1:
                second=float(np.partition(scores,-2)[-2])
                if float(scores[best]-second)<float(min_margin): continue
            chosen[pos]=int(c[best]); anchors+=1
        if anchors==0:
            fallback,work=self.dflash3_mobs_select_path(draft_logits,context,prev_token,top_k=top_k,refine_passes=0); return fallback,int(work),count,0
        chosen,pairs=self._anchored_gap_fill(top_ids,top_vals,chosen,context,prev_token); return chosen,int(pairs),count,anchors

    def dflash6_boltzmann_select_path(self,draft_logits,context,prev_token,top_k=8,temperature=0.15):
        _,top_ids,top_vals=self._top_k(draft_logits,top_k)
        if top_ids.size==0:return np.empty(0,dtype=np.int64),0
        temps=_adaptive_temperature(top_vals,temperature); seed=_context_seed(context,prev_token,salt=0xD6B017); chosen=np.empty(top_ids.shape[0],dtype=np.int64)
        for pos in range(top_ids.shape[0]):
            c=top_ids[pos]; scores=top_vals[pos].astype(np.float64)/temps[pos]+_gumbel_for_ids(c,seed,pos); chosen[pos]=int(c[int(np.argmax(scores))])
        return chosen,int(top_ids.size)

    def dflash6_bmobs_select_path(self,draft_logits,context,prev_token,top_k=8,temperature=0.35):
        _,top_ids,top_vals=self._top_k(draft_logits,top_k); block=int(top_ids.shape[0])
        if block==0:return np.empty(0,dtype=np.int64),0,0
        centers=[block//2] if block%2 else [block//2-1,block//2]
        if len(centers)==2 and top_vals.shape[1]>1:
            margins=[float(top_vals[p,0]-top_vals[p,1]) for p in centers]; anchor=centers[int(np.argmin(margins))]
        else: anchor=centers[0]
        temps=_adaptive_temperature(top_vals,temperature); c=top_ids[anchor]; seed=_context_seed(context,prev_token,salt=0xB00B5); scores=top_vals[anchor].astype(np.float64)/temps[anchor]+_gumbel_for_ids(c,seed,anchor); chosen=np.full(block,-1,dtype=np.int64); chosen[anchor]=int(c[int(np.argmax(scores))]); chosen,pairs=self._anchored_gap_fill(top_ids,top_vals,chosen,context,prev_token)
        return chosen,int(pairs),int(c.size)
