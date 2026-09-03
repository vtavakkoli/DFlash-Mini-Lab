from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .qwen17_aux import load_bundle


METHODS = (
    "normal_cached",
    "dflash",
    "dflash2",
    "dflash3_mobs",
    "dflash4_jump_mobs",
    "dflash5_fused_jump_mobs",
    "dflash6_boltzmann",
    "dflash6_bmobs",
    "dflash7_act",
)


@dataclass
class Qwen17DecodeStats:
    method: str
    new_tokens: int
    wall_seconds: float
    prefill_seconds: float
    target_forward_passes: int
    target_input_tokens: int
    draft_forward_passes: int
    jump_forward_passes: int
    accepted_draft_tokens: int
    proposed_draft_tokens: int
    mean_verify_drafts: float
    selector_pair_scores: int = 0
    jump_candidate_scores: int = 0
    fused_candidate_scores: int = 0
    boltzmann_candidate_scores: int = 0
    v7_margin_threshold: float | None = None

    @property
    def tokens_per_second(self) -> float:
        return self.new_tokens / max(self.wall_seconds, 1e-12)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_draft_tokens / max(self.proposed_draft_tokens, 1)

    @property
    def tokens_per_target_pass(self) -> float:
        return self.new_tokens / max(self.target_forward_passes, 1)

    @property
    def total_guidance_scores(self) -> int:
        return self.selector_pair_scores + self.jump_candidate_scores + self.fused_candidate_scores + self.boltzmann_candidate_scores

    def to_dict(self) -> dict:
        result = asdict(self)
        result.update(
            tokens_per_second=self.tokens_per_second,
            acceptance_rate=self.acceptance_rate,
            tokens_per_target_pass=self.tokens_per_target_pass,
            total_guidance_scores=self.total_guidance_scores,
        )
        return result


def _top_k(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float32)
    k = min(max(1, int(k)), int(logits.shape[-1]))
    ids = np.argpartition(logits, -k, axis=-1)[:, -k:]
    vals = np.take_along_axis(logits, ids, axis=-1)
    order = np.argsort(vals, axis=-1)[:, ::-1]
    ids = np.take_along_axis(ids, order, axis=-1)
    vals = np.take_along_axis(vals, order, axis=-1)
    return ids.astype(np.int64, copy=False), vals.astype(np.float32, copy=False)


def _gumbel_for(global_ids: np.ndarray, seed: int) -> np.ndarray:
    x = np.asarray(global_ids, dtype=np.uint64)
    z = x ^ np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    u = ((z >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    return (-np.log(-np.log(u))).astype(np.float32)


class Qwen17Runtime:
    def __init__(self, aux_path: str, *, model_id: str | None = None, cpu_threads: int | None = None, dtype: str = "bfloat16"):
        threads = int(cpu_threads or os.getenv("CPU_THREADS", "2"))
        torch.set_num_threads(max(1, threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        config, candidate_ids, drafter, selector, jump, fused, metadata = load_bundle(aux_path)
        self.config = config
        self.metadata = metadata
        self.candidate_ids_t = candidate_ids.long().cpu()
        self.candidate_ids = self.candidate_ids_t.numpy().astype(np.int64, copy=False)
        self.candidate_set = set(int(x) for x in self.candidate_ids.tolist())
        self.drafter = drafter.eval(); self.selector = selector.eval(); self.jump = jump.eval(); self.fused = fused.eval()
        self.model_id = model_id or config.model_id
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.target = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch_dtype)
        self.target.eval().to("cpu")
        self.embedding = self.target.get_input_embeddings().weight.detach()
        self.target_parameter_count = sum(p.numel() for p in self.target.parameters())
        self.aux_parameter_count = sum(p.numel() for module in (self.drafter, self.selector, self.jump, self.fused) for p in module.parameters())
        self.head = self.drafter.candidate_head_weight.detach().float().cpu()

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(self.tokenizer.encode(text, add_special_tokens=True), dtype=np.int64)

    def _raw_hidden(self, outputs) -> torch.Tensor:
        return torch.cat([outputs.hidden_states[layer_id + 1][0] for layer_id in self.config.target_layer_ids], dim=-1).detach().to(torch.float16).cpu()

    @torch.inference_mode()
    def _prefill(self, prompt_ids: np.ndarray, *, need_hidden: bool):
        cache = DynamicCache(config=self.target.config)
        ids = torch.from_numpy(np.asarray(prompt_ids, dtype=np.int64)).long().unsqueeze(0)
        start = time.perf_counter_ns()
        outputs = self.target(input_ids=ids, past_key_values=cache, use_cache=True, output_hidden_states=need_hidden, return_dict=True, cache_position=torch.arange(ids.shape[-1], dtype=torch.long))
        seconds = (time.perf_counter_ns() - start) / 1e9
        anchor = int(torch.argmax(outputs.logits[0, -1].float()).item())
        return outputs.past_key_values, anchor, self._raw_hidden(outputs) if need_hidden else None, seconds

    @torch.inference_mode()
    def _target_step(self, cache, input_tokens: list[int], *, need_hidden: bool):
        ids = torch.tensor([input_tokens], dtype=torch.long)
        base = int(cache.get_seq_length())
        return self.target(input_ids=ids, past_key_values=cache, use_cache=True, output_hidden_states=need_hidden, return_dict=True, cache_position=torch.arange(base, base + len(input_tokens), dtype=torch.long))

    def _aux_inputs(self, raw_memory: torch.Tensor, anchor_token: int):
        keep = raw_memory[-self.config.memory_tokens :]
        pad_count = self.config.memory_tokens - int(keep.shape[0])
        if pad_count:
            keep = torch.cat([torch.zeros(pad_count, self.config.raw_memory_dim, dtype=keep.dtype), keep], dim=0)
        mask = torch.zeros(self.config.memory_tokens, dtype=torch.bool)
        if pad_count:
            mask[:pad_count] = True
        valid = ~mask
        summary = keep[valid].float().mean(dim=0) if bool(valid.any()) else torch.zeros(self.config.raw_memory_dim, dtype=torch.float32)
        anchor_embedding = self.embedding[int(anchor_token)].detach().float().cpu()
        return keep.unsqueeze(0), mask.unsqueeze(0), summary.unsqueeze(0), anchor_embedding.unsqueeze(0)

    @torch.inference_mode()
    def draft_parts(self, raw_memory: torch.Tensor, anchor_token: int):
        mem, mask, summary, anchor_emb = self._aux_inputs(raw_memory, anchor_token)
        latent, base_future, logits = self.drafter.hidden_and_logits(mem, mask, anchor_emb)
        return latent[0].float().cpu(), base_future[0].float().cpu(), logits[0].float().cpu().numpy(), summary, anchor_emb

    @staticmethod
    def choose_v7_verify_count(draft_logits: np.ndarray, maximum: int, threshold: float) -> int:
        maximum = min(int(maximum), int(draft_logits.shape[0]))
        if maximum <= 0: return 0
        if float(threshold) <= 0.0: return maximum
        logits = np.asarray(draft_logits[:maximum], dtype=np.float32)
        if logits.shape[-1] < 2: return maximum
        top2 = np.partition(logits, -2, axis=-1)[:, -2:]
        margins = np.max(top2, axis=-1) - np.min(top2, axis=-1)
        count = 0
        for margin in margins.tolist():
            if float(margin) < float(threshold): break
            count += 1
        return count

    @torch.inference_mode()
    def _selector_arrays(self, summary: torch.Tensor, anchor_emb: torch.Tensor):
        gate = self.selector.gate(summary)[0].float().cpu().numpy()
        anchor = self.selector.anchor(anchor_emb.float())[0].float().cpu().numpy() * gate
        return gate, anchor, self.selector.prev.weight.detach().float().cpu().numpy(), self.selector.next.weight.detach().float().cpu().numpy(), float(self.selector.scale)

    def _first_scores(self, cur: np.ndarray, arrays) -> np.ndarray:
        _, anchor, _, nxt, scale = arrays
        return (nxt[np.asarray(cur, dtype=np.int64)] * anchor[None, :]).sum(axis=-1) * scale

    def _pair_matrix(self, prev_ids: np.ndarray, cur_ids: np.ndarray, arrays) -> np.ndarray:
        gate, _, prev, nxt, scale = arrays
        a = prev[np.asarray(prev_ids, dtype=np.int64)] * gate[None, :]
        b = nxt[np.asarray(cur_ids, dtype=np.int64)]
        return (a[:, None, :] * b[None, :, :]).sum(axis=-1) * scale

    def _pair_from_one(self, prev_id: int, cur_ids: np.ndarray, arrays) -> np.ndarray:
        return self._pair_matrix(np.asarray([prev_id]), cur_ids, arrays)[0]

    def _pair_to_one(self, prev_ids: np.ndarray, next_id: int, arrays) -> np.ndarray:
        return self._pair_matrix(prev_ids, np.asarray([next_id]), arrays)[:, 0]

    def _dflash2_path(self, logits, summary, anchor_emb, top_k):
        top_ids, top_vals = _top_k(logits, top_k); arrays = self._selector_arrays(summary, anchor_emb)
        k = int(top_ids.shape[1]); block = int(top_ids.shape[0]); dp = top_vals[0] + self._first_scores(top_ids[0], arrays); work = k; backs=[]
        for pos in range(1, block):
            pair = self._pair_matrix(top_ids[pos - 1], top_ids[pos], arrays); work += k*k
            scores = dp[:, None] + pair + top_vals[pos][None, :]; best = np.argmax(scores, axis=0); dp = scores[best, np.arange(k)]; backs.append(best)
        idx = int(np.argmax(dp)); chosen=[idx]
        for back in reversed(backs): idx=int(back[idx]); chosen.append(idx)
        chosen.reverse(); return np.asarray([top_ids[p, chosen[p]] for p in range(block)], dtype=np.int64), work

    def _mobs_path(self, logits, summary, anchor_emb, top_k):
        top_ids, top_vals = _top_k(logits, top_k); arrays = self._selector_arrays(summary, anchor_emb)
        block=int(top_ids.shape[0]); k=int(top_ids.shape[1])
        if block == 0: return np.empty(0,dtype=np.int64),0
        if block % 2: center=block//2
        else:
            centers=(block//2-1,block//2); m0=float(top_vals[centers[0],0]-top_vals[centers[0],1]); m1=float(top_vals[centers[1],0]-top_vals[centers[1],1]); center=centers[0] if m0<=m1 else centers[1]
        chosen=np.full(block,-1,dtype=np.int64); score=top_vals[center]+self._first_scores(top_ids[center],arrays); work=k; chosen[center]=int(top_ids[center,int(np.argmax(score))])
        for dist in range(1,block+1):
            left=center-dist
            if left>=0:
                score=top_vals[left]+self._pair_to_one(top_ids[left],int(chosen[left+1]),arrays); work+=k
                if left==0: score=score+self._first_scores(top_ids[left],arrays); work+=k
                chosen[left]=int(top_ids[left,int(np.argmax(score))])
            right=center+dist
            if right<block:
                score=top_vals[right]+self._pair_from_one(int(chosen[right-1]),top_ids[right],arrays); work+=k; chosen[right]=int(top_ids[right,int(np.argmax(score))])
        return chosen,work

    def _gap_fill(self, top_ids, top_vals, chosen, summary, anchor_emb):
        arrays=self._selector_arrays(summary,anchor_emb); block=int(top_ids.shape[0]); work=0
        while np.any(chosen<0):
            snap=chosen.copy(); frontier=[]
            for pos in range(block):
                if snap[pos]>=0: continue
                if pos==0 or snap[pos-1]>=0 or (pos+1<block and snap[pos+1]>=0): frontier.append(pos)
            if not frontier: raise RuntimeError("MOBS gap fill could not advance")
            for pos in frontier:
                ids=top_ids[pos]; score=top_vals[pos].copy()
                if pos==0: score+=self._first_scores(ids,arrays); work+=len(ids)
                elif snap[pos-1]>=0: score+=self._pair_from_one(int(snap[pos-1]),ids,arrays); work+=len(ids)
                if pos+1<block and snap[pos+1]>=0: score+=self._pair_to_one(ids,int(snap[pos+1]),arrays); work+=len(ids)
                chosen[pos]=int(ids[int(np.argmax(score))])
        return chosen,work

    @torch.inference_mode()
    def _jump_path(self, logits, summary, anchor_emb, top_k, jump_weight):
        top_ids,top_vals=_top_k(logits,top_k); chosen=np.full(int(top_ids.shape[0]),-1,dtype=np.int64); hidden=self.jump(summary,anchor_emb)[0].float().cpu(); score_count=0
        for j,offset in enumerate(self.config.jump_offsets):
            pos=int(offset)-1
            if pos<0 or pos>=len(chosen) or j>=hidden.shape[0]: continue
            ids=top_ids[pos]; bonus=torch.mv(self.head[torch.from_numpy(ids).long()],hidden[j]).numpy().astype(np.float32); bonus-=float(np.max(bonus)); score=top_vals[pos]+float(jump_weight)*bonus; chosen[pos]=int(ids[int(np.argmax(score))]); score_count+=len(ids)
        if not np.any(chosen>=0):
            path,pairs=self._mobs_path(logits,summary,anchor_emb,top_k); return path,pairs,0
        path,pairs=self._gap_fill(top_ids,top_vals,chosen,summary,anchor_emb); return path,pairs,score_count

    @torch.inference_mode()
    def _fused_path(self, latent, base_future, logits, summary, anchor_emb, top_k, fused_weight):
        top_ids,top_vals=_top_k(logits,top_k); chosen=np.full(int(top_ids.shape[0]),-1,dtype=np.int64); hidden=self.fused(latent.unsqueeze(0),base_future.unsqueeze(0))[0].float().cpu(); score_count=0
        for j,offset in enumerate(self.config.jump_offsets):
            pos=int(offset)-1
            if pos<0 or pos>=len(chosen) or j>=hidden.shape[0]: continue
            ids=top_ids[pos]; bonus=torch.mv(self.head[torch.from_numpy(ids).long()],hidden[j]).numpy().astype(np.float32); bonus-=float(np.max(bonus)); score=top_vals[pos]+float(fused_weight)*bonus; chosen[pos]=int(ids[int(np.argmax(score))]); score_count+=len(ids)
        if not np.any(chosen>=0):
            path,pairs=self._mobs_path(logits,summary,anchor_emb,top_k); return path,pairs,0
        path,pairs=self._gap_fill(top_ids,top_vals,chosen,summary,anchor_emb); return path,pairs,score_count

    def _boltzmann_path(self, logits, anchor_token, top_k, temperature):
        top_ids,top_vals=_top_k(logits,top_k); chosen=np.empty(int(top_ids.shape[0]),dtype=np.int64); work=0
        for pos in range(len(chosen)):
            ids=top_ids[pos]; vals=top_vals[pos]; margin=float(vals[0]-vals[1]) if len(vals)>1 else 100.0; temp=max(float(temperature)*math.exp(-max(margin,0.0)),0.02*float(temperature),1e-5); global_ids=self.candidate_ids[ids]; seed=(int(anchor_token)*0x9E3779B1+pos*0x85EBCA77+0xD1B54A32)&0xFFFFFFFFFFFFFFFF; score=vals/temp+_gumbel_for(global_ids,seed); chosen[pos]=int(ids[int(np.argmax(score))]); work+=len(ids)
        return chosen,work

    def _bmobs_path(self, logits, summary, anchor_emb, anchor_token, top_k, temperature):
        top_ids,top_vals=_top_k(logits,top_k); block=int(top_ids.shape[0]); chosen=np.full(block,-1,dtype=np.int64)
        if block==0: return chosen,0,0
        if block%2: pos=block//2
        else:
            centers=(block//2-1,block//2); margins=[float(top_vals[c,0]-top_vals[c,1]) if top_vals.shape[1]>1 else 100.0 for c in centers]; pos=centers[int(margins[1]<margins[0])]
        ids=top_ids[pos]; vals=top_vals[pos]; margin=float(vals[0]-vals[1]) if len(vals)>1 else 100.0; temp=max(float(temperature)*math.exp(-max(margin,0.0)),0.02*float(temperature),1e-5); global_ids=self.candidate_ids[ids]; seed=(int(anchor_token)*0x9E3779B1+pos*0x85EBCA77+0xB5297A4D)&0xFFFFFFFFFFFFFFFF; score=vals/temp+_gumbel_for(global_ids,seed); chosen[pos]=int(ids[int(np.argmax(score))]); path,pairs=self._gap_fill(top_ids,top_vals,chosen,summary,anchor_emb); return path,pairs,len(ids)

    def normal_cached_decode(self, prompt_ids, max_new_tokens):
        cache,anchor,_,prefill_seconds=self._prefill(prompt_ids,need_hidden=False); generated=[]; calls=0; target_input_tokens=0; start=time.perf_counter_ns()
        while len(generated)<int(max_new_tokens):
            generated.append(int(anchor))
            if len(generated)>=int(max_new_tokens): break
            outputs=self._target_step(cache,[int(anchor)],need_hidden=False); cache=outputs.past_key_values; calls+=1; target_input_tokens+=1; anchor=int(torch.argmax(outputs.logits[0,-1].float()).item())
        seconds=(time.perf_counter_ns()-start)/1e9
        return np.asarray(generated,dtype=np.int64),Qwen17DecodeStats(method="normal_cached",new_tokens=int(max_new_tokens),wall_seconds=seconds,prefill_seconds=prefill_seconds,target_forward_passes=calls,target_input_tokens=target_input_tokens,draft_forward_passes=0,jump_forward_passes=0,accepted_draft_tokens=0,proposed_draft_tokens=0,mean_verify_drafts=0.0)

    def speculative_decode(self,prompt_ids,max_new_tokens,*,method,top_k=8,jump_weight=0.5,fused_weight=1.0,boltzmann_temp=0.1,bmobs_temp=0.1,v7_margin_threshold=1.0):
        if method not in METHODS or method=="normal_cached": raise ValueError(f"unsupported speculative method: {method}")
        cache,anchor,raw_memory,prefill_seconds=self._prefill(prompt_ids,need_hidden=True); assert raw_memory is not None
        generated=[]; calls=target_input_tokens=draft_calls=jump_calls=accepted_total=proposed_total=selector_work=jump_work=fused_work=boltzmann_work=0; verify_counts=[]; start=time.perf_counter_ns()
        while len(generated)<int(max_new_tokens):
            generated.append(int(anchor))
            if len(generated)>=int(max_new_tokens): break
            remaining=int(max_new_tokens)-len(generated); latent,base_future,draft_logits,summary,anchor_emb=self.draft_parts(raw_memory,int(anchor)); draft_calls+=1; maximum=min(self.config.speculative_tokens,remaining); logits=draft_logits[:maximum]
            if method=="dflash7_act":
                verify_count=self.choose_v7_verify_count(logits,maximum,v7_margin_threshold); local=np.argmax(logits[:verify_count],axis=-1).astype(np.int64) if verify_count else np.empty(0,dtype=np.int64)
            else:
                verify_count=maximum
                if method=="dflash": local=np.argmax(logits,axis=-1).astype(np.int64)
                elif method=="dflash2": local,work=self._dflash2_path(logits,summary,anchor_emb,top_k); selector_work+=work
                elif method=="dflash3_mobs": local,work=self._mobs_path(logits,summary,anchor_emb,top_k); selector_work+=work
                elif method=="dflash4_jump_mobs": local,pairs,scores=self._jump_path(logits,summary,anchor_emb,top_k,jump_weight); selector_work+=pairs; jump_work+=scores; jump_calls+=1
                elif method=="dflash5_fused_jump_mobs": local,pairs,scores=self._fused_path(latent[:maximum],base_future[:maximum],logits,summary,anchor_emb,top_k,fused_weight); selector_work+=pairs; fused_work+=scores
                elif method=="dflash6_boltzmann": local,work=self._boltzmann_path(logits,int(anchor),top_k,boltzmann_temp); boltzmann_work+=work
                elif method=="dflash6_bmobs": local,pairs,work=self._bmobs_path(logits,summary,anchor_emb,int(anchor),top_k,bmobs_temp); selector_work+=pairs; boltzmann_work+=work
                else: raise AssertionError(method)
            verify_counts.append(int(verify_count)); proposal=self.candidate_ids[local] if verify_count else np.empty(0,dtype=np.int64); proposed_total+=int(verify_count); base_len=int(cache.get_seq_length()); verifier_input=[int(anchor)]+[int(x) for x in proposal.tolist()]; outputs=self._target_step(cache,verifier_input,need_hidden=True); cache=outputs.past_key_values; calls+=1; target_input_tokens+=len(verifier_input)
            if verify_count:
                target_next=torch.argmax(outputs.logits[0,:verify_count].float(),dim=-1).cpu().numpy().astype(np.int64); mismatch=np.flatnonzero(proposal!=target_next); accepted=verify_count if mismatch.size==0 else int(mismatch[0])
            else: accepted=0
            accepted_total+=accepted
            if accepted: generated.extend(int(x) for x in proposal[:accepted].tolist())
            keep_processed=1+accepted; cache.crop(base_len+keep_processed); current_hidden=self._raw_hidden(outputs); raw_memory=torch.cat([raw_memory,current_hidden[:keep_processed]],dim=0); anchor=int(torch.argmax(outputs.logits[0,accepted if accepted<verify_count else verify_count].float()).item())
        generated=generated[:int(max_new_tokens)]; seconds=(time.perf_counter_ns()-start)/1e9
        return np.asarray(generated,dtype=np.int64),Qwen17DecodeStats(method=method,new_tokens=int(max_new_tokens),wall_seconds=seconds,prefill_seconds=prefill_seconds,target_forward_passes=calls,target_input_tokens=target_input_tokens,draft_forward_passes=draft_calls,jump_forward_passes=jump_calls,accepted_draft_tokens=accepted_total,proposed_draft_tokens=proposed_total,mean_verify_drafts=float(np.mean(verify_counts)) if verify_counts else 0.0,selector_pair_scores=selector_work,jump_candidate_scores=jump_work,fused_candidate_scores=fused_work,boltzmann_candidate_scores=boltzmann_work,v7_margin_threshold=float(v7_margin_threshold) if method=="dflash7_act" else None)
