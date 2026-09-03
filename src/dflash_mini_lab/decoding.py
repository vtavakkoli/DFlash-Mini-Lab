from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np

from .boltzmann import dflash6_bmobs_select_path, dflash6_boltzmann_select_path
from .runtime import CpuReferenceRuntime


@dataclass
class DecodeStats:
    method: str
    new_tokens: int
    target_forward_passes: int
    draft_forward_passes: int
    accepted_draft_tokens: int
    proposed_draft_tokens: int
    wall_seconds: float
    selector_pair_scores: int = 0
    jump_forward_passes: int = 0
    jump_candidate_scores: int = 0
    fused_anchor_uses: int = 0
    boltzmann_candidate_scores: int = 0

    @property
    def tokens_per_second(self) -> float:
        return self.new_tokens / max(self.wall_seconds, 1e-12)

    @property
    def latency_ms(self) -> float:
        return self.wall_seconds * 1000.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_draft_tokens / max(self.proposed_draft_tokens, 1)

    @property
    def tokens_per_target_pass(self) -> float:
        return self.new_tokens / max(self.target_forward_passes, 1)

    @property
    def total_guidance_scores(self) -> int:
        return self.selector_pair_scores + self.jump_candidate_scores + self.boltzmann_candidate_scores

    def to_dict(self) -> dict:
        out = asdict(self)
        out.update(
            tokens_per_second=self.tokens_per_second,
            latency_ms=self.latency_ms,
            acceptance_rate=self.acceptance_rate,
            tokens_per_target_pass=self.tokens_per_target_pass,
            total_guidance_scores=self.total_guidance_scores,
        )
        return out


def normal_decode(runtime: CpuReferenceRuntime, input_ids: list[int] | np.ndarray, max_new_tokens: int) -> tuple[np.ndarray, DecodeStats]:
    seq = np.asarray(input_ids, dtype=np.int64).copy()
    calls = 0
    t0 = time.perf_counter_ns()
    for _ in range(max_new_tokens):
        logits = runtime.target_logits(seq)
        calls += 1
        seq = np.append(seq, int(np.argmax(logits[-1])))
    return seq, DecodeStats("normal", max_new_tokens, calls, 0, 0, 0, (time.perf_counter_ns() - t0) / 1e9)


def _speculative_decode(
    runtime,
    input_ids,
    max_new_tokens,
    method,
    top_k=4,
    mobs_refine_passes=1,
    jump_weight=0.5,
    fused_weight=1.0,
    fused_min_margin=0.0,
    boltzmann_temperature=0.15,
):
    seq = np.asarray(input_ids, dtype=np.int64).copy()
    start_len = int(seq.size)
    target_calls = draft_calls = accepted_total = proposed_total = selector_pair_scores = 0
    jump_calls = jump_candidate_scores = fused_anchor_uses = boltzmann_candidate_scores = 0
    t0 = time.perf_counter_ns()

    while int(seq.size) - start_len < max_new_tokens:
        remaining = max_new_tokens - (int(seq.size) - start_len)
        context = runtime.context_features(seq)
        if method == "dflash5_fused_jump_mobs":
            draft_hidden, draft_logits = runtime.draft_hidden_and_logits(context)
        else:
            draft_hidden = None
            draft_logits = runtime.draft_logits(context)
        draft_calls += 1

        if method == "dflash":
            proposal = np.argmax(draft_logits, axis=-1).astype(np.int64)
        elif method == "dflash2":
            k = min(int(top_k), int(draft_logits.shape[-1]))
            block = int(draft_logits.shape[0])
            proposal = runtime.dflash2_select_path(draft_logits, context, int(seq[-1]), top_k=k)
            selector_pair_scores += k + max(0, block - 1) * k * k
        elif method == "dflash3_mobs":
            proposal, pairs = runtime.dflash3_mobs_select_path(
                draft_logits, context, int(seq[-1]), top_k=top_k, refine_passes=mobs_refine_passes
            )
            selector_pair_scores += int(pairs)
        elif method == "dflash4_jump_mobs":
            offsets, sparse = runtime.jump_logits(context)
            jump_calls += 1
            proposal, pairs, jumps = runtime.dflash4_jump_mobs_select_path(
                draft_logits, offsets, sparse, context, int(seq[-1]), top_k=top_k, jump_weight=jump_weight
            )
            selector_pair_scores += int(pairs)
            jump_candidate_scores += int(jumps)
        elif method == "dflash5_fused_jump_mobs":
            assert draft_hidden is not None
            proposal, pairs, jumps, anchors = runtime.dflash5_fused_jump_mobs_select_path(
                draft_hidden, draft_logits, context, int(seq[-1]), top_k=top_k,
                fused_weight=fused_weight, min_margin=fused_min_margin,
            )
            selector_pair_scores += int(pairs)
            jump_candidate_scores += int(jumps)
            fused_anchor_uses += int(anchors)
        elif method == "dflash6_boltzmann":
            proposal, sampled = dflash6_boltzmann_select_path(
                draft_logits, context, int(seq[-1]), top_k=top_k, temperature=boltzmann_temperature
            )
            boltzmann_candidate_scores += int(sampled)
        elif method == "dflash6_bmobs":
            proposal, pairs, sampled = dflash6_bmobs_select_path(
                runtime, draft_logits, context, int(seq[-1]), top_k=top_k, temperature=boltzmann_temperature
            )
            selector_pair_scores += int(pairs)
            boltzmann_candidate_scores += int(sampled)
        else:
            raise ValueError(f"unknown method: {method}")

        proposal = proposal[: min(int(proposal.size), remaining)]
        proposed_total += int(proposal.size)
        verify_input = np.concatenate([seq, proposal])
        logits = runtime.target_logits(verify_input)
        target_calls += 1
        p, k = int(seq.size), int(proposal.size)
        verifier = np.argmax(logits[p - 1 : p - 1 + k], axis=-1).astype(np.int64)
        mismatch = np.flatnonzero(proposal != verifier)
        accepted = k if mismatch.size == 0 else int(mismatch[0])
        accepted_total += accepted
        if accepted:
            seq = np.concatenate([seq, proposal[:accepted]])
        if accepted < k and int(seq.size) - start_len < max_new_tokens:
            seq = np.append(seq, verifier[accepted])

    seq = seq[: start_len + max_new_tokens]
    seconds = (time.perf_counter_ns() - t0) / 1e9
    return seq, DecodeStats(
        method, max_new_tokens, target_calls, draft_calls, accepted_total, proposed_total, seconds,
        selector_pair_scores=selector_pair_scores,
        jump_forward_passes=jump_calls,
        jump_candidate_scores=jump_candidate_scores,
        fused_anchor_uses=fused_anchor_uses,
        boltzmann_candidate_scores=boltzmann_candidate_scores,
    )


def dflash_decode(runtime, input_ids, max_new_tokens: int):
    return _speculative_decode(runtime, input_ids, max_new_tokens, method="dflash")


def dflash2_decode(runtime, input_ids, max_new_tokens: int, top_k: int = 4):
    return _speculative_decode(runtime, input_ids, max_new_tokens, method="dflash2", top_k=top_k)


def dflash3_mobs_decode(runtime, input_ids, max_new_tokens: int, top_k: int = 4, refine_passes: int = 1):
    return _speculative_decode(runtime, input_ids, max_new_tokens, method="dflash3_mobs", top_k=top_k, mobs_refine_passes=refine_passes)


def dflash4_jump_mobs_decode(runtime, input_ids, max_new_tokens: int, top_k: int = 4, jump_weight: float = 0.5):
    return _speculative_decode(runtime, input_ids, max_new_tokens, method="dflash4_jump_mobs", top_k=top_k, jump_weight=jump_weight)


def dflash5_fused_jump_mobs_decode(runtime, input_ids, max_new_tokens: int, top_k: int = 4, fused_weight: float = 1.0, min_margin: float = 0.0):
    return _speculative_decode(runtime, input_ids, max_new_tokens, method="dflash5_fused_jump_mobs", top_k=top_k, fused_weight=fused_weight, fused_min_margin=min_margin)


def dflash6_boltzmann_decode(runtime, input_ids, max_new_tokens: int, top_k: int = 8, temperature: float = 0.15):
    return _speculative_decode(runtime, input_ids, max_new_tokens, method="dflash6_boltzmann", top_k=top_k, boltzmann_temperature=temperature)


def dflash6_bmobs_decode(runtime, input_ids, max_new_tokens: int, top_k: int = 8, temperature: float = 0.15):
    return _speculative_decode(runtime, input_ids, max_new_tokens, method="dflash6_bmobs", top_k=top_k, boltzmann_temperature=temperature)
