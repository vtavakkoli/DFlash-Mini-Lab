from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np

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

    def to_dict(self) -> dict:
        out = asdict(self)
        out.update(tokens_per_second=self.tokens_per_second, latency_ms=self.latency_ms, acceptance_rate=self.acceptance_rate, tokens_per_target_pass=self.tokens_per_target_pass)
        return out


def normal_decode(runtime: CpuReferenceRuntime, input_ids: list[int] | np.ndarray, max_new_tokens: int) -> tuple[np.ndarray, DecodeStats]:
    seq = np.asarray(input_ids, dtype=np.int64).copy()
    calls = 0
    t0 = time.perf_counter_ns()
    for _ in range(max_new_tokens):
        logits = runtime.target_logits(seq)
        calls += 1
        seq = np.append(seq, int(np.argmax(logits[-1])))
    seconds = (time.perf_counter_ns() - t0) / 1e9
    return seq, DecodeStats("normal", max_new_tokens, calls, 0, 0, 0, seconds)


def _speculative_decode(runtime: CpuReferenceRuntime, input_ids, max_new_tokens: int, method: str, top_k: int = 4):
    seq = np.asarray(input_ids, dtype=np.int64).copy()
    start_len = int(seq.size)
    target_calls = draft_calls = accepted_total = proposed_total = 0
    t0 = time.perf_counter_ns()
    while int(seq.size) - start_len < max_new_tokens:
        remaining = max_new_tokens - (int(seq.size) - start_len)
        context = runtime.context_features(seq)
        draft_logits = runtime.draft_logits(context)
        draft_calls += 1
        if method == "dflash":
            proposal = np.argmax(draft_logits, axis=-1).astype(np.int64)
        elif method == "dflash2":
            proposal = runtime.dflash2_select_path(draft_logits, context, int(seq[-1]), top_k=top_k)
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
    return seq, DecodeStats(method, max_new_tokens, target_calls, draft_calls, accepted_total, proposed_total, seconds)


def dflash_decode(runtime: CpuReferenceRuntime, input_ids, max_new_tokens: int):
    return _speculative_decode(runtime, input_ids, max_new_tokens, method="dflash")


def dflash2_decode(runtime: CpuReferenceRuntime, input_ids, max_new_tokens: int, top_k: int = 4):
    return _speculative_decode(runtime, input_ids, max_new_tokens, method="dflash2", top_k=top_k)
