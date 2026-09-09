"""Shared exact greedy verification for the LFM speculative methods.

The last target row is a bonus token only after every proposed token matches.
The uncached full-logit API remains available as an independent reference.
"""
from __future__ import annotations

import time

import numpy as np


def target_token_ids(runtime, input_ids: np.ndarray, start: int) -> np.ndarray:
    """Return greedy IDs for a suffix, reducing on the target's own device."""
    fast = getattr(runtime, "target_greedy_tokens", None)
    if fast is not None:
        return fast(input_ids, start)
    # Keep small research runtimes and downstream callers compatible.
    return np.argmax(runtime.target_logits(input_ids)[start:], axis=-1).astype(np.int64)


def trim_proposal(runtime, proposal: np.ndarray, remaining: int) -> np.ndarray:
    proposal = np.asarray(proposal, dtype=np.int64)
    width = int(getattr(runtime, "max_verify_tokens", 0))
    if width < 0:
        raise ValueError("max_verify_tokens must be non-negative")
    limit = min(int(proposal.size), int(remaining))
    if width:
        limit = min(limit, width)
    if proposal.ndim != 1 or limit <= 0:
        raise ValueError("verification requires a non-empty 1D proposal and token budget")
    return proposal[:limit]


def verify_draft(runtime, seq: np.ndarray, proposal: np.ndarray):
    """One target call, accepted prefix length, and optional bonus prediction.

    A rejection at position a commits proposal[:a] followed by verifier[a].
    Full acceptance commits proposal followed by verifier[len(proposal)].
    Callers must still enforce their remaining output-token budget.
    """
    prefix, width = int(seq.size), int(proposal.size)
    if prefix == 0 or width == 0:
        raise ValueError("verification requires a non-empty prefix and proposal")
    verify_input = np.concatenate([seq, proposal])
    start = time.perf_counter()
    verifier = target_token_ids(runtime, verify_input, prefix - 1)
    elapsed = time.perf_counter() - start
    if verifier.shape != (width + 1,):
        raise ValueError("target must return one prediction per draft token plus a bonus row")
    mismatch = np.flatnonzero(proposal != verifier[:width])
    accepted = width if mismatch.size == 0 else int(mismatch[0])
    if not getattr(runtime, "use_bonus_token", True):
        verifier = verifier[:width]
    return verifier, accepted, elapsed
