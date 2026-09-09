"""Measure verifier widths on calibration prompts without fitting to test data."""
from __future__ import annotations

import numpy as np


def calibrate_verify_widths(runtime, prompts, methods, run_method, *, tokens: int, repeats: int = 2):
    """Choose a measured width for each speculative method on this backend.

    run_method(method, input_ids, tokens) returns (output, stats, metadata).
    Every width is warmed and order is rotated. Timings include the complete
    decoder, not just target forwards. Invalid outputs can never win.
    """
    if not prompts or tokens < 1 or repeats < 1:
        raise ValueError("width calibration requires prompts, positive tokens and repeats")
    block = int(runtime.block_size)
    if block < 1:
        raise ValueError("runtime block_size must be positive")
    widths = sorted({min(n, block) for n in (1, 2, 4, block)})
    encoded = [runtime.encode(prompt) for prompt in prompts]
    references = []
    for ids in encoded:
        seq = ids.copy()
        for _ in range(tokens):
            # Independent full-logit reference, never the optimized verifier.
            seq = np.append(seq, int(np.argmax(runtime.target_logits(seq)[-1])))
        references.append(seq)

    selected, trials = {}, {}
    previous = getattr(runtime, "max_verify_tokens", 0)
    try:
        for method in methods:
            rows = {width: [] for width in widths}
            for width in widths:
                runtime.max_verify_tokens = width
                run_method(method, encoded[0], min(tokens, block + 1))
            for prompt_index, ids in enumerate(encoded):
                for repeat in range(repeats):
                    shift = (prompt_index * repeats + repeat) % len(widths)
                    for width in widths[shift:] + widths[:shift]:
                        runtime.max_verify_tokens = width
                        output, stats, _ = run_method(method, ids, tokens)
                        rows[width].append({
                            "prompt_index": prompt_index,
                            "repeat": repeat,
                            "wall_seconds": float(stats.wall_seconds),
                            "target_forward_passes": int(stats.target_forward_passes),
                            "exact_match": bool(np.array_equal(output, references[prompt_index])),
                        })
            candidates = []
            for width, samples in rows.items():
                elapsed = sum(row["wall_seconds"] for row in samples)
                candidates.append({
                    "verify_width": width,
                    "tokens_per_second": tokens * len(samples) / max(elapsed, 1e-12),
                    "all_exact": all(row["exact_match"] for row in samples),
                    "rows": samples,
                })
            valid = [candidate for candidate in candidates if candidate["all_exact"]]
            if not valid:
                raise RuntimeError(f"No exact verification width for {method}")
            best = max(valid, key=lambda candidate: (candidate["tokens_per_second"], -candidate["verify_width"]))
            selected[method] = best["verify_width"]
            trials[method] = candidates
    finally:
        runtime.max_verify_tokens = previous
    return selected, {"widths": widths, "tokens": tokens, "repeats": repeats, "trials": trials}
