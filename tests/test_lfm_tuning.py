from types import SimpleNamespace

import numpy as np
import pytest

from dflash_mini_lab.lfm_tuning import calibrate_verify_widths


def test_tuning_uses_exact_candidates_and_restores_runtime():
    runtime = SimpleNamespace(block_size=4, max_verify_tokens=3,
        encode=lambda prompt: np.array([1]),
        target_logits=lambda ids: np.eye(16)[ids + 1])
    visits = []
    def run(method, ids, tokens):
        width = runtime.max_verify_tokens
        visits.append((method, width))
        output = np.arange(1, tokens + 2)
        if width == 4:
            output[-1] = 0  # Fastest but incorrect candidate must be excluded.
        seconds = {1: 4.0, 2: 2.0, 4: 0.1}[width]
        return output, SimpleNamespace(wall_seconds=seconds, target_forward_passes=2), {}
    selected, calibration = calibrate_verify_widths(runtime, ["calibration"], ["test"], run, tokens=5)
    assert selected == {"test": 2}
    assert runtime.max_verify_tokens == 3
    assert calibration["trials"]["test"][2]["all_exact"] is False
    assert visits[3:6] != visits[6:9]  # Rotated measured order.


def test_tuning_failure_restores_width():
    runtime = SimpleNamespace(block_size=4, max_verify_tokens=3,
        encode=lambda prompt: np.array([1]),
        target_logits=lambda ids: np.eye(16)[ids + 1])
    def fail(*args):
        raise RuntimeError("target failed")
    with pytest.raises(RuntimeError, match="target failed"):
        calibrate_verify_widths(runtime, ["calibration"], ["test"], fail, tokens=2)
    assert runtime.max_verify_tokens == 3
