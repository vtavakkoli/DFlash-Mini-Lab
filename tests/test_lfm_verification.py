from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from dflash_mini_lab.lfm_all14_benchmark import METHODS, _run_method, _greedy_reference
from dflash_mini_lab.lfm_verification import trim_proposal, verify_draft


@pytest.mark.parametrize("start", [0, 2, 5])
def test_suffix_projection_matches_full_logits(runtime, start):
    ids = np.array([1, 3, 5, 7, 9, 11])
    expected = np.argmax(runtime.target_logits(ids)[start:], axis=-1)
    projected = []
    handle = runtime.target.lm_head.register_forward_pre_hook(
        lambda module, args: projected.append(int(args[0].shape[1])))
    try:
        actual = runtime.target_greedy_tokens(ids, start)
    finally:
        handle.remove()
    np.testing.assert_array_equal(actual, expected)
    assert projected == [len(ids) - start]


@pytest.mark.parametrize("fast,supports", [(False, True), (True, False)])
def test_full_projection_fallback(runtime, fast, supports):
    ids = np.array([1, 5, 7])
    expected = np.argmax(runtime.target_logits(ids)[1:], axis=-1)
    runtime.optimize_target_logits = fast
    runtime._supports_logits_to_keep = supports
    np.testing.assert_array_equal(runtime.target_greedy_tokens(ids, 1), expected)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("width,tokens", [(1, 0), (1, 1), (1, 7), (4, 7)])
def test_all_methods_match_independent_hybrid_lfm_greedy(runtime, method_options, method, width, tokens):
    ids = np.array([1, 3, 5])
    reference = _greedy_reference(runtime, ids, tokens)
    runtime.verify_widths = {method: width}
    output, stats, _ = _run_method(method, runtime, ids, tokens, **method_options)
    np.testing.assert_array_equal(output, reference)
    assert stats.new_tokens == tokens
    assert runtime.max_verify_tokens == 0


@pytest.mark.parametrize("method", METHODS[1:])
def test_all_methods_commit_bonus_and_respect_final_budget(runtime, method_options, method):
    # Strong drafter margin makes every selector propose token zero. The target
    # is controlled so full acceptance is certain on every block.
    with torch.no_grad():
        runtime.drafter.head.weight.zero_()
        runtime.drafter.head.bias.fill_(-100)
        runtime.drafter.head.bias[0] = 100
    calls = []
    def target(ids, start):
        calls.append(len(ids))
        return np.zeros(len(ids) - start, dtype=np.int64)
    with patch.object(runtime, "target_greedy_tokens", side_effect=target):
        output, stats, _ = _run_method(method, runtime, np.array([1, 3]), 11, **method_options)
    np.testing.assert_array_equal(output, [1, 3] + [0] * 11)
    assert stats.target_forward_passes == len(calls) == 3
    assert stats.accepted_draft_tokens == 9


@pytest.mark.parametrize("rejection", [0, 1, 2, 3, None])
def test_reject_first_middle_last_or_accept_all(rejection):
    seq = np.array([6, 7])
    proposal = np.array([8, 9, 10, 11])
    predicted = np.array([8, 9, 10, 11, 12])
    if rejection is not None:
        predicted[rejection] = 15
    runtime = SimpleNamespace(target_greedy_tokens=lambda ids, start: predicted)
    verifier, accepted, _ = verify_draft(runtime, seq, proposal)
    assert accepted == (4 if rejection is None else rejection)
    committed = np.concatenate([proposal[:accepted], verifier[accepted:accepted + 1]])
    expected = [8, 9, 10, 11, 12] if rejection is None else [8, 9, 10, 11][:rejection] + [15]
    np.testing.assert_array_equal(committed, expected)


def test_legacy_bonus_switch_and_numpy_runtime():
    runtime = SimpleNamespace(target_logits=lambda ids: np.eye(16)[ids + 1], use_bonus_token=False)
    verifier, accepted, _ = verify_draft(runtime, np.array([1, 2]), np.array([3, 4]))
    np.testing.assert_array_equal(verifier, [3, 4])
    assert accepted == 2


def test_invalid_proposals_and_widths():
    with pytest.raises(ValueError):
        trim_proposal(SimpleNamespace(), np.array([], dtype=int), 1)
    with pytest.raises(ValueError):
        trim_proposal(SimpleNamespace(max_verify_tokens=-1), np.array([1]), 1)
    np.testing.assert_array_equal(trim_proposal(SimpleNamespace(max_verify_tokens=2), np.arange(4), 1), [0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_suffix_argmax(runtime, dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 device required")
    runtime.target.to(device="cuda", dtype=dtype)
    ids = np.array([1, 3, 5, 7, 9])
    with torch.inference_mode():
        reference = runtime.target(input_ids=torch.tensor(ids, device="cuda").unsqueeze(0), use_cache=False).logits[0, 2:].argmax(-1).cpu().numpy()
    np.testing.assert_array_equal(runtime.target_greedy_tokens(ids, 2), reference)
