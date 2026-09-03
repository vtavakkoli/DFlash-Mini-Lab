import json
from pathlib import Path

import numpy as np

from dflash_mini_lab.decoding import (
    dflash2_decode,
    dflash3_mobs_decode,
    dflash4_jump_mobs_decode,
    dflash5_fused_jump_mobs_decode,
    dflash6_bmobs_decode,
    dflash6_boltzmann_decode,
    dflash_decode,
    normal_decode,
)
from dflash_mini_lab.runtime import CpuReferenceRuntime
from dflash_mini_lab.tokenizer import WordTokenizer

ROOT = Path(__file__).resolve().parents[1]


def test_speculative_modes_match_normal_greedy_output():
    rt = CpuReferenceRuntime(ROOT / "models/tiny_dflash_lab.npz")
    tok = WordTokenizer.load(ROOT / "models/tokenizer.json")
    prompts = json.loads((ROOT / "benchmarks/prompts.json").read_text())["prompts"][:4]
    for prompt in prompts:
        ids = tok.encode(prompt)
        normal, _ = normal_decode(rt, ids, 12)
        outputs = [
            dflash_decode(rt, ids, 12),
            dflash2_decode(rt, ids, 12, top_k=8),
            dflash3_mobs_decode(rt, ids, 12, top_k=8, refine_passes=0),
            dflash4_jump_mobs_decode(rt, ids, 12, top_k=8, jump_weight=0.5),
            dflash5_fused_jump_mobs_decode(rt, ids, 12, top_k=8, fused_weight=1.0, min_margin=0.0),
            dflash6_boltzmann_decode(rt, ids, 12, top_k=8, temperature=0.10),
            dflash6_bmobs_decode(rt, ids, 12, top_k=8, temperature=0.10),
        ]
        for output, _ in outputs:
            assert np.array_equal(normal, output)
        assert outputs[2][1].selector_pair_scores > 0
        assert outputs[3][1].jump_forward_passes > 0
        assert outputs[4][1].jump_forward_passes == 0
        assert outputs[5][1].boltzmann_candidate_scores > 0
        assert outputs[5][1].selector_pair_scores == 0
        assert outputs[6][1].boltzmann_candidate_scores > 0
        assert outputs[6][1].selector_pair_scores > 0


def test_linear_guidance_uses_less_work_than_dflash2():
    rt = CpuReferenceRuntime(ROOT / "models/tiny_dflash_lab.npz")
    tok = WordTokenizer.load(ROOT / "models/tokenizer.json")
    ids = tok.encode("the city model predicts")
    _, d2 = dflash2_decode(rt, ids, 16, top_k=8)
    _, d3 = dflash3_mobs_decode(rt, ids, 16, top_k=8, refine_passes=0)
    _, d4 = dflash4_jump_mobs_decode(rt, ids, 16, top_k=8, jump_weight=0.5)
    _, d5 = dflash5_fused_jump_mobs_decode(rt, ids, 16, top_k=8, fused_weight=1.0, min_margin=0.0)
    _, d6b = dflash6_boltzmann_decode(rt, ids, 16, top_k=8, temperature=0.10)
    _, d6m = dflash6_bmobs_decode(rt, ids, 16, top_k=8, temperature=0.10)
    assert d3.total_guidance_scores < d2.selector_pair_scores
    assert d4.total_guidance_scores < d2.selector_pair_scores
    assert d5.total_guidance_scores < d2.selector_pair_scores
    assert d6b.total_guidance_scores < d2.selector_pair_scores
    assert d6m.total_guidance_scores < d2.selector_pair_scores
    assert d6b.jump_forward_passes == 0
    assert d6m.jump_forward_passes == 0
