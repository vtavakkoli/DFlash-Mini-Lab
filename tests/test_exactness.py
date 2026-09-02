import json
from pathlib import Path

import numpy as np

from dflash_mini_lab.decoding import dflash2_decode, dflash3_mobs_decode, dflash_decode, normal_decode
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
        dflash, _ = dflash_decode(rt, ids, 12)
        dflash2, _ = dflash2_decode(rt, ids, 12, top_k=8)
        dflash3, stats3 = dflash3_mobs_decode(rt, ids, 12, top_k=8, refine_passes=1)
        assert np.array_equal(normal, dflash)
        assert np.array_equal(normal, dflash2)
        assert np.array_equal(normal, dflash3)
        assert stats3.selector_pair_scores > 0


def test_mobs_selector_uses_fewer_pair_scores_than_dflash2():
    rt = CpuReferenceRuntime(ROOT / "models/tiny_dflash_lab.npz")
    tok = WordTokenizer.load(ROOT / "models/tokenizer.json")
    ids = tok.encode("the city model predicts")
    _, d2 = dflash2_decode(rt, ids, 16, top_k=8)
    _, d3 = dflash3_mobs_decode(rt, ids, 16, top_k=8, refine_passes=1)
    assert d3.selector_pair_scores < d2.selector_pair_scores
