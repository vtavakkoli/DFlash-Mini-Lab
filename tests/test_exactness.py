import json
from pathlib import Path

import numpy as np

from dflash_mini_lab.decoding import dflash2_decode, dflash_decode, normal_decode
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
        dflash2, _ = dflash2_decode(rt, ids, 12)
        assert np.array_equal(normal, dflash)
        assert np.array_equal(normal, dflash2)
