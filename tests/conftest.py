from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from transformers import Lfm2Config, Lfm2ForCausalLM

from dflash_mini_lab.lfm_aux import (
    AuxConfig, CompactParallelDrafter, PairwisePathSelector,
    CompactJumpHead, CompactFusedResidual,
)
from dflash_mini_lab.lfm_dspark import (
    LfmDSparkRuntime, DSparkLiteConfig, MarkovHead, ConfidenceHead,
)
from dflash_mini_lab.lfm_v10 import V10Config
from dflash_mini_lab.v11_boltzmann_mobs import V11Config
from dflash_mini_lab.v12_parareal import FEATURE_NAMES, PararealLinearModel, V12Config
from dflash_mini_lab.v14_simple_parareal import V14Estimator


@pytest.fixture
def runtime():
    """Actual tiny hybrid LFM plus real auxiliary modules; no network or weights."""
    torch.set_num_threads(1)
    torch.manual_seed(17)
    config = AuxConfig("test-lfm", 32, 32, 32, drafter_dim=16,
                       selector_rank=4, jump_dim=8, fused_rank=4)
    target = Lfm2ForCausalLM(Lfm2Config(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        layer_types=["full_attention", "conv"], block_multiple_of=16,
    )).eval()
    bundle = (config, torch.arange(32), CompactParallelDrafter(config),
              PairwisePathSelector(config), CompactJumpHead(config),
              CompactFusedResidual(config), {})
    ds = DSparkLiteConfig("test-lfm", 32, 32, 16, 4, markov_rank=4)
    tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: [1, 3, 5],
        decode=lambda ids, **kwargs: " ".join(map(str, ids)),
    )
    with patch("dflash_mini_lab.lfm_runtime.load_aux_bundle", return_value=bundle), \
         patch("dflash_mini_lab.lfm_runtime.AutoModelForCausalLM.from_pretrained", return_value=target), \
         patch("dflash_mini_lab.lfm_runtime.AutoTokenizer.from_pretrained", return_value=tokenizer), \
         patch("dflash_mini_lab.lfm_dspark.load_dspark_lite", return_value=(ds, MarkovHead(ds), ConfidenceHead(ds), {})):
        result = LfmDSparkRuntime("unused", "unused", cpu_threads=1)
    result.optimize_target_logits = True
    result.use_bonus_token = True
    return result


@pytest.fixture
def method_options():
    count = len(FEATURE_NAMES)
    return dict(
        top_k=4, jump_weight=0.5, fused_weight=1.0,
        boltzmann_temperature=0.15, bmobs_temperature=0.35,
        act_threshold=0.0, dspark_floor=0.0,
        v10_config=V10Config(), v11_config=V11Config(),
        v12_model=PararealLinearModel(np.zeros(count + 1), np.zeros(count), np.ones(count)),
        v12_config=V12Config(top_k=4),
        v14_estimator=V14Estimator(0.0, 1.0, 0.0),
    )
