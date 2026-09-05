from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import gc
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .lfm_aux import AuxConfig, TrainingTensors, load_aux_bundle, make_training_tensors
from .lfm_prepare import _read_list, generate_distillation_sequences
from .lfm_runtime import LfmReferenceRuntime


@dataclass(frozen=True)
class DSparkLiteConfig:
    model_id: str
    target_vocab_size: int
    candidate_size: int
    drafter_dim: int
    block_size: int
    markov_rank: int = 16


class MarkovHead(nn.Module):
    """Low-rank previous-token -> candidate-logit bias used by DSpark-Lite."""

    def __init__(self, config: DSparkLiteConfig):
        super().__init__()
        self.config = config
        self.prev = nn.Embedding(config.target_vocab_size, config.markov_rank)
        self.next = nn.Embedding(config.candidate_size, config.markov_rank)
        self.position_scale = nn.Parameter(torch.ones(config.block_size))
        self.scale = config.markov_rank ** -0.5

    def full_bias(self, prev_tokens: torch.Tensor) -> torch.Tensor:
        p = self.prev(prev_tokens)
        bias = torch.einsum("bpr,cr->bpc", p, self.next.weight) * self.scale
        return bias * self.position_scale.view(1, -1, 1)


class ConfidenceHead(nn.Module):
    """Scalar prefix-survival confidence head over draft hidden + Markov state."""

    def __init__(self, config: DSparkLiteConfig):
        super().__init__()
        self.proj = nn.Linear(config.drafter_dim + config.markov_rank, 1)

    def forward(self, draft_hidden: torch.Tensor, prev_state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([draft_hidden, prev_state], dim=-1)
        return self.proj(x).squeeze(-1)


def _batch_indices(size: int, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randint(0, size, (min(int(batch_size), int(size)),), generator=generator)


def train_dspark_lite(
    base_config: AuxConfig,
    candidate_ids: torch.Tensor,
    tensors: TrainingTensors,
    drafter: nn.Module,
    *,
    markov_rank: int = 16,
    markov_steps: int = 180,
    confidence_steps: int = 100,
    batch_size: int = 64,
    seed: int = 17,
) -> tuple[MarkovHead, ConfidenceHead, dict]:
    """Train only DSpark's cheap Markov + confidence heads.

    The DFlash backbone is frozen. Confidence labels are prefix-survival labels:
    once a teacher-conditioned DSpark prediction misses, later positions in that
    block are labeled non-surviving. This directly matches speculative prefix
    acceptance rather than only token-wise accuracy.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    config = DSparkLiteConfig(
        model_id=base_config.model_id,
        target_vocab_size=base_config.target_vocab_size,
        candidate_size=base_config.candidate_size,
        drafter_dim=base_config.drafter_dim,
        block_size=base_config.block_size,
        markov_rank=int(markov_rank),
    )
    markov = MarkovHead(config)
    confidence = ConfidenceHead(config)
    drafter.eval()
    for p in drafter.parameters():
        p.requires_grad_(False)

    markov.train()
    opt = torch.optim.AdamW(markov.parameters(), lr=3e-3, weight_decay=1e-4)
    last_markov_loss = 0.0
    for _ in range(max(1, int(markov_steps))):
        idx = _batch_indices(len(tensors), batch_size, generator)
        context = tensors.context[idx]
        with torch.no_grad():
            base_logits = drafter(context)
        corrected = base_logits + markov.full_bias(tensors.prev_target[idx])
        loss = F.cross_entropy(corrected.reshape(-1, base_config.candidate_size), tensors.future_local[idx].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(markov.parameters(), 1.0)
        opt.step()
        last_markov_loss = float(loss.detach())
    markov.eval()
    for p in markov.parameters():
        p.requires_grad_(False)

    position_correct = torch.zeros(base_config.block_size, dtype=torch.float64)
    position_total = 0
    positive_survival = 0.0
    total_survival = 0
    with torch.no_grad():
        for start in range(0, len(tensors), 128):
            sl = slice(start, min(start + 128, len(tensors)))
            hidden = drafter.encode(tensors.context[sl])
            base = drafter.head(hidden)
            corrected = base + markov.full_bias(tensors.prev_target[sl])
            ok = corrected.argmax(dim=-1).eq(tensors.future_local[sl])
            position_correct += ok.double().sum(dim=0)
            position_total += int(ok.shape[0])
            surv = ok.long().cumprod(dim=1).float()
            positive_survival += float(surv.sum())
            total_survival += int(surv.numel())

    confidence.train()
    opt = torch.optim.AdamW(confidence.parameters(), lr=3e-3, weight_decay=1e-4)
    last_confidence_loss = 0.0
    for _ in range(max(1, int(confidence_steps))):
        idx = _batch_indices(len(tensors), batch_size, generator)
        context = tensors.context[idx]
        with torch.no_grad():
            hidden = drafter.encode(context)
            base = drafter.head(hidden)
            prev_state = markov.prev(tensors.prev_target[idx])
            corrected = base + markov.full_bias(tensors.prev_target[idx])
            ok = corrected.argmax(dim=-1).eq(tensors.future_local[idx])
            survival = ok.long().cumprod(dim=1).float()
        logits = confidence(hidden, prev_state)
        loss = F.binary_cross_entropy_with_logits(logits, survival)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(confidence.parameters(), 1.0)
        opt.step()
        last_confidence_loss = float(loss.detach())
    confidence.eval()

    metrics = {
        "training_examples": len(tensors),
        "markov_rank": int(markov_rank),
        "markov_steps": int(markov_steps),
        "confidence_steps": int(confidence_steps),
        "batch_size": int(batch_size),
        "final_markov_loss": last_markov_loss,
        "final_confidence_loss": last_confidence_loss,
        "teacher_conditioned_position_accuracy": [float(x / max(position_total, 1)) for x in position_correct.tolist()],
        "prefix_survival_positive_rate": float(positive_survival / max(total_survival, 1)),
        "markov_parameter_count": sum(p.numel() for p in markov.parameters()),
        "confidence_parameter_count": sum(p.numel() for p in confidence.parameters()),
        "dflash_backbone_frozen": True,
    }
    return markov, confidence, metrics


def save_dspark_lite(path: str | Path, config: DSparkLiteConfig, markov: MarkovHead, confidence: ConfidenceHead, metadata: dict) -> None:
    payload = {
        "format_version": 1,
        "config": asdict(config),
        "markov": markov.state_dict(),
        "confidence": confidence.state_dict(),
        "metadata": metadata,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_dspark_lite(path: str | Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = DSparkLiteConfig(**payload["config"])
    markov = MarkovHead(config)
    confidence = ConfidenceHead(config)
    markov.load_state_dict(payload["markov"])
    confidence.load_state_dict(payload["confidence"])
    markov.eval(); confidence.eval()
    return config, markov, confidence, payload.get("metadata", {})


class LfmDSparkRuntime(LfmReferenceRuntime):
    """Existing LFM DFlash runtime plus CPU-cheap DSpark-Lite heads."""

    def __init__(self, aux_path: str | Path, dspark_path: str | Path, **kwargs):
        super().__init__(aux_path, **kwargs)
        config, markov, confidence, metadata = load_dspark_lite(dspark_path)
        if config.model_id != self.model_id:
            raise ValueError(f"DSpark target mismatch: {config.model_id} != {self.model_id}")
        if config.candidate_size != self.candidate_size:
            raise ValueError("DSpark candidate vocabulary size mismatch")
        self.dspark_config = config
        self.dspark_metadata = metadata
        self.markov = markov.eval()
        self.confidence = confidence.eval()
        self.markov_prev = markov.prev.weight.detach().float().cpu().numpy()
        self.markov_next = markov.next.weight.detach().float().cpu().numpy()
        self.markov_position_scale = markov.position_scale.detach().float().cpu().numpy()
        self.markov_scale = float(markov.scale)
        self.confidence_w = confidence.proj.weight.detach().float().cpu().numpy()[0]
        self.confidence_b = float(confidence.proj.bias.detach().float().cpu().item())
        self.dspark_parameter_count = sum(p.numel() for p in markov.parameters()) + sum(p.numel() for p in confidence.parameters())

    def dspark_select_path(
        self,
        draft_hidden: np.ndarray,
        draft_logits: np.ndarray,
        prev_token: int,
        *,
        top_k: int = 8,
        markov_weight: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        top_local, top_ids, top_vals = self._top_k(draft_logits, top_k)
        block = int(top_ids.shape[0])
        chosen = np.empty(block, dtype=np.int64)
        confidences = np.empty(block, dtype=np.float32)
        prev = int(prev_token)
        score_ops = 0
        for pos in range(block):
            prev_state = self.markov_prev[prev]
            local = top_local[pos]
            ids = top_ids[pos]
            correction = (self.markov_next[local] * prev_state[None, :]).sum(axis=-1)
            correction = correction * self.markov_scale * float(self.markov_position_scale[pos])
            scores = top_vals[pos] + float(markov_weight) * correction.astype(np.float32)
            best = int(np.argmax(scores))
            token = int(ids[best])
            chosen[pos] = token
            x = np.concatenate([np.asarray(draft_hidden[pos], dtype=np.float32), prev_state.astype(np.float32, copy=False)])
            logit = float(np.dot(self.confidence_w, x) + self.confidence_b)
            if logit >= 0:
                z = math.exp(-logit); conf = 1.0 / (1.0 + z)
            else:
                z = math.exp(logit); conf = z / (1.0 + z)
            confidences[pos] = float(conf)
            prev = token
            score_ops += int(ids.size)
        return chosen, confidences, score_ops


def confidence_verify_length(confidences: np.ndarray, survival_floor: float, limit: int) -> int:
    n = min(int(limit), int(np.asarray(confidences).size))
    if n <= 0:
        return 0
    if float(survival_floor) <= 0:
        return n
    survival = 1.0
    keep = 0
    for conf in np.asarray(confidences[:n], dtype=np.float64):
        survival *= max(1e-6, min(1.0, float(conf)))
        if survival < float(survival_floor):
            break
        keep += 1
    return max(1, keep)


def prepare_dspark(
    *,
    aux_path: str | Path,
    output_path: str | Path,
    seeds_path: str | Path,
    max_seed_count: int = 40,
    generation_tokens: int = 32,
    markov_rank: int = 16,
    markov_steps: int = 180,
    confidence_steps: int = 100,
    batch_size: int = 64,
    cpu_threads: int = 2,
    seed: int = 17,
) -> dict:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(cpu_threads)))
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass

    base_config, candidate_ids, drafter, _, _, _, base_metadata = load_aux_bundle(aux_path)
    tokenizer = AutoTokenizer.from_pretrained(base_config.model_id)
    model = AutoModelForCausalLM.from_pretrained(base_config.model_id, dtype=torch.float32)
    model.eval(); model.to("cpu")
    seeds = _read_list(seeds_path, "seeds")[: max(1, int(max_seed_count))]
    sequences, prompt_lengths, _, _ = generate_distillation_sequences(
        model, tokenizer, seeds, int(generation_tokens), top_candidate_k=8,
    )
    embedding = model.get_input_embeddings().weight.detach().float().cpu()
    tensors = make_training_tensors(
        sequences,
        embedding,
        [int(x) for x in candidate_ids.tolist()],
        training_start_indices=prompt_lengths,
        block_size=base_config.block_size,
        context_tokens=base_config.context_tokens,
    )
    del embedding, model
    gc.collect()

    markov, confidence, train_metrics = train_dspark_lite(
        base_config,
        candidate_ids,
        tensors,
        drafter,
        markov_rank=int(markov_rank),
        markov_steps=int(markov_steps),
        confidence_steps=int(confidence_steps),
        batch_size=int(batch_size),
        seed=int(seed),
    )
    ds_config = DSparkLiteConfig(
        model_id=base_config.model_id,
        target_vocab_size=base_config.target_vocab_size,
        candidate_size=base_config.candidate_size,
        drafter_dim=base_config.drafter_dim,
        block_size=base_config.block_size,
        markov_rank=int(markov_rank),
    )
    metadata = {
        "algorithm": "DSpark-Lite",
        "mechanism": "Frozen DFlash backbone + low-rank previous-token Markov correction + scalar prefix-survival confidence head",
        "inference_candidate_rescoring": "top-k only",
        "model_id": base_config.model_id,
        "base_aux_training": base_metadata.get("training", {}),
        "distillation_seed_count": len(seeds),
        "generation_tokens_per_seed": int(generation_tokens),
        "training": train_metrics,
        "target_weights_redistributed": False,
    }
    save_dspark_lite(output_path, ds_config, markov, confidence, metadata)
    print(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DSpark-Lite heads on the frozen LFM DFlash backbone")
    parser.add_argument("--aux", default="lfm-artifacts/lfm_aux.pt")
    parser.add_argument("--output", default="lfm-artifacts/lfm_dspark.pt")
    parser.add_argument("--seeds", default="real_benchmarks/train_seeds.json")
    parser.add_argument("--max-seed-count", type=int, default=40)
    parser.add_argument("--generation-tokens", type=int, default=32)
    parser.add_argument("--markov-rank", type=int, default=16)
    parser.add_argument("--markov-steps", type=int, default=180)
    parser.add_argument("--confidence-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=int(os.getenv("CPU_THREADS", "2")))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    prepare_dspark(
        aux_path=args.aux,
        output_path=args.output,
        seeds_path=args.seeds,
        max_seed_count=args.max_seed_count,
        generation_tokens=args.generation_tokens,
        markov_rank=args.markov_rank,
        markov_steps=args.markov_steps,
        confidence_steps=args.confidence_steps,
        batch_size=args.batch_size,
        cpu_threads=args.cpu_threads,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
