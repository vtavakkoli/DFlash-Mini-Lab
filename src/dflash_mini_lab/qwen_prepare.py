from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .qwen_aux import QwenDFlashConfig, build_training_tensors, save_bundle, train_drafter


DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B-Base"


def _read_list(path: str | Path, key: str) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path} must contain a non-empty '{key}' list")
    return [str(value) for value in values]


def _selected_hidden(outputs, layer_ids: tuple[int, ...]) -> torch.Tensor:
    # hidden_states[0] is the embedding output. Layer N therefore lives at N+1.
    return torch.cat([outputs.hidden_states[layer_id + 1][0] for layer_id in layer_ids], dim=-1)


def generate_teacher_trajectories(
    model,
    tokenizer,
    seeds: list[str],
    *,
    layer_ids: tuple[int, ...],
    generation_tokens: int,
    top_candidate_k: int,
) -> tuple[list[dict], Counter[int], set[int], list[dict]]:
    trajectories: list[dict] = []
    candidate_counts: Counter[int] = Counter()
    required_tokens: set[int] = set()
    records: list[dict] = []

    for seed in seeds:
        encoded = tokenizer(seed, return_tensors="pt", add_special_tokens=True)
        prompt_ids = encoded["input_ids"].to("cpu")
        prompt_len = int(prompt_ids.shape[-1])
        cache = DynamicCache(config=model.config)
        with torch.inference_mode():
            outputs = model(
                input_ids=prompt_ids,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                cache_position=torch.arange(prompt_len, dtype=torch.long),
            )
        cache = outputs.past_key_values
        token_ids = [int(x) for x in prompt_ids[0].tolist()]
        hidden_chunks = [_selected_hidden(outputs, layer_ids).detach().to(torch.float16).cpu()]
        next_logits = outputs.logits[0, -1].float()

        for _ in range(int(generation_tokens)):
            top = torch.topk(next_logits, k=min(int(top_candidate_k), int(next_logits.numel())))
            candidate_counts.update(int(x) for x in top.indices.tolist())
            next_token = int(torch.argmax(next_logits).item())
            required_tokens.add(next_token)
            token_ids.append(next_token)

            past_len = int(cache.get_seq_length())
            step_ids = torch.tensor([[next_token]], dtype=torch.long)
            with torch.inference_mode():
                outputs = model(
                    input_ids=step_ids,
                    past_key_values=cache,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                    cache_position=torch.arange(past_len, past_len + 1, dtype=torch.long),
                )
            cache = outputs.past_key_values
            hidden_chunks.append(_selected_hidden(outputs, layer_ids).detach().to(torch.float16).cpu())
            next_logits = outputs.logits[0, -1].float()

        raw_hidden = torch.cat(hidden_chunks, dim=0)
        trajectories.append({
            "token_ids": token_ids,
            "prompt_len": prompt_len,
            "raw_hidden": raw_hidden,
        })
        records.append({
            "seed": seed,
            "prompt_tokens": prompt_len,
            "generated_tokens": int(generation_tokens),
            "generated_text": tokenizer.decode(token_ids[prompt_len:], skip_special_tokens=True),
        })

    return trajectories, candidate_counts, required_tokens, records


def _build_candidate_ids(
    candidate_counts: Counter[int],
    required_tokens: set[int],
    candidate_limit: int,
) -> list[int]:
    required = sorted(int(x) for x in required_tokens)
    if len(required) > int(candidate_limit):
        raise RuntimeError("candidate limit is smaller than the required teacher token set")
    ordered = [token for token, _ in candidate_counts.most_common() if token not in required_tokens]
    room = int(candidate_limit) - len(required)
    return required + ordered[:room]


def prepare(
    *,
    model_id: str,
    seeds_path: str | Path,
    output_dir: str | Path,
    max_seed_count: int,
    generation_tokens: int,
    top_candidate_k: int,
    candidate_limit: int,
    layer_ids: tuple[int, ...],
    memory_tokens: int,
    block_size: int,
    draft_dim: int,
    draft_layers: int,
    train_steps: int,
    cpu_threads: int,
    seed: int,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(cpu_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    seeds = _read_list(seeds_path, "seeds")[: max(1, int(max_seed_count))]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    model.to("cpu")
    target_parameter_count = sum(parameter.numel() for parameter in model.parameters())

    trajectories, candidate_counts, required, records = generate_teacher_trajectories(
        model,
        tokenizer,
        seeds,
        layer_ids=layer_ids,
        generation_tokens=generation_tokens,
        top_candidate_k=top_candidate_k,
    )
    candidate_ids = _build_candidate_ids(candidate_counts, required, candidate_limit)
    hidden_size = int(model.config.hidden_size)
    config = QwenDFlashConfig(
        model_id=model_id,
        target_hidden_size=hidden_size,
        target_vocab_size=int(model.config.vocab_size),
        candidate_size=len(candidate_ids),
        target_layer_ids=tuple(int(x) for x in layer_ids),
        memory_tokens=int(memory_tokens),
        block_size=int(block_size),
        draft_dim=int(draft_dim),
        draft_heads=8,
        draft_layers=int(draft_layers),
    )
    embedding_weight = model.get_input_embeddings().weight.detach().float().cpu()
    candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long)
    candidate_head_weight = model.lm_head.weight.detach().float().cpu()[candidate_tensor].contiguous()
    tensors = build_training_tensors(trajectories, candidate_ids, embedding_weight, config)

    metadata = {
        "model_id": model_id,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "target_parameter_count": int(target_parameter_count),
        "target_hidden_size": hidden_size,
        "target_vocab_size": int(model.config.vocab_size),
        "target_num_hidden_layers": int(model.config.num_hidden_layers),
        "target_num_attention_heads": int(model.config.num_attention_heads),
        "target_num_key_value_heads": int(model.config.num_key_value_heads),
        "target_layer_ids": list(layer_ids),
        "seed": int(seed),
        "distillation_seed_count": len(seeds),
        "generation_tokens_per_seed": int(generation_tokens),
        "top_candidate_k": int(top_candidate_k),
        "candidate_limit": int(candidate_limit),
        "candidate_size": len(candidate_ids),
        "memory_tokens": int(memory_tokens),
        "block_size": int(block_size),
        "draft_dim": int(draft_dim),
        "draft_layers": int(draft_layers),
        "conditioning": "selected frozen Qwen hidden layers -> learned fusion -> bidirectional block decoder cross-memory K/V",
        "anchor_semantics": "known verifier bonus token; mask slots predict only tokens after the anchor",
        "target_head": "frozen target LM-head rows for retained candidate vocabulary",
        "target_weights_redistributed": False,
    }

    # Free the 0.6B verifier before auxiliary optimization. All required target
    # hidden memories, embeddings and retained head rows are now cached locally.
    del model, trajectories
    gc.collect()

    drafter, train_metrics = train_drafter(
        config,
        candidate_head_weight,
        tensors,
        steps=train_steps,
        batch_size=16,
        seed=seed,
    )
    metadata["training"] = train_metrics
    save_bundle(output / "qwen_dflash.pt", config, candidate_ids, drafter, metadata)
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "distillation.json").write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a hidden-fusion DFlash auxiliary for Qwen3-0.6B-Base")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--seeds", default="qwen_benchmarks/train_seeds.json")
    parser.add_argument("--output-dir", default="qwen-artifacts")
    parser.add_argument("--max-seed-count", type=int, default=24)
    parser.add_argument("--generation-tokens", type=int, default=18)
    parser.add_argument("--top-candidate-k", type=int, default=32)
    parser.add_argument("--candidate-limit", type=int, default=8192)
    parser.add_argument("--layer-ids", type=int, nargs="+", default=[4, 14, 27])
    parser.add_argument("--memory-tokens", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=6)
    parser.add_argument("--draft-dim", type=int, default=256)
    parser.add_argument("--draft-layers", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=180)
    parser.add_argument("--cpu-threads", type=int, default=int(os.getenv("CPU_THREADS", "2")))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    prepare(
        model_id=args.model_id,
        seeds_path=args.seeds,
        output_dir=args.output_dir,
        max_seed_count=args.max_seed_count,
        generation_tokens=args.generation_tokens,
        top_candidate_k=args.top_candidate_k,
        candidate_limit=args.candidate_limit,
        layer_ids=tuple(args.layer_ids),
        memory_tokens=args.memory_tokens,
        block_size=args.block_size,
        draft_dim=args.draft_dim,
        draft_layers=args.draft_layers,
        train_steps=args.train_steps,
        cpu_threads=args.cpu_threads,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
