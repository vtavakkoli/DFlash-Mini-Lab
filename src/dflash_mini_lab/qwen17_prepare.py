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

from .qwen17_aux import Qwen17Config, build_training_tensors, save_bundle, train_bundle


DEFAULT_MODEL_ID = "Qwen/Qwen3-1.7B-Base"


def _read_list(path: str | Path, key: str) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path} must contain a non-empty '{key}' list")
    return [str(value) for value in values]


def _selected_hidden(outputs, layer_ids: tuple[int, ...]) -> torch.Tensor:
    return torch.cat([outputs.hidden_states[layer_id + 1] for layer_id in layer_ids], dim=-1)


def generate_teacher_trajectories_batched(
    model,
    tokenizer,
    seeds: list[str],
    *,
    layer_ids: tuple[int, ...],
    generation_tokens: int,
    top_candidate_k: int,
    teacher_batch_size: int,
) -> tuple[list[dict], Counter[int], set[int], list[dict]]:
    trajectories: list[dict] = []
    candidate_counts: Counter[int] = Counter()
    required_tokens: set[int] = set()
    records: list[dict] = []
    original_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for batch_start in range(0, len(seeds), max(1, int(teacher_batch_size))):
        batch_seeds = seeds[batch_start : batch_start + max(1, int(teacher_batch_size))]
        encoded = tokenizer(batch_seeds, return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = encoded["input_ids"].to("cpu")
        attention_mask = encoded["attention_mask"].to("cpu")
        prompt_lens = attention_mask.sum(dim=-1).long()
        batch, padded_prompt = input_ids.shape
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        cache = DynamicCache(config=model.config)
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                cache_position=torch.arange(padded_prompt, dtype=torch.long),
            )
        cache = outputs.past_key_values
        selected = _selected_hidden(outputs, layer_ids).detach().to(torch.float16).cpu()
        token_lists: list[list[int]] = []
        hidden_lists: list[list[torch.Tensor]] = []
        for row in range(batch):
            plen = int(prompt_lens[row].item())
            token_lists.append([int(x) for x in input_ids[row, -plen:].tolist()])
            hidden_lists.append([selected[row, -plen:, :].contiguous()])
        next_logits = outputs.logits[:, -1, :].float().cpu()
        running_mask = attention_mask.clone()

        for step in range(int(generation_tokens)):
            k = min(int(top_candidate_k), int(next_logits.shape[-1]))
            top = torch.topk(next_logits, k=k, dim=-1)
            next_tokens = torch.argmax(next_logits, dim=-1).long()
            for row in range(batch):
                candidate_counts.update(int(x) for x in top.indices[row].tolist())
                token = int(next_tokens[row].item())
                required_tokens.add(token)
                token_lists[row].append(token)

            base = int(cache.get_seq_length())
            step_ids = next_tokens.unsqueeze(1)
            running_mask = torch.cat([running_mask, torch.ones(batch, 1, dtype=running_mask.dtype)], dim=1)
            step_position_ids = (prompt_lens + step).unsqueeze(1)
            with torch.inference_mode():
                outputs = model(
                    input_ids=step_ids,
                    attention_mask=running_mask,
                    position_ids=step_position_ids,
                    past_key_values=cache,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                    cache_position=torch.arange(base, base + 1, dtype=torch.long),
                )
            cache = outputs.past_key_values
            selected_step = _selected_hidden(outputs, layer_ids).detach().to(torch.float16).cpu()
            for row in range(batch):
                hidden_lists[row].append(selected_step[row, :, :].contiguous())
            next_logits = outputs.logits[:, -1, :].float().cpu()

        for row, seed_text in enumerate(batch_seeds):
            plen = int(prompt_lens[row].item())
            raw_hidden = torch.cat(hidden_lists[row], dim=0)
            token_ids = token_lists[row]
            trajectories.append({"token_ids": token_ids, "prompt_len": plen, "raw_hidden": raw_hidden})
            records.append({
                "seed": seed_text,
                "prompt_tokens": plen,
                "generated_tokens": int(generation_tokens),
                "generated_text": tokenizer.decode(token_ids[plen:], skip_special_tokens=True),
            })
        del outputs, cache, selected, next_logits
        gc.collect()

    tokenizer.padding_side = original_padding
    return trajectories, candidate_counts, required_tokens, records


def _build_candidate_ids(candidate_counts: Counter[int], required_tokens: set[int], candidate_limit: int) -> list[int]:
    required = sorted(int(x) for x in required_tokens)
    if len(required) > int(candidate_limit):
        raise RuntimeError("candidate limit is smaller than required teacher token set")
    ordered = [token for token, _ in candidate_counts.most_common() if token not in required_tokens]
    return required + ordered[: int(candidate_limit) - len(required)]


def prepare(
    *,
    model_id: str,
    seeds_path: str | Path,
    output_dir: str | Path,
    max_seed_count: int,
    generation_tokens: int,
    teacher_batch_size: int,
    top_candidate_k: int,
    candidate_limit: int,
    layer_ids: tuple[int, ...],
    memory_tokens: int,
    block_size: int,
    draft_dim: int,
    draft_layers: int,
    drafter_steps: int,
    selector_steps: int,
    jump_steps: int,
    fused_steps: int,
    train_batch_size: int,
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
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    model.eval().to("cpu")
    target_parameter_count = sum(p.numel() for p in model.parameters())

    trajectories, candidate_counts, required, records = generate_teacher_trajectories_batched(
        model,
        tokenizer,
        seeds,
        layer_ids=layer_ids,
        generation_tokens=generation_tokens,
        top_candidate_k=top_candidate_k,
        teacher_batch_size=teacher_batch_size,
    )
    candidate_ids = _build_candidate_ids(candidate_counts, required, candidate_limit)
    hidden_size = int(model.config.hidden_size)
    config = Qwen17Config(
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
    candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long)
    candidate_embedding_weight = model.get_input_embeddings().weight.detach().cpu()[candidate_tensor].float().contiguous()
    candidate_head_weight = model.lm_head.weight.detach().cpu()[candidate_tensor].float().contiguous()
    tensors = build_training_tensors(trajectories, candidate_ids, candidate_embedding_weight, config)

    expected_examples = len(seeds) * max(0, int(generation_tokens) - config.speculative_tokens)
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
        "target_dtype": "bfloat16",
        "seed": int(seed),
        "distillation_seed_count": len(seeds),
        "generation_tokens_per_seed": int(generation_tokens),
        "teacher_batch_size": int(teacher_batch_size),
        "expected_sliding_examples": int(expected_examples),
        "constructed_training_examples": int(len(tensors)),
        "reference_qwen06_examples": 312,
        "training_scale_vs_qwen06": float(len(tensors) / 312.0),
        "top_candidate_k": int(top_candidate_k),
        "candidate_limit": int(candidate_limit),
        "candidate_size": len(candidate_ids),
        "memory_tokens": int(memory_tokens),
        "block_size": int(block_size),
        "draft_dim": int(draft_dim),
        "draft_layers": int(draft_layers),
        "methods": [
            "normal_cached", "dflash", "dflash2", "dflash3_mobs", "dflash4_jump_mobs",
            "dflash5_fused_jump_mobs", "dflash6_boltzmann", "dflash6_bmobs", "dflash7_act",
        ],
        "conditioning": "selected frozen Qwen hidden layers -> learned fusion -> bidirectional block decoder cross-memory K/V",
        "selector": "context-gated low-rank predecessor model trained on teacher transitions",
        "jump": "separate sparse +2/+4 candidate predictor for DFlash4; fused residual over existing draft slots for DFlash5",
        "anchor_semantics": "known verifier bonus token; drafter predicts only tokens after anchor",
        "target_head": "frozen target LM-head rows for retained candidate vocabulary",
        "target_weights_redistributed": False,
    }

    del model, trajectories, candidate_embedding_weight
    gc.collect()
    drafter, selector, jump, fused, train_metrics = train_bundle(
        config,
        candidate_head_weight,
        tensors,
        drafter_steps=drafter_steps,
        selector_steps=selector_steps,
        jump_steps=jump_steps,
        fused_steps=fused_steps,
        batch_size=train_batch_size,
        seed=seed,
    )
    metadata["training"] = train_metrics
    save_bundle(output / "qwen17_all_methods.pt", config, candidate_ids, drafter, selector, jump, fused, metadata)
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "distillation.json").write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all DFlash-lab auxiliaries for Qwen3-1.7B-Base")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--seeds", default="qwen_benchmarks/train_seeds.json")
    parser.add_argument("--output-dir", default="qwen17-artifacts")
    parser.add_argument("--max-seed-count", type=int, default=32)
    parser.add_argument("--generation-tokens", type=int, default=100)
    parser.add_argument("--teacher-batch-size", type=int, default=4)
    parser.add_argument("--top-candidate-k", type=int, default=32)
    parser.add_argument("--candidate-limit", type=int, default=8192)
    parser.add_argument("--layer-ids", type=int, nargs="+", default=[4, 14, 27])
    parser.add_argument("--memory-tokens", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=6)
    parser.add_argument("--draft-dim", type=int, default=256)
    parser.add_argument("--draft-layers", type=int, default=2)
    parser.add_argument("--drafter-steps", type=int, default=320)
    parser.add_argument("--selector-steps", type=int, default=160)
    parser.add_argument("--jump-steps", type=int, default=140)
    parser.add_argument("--fused-steps", type=int, default=140)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=int(os.getenv("CPU_THREADS", "2")))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    prepare(
        model_id=args.model_id,
        seeds_path=args.seeds,
        output_dir=args.output_dir,
        max_seed_count=args.max_seed_count,
        generation_tokens=args.generation_tokens,
        teacher_batch_size=args.teacher_batch_size,
        top_candidate_k=args.top_candidate_k,
        candidate_limit=args.candidate_limit,
        layer_ids=tuple(args.layer_ids),
        memory_tokens=args.memory_tokens,
        block_size=args.block_size,
        draft_dim=args.draft_dim,
        draft_layers=args.draft_layers,
        drafter_steps=args.drafter_steps,
        selector_steps=args.selector_steps,
        jump_steps=args.jump_steps,
        fused_steps=args.fused_steps,
        train_batch_size=args.train_batch_size,
        cpu_threads=args.cpu_threads,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
