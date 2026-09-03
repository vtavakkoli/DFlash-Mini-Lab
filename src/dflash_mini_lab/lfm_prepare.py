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
from transformers import AutoModelForCausalLM, AutoTokenizer

from .lfm_aux import AuxConfig, BLOCK_SIZE, make_training_tensors, save_aux_bundle, train_auxiliary_models


DEFAULT_MODEL_ID = "LiquidAI/LFM2.5-350M-Base"


def _read_list(path: str | Path, key: str) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8")); values = payload[key]
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path} must contain a non-empty '{key}' list")
    return [str(x) for x in values]


def _candidate_ids(
    sequences: list[list[int]],
    teacher_alternatives: Counter[int],
    benchmark_prompts: list[str],
    tokenizer,
    limit: int,
) -> list[int]:
    # All greedy teacher-trajectory tokens are mandatory so distillation labels
    # never disappear from the compact output space. High-probability teacher
    # alternatives then fill the remaining capacity.
    required: set[int] = set()
    for seq in sequences:
        required.update(int(x) for x in seq[1:])
    for prompt in benchmark_prompts:
        required.update(int(x) for x in tokenizer.encode(prompt, add_special_tokens=True))
    for special in (tokenizer.pad_token_id, tokenizer.bos_token_id):
        if special is not None:
            required.discard(int(special)); teacher_alternatives.pop(int(special), None)
    if tokenizer.eos_token_id is not None:
        required.add(int(tokenizer.eos_token_id))
    ordered = [tok for tok, _ in teacher_alternatives.most_common() if tok not in required]
    keep = list(sorted(required)); room = max(0, int(limit) - len(keep)); keep.extend(ordered[:room])
    if not keep:
        raise RuntimeError("Candidate vocabulary is empty")
    return sorted(set(keep))


def generate_distillation_sequences(
    model,
    tokenizer,
    seeds: list[str],
    max_new_tokens: int,
    top_candidate_k: int,
) -> tuple[list[list[int]], list[int], list[dict], Counter[int]]:
    sequences: list[list[int]] = []; prompt_lengths: list[int] = []; records: list[dict] = []; alternatives: Counter[int] = Counter()
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for seed in seeds:
        encoded = tokenizer(seed, return_tensors="pt", add_special_tokens=True); input_ids = encoded["input_ids"].to(model.device); prompt_len = int(input_ids.shape[-1])
        attention_mask = encoded.get("attention_mask", None)
        if attention_mask is not None: attention_mask = attention_mask.to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                use_cache=True,
                pad_token_id=pad,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        ids = [int(x) for x in generated.sequences[0].cpu().tolist()]
        for score in generated.scores:
            k = min(int(top_candidate_k), int(score.shape[-1]))
            alternatives.update(int(x) for x in torch.topk(score[0], k=k).indices.cpu().tolist())
        sequences.append(ids); prompt_lengths.append(prompt_len)
        records.append({"seed": seed, "prompt_tokens": prompt_len, "generated_tokens": len(ids) - prompt_len, "total_tokens": len(ids), "text": tokenizer.decode(ids, skip_special_tokens=True)})
    return sequences, prompt_lengths, records, alternatives


def prepare(
    *,
    model_id: str,
    seeds_path: str | Path,
    benchmark_prompts_path: str | Path,
    output_dir: str | Path,
    max_seed_count: int = 24,
    generation_tokens: int = 20,
    top_candidate_k: int = 24,
    candidate_limit: int = 2048,
    drafter_steps: int = 180,
    selector_steps: int = 100,
    jump_steps: int = 80,
    fused_steps: int = 100,
    cpu_threads: int = 2,
    seed: int = 7,
) -> dict:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(max(1, int(cpu_threads)))
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    seeds = _read_list(seeds_path, "seeds")[: max(1, int(max_seed_count))]; benchmark_prompts = _read_list(benchmark_prompts_path, "prompts")
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32); model.eval(); model.to("cpu")
    sequences, prompt_lengths, records, alternatives = generate_distillation_sequences(model, tokenizer, seeds, generation_tokens, top_candidate_k)
    candidates = _candidate_ids(sequences, alternatives, benchmark_prompts, tokenizer, candidate_limit)

    hidden_size = int(model.config.hidden_size); vocab_size = int(model.config.vocab_size)
    config = AuxConfig(model_id=model_id, target_hidden_size=hidden_size, target_vocab_size=vocab_size, candidate_size=len(candidates))
    embedding_weight = model.get_input_embeddings().weight.detach().float().cpu()
    tensors = make_training_tensors(
        sequences,
        embedding_weight,
        candidates,
        training_start_indices=prompt_lengths,
        block_size=BLOCK_SIZE,
        context_tokens=config.context_tokens,
    )
    model_commit = getattr(model.config, "_commit_hash", None)
    model_config = {
        "architectures": list(getattr(model.config, "architectures", []) or []), "hidden_size": hidden_size, "vocab_size": vocab_size,
        "num_hidden_layers": int(getattr(model.config, "num_hidden_layers", 0)), "num_attention_heads": int(getattr(model.config, "num_attention_heads", 0)),
        "max_position_embeddings": int(getattr(model.config, "max_position_embeddings", 0)), "layer_types": list(getattr(model.config, "layer_types", []) or []),
    }
    del embedding_weight, model; gc.collect()

    drafter, selector, jump, fused, train_metrics = train_auxiliary_models(
        config, candidates, tensors, seed=seed, drafter_steps=drafter_steps, selector_steps=selector_steps, jump_steps=jump_steps, fused_steps=fused_steps,
    )
    metadata = {
        "model_id": model_id, "model_commit": model_commit, "seed": seed, "distillation_seed_count": len(seeds), "generation_tokens_per_seed": int(generation_tokens),
        "teacher_top_candidate_k": int(top_candidate_k), "candidate_limit": int(candidate_limit), "candidate_size": len(candidates), "teacher_alternative_unique_tokens": len(alternatives),
        "target_config": model_config, "training": train_metrics,
        "conditioning": f"last {config.context_tokens} frozen LFM input embeddings in order + mean frozen prefix embedding",
        "training_examples_scope": "only contexts whose future block is on the LFM greedy generated trajectory",
        "target_weights_redistributed": False,
    }
    save_aux_bundle(out / "lfm_aux.pt", config, candidates, drafter, selector, jump, fused, metadata)
    (out / "distillation.json").write_text(json.dumps({"records": records}, indent=2), encoding="utf-8"); (out / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2)); return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare compact DFlash auxiliaries for a frozen LFM2.5 target")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID); parser.add_argument("--seeds", default="real_benchmarks/train_seeds.json"); parser.add_argument("--benchmark-prompts", default="real_benchmarks/prompts.json"); parser.add_argument("--output-dir", default="lfm-artifacts")
    parser.add_argument("--max-seed-count", type=int, default=24); parser.add_argument("--generation-tokens", type=int, default=20); parser.add_argument("--top-candidate-k", type=int, default=24); parser.add_argument("--candidate-limit", type=int, default=2048)
    parser.add_argument("--drafter-steps", type=int, default=180); parser.add_argument("--selector-steps", type=int, default=100); parser.add_argument("--jump-steps", type=int, default=80); parser.add_argument("--fused-steps", type=int, default=100)
    parser.add_argument("--cpu-threads", type=int, default=int(os.getenv("CPU_THREADS", "2"))); parser.add_argument("--seed", type=int, default=7); args = parser.parse_args()
    prepare(model_id=args.model_id, seeds_path=args.seeds, benchmark_prompts_path=args.benchmark_prompts, output_dir=args.output_dir, max_seed_count=args.max_seed_count, generation_tokens=args.generation_tokens, top_candidate_k=args.top_candidate_k, candidate_limit=args.candidate_limit, drafter_steps=args.drafter_steps, selector_steps=args.selector_steps, jump_steps=args.jump_steps, fused_steps=args.fused_steps, cpu_threads=args.cpu_threads, seed=args.seed)


if __name__ == "__main__": main()
