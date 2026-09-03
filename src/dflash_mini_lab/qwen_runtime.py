from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .qwen_aux import load_bundle


@dataclass
class QwenDecodeStats:
    method: str
    new_tokens: int
    wall_seconds: float
    prefill_seconds: float
    target_forward_passes: int
    target_input_tokens: int
    draft_forward_passes: int
    anchor_tokens: int
    accepted_draft_tokens: int
    proposed_draft_tokens: int
    mean_verify_drafts: float
    v7_margin_threshold: float | None = None

    @property
    def tokens_per_second(self) -> float:
        return self.new_tokens / max(self.wall_seconds, 1e-12)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_draft_tokens / max(self.proposed_draft_tokens, 1)

    @property
    def tokens_per_target_pass(self) -> float:
        return self.new_tokens / max(self.target_forward_passes, 1)

    def to_dict(self) -> dict:
        result = asdict(self)
        result.update(
            tokens_per_second=self.tokens_per_second,
            acceptance_rate=self.acceptance_rate,
            tokens_per_target_pass=self.tokens_per_target_pass,
        )
        return result


class QwenDFlashRuntime:
    def __init__(
        self,
        aux_path: str,
        *,
        model_id: str | None = None,
        cpu_threads: int | None = None,
        dtype: str = "float32",
    ):
        threads = int(cpu_threads or os.getenv("CPU_THREADS", "2"))
        torch.set_num_threads(max(1, threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        config, candidate_ids, drafter, metadata = load_bundle(aux_path)
        self.config = config
        self.metadata = metadata
        self.candidate_ids_t = candidate_ids.long().cpu()
        self.candidate_ids = self.candidate_ids_t.numpy().astype(np.int64, copy=False)
        self.candidate_set = set(int(x) for x in self.candidate_ids.tolist())
        self.drafter = drafter.eval()
        self.model_id = model_id or config.model_id
        torch_dtype = torch.float32 if dtype == "float32" else torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.target = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch_dtype)
        self.target.eval()
        self.target.to("cpu")
        self.embedding = self.target.get_input_embeddings().weight.detach()
        self.target_parameter_count = sum(parameter.numel() for parameter in self.target.parameters())
        self.aux_parameter_count = sum(parameter.numel() for parameter in self.drafter.parameters())

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(self.tokenizer.encode(text, add_special_tokens=True), dtype=np.int64)

    def decode_text(self, ids: list[int] | np.ndarray) -> str:
        return self.tokenizer.decode([int(x) for x in ids], skip_special_tokens=True)

    def _raw_hidden(self, outputs) -> torch.Tensor:
        return torch.cat(
            [outputs.hidden_states[layer_id + 1][0] for layer_id in self.config.target_layer_ids],
            dim=-1,
        ).detach().float().cpu()

    @torch.inference_mode()
    def _prefill(self, prompt_ids: np.ndarray, *, need_hidden: bool):
        cache = DynamicCache(config=self.target.config)
        ids = torch.from_numpy(np.asarray(prompt_ids, dtype=np.int64)).long().unsqueeze(0)
        start = time.perf_counter_ns()
        outputs = self.target(
            input_ids=ids,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=need_hidden,
            return_dict=True,
            cache_position=torch.arange(ids.shape[-1], dtype=torch.long),
        )
        seconds = (time.perf_counter_ns() - start) / 1e9
        anchor = int(torch.argmax(outputs.logits[0, -1].float()).item())
        memory = self._raw_hidden(outputs) if need_hidden else None
        return outputs.past_key_values, anchor, memory, seconds

    @torch.inference_mode()
    def _target_step(self, cache, input_tokens: list[int], *, need_hidden: bool):
        ids = torch.tensor([input_tokens], dtype=torch.long)
        base = int(cache.get_seq_length())
        outputs = self.target(
            input_ids=ids,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=need_hidden,
            return_dict=True,
            cache_position=torch.arange(base, base + len(input_tokens), dtype=torch.long),
        )
        return outputs

    @torch.inference_mode()
    def draft_logits(self, raw_memory: torch.Tensor, anchor_token: int) -> np.ndarray:
        keep = raw_memory[-self.config.memory_tokens :]
        pad_count = self.config.memory_tokens - int(keep.shape[0])
        if pad_count:
            pad = torch.zeros(pad_count, self.config.raw_memory_dim, dtype=torch.float32)
            keep = torch.cat([pad, keep], dim=0)
        mask = torch.zeros(self.config.memory_tokens, dtype=torch.bool)
        if pad_count:
            mask[:pad_count] = True
        anchor_embedding = self.embedding[int(anchor_token)].detach().float().cpu()
        logits = self.drafter(
            keep.unsqueeze(0),
            mask.unsqueeze(0),
            anchor_embedding.unsqueeze(0),
        )[0]
        return logits.float().cpu().numpy()

    @staticmethod
    def choose_v7_verify_count(draft_logits: np.ndarray, maximum: int, threshold: float) -> int:
        maximum = min(int(maximum), int(draft_logits.shape[0]))
        if maximum <= 0:
            return 0
        if float(threshold) <= 0.0:
            return maximum
        logits = np.asarray(draft_logits[:maximum], dtype=np.float32)
        if logits.shape[-1] < 2:
            return maximum
        top2 = np.partition(logits, -2, axis=-1)[:, -2:]
        margins = np.max(top2, axis=-1) - np.min(top2, axis=-1)
        count = 0
        for margin in margins.tolist():
            if float(margin) < float(threshold):
                break
            count += 1
        return count

    def normal_cached_decode(self, prompt_ids: np.ndarray, max_new_tokens: int):
        cache, anchor, _, prefill_seconds = self._prefill(prompt_ids, need_hidden=False)
        generated: list[int] = []
        calls = 0
        target_input_tokens = 0
        start = time.perf_counter_ns()
        while len(generated) < int(max_new_tokens):
            generated.append(int(anchor))
            if len(generated) >= int(max_new_tokens):
                break
            outputs = self._target_step(cache, [int(anchor)], need_hidden=False)
            cache = outputs.past_key_values
            calls += 1
            target_input_tokens += 1
            anchor = int(torch.argmax(outputs.logits[0, -1].float()).item())
        seconds = (time.perf_counter_ns() - start) / 1e9
        return np.asarray(generated, dtype=np.int64), QwenDecodeStats(
            method="normal_cached",
            new_tokens=int(max_new_tokens),
            wall_seconds=seconds,
            prefill_seconds=prefill_seconds,
            target_forward_passes=calls,
            target_input_tokens=target_input_tokens,
            draft_forward_passes=0,
            anchor_tokens=int(max_new_tokens),
            accepted_draft_tokens=0,
            proposed_draft_tokens=0,
            mean_verify_drafts=0.0,
        )

    def dflash_decode(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int,
        *,
        method: str = "dflash_qwen",
        v7_margin_threshold: float | None = None,
    ):
        cache, anchor, raw_memory, prefill_seconds = self._prefill(prompt_ids, need_hidden=True)
        assert raw_memory is not None
        generated: list[int] = []
        calls = 0
        target_input_tokens = 0
        draft_calls = 0
        accepted_total = 0
        proposed_total = 0
        verify_draft_counts: list[int] = []
        start = time.perf_counter_ns()

        while len(generated) < int(max_new_tokens):
            # The anchor is the target verifier's already-known greedy bonus
            # token. It is exact by construction and has not yet entered cache.
            generated.append(int(anchor))
            if len(generated) >= int(max_new_tokens):
                break
            remaining = int(max_new_tokens) - len(generated)
            draft_logits = self.draft_logits(raw_memory, int(anchor))
            draft_calls += 1
            maximum = min(self.config.speculative_tokens, remaining)
            if method == "dflash7_act":
                verify_count = self.choose_v7_verify_count(
                    draft_logits,
                    maximum,
                    float(v7_margin_threshold or 0.0),
                )
            else:
                verify_count = maximum
            verify_draft_counts.append(int(verify_count))
            local = np.argmax(draft_logits[:verify_count], axis=-1).astype(np.int64)
            proposal = self.candidate_ids[local] if verify_count else np.empty(0, dtype=np.int64)
            proposed_total += int(verify_count)

            base_len = int(cache.get_seq_length())
            verifier_input = [int(anchor)] + [int(x) for x in proposal.tolist()]
            outputs = self._target_step(cache, verifier_input, need_hidden=True)
            cache = outputs.past_key_values
            calls += 1
            target_input_tokens += len(verifier_input)

            if verify_count:
                target_next = torch.argmax(outputs.logits[0, :verify_count].float(), dim=-1).cpu().numpy().astype(np.int64)
                mismatch = np.flatnonzero(proposal != target_next)
                accepted = verify_count if mismatch.size == 0 else int(mismatch[0])
            else:
                accepted = 0
                mismatch = np.empty(0, dtype=np.int64)
            accepted_total += int(accepted)
            if accepted:
                generated.extend(int(x) for x in proposal[:accepted].tolist())

            keep_processed = 1 + int(accepted)
            cache.crop(base_len + keep_processed)
            current_hidden = self._raw_hidden(outputs)
            raw_memory = torch.cat([raw_memory, current_hidden[:keep_processed]], dim=0)

            if accepted < verify_count:
                # logits[accepted] predicts the first rejected position after
                # the known anchor plus accepted speculative prefix.
                anchor = int(torch.argmax(outputs.logits[0, accepted].float()).item())
            else:
                # The final verifier logit is a free bonus token and becomes
                # the next known anchor without an extra target forward.
                anchor = int(torch.argmax(outputs.logits[0, verify_count].float()).item())

        generated = generated[: int(max_new_tokens)]
        seconds = (time.perf_counter_ns() - start) / 1e9
        return np.asarray(generated, dtype=np.int64), QwenDecodeStats(
            method=method,
            new_tokens=int(max_new_tokens),
            wall_seconds=seconds,
            prefill_seconds=prefill_seconds,
            target_forward_passes=calls,
            target_input_tokens=target_input_tokens,
            draft_forward_passes=draft_calls,
            anchor_tokens=int(max_new_tokens) - accepted_total,
            accepted_draft_tokens=accepted_total,
            proposed_draft_tokens=proposed_total,
            mean_verify_drafts=float(np.mean(verify_draft_counts)) if verify_draft_counts else 0.0,
            v7_margin_threshold=float(v7_margin_threshold) if method == "dflash7_act" else None,
        )
