from __future__ import annotations

from dataclasses import asdict, dataclass
import html
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from .lfm_runtime import LfmReferenceRuntime


METHODS = (
    "normal",
    "dflash",
    "dflash2",
    "dflash3_mobs",
    "dflash4_jump_mobs",
    "dflash5_fused_jump_mobs",
    "dflash6_boltzmann",
    "dflash6_bmobs",
)

LABELS = {
    "normal": "Normal",
    "dflash": "DFlash",
    "dflash2": "DFlash2",
    "dflash3_mobs": "DFlash3-MOBS",
    "dflash4_jump_mobs": "DFlash4-JUMP",
    "dflash5_fused_jump_mobs": "DFlash5-FUSED",
    "dflash6_boltzmann": "DFlash6-Boltzmann",
    "dflash6_bmobs": "DFlash6-BMOBS",
}


@dataclass
class RealDecodeStats:
    method: str
    new_tokens: int
    target_forward_passes: int
    draft_forward_passes: int
    accepted_draft_tokens: int
    proposed_draft_tokens: int
    wall_seconds: float
    target_seconds: float = 0.0
    context_seconds: float = 0.0
    draft_seconds: float = 0.0
    selection_seconds: float = 0.0
    selector_pair_scores: int = 0
    jump_forward_passes: int = 0
    jump_candidate_scores: int = 0
    fused_anchor_uses: int = 0
    boltzmann_candidate_scores: int = 0

    @property
    def tokens_per_second(self) -> float:
        return self.new_tokens / max(self.wall_seconds, 1e-12)

    @property
    def latency_ms(self) -> float:
        return self.wall_seconds * 1000.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_draft_tokens / max(self.proposed_draft_tokens, 1)

    @property
    def tokens_per_target_pass(self) -> float:
        return self.new_tokens / max(self.target_forward_passes, 1)

    @property
    def total_guidance_scores(self) -> int:
        return self.selector_pair_scores + self.jump_candidate_scores + self.boltzmann_candidate_scores

    def to_dict(self) -> dict:
        out = asdict(self)
        out.update(
            tokens_per_second=self.tokens_per_second,
            latency_ms=self.latency_ms,
            acceptance_rate=self.acceptance_rate,
            tokens_per_target_pass=self.tokens_per_target_pass,
            total_guidance_scores=self.total_guidance_scores,
        )
        return out


def normal_decode(runtime: LfmReferenceRuntime, input_ids: np.ndarray, max_new_tokens: int):
    seq = np.asarray(input_ids, dtype=np.int64).copy()
    calls = 0; target_seconds = 0.0
    t0 = time.perf_counter()
    for _ in range(max_new_tokens):
        t = time.perf_counter(); logits = runtime.target_logits(seq); target_seconds += time.perf_counter() - t
        calls += 1
        seq = np.append(seq, int(np.argmax(logits[-1])))
    wall = time.perf_counter() - t0
    return seq, RealDecodeStats("normal", max_new_tokens, calls, 0, 0, 0, wall, target_seconds=target_seconds)


def speculative_decode(
    runtime: LfmReferenceRuntime,
    input_ids: np.ndarray,
    max_new_tokens: int,
    method: str,
    *,
    top_k: int = 8,
    jump_weight: float = 0.5,
    fused_weight: float = 1.0,
    fused_min_margin: float = 0.0,
    boltzmann_temperature: float = 0.15,
    bmobs_temperature: float = 0.35,
):
    seq = np.asarray(input_ids, dtype=np.int64).copy(); start_len = int(seq.size)
    target_calls = draft_calls = accepted_total = proposed_total = selector_pair_scores = 0
    jump_calls = jump_candidate_scores = fused_anchor_uses = boltzmann_candidate_scores = 0
    target_seconds = context_seconds = draft_seconds = selection_seconds = 0.0
    wall0 = time.perf_counter()

    while int(seq.size) - start_len < max_new_tokens:
        remaining = max_new_tokens - (int(seq.size) - start_len)
        t = time.perf_counter(); context = runtime.context_features(seq); context_seconds += time.perf_counter() - t
        t = time.perf_counter()
        if method == "dflash5_fused_jump_mobs":
            draft_hidden, draft_logits = runtime.draft_hidden_and_logits(context)
        else:
            draft_hidden = None; draft_logits = runtime.draft_logits(context)
        draft_seconds += time.perf_counter() - t; draft_calls += 1

        t = time.perf_counter()
        if method == "dflash":
            proposal = runtime.proposal_argmax(draft_logits)
        elif method == "dflash2":
            proposal = runtime.dflash2_select_path(draft_logits, context, int(seq[-1]), top_k=top_k)
            k = min(int(top_k), runtime.candidate_size); block = int(draft_logits.shape[0])
            selector_pair_scores += k + max(0, block - 1) * k * k
        elif method == "dflash3_mobs":
            proposal, pairs = runtime.dflash3_mobs_select_path(draft_logits, context, int(seq[-1]), top_k=top_k, refine_passes=0)
            selector_pair_scores += int(pairs)
        elif method == "dflash4_jump_mobs":
            offsets, jump_logits = runtime.jump_logits(context); jump_calls += 1
            proposal, pairs, jumps = runtime.dflash4_jump_mobs_select_path(
                draft_logits, offsets, jump_logits, context, int(seq[-1]), top_k=top_k, jump_weight=jump_weight
            )
            selector_pair_scores += int(pairs); jump_candidate_scores += int(jumps)
        elif method == "dflash5_fused_jump_mobs":
            proposal, pairs, jumps, anchors = runtime.dflash5_fused_jump_mobs_select_path(
                draft_hidden, draft_logits, context, int(seq[-1]), top_k=top_k,
                fused_weight=fused_weight, min_margin=fused_min_margin,
            )
            selector_pair_scores += int(pairs); jump_candidate_scores += int(jumps); fused_anchor_uses += int(anchors)
        elif method == "dflash6_boltzmann":
            proposal, sampled = runtime.dflash6_boltzmann_select_path(
                draft_logits, context, int(seq[-1]), top_k=top_k, temperature=boltzmann_temperature
            )
            boltzmann_candidate_scores += int(sampled)
        elif method == "dflash6_bmobs":
            proposal, pairs, sampled = runtime.dflash6_bmobs_select_path(
                draft_logits, context, int(seq[-1]), top_k=top_k, temperature=bmobs_temperature
            )
            selector_pair_scores += int(pairs); boltzmann_candidate_scores += int(sampled)
        else:
            raise ValueError(method)
        selection_seconds += time.perf_counter() - t

        proposal = np.asarray(proposal, dtype=np.int64)[: min(int(np.asarray(proposal).size), remaining)]
        proposed_total += int(proposal.size)
        verify_input = np.concatenate([seq, proposal])
        t = time.perf_counter(); logits = runtime.target_logits(verify_input); target_seconds += time.perf_counter() - t
        target_calls += 1
        p, k = int(seq.size), int(proposal.size)
        verifier = np.argmax(logits[p - 1 : p - 1 + k], axis=-1).astype(np.int64)
        mismatch = np.flatnonzero(proposal != verifier)
        accepted = k if mismatch.size == 0 else int(mismatch[0])
        accepted_total += accepted
        if accepted:
            seq = np.concatenate([seq, proposal[:accepted]])
        if accepted < k and int(seq.size) - start_len < max_new_tokens:
            seq = np.append(seq, verifier[accepted])

    seq = seq[: start_len + max_new_tokens]
    wall = time.perf_counter() - wall0
    return seq, RealDecodeStats(
        method=method,
        new_tokens=max_new_tokens,
        target_forward_passes=target_calls,
        draft_forward_passes=draft_calls,
        accepted_draft_tokens=accepted_total,
        proposed_draft_tokens=proposed_total,
        wall_seconds=wall,
        target_seconds=target_seconds,
        context_seconds=context_seconds,
        draft_seconds=draft_seconds,
        selection_seconds=selection_seconds,
        selector_pair_scores=selector_pair_scores,
        jump_forward_passes=jump_calls,
        jump_candidate_scores=jump_candidate_scores,
        fused_anchor_uses=fused_anchor_uses,
        boltzmann_candidate_scores=boltzmann_candidate_scores,
    )


def _read_prompts(path: str | Path, limit: int | None = None) -> list[str]:
    prompts = [str(x) for x in json.loads(Path(path).read_text(encoding="utf-8"))["prompts"]]
    return prompts if limit is None else prompts[: int(limit)]


def _median(rows: list[dict], key: str) -> float:
    return float(statistics.median(float(r[key]) for r in rows))


def _mean(rows: list[dict], key: str) -> float:
    return float(statistics.fmean(float(r[key]) for r in rows))


def run_real_benchmark(
    *,
    aux_path: str | Path,
    prompts_path: str | Path,
    output_dir: str | Path,
    model_id: str | None = None,
    max_new_tokens: int = 8,
    repeats: int = 1,
    prompt_limit: int = 4,
    top_k: int = 8,
    jump_weight: float = 0.5,
    fused_weight: float = 1.0,
    fused_min_margin: float = 0.0,
    boltzmann_temperature: float = 0.15,
    bmobs_temperature: float = 0.35,
    cpu_threads: int = 2,
    dtype: str = "float32",
) -> dict:
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    runtime = LfmReferenceRuntime(aux_path, model_id=model_id, cpu_threads=cpu_threads, dtype=dtype)
    prompts = _read_prompts(prompts_path, prompt_limit)

    # One unmeasured target/drafter warmup avoids charging lazy initialization to
    # only the first benchmark method.
    warm_ids = runtime.encode(prompts[0])
    _ = runtime.target_logits(warm_ids)
    _ = runtime.draft_logits(runtime.context_features(warm_ids))

    runs: list[dict] = []; exactness: list[dict] = []; outputs: list[dict] = []
    candidate_set = set(int(x) for x in runtime.candidate_ids.tolist())
    for prompt_index, prompt in enumerate(prompts):
        ids = runtime.encode(prompt)
        normal_outputs: list[np.ndarray] = []
        for repeat in range(max(1, int(repeats))):
            normal, stats = normal_decode(runtime, ids, max_new_tokens)
            normal_outputs.append(normal)
            row = stats.to_dict(); row.update(prompt=prompt, prompt_index=prompt_index, repeat=repeat, exact_match=True); runs.append(row)
        reference = normal_outputs[0]
        continuation = reference[len(ids):]
        coverage = sum(int(tok) in candidate_set for tok in continuation) / max(1, len(continuation))
        first_outputs = {"normal": runtime.decode(reference)}
        exact_row = {"prompt": prompt, "candidate_coverage_of_normal_continuation": coverage}

        for method in METHODS[1:]:
            method_outputs = []
            for repeat in range(max(1, int(repeats))):
                output, stats = speculative_decode(
                    runtime, ids, max_new_tokens, method,
                    top_k=top_k, jump_weight=jump_weight,
                    fused_weight=fused_weight, fused_min_margin=fused_min_margin,
                    boltzmann_temperature=boltzmann_temperature, bmobs_temperature=bmobs_temperature,
                )
                method_outputs.append(output)
                exact = bool(np.array_equal(output, reference))
                row = stats.to_dict(); row.update(prompt=prompt, prompt_index=prompt_index, repeat=repeat, exact_match=exact); runs.append(row)
            exact_row[f"normal_equals_{method}"] = all(bool(np.array_equal(x, reference)) for x in method_outputs)
            first_outputs[method] = runtime.decode(method_outputs[0])
        exactness.append(exact_row)
        outputs.append({"prompt": prompt, "candidate_coverage": coverage, "texts": first_outputs})

    summary = {}
    for method in METHODS:
        rows = [r for r in runs if r["method"] == method]
        summary[method] = {
            "tokens_per_second_median": _median(rows, "tokens_per_second"),
            "latency_seconds_median": _median(rows, "wall_seconds"),
            "target_seconds_median": _median(rows, "target_seconds"),
            "draft_seconds_median": _median(rows, "draft_seconds"),
            "selection_seconds_median": _median(rows, "selection_seconds"),
            "mean_target_forward_passes": _mean(rows, "target_forward_passes"),
            "mean_draft_forward_passes": _mean(rows, "draft_forward_passes"),
            "mean_acceptance_rate": _mean(rows, "acceptance_rate"),
            "mean_tokens_per_target_pass": _mean(rows, "tokens_per_target_pass"),
            "mean_total_guidance_scores": _mean(rows, "total_guidance_scores"),
            "all_exact": all(bool(r["exact_match"]) for r in rows),
        }
    baseline = summary["normal"]["tokens_per_second_median"]
    for method in METHODS:
        summary[method]["speedup_vs_normal"] = summary[method]["tokens_per_second_median"] / max(baseline, 1e-12)

    payload = {
        "schema_version": 1,
        "benchmark_name": "DFlash Mini Lab - LFM2.5 real target benchmark",
        "generated_unix": int(time.time()),
        "model": {
            "id": runtime.model_id,
            "target_parameter_count": runtime.target_parameter_count,
            "target_hidden_size": runtime.config.target_hidden_size,
            "target_vocab_size": runtime.config.target_vocab_size,
            "candidate_size": runtime.candidate_size,
            "candidate_fraction_of_target_vocab": runtime.candidate_size / runtime.config.target_vocab_size,
            "aux_parameter_count": runtime.aux_parameter_count,
            "metadata": runtime.metadata,
        },
        "config": {
            "max_new_tokens": max_new_tokens,
            "repeats": repeats,
            "prompt_count": len(prompts),
            "block_size": runtime.block_size,
            "top_k": top_k,
            "jump_weight": jump_weight,
            "fused_weight": fused_weight,
            "fused_min_margin": fused_min_margin,
            "boltzmann_temperature": boltzmann_temperature,
            "bmobs_temperature": bmobs_temperature,
            "cpu_threads": cpu_threads,
            "dtype": dtype,
            "target_use_cache_for_timed_forward": False,
        },
        "method_scope": "Mechanism-level DFlash family with a real frozen LFM2.5-350M-Base target. Auxiliary conditioning uses frozen LFM input embeddings; it is not the upstream DFlash hidden-state fusion/training recipe.",
        "timing_note": "Target verification uses full-sequence PyTorch forwards with use_cache=False for controlled exact comparison. These are reference CPU timings, not optimized llama.cpp/vLLM serving numbers.",
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_threads": cpu_threads,
            "omp_num_threads": os.getenv("OMP_NUM_THREADS"),
        },
        "summary": summary,
        "mean_candidate_coverage": statistics.fmean(float(x["candidate_coverage_of_normal_continuation"]) for x in exactness),
        "exactness": exactness,
        "outputs": outputs,
        "runs": runs,
    }
    (out_dir / "benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    build_real_report(payload, out_dir / "report.html")
    return payload


def build_real_report(data: dict, output_path: str | Path) -> None:
    summary = data["summary"]
    rows = []
    for method in METHODS:
        s = summary[method]
        rows.append(
            "<tr>"
            f"<th>{html.escape(LABELS[method])}</th>"
            f"<td>{s['tokens_per_second_median']:.2f}</td>"
            f"<td>{s['speedup_vs_normal']:.2f}×</td>"
            f"<td>{s['latency_seconds_median']:.3f}s</td>"
            f"<td>{100*s['mean_acceptance_rate']:.1f}%</td>"
            f"<td>{s['mean_tokens_per_target_pass']:.2f}</td>"
            f"<td>{s['mean_target_forward_passes']:.1f}</td>"
            f"<td>{s['mean_total_guidance_scores']:.0f}</td>"
            f"<td>{'✓' if s['all_exact'] else '✗'}</td>"
            "</tr>"
        )
    best = max(METHODS, key=lambda m: summary[m]["tokens_per_second_median"])
    model = data["model"]
    output_rows = []
    for item in data["outputs"]:
        output_rows.append(
            f"<h3>{html.escape(item['prompt'])}</h3>"
            f"<p>Candidate coverage of normal continuation: <b>{100*item['candidate_coverage']:.1f}%</b></p>"
            f"<pre>{html.escape(item['texts']['normal'])}</pre>"
        )
    css = """
    :root{font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#172033;background:#f5f7fb}
    body{margin:0}.wrap{max-width:1250px;margin:auto;padding:32px}.card{background:#fff;border:1px solid #dbe2ec;border-radius:16px;padding:24px;margin:0 0 20px;box-shadow:0 4px 18px #17203310}
    h1{margin-top:0}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px;border-bottom:1px solid #e5eaf1;text-align:right}th:first-child{text-align:left}thead th{background:#f4f7fb}.badge{display:inline-block;background:#eaf2ff;color:#245ba7;padding:6px 10px;border-radius:999px;margin-right:7px;font-weight:700}.good{border-left:4px solid #1b9c5b;background:#effcf5;padding:13px}.warn{border-left:4px solid #db8a25;background:#fff8ec;padding:13px}pre{white-space:pre-wrap;background:#111827;color:#e8edf7;padding:13px;border-radius:10px}@media(max-width:800px){.wrap{padding:14px}table{display:block;overflow:auto}}
    """
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>LFM2.5 DFlash real benchmark</title><style>{css}</style></head><body><main class='wrap'>
    <section class='card'><span class='badge'>Real pretrained target</span><span class='badge'>CPU reference</span><span class='badge'>Exact greedy verification</span><h1>DFlash Mini Lab × LFM2.5-350M-Base</h1>
    <p>Frozen target: <b>{html.escape(model['id'])}</b> ({model['target_parameter_count']/1e6:.1f}M parameters). Compact drafter candidate set: {model['candidate_size']} / {model['target_vocab_size']} target tokens ({100*model['candidate_fraction_of_target_vocab']:.2f}%). Auxiliary models: {model['aux_parameter_count']/1e6:.2f}M parameters.</p>
    <div class='good'><b>Fastest measured method:</b> {html.escape(LABELS[best])} at {summary[best]['tokens_per_second_median']:.2f} tok/s ({summary[best]['speedup_vs_normal']:.2f}× normal).</div>
    <div class='warn'><b>Scope:</b> {html.escape(data['method_scope'])}<br><b>Timing:</b> {html.escape(data['timing_note'])}</div></section>
    <section class='card'><h2>Performance matrix</h2><table><thead><tr><th>Method</th><th>tok/s</th><th>Speedup</th><th>Latency</th><th>Acceptance</th><th>Tokens/target pass</th><th>Target passes</th><th>Guidance</th><th>Exact</th></tr></thead><tbody>{''.join(rows)}</tbody></table><p>Mean candidate coverage of normal target continuations: <b>{100*data['mean_candidate_coverage']:.1f}%</b>.</p></section>
    <section class='card'><h2>Normal target outputs</h2>{''.join(output_rows)}</section>
    </main></body></html>"""
    Path(output_path).write_text(doc, encoding="utf-8")
