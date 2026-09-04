from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import statistics
import time

import torch

try:
    from eagle.model.ea_model import EaModel
except Exception as exc:  # pragma: no cover - exercised in Docker/CI
    raise RuntimeError(
        "Version 8 requires the pinned SafeAILab/EAGLE checkout on PYTHONPATH. "
        "Use Dockerfile.eagle3."
    ) from exc


DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_EAGLE_MODEL = "AngelSlim/Qwen3-1.7B_eagle3"
UPSTREAM_EAGLE_COMMIT = "cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b"


def _read_prompts(path: str | Path) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts = data.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"{path} must contain a non-empty prompts list")
    return [str(x) for x in prompts]


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(name)


def _one_generation(model: EaModel, input_ids: torch.Tensor, *, method: str, tokens: int, max_length: int) -> dict:
    if tokens < 1:
        raise ValueError("tokens must be positive")
    # SafeAILab EAGLE stops when new_token > max_new_tokens, so tokens-1 asks
    # naivegenerate for exactly `tokens` in the non-EOS case. EAGLE may finish
    # a whole accepted tree path and overshoot; the requested prefix is what we
    # compare, while wall time intentionally includes that overshoot overhead.
    internal_max_new = max(0, int(tokens) - 1)
    input_len = int(input_ids.shape[1])
    fn = model.naivegenerate if method == "normal_eagle_runtime" else model.eagenerate

    start = time.perf_counter_ns()
    output_ids, reported_new_tokens, tree_index = fn(
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=internal_max_new,
        max_length=int(max_length),
        log=True,
    )
    seconds = (time.perf_counter_ns() - start) / 1e9

    full_suffix = output_ids[0, input_len:].detach().cpu().tolist()
    requested_suffix = [int(x) for x in full_suffix[: int(tokens)]]
    used_tokens = len(requested_suffix)
    iterations = int(tree_index) + 1 if method == "eagle3" else int(reported_new_tokens)
    return {
        "method": method,
        "wall_seconds": float(seconds),
        "requested_tokens": int(tokens),
        "used_tokens": int(used_tokens),
        "raw_generated_tokens": int(len(full_suffix)),
        "reported_new_tokens": int(reported_new_tokens),
        "iterations": max(int(iterations), 0),
        "tokens_per_second": float(used_tokens / max(seconds, 1e-12)),
        "tokens_per_iteration": float(used_tokens / max(iterations, 1)),
        "tokens": requested_suffix,
    }


def _write_html(payload: dict, path: Path) -> None:
    n = payload["summary"]["normal_eagle_runtime"]
    e = payload["summary"]["eagle3"]
    verdict = (
        "EAGLE-3 is faster than its matched normal baseline on this runner."
        if e["speedup_vs_normal"] > 1.0
        else "EAGLE-3 does not beat its matched normal baseline on this CPU runner."
    )
    rows = []
    for key, label in (("normal_eagle_runtime", "Normal Qwen in EAGLE runtime"), ("eagle3", "Version 8 — EAGLE-3")):
        s = payload["summary"][key]
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{s['median_tokens_per_second']:.3f}</td>"
            f"<td>{s['speedup_vs_normal']:.3f}×</td>"
            f"<td>{s['median_wall_seconds']*1000:.1f}</td>"
            f"<td>{s['mean_tokens_per_iteration']:.2f}</td>"
            f"<td>{'✓' if s['all_exact'] else '✗'}</td>"
            "</tr>"
        )

    path.write_text(
        f"""<!doctype html><html><head><meta charset='utf-8'><title>DFlash Mini Lab · Version 8 EAGLE-3</title>
<style>body{{font-family:system-ui,-apple-system,sans-serif;max-width:1050px;margin:40px auto;padding:0 20px;color:#171717}}.hero,.note{{padding:18px 22px;border-radius:14px;background:#f5f6f8;margin:18px 0}}.note{{background:#fff7df}}table{{border-collapse:collapse;width:100%;margin:22px 0}}th,td{{padding:10px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{background:#eee;padding:2px 5px;border-radius:4px}}</style></head><body>
<h1>Version 8 — EAGLE-3 baseline</h1>
<div class='hero'><p><b>{html.escape(verdict)}</b></p><p>Matched target: <code>{html.escape(payload['model']['base_model'])}</code> · draft: <code>{html.escape(payload['model']['eagle_model'])}</code>.</p><p>Measured EAGLE speedup: <b>{e['speedup_vs_normal']:.3f}×</b>. Timing includes prefill, EAGLE draft-tree work, target verification, cache maintenance and Python overhead.</p></div>
<table><thead><tr><th>Method</th><th>tok/s</th><th>vs normal</th><th>median wall ms</th><th>tokens / iteration</th><th>exact</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Configuration</h2><ul><li>dtype: <code>{html.escape(payload['config']['dtype'])}</code></li><li>CPU threads: {payload['config']['cpu_threads']}</li><li>requested tokens: {payload['config']['tokens']}</li><li>prompts: {payload['config']['prompt_count']}</li><li>EAGLE tree budget: {payload['config']['total_token']}</li><li>depth: {payload['config']['depth']}</li><li>draft top-k: {payload['config']['draft_top_k']}</li></ul>
<div class='note'><b>Comparison scope.</b> The public AngelSlim checkpoint is paired with <code>Qwen/Qwen3-1.7B</code>, not <code>Qwen/Qwen3-1.7B-Base</code>. Therefore the EAGLE <i>speedup ratio</i> can be compared with the earlier DFlash study, but raw tok/s should not be presented as an apples-to-apples model comparison.</div>
<h2>Exactness</h2><p>For every held-out prompt, the first requested greedy tokens from EAGLE-3 must equal the same target model's <code>naivegenerate</code> output token-for-token. CI fails if this invariant is violated.</p>
<h2>Reproducibility</h2><p>SafeAILab/EAGLE source is pinned to <code>{UPSTREAM_EAGLE_COMMIT}</code>. Model weights are downloaded at runtime and are not redistributed by DFlash Mini Lab.</p>
</body></html>""",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(max(1, int(args.cpu_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    prompts = _read_prompts(args.prompts)[: max(1, int(args.prompt_limit))]
    model = EaModel.from_pretrained(
        use_eagle3=True,
        base_model_path=args.base_model,
        ea_model_path=args.eagle_model,
        total_token=int(args.total_token),
        depth=int(args.depth),
        top_k=int(args.draft_top_k),
        threshold=float(args.threshold),
        torch_dtype=_dtype(args.dtype),
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Allocate both cache paths before measurement so neither method pays one-off
    # tensor/cache construction in the reported timings.
    warm = torch.as_tensor(model.tokenizer([prompts[0]], add_special_tokens=True).input_ids, dtype=torch.long)
    _one_generation(model, warm, method="normal_eagle_runtime", tokens=2, max_length=args.max_length)
    _one_generation(model, warm, method="eagle3", tokens=2, max_length=args.max_length)

    records: list[dict] = []
    exact_rows: list[bool] = []
    for repeat in range(max(1, int(args.repeats))):
        for prompt_index, prompt in enumerate(prompts):
            ids = torch.as_tensor(model.tokenizer([prompt], add_special_tokens=True).input_ids, dtype=torch.long)
            # Alternate measured order to reduce systematic first/second-method bias.
            order = ("normal_eagle_runtime", "eagle3") if (prompt_index + repeat) % 2 == 0 else ("eagle3", "normal_eagle_runtime")
            by_method = {}
            for method in order:
                row = _one_generation(model, ids, method=method, tokens=args.tokens, max_length=args.max_length)
                row.update(prompt=prompt, prompt_index=prompt_index, repeat=repeat)
                by_method[method] = row
            normal_tokens = by_method["normal_eagle_runtime"]["tokens"]
            eagle_tokens = by_method["eagle3"]["tokens"]
            exact = normal_tokens == eagle_tokens and len(normal_tokens) > 0
            exact_rows.append(bool(exact))
            for method in ("normal_eagle_runtime", "eagle3"):
                row = by_method[method]
                row["exact"] = True if method == "normal_eagle_runtime" else bool(exact)
                records.append(row)

    summary = {}
    normal_tps = statistics.median(r["tokens_per_second"] for r in records if r["method"] == "normal_eagle_runtime")
    for method in ("normal_eagle_runtime", "eagle3"):
        rows = [r for r in records if r["method"] == method]
        tps = statistics.median(r["tokens_per_second"] for r in rows)
        summary[method] = {
            "median_tokens_per_second": float(tps),
            "speedup_vs_normal": float(tps / normal_tps),
            "median_wall_seconds": float(statistics.median(r["wall_seconds"] for r in rows)),
            "mean_tokens_per_iteration": float(statistics.mean(r["tokens_per_iteration"] for r in rows)),
            "mean_raw_generated_tokens": float(statistics.mean(r["raw_generated_tokens"] for r in rows)),
            "all_exact": bool(all(r["exact"] for r in rows)),
        }

    target_params = int(sum(p.numel() for p in model.base_model.parameters()))
    draft_params = int(sum(p.numel() for p in model.ea_layer.parameters()))
    payload = {
        "schema_version": 1,
        "version": "8",
        "method": "EAGLE-3 baseline",
        "model": {
            "base_model": args.base_model,
            "eagle_model": args.eagle_model,
            "target_parameter_count": target_params,
            "draft_parameter_count": draft_params,
            "target_architecture": model.config.architectures[0] if getattr(model.config, "architectures", None) else None,
        },
        "upstream": {
            "repository": "SafeAILab/EAGLE",
            "commit": UPSTREAM_EAGLE_COMMIT,
            "checkpoint_pair_is_official_safeailab": False,
            "checkpoint_pair_is_listed_by_safeailab": True,
        },
        "config": {
            "dtype": args.dtype,
            "cpu_threads": int(args.cpu_threads),
            "tokens": int(args.tokens),
            "prompt_count": len(prompts),
            "repeats": int(args.repeats),
            "total_token": int(args.total_token),
            "depth": int(args.depth),
            "draft_top_k": int(args.draft_top_k),
            "threshold": float(args.threshold),
            "max_length": int(args.max_length),
            "timing_scope": "end-to-end generation call after model/cache warmup; includes prefill, draft, verification and runtime overhead",
        },
        "summary": summary,
        "all_exact": bool(all(exact_rows)),
        "records": records,
        "comparison_note": "EAGLE uses its matched Qwen/Qwen3-1.7B checkpoint; earlier DFlash 1.7B experiments use Qwen/Qwen3-1.7B-Base. Compare relative speedup, not raw tok/s.",
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_html(payload, out / "report.html")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Version 8 EAGLE-3 end-to-end baseline")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--eagle-model", default=DEFAULT_EAGLE_MODEL)
    parser.add_argument("--prompts", default="qwen_benchmarks/prompts.json")
    parser.add_argument("--output-dir", default="eagle3-reports")
    parser.add_argument("--prompt-limit", type=int, default=6)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--cpu-threads", type=int, default=int(os.getenv("CPU_THREADS", "2")))
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--draft-top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"model": payload["model"], "config": payload["config"], "summary": payload["summary"], "all_exact": payload["all_exact"]}, indent=2))


if __name__ == "__main__":
    main()
