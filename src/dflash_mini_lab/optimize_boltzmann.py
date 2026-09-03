from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from .decoding import dflash6_bmobs_decode, dflash6_boltzmann_decode, normal_decode
from .runtime import CpuReferenceRuntime
from .tokenizer import WordTokenizer


def _measure(decode, runtime, tok, prompts, tokens, repeats, top_k, temperature):
    tps = []
    acceptance = []
    target_eff = []
    exact = True
    for prompt in prompts:
        ids = tok.encode(prompt)
        reference, _ = normal_decode(runtime, ids, tokens)
        for _ in range(repeats):
            out, stats = decode(runtime, ids, tokens, top_k=top_k, temperature=temperature)
            exact = exact and bool(np.array_equal(reference, out))
            tps.append(stats.tokens_per_second)
            acceptance.append(stats.acceptance_rate)
            target_eff.append(stats.tokens_per_target_pass)
    return {
        "temperature": float(temperature),
        "all_exact": exact,
        "tokens_per_second_median": statistics.median(tps),
        "acceptance_rate_mean": statistics.fmean(acceptance),
        "tokens_per_target_pass_mean": statistics.fmean(target_eff),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="models/tiny_dflash_lab.npz")
    p.add_argument("--tokenizer", default="models/tokenizer.json")
    p.add_argument("--prompts", default="benchmarks/prompts.json")
    p.add_argument("--output", required=True)
    p.add_argument("--tokens", type=int, default=12)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--temperatures", default="0.02,0.05,0.10,0.20,0.35")
    args = p.parse_args()

    runtime = CpuReferenceRuntime(args.weights)
    tok = WordTokenizer.load(args.tokenizer)
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))["prompts"]
    temperatures = [float(x) for x in args.temperatures.split(",") if x.strip()]

    boltzmann = [_measure(dflash6_boltzmann_decode, runtime, tok, prompts, args.tokens, args.repeats, args.top_k, t) for t in temperatures]
    bmobs = [_measure(dflash6_bmobs_decode, runtime, tok, prompts, args.tokens, args.repeats, args.top_k, t) for t in temperatures]
    valid_b = [x for x in boltzmann if x["all_exact"]]
    valid_m = [x for x in bmobs if x["all_exact"]]
    if not valid_b or not valid_m:
        raise SystemExit("No exact DFlash6 temperature candidate")
    best_b = max(valid_b, key=lambda x: x["tokens_per_second_median"])
    best_m = max(valid_m, key=lambda x: x["tokens_per_second_median"])
    result = {"boltzmann_candidates": boltzmann, "bmobs_candidates": bmobs, "best_boltzmann": best_b, "best_bmobs": best_m}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
