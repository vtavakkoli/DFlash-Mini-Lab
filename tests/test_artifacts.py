from pathlib import Path

from dflash_mini_lab.benchmark import run_benchmark
from dflash_mini_lab.report import build_report
from dflash_mini_lab.visuals import make_architecture_gif, make_speedup_gif

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_and_report_are_generated(tmp_path):
    data = tmp_path / "benchmark.json"; speed = tmp_path / "speed.gif"; arch = tmp_path / "arch.gif"; report = tmp_path / "report.html"
    payload = run_benchmark(
        ROOT / "models/tiny_dflash_lab.npz", ROOT / "models/tokenizer.json", ROOT / "benchmarks/prompts.json", data,
        max_new_tokens=4, warmups=0, repeats=1, top_k=8, mobs_refine_passes=0,
        jump_weight=0.5, fused_weight=1.0, fused_min_margin=0.0,
    )
    for method in ("normal", "dflash", "dflash2", "dflash3_mobs", "dflash4_jump_mobs", "dflash5_fused_jump_mobs"):
        assert payload["summary"][method]["all_exact"]
    assert payload["summary"]["dflash3_mobs"]["mean_selector_pair_scores"] < payload["summary"]["dflash2"]["mean_selector_pair_scores"]
    assert payload["summary"]["dflash4_jump_mobs"]["mean_total_guidance_scores"] < payload["summary"]["dflash2"]["mean_selector_pair_scores"]
    assert payload["summary"]["dflash5_fused_jump_mobs"]["mean_total_guidance_scores"] < payload["summary"]["dflash2"]["mean_selector_pair_scores"]
    assert payload["summary"]["dflash5_fused_jump_mobs"]["mean_jump_forward_passes"] == 0
    make_speedup_gif(data, speed); make_architecture_gif(arch); build_report(data, speed, arch, report)
    text = report.read_text(encoding="utf-8")
    assert "data:image/gif;base64," in text
    assert "Interactive comparison" in text
    assert "DFlash5-FUSED-JUMP" in text
