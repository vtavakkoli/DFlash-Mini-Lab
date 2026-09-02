from pathlib import Path

from dflash_mini_lab.benchmark import run_benchmark
from dflash_mini_lab.report import build_report
from dflash_mini_lab.visuals import make_architecture_gif, make_speedup_gif

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_and_report_are_generated(tmp_path):
    data = tmp_path / "benchmark.json"
    speed = tmp_path / "speed.gif"
    arch = tmp_path / "arch.gif"
    report = tmp_path / "report.html"
    payload = run_benchmark(ROOT / "models/tiny_dflash_lab.npz", ROOT / "models/tokenizer.json", ROOT / "benchmarks/prompts.json", data, max_new_tokens=4, warmups=0, repeats=1)
    assert payload["summary"]["normal"]["all_exact"]
    assert payload["summary"]["dflash"]["all_exact"]
    assert payload["summary"]["dflash2"]["all_exact"]
    make_speedup_gif(data, speed)
    make_architecture_gif(arch)
    build_report(data, speed, arch, report)
    text = report.read_text(encoding="utf-8")
    assert "data:image/gif;base64," in text
    assert "Interactive comparison" in text
