from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 960, 540
BG = (247, 249, 252); TEXT = (24, 32, 48); MUTED = (91, 104, 126)
BLUE = (54, 111, 210); ORANGE = (235, 132, 55); GREEN = (55, 157, 100); BORDER = (213, 220, 232)


def _font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _base(title: str, subtitle: str):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((36, 24), title, font=_font(28, True), fill=TEXT)
    d.text((36, 62), subtitle, font=_font(16), fill=MUTED)
    return im, d


def make_speedup_gif(benchmark_json: str | Path, out_path: str | Path) -> None:
    data = json.loads(Path(benchmark_json).read_text(encoding="utf-8"))
    names = [("Normal", "normal", BLUE), ("DFlash", "dflash", ORANGE), ("DFlash v2", "dflash2", GREEN)]
    values = [float(data["summary"][key]["tokens_per_second_median"]) for _, key, _ in names]
    max_v = max(values) * 1.15
    frames = []
    for step in range(31):
        progress = step / 30.0
        im, d = _base("CPU decoding speed comparison", "Median generated tokens per second; identical prompts and token budget")
        for i, (label, _, color) in enumerate(names):
            y = 145 + i * 105
            d.text((62, y + 10), label, font=_font(20, True), fill=TEXT)
            x0, x1 = 230, 875
            d.rounded_rectangle((x0, y, x1, y + 52), radius=9, fill=(233, 237, 244))
            width = int((x1 - x0) * min(values[i] / max_v, 1.0) * progress)
            if width > 0:
                d.rounded_rectangle((x0, y, x0 + width, y + 52), radius=9, fill=color)
            shown = values[i] * progress
            d.text((x0 + 14, y + 14), f"{shown:.1f} tok/s", font=_font(17, True), fill=(255, 255, 255) if width > 150 else TEXT)
        normal = max(values[0], 1e-9)
        d.text((62, 477), f"Final speedup: DFlash {values[1]/normal:.2f}x   |   DFlash v2 {values[2]/normal:.2f}x", font=_font(18, True), fill=TEXT)
        frames.append(im)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=70, loop=0, optimize=True)


def _token_box(d, x: int, y: int, label: str, color, active: bool = True):
    fill = color if active else (226, 231, 239)
    d.rounded_rectangle((x, y, x + 74, y + 46), radius=8, fill=fill, outline=BORDER)
    d.text((x + 24, y + 12), label, font=_font(16, True), fill=(255, 255, 255) if active else MUTED)


def make_architecture_gif(out_path: str | Path) -> None:
    frames = []
    for step in range(36):
        im, d = _base("How DFlash changes speculative decoding", "Parallel draft block first, then one autoregressive target verifies the proposed block")
        d.text((40, 116), "Normal autoregressive", font=_font(19, True), fill=BLUE)
        active_normal = min(6, step // 5 + 1)
        for i in range(6):
            x = 250 + i * 105
            _token_box(d, x, 104, f"t{i+1}", BLUE, i < active_normal)
            if i < 5:
                d.line((x + 76, 127, x + 101, 127), fill=MUTED, width=3)
        d.text((40, 157), "One target pass per next token", font=_font(15), fill=MUTED)
        d.text((40, 247), "DFlash", font=_font(19, True), fill=ORANGE)
        d.rounded_rectangle((250, 226, 875, 286), radius=10, outline=ORANGE, width=3)
        d.text((424, 244), "one parallel draft forward pass", font=_font(17, True), fill=ORANGE)
        reveal = min(6, max(0, (step - 8) // 2 + 1))
        for i in range(6):
            x = 250 + i * 105
            _token_box(d, x, 309, f"d{i+1}", ORANGE, i < reveal)
            d.line((x + 37, 288, x + 37, 307), fill=ORANGE, width=2)
        verify_on = step >= 22
        d.rounded_rectangle((250, 385, 875, 447), radius=10, fill=GREEN if verify_on else (226, 231, 239), outline=BORDER)
        d.text((383, 405), "target verifies the whole block once", font=_font(18, True), fill=(255, 255, 255) if verify_on else MUTED)
        if step >= 27:
            d.text((310, 478), "accept ✓   accept ✓   accept ✓   reject ✗ → target correction", font=_font(18, True), fill=TEXT)
        frames.append(im)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=110, loop=0, optimize=True)
