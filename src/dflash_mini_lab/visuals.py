from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 960, 540
BG = (247, 249, 252); TEXT = (24, 32, 48); MUTED = (91, 104, 126)
BLUE = (54, 111, 210); ORANGE = (235, 132, 55); GREEN = (55, 157, 100); PURPLE = (126, 87, 194); BORDER = (213, 220, 232)


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
    names = [
        ("Normal", "normal", BLUE),
        ("DFlash", "dflash", ORANGE),
        ("DFlash v2", "dflash2", GREEN),
        ("DFlash3-MOBS", "dflash3_mobs", PURPLE),
    ]
    values = [float(data["summary"][key]["tokens_per_second_median"]) for _, key, _ in names]
    max_v = max(values) * 1.15
    frames = []
    for step in range(31):
        progress = step / 30.0
        im, d = _base("CPU decoding speed comparison", "Median generated tokens per second; identical prompts and token budget")
        for i, (label, _, color) in enumerate(names):
            y = 120 + i * 82
            d.text((48, y + 8), label, font=_font(18, True), fill=TEXT)
            x0, x1 = 225, 875
            d.rounded_rectangle((x0, y, x1, y + 46), radius=8, fill=(233, 237, 244))
            width = int((x1 - x0) * min(values[i] / max_v, 1.0) * progress)
            if width > 0:
                d.rounded_rectangle((x0, y, x0 + width, y + 46), radius=8, fill=color)
            shown = values[i] * progress
            d.text((x0 + 12, y + 12), f"{shown:.1f} tok/s", font=_font(15, True), fill=(255, 255, 255) if width > 145 else TEXT)
        normal = max(values[0], 1e-9)
        d.text(
            (48, 466),
            f"Speedup: DFlash {values[1]/normal:.2f}x | v2 {values[2]/normal:.2f}x | MOBS {values[3]/normal:.2f}x",
            font=_font(17, True),
            fill=TEXT,
        )
        frames.append(im)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=70, loop=0, optimize=True)


def _token_box(d, x: int, y: int, label: str, color, active: bool = True):
    fill = color if active else (226, 231, 239)
    d.rounded_rectangle((x, y, x + 74, y + 46), radius=8, fill=fill, outline=BORDER)
    d.text((x + 24, y + 12), label, font=_font(16, True), fill=(255, 255, 255) if active else MUTED)


def make_architecture_gif(out_path: str | Path) -> None:
    frames = []
    for step in range(42):
        im, d = _base("DFlash3-MOBS: middle-out speculative path selection", "Parallel block draft + O(BK) bidirectional neighbor scoring + exact target verification")
        d.text((38, 112), "Parallel draft candidates", font=_font(18, True), fill=ORANGE)
        for pos in range(6):
            x = 220 + pos * 112
            d.rounded_rectangle((x, 100, x + 88, 154), radius=9, outline=ORANGE, width=2)
            d.text((x + 12, 112), f"K @ p{pos+1}", font=_font(14, True), fill=ORANGE)

        anchor = 2
        if step >= 7:
            x = 220 + anchor * 112
            d.rounded_rectangle((x - 4, 96, x + 92, 158), radius=10, outline=PURPLE, width=5)
            d.text((37, 175), "1. Reproducible random-middle anchor", font=_font(16, True), fill=PURPLE)

        if step >= 13:
            d.text((37, 229), "2. Expand left/right", font=_font(16, True), fill=PURPLE)
            center_x = 220 + anchor * 112 + 44
            reveal = min(3, max(0, (step - 13) // 4 + 1))
            for dist in range(1, reveal + 1):
                for pos in (anchor - dist, anchor + dist):
                    if 0 <= pos < 6:
                        x = 220 + pos * 112 + 44
                        d.line((center_x, 206, x, 250), fill=PURPLE, width=3)
                        d.ellipse((x - 7, 243, x + 7, 257), fill=PURPLE)
            d.text((355, 270), "K scores per new position", font=_font(15, True), fill=MUTED)

        if step >= 27:
            d.text((37, 329), "3. One odd/even bubble-like local refinement", font=_font(16, True), fill=PURPLE)
            for pos in range(6):
                x = 220 + pos * 112
                color = PURPLE if (pos + step) % 2 == 0 else (202, 191, 225)
                d.rounded_rectangle((x, 318, x + 88, 360), radius=8, fill=color)
                d.text((x + 32, 329), f"t{pos+1}", font=_font(14, True), fill=(255, 255, 255))

        if step >= 34:
            d.rounded_rectangle((220, 405, 892, 461), radius=10, fill=GREEN)
            d.text((376, 423), "TARGET verifies selected path exactly", font=_font(17, True), fill=(255, 255, 255))
            d.text((265, 484), "Selector complexity: O(BK), versus DFlash2 transition grid O(BK²)", font=_font(16, True), fill=TEXT)
        frames.append(im)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
