from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 960, 540
BG = (247, 249, 252); TEXT = (24, 32, 48); MUTED = (91, 104, 126)
BLUE = (54, 111, 210); ORANGE = (235, 132, 55); GREEN = (55, 157, 100); PURPLE = (126, 87, 194)
CYAN = (24, 167, 184); EMERALD = (32, 165, 98); PINK = (201, 93, 136); INDIGO = (102, 87, 200)


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
    d.text((36, 22), title, font=_font(27, True), fill=TEXT)
    d.text((36, 58), subtitle, font=_font(15), fill=MUTED)
    return im, d


def make_speedup_gif(benchmark_json: str | Path, out_path: str | Path) -> None:
    data = json.loads(Path(benchmark_json).read_text(encoding="utf-8"))
    names = [
        ("Normal", "normal", BLUE), ("DFlash", "dflash", ORANGE), ("DFlash v2", "dflash2", GREEN),
        ("DFlash3-MOBS", "dflash3_mobs", PURPLE), ("DFlash4-JUMP", "dflash4_jump_mobs", CYAN),
        ("DFlash5-FUSED", "dflash5_fused_jump_mobs", EMERALD), ("DFlash6-Boltz", "dflash6_boltzmann", PINK),
        ("DFlash6-BMOBS", "dflash6_bmobs", INDIGO),
    ]
    values = [float(data["summary"][key]["tokens_per_second_median"]) for _, key, _ in names]
    max_v = max(values) * 1.15
    frames = []
    for step in range(31):
        progress = step / 30.0
        im, d = _base("CPU decoding speed comparison", "Eight modes • same prompts, target, token budget and one CPU thread")
        for i, (label, _, color) in enumerate(names):
            y = 92 + i * 49
            d.text((34, y + 5), label, font=_font(13, True), fill=TEXT)
            x0, x1 = 205, 880
            d.rounded_rectangle((x0, y, x1, y + 31), radius=7, fill=(233, 237, 244))
            width = int((x1 - x0) * min(values[i] / max_v, 1.0) * progress)
            if width > 0:
                d.rounded_rectangle((x0, y, x0 + width, y + 31), radius=7, fill=color)
            d.text((x0 + 8, y + 7), f"{values[i] * progress:.1f} tok/s", font=_font(12, True), fill=(255,255,255) if width > 120 else TEXT)
        normal = max(values[0], 1e-9)
        d.text((34, 493), f"DFlash {values[1]/normal:.2f}x • Boltzmann {values[6]/normal:.2f}x • BMOBS {values[7]/normal:.2f}x", font=_font(14, True), fill=TEXT)
        frames.append(im)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=70, loop=0, optimize=True)


def make_architecture_gif(out_path: str | Path) -> None:
    frames = []
    positions = [245, 385, 525, 665]
    for step in range(44):
        im, d = _base("DFlash6-Boltzmann / BMOBS", "Training-free deterministic exploration from existing draft logits; exact target verification")
        d.text((38, 105), "1. One parallel DFlash draft", font=_font(18, True), fill=ORANGE)
        for i, x in enumerate(positions):
            d.rounded_rectangle((x, 94, x + 100, 140), radius=9, outline=ORANGE, width=2)
            d.text((x + 22, 108), f"top-k +{i+1}", font=_font(13, True), fill=ORANGE)
        if step >= 8:
            d.text((38, 182), "2. Margin → adaptive temperature", font=_font(18, True), fill=PINK)
            d.rounded_rectangle((245, 169, 765, 216), radius=10, outline=PINK, width=3)
            d.text((320, 184), "large margin → T↓   |   uncertain → T↑", font=_font(15, True), fill=PINK)
        if step >= 15:
            d.text((38, 253), "3A. Boltzmann", font=_font(17, True), fill=PINK)
            d.text((210, 253), "context-seeded Gumbel-Max at every slot", font=_font(15), fill=TEXT)
            d.line((245, 286, 765, 286), fill=PINK, width=4)
            for x in positions:
                d.ellipse((x + 43, 279, x + 57, 293), fill=PINK)
        if step >= 23:
            d.text((38, 334), "3B. BMOBS", font=_font(17, True), fill=INDIGO)
            d.text((210, 334), "sample one uncertain middle anchor, then O(BK) gap fill", font=_font(15), fill=TEXT)
            d.line((300, 372, 690, 372), fill=INDIGO, width=4)
            for x in (300, 430, 560, 690):
                d.ellipse((x-7, 365, x+7, 379), fill=INDIGO)
        if step >= 33:
            d.rounded_rectangle((245, 430, 770, 478), radius=10, fill=GREEN)
            d.text((350, 444), "TARGET verifies every proposed token", font=_font(16, True), fill=(255,255,255))
            d.text((215, 497), "No new model weights • no extra forward pass • deterministic benchmark", font=_font(14, True), fill=TEXT)
        frames.append(im)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
