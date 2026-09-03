from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 960, 540
BG = (247, 249, 252); TEXT = (24, 32, 48); MUTED = (91, 104, 126)
BLUE = (54, 111, 210); ORANGE = (235, 132, 55); GREEN = (55, 157, 100); PURPLE = (126, 87, 194); CYAN = (24, 167, 184); EMERALD = (32, 165, 98); BORDER = (213, 220, 232)


def _font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists(): return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _base(title: str, subtitle: str):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((36, 24), title, font=_font(28, True), fill=TEXT); d.text((36, 62), subtitle, font=_font(16), fill=MUTED)
    return im, d


def make_speedup_gif(benchmark_json: str | Path, out_path: str | Path) -> None:
    data = json.loads(Path(benchmark_json).read_text(encoding="utf-8"))
    names = [
        ("Normal", "normal", BLUE), ("DFlash", "dflash", ORANGE), ("DFlash v2", "dflash2", GREEN),
        ("DFlash3-MOBS", "dflash3_mobs", PURPLE), ("DFlash4-JUMP", "dflash4_jump_mobs", CYAN),
        ("DFlash5-FUSED", "dflash5_fused_jump_mobs", EMERALD),
    ]
    values = [float(data["summary"][key]["tokens_per_second_median"]) for _, key, _ in names]
    max_v = max(values) * 1.15; frames = []
    for step in range(31):
        progress = step / 30.0
        im, d = _base("CPU decoding speed comparison", "Median generated tokens per second; identical prompts, model and token budget")
        for i, (label, _, color) in enumerate(names):
            y = 100 + i * 60
            d.text((40, y + 7), label, font=_font(15, True), fill=TEXT)
            x0, x1 = 225, 875
            d.rounded_rectangle((x0, y, x1, y + 35), radius=8, fill=(233, 237, 244))
            width = int((x1 - x0) * min(values[i] / max_v, 1.0) * progress)
            if width > 0: d.rounded_rectangle((x0, y, x0 + width, y + 35), radius=8, fill=color)
            d.text((x0 + 10, y + 8), f"{values[i] * progress:.1f} tok/s", font=_font(13, True), fill=(255,255,255) if width > 130 else TEXT)
        normal = max(values[0], 1e-9)
        d.text((40, 472), f"Speedup: DFlash {values[1]/normal:.2f}x | v2 {values[2]/normal:.2f}x | MOBS {values[3]/normal:.2f}x | JUMP {values[4]/normal:.2f}x | FUSED {values[5]/normal:.2f}x", font=_font(13, True), fill=TEXT)
        frames.append(im)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=70, loop=0, optimize=True)


def make_architecture_gif(out_path: str | Path) -> None:
    frames = []; positions = [250, 390, 530, 670]
    for step in range(44):
        im, d = _base("DFlash5-FUSED-JUMP-MOBS", "One drafter pass → normal logits + low-rank candidate-only jump residuals → exact verification")
        d.text((42, 108), "1. One parallel drafter pass", font=_font(18, True), fill=ORANGE)
        d.rounded_rectangle((230, 96, 790, 148), radius=10, outline=ORANGE, width=3)
        d.text((370, 112), "shared non-causal drafter hidden states", font=_font(15, True), fill=ORANGE)

        if step >= 7:
            d.text((42, 184), "2. Shared hidden state branches", font=_font(18, True), fill=EMERALD)
            d.line((510, 150, 510, 205), fill=EMERALD, width=4)
            d.line((510, 205, 340, 240), fill=ORANGE, width=4); d.line((510, 205, 680, 240), fill=EMERALD, width=4)
            d.rounded_rectangle((245, 238, 435, 282), radius=9, fill=ORANGE); d.text((280, 251), "normal draft logits", font=_font(14, True), fill=(255,255,255))
            d.rounded_rectangle((585, 238, 775, 282), radius=9, fill=EMERALD); d.text((608, 251), "rank-16 jump residual", font=_font(14, True), fill=(255,255,255))

        if step >= 14:
            d.text((42, 320), "3. Score only retained top-k at +2 / +4", font=_font(18, True), fill=EMERALD)
            for i, x in enumerate(positions):
                active = i in (1, 3)
                color = EMERALD if active else (220, 225, 232)
                d.rounded_rectangle((x, 310, x + 96, 354), radius=9, fill=color)
                d.text((x + 24, 323), f"K @ +{i+1}", font=_font(13, True), fill=(255,255,255) if active else MUTED)
            d.text((285, 370), "no second Transformer/MLP pass • no full-vocabulary jump matmul", font=_font(14, True), fill=MUTED)

        if step >= 24:
            d.text((42, 414), "4. Confidence-gated anchors → O(BK) gap filling", font=_font(17, True), fill=PURPLE)
            d.line((300, 440, 690, 440), fill=PURPLE, width=4)
            for x in (300, 430, 560, 690): d.ellipse((x-7, 433, x+7, 447), fill=PURPLE)

        if step >= 33:
            d.rounded_rectangle((250, 466, 766, 510), radius=10, fill=GREEN)
            d.text((354, 479), "TARGET verifies the proposal exactly", font=_font(16, True), fill=(255,255,255))
            d.text((258, 517), "DFlash5 removes DFlash4's separate jump-forward bottleneck.", font=_font(13, True), fill=TEXT)
        frames.append(im)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
