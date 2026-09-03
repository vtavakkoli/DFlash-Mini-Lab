from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 960, 540
BG = (247, 249, 252); TEXT = (24, 32, 48); MUTED = (91, 104, 126)
BLUE = (54, 111, 210); ORANGE = (235, 132, 55); GREEN = (55, 157, 100); PURPLE = (126, 87, 194); CYAN = (24, 167, 184); BORDER = (213, 220, 232)


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
    names = [("Normal", "normal", BLUE), ("DFlash", "dflash", ORANGE), ("DFlash v2", "dflash2", GREEN), ("DFlash3-MOBS", "dflash3_mobs", PURPLE), ("DFlash4-JUMP", "dflash4_jump_mobs", CYAN)]
    values = [float(data["summary"][key]["tokens_per_second_median"]) for _, key, _ in names]
    max_v = max(values) * 1.15; frames = []
    for step in range(31):
        progress = step / 30.0
        im, d = _base("CPU decoding speed comparison", "Median generated tokens per second; identical prompts, model and token budget")
        for i, (label, _, color) in enumerate(names):
            y = 108 + i * 70
            d.text((43, y + 8), label, font=_font(16, True), fill=TEXT)
            x0, x1 = 225, 875
            d.rounded_rectangle((x0, y, x1, y + 40), radius=8, fill=(233, 237, 244))
            width = int((x1 - x0) * min(values[i] / max_v, 1.0) * progress)
            if width > 0: d.rounded_rectangle((x0, y, x0 + width, y + 40), radius=8, fill=color)
            d.text((x0 + 10, y + 10), f"{values[i] * progress:.1f} tok/s", font=_font(14, True), fill=(255,255,255) if width > 135 else TEXT)
        normal = max(values[0], 1e-9)
        d.text((43, 472), f"Speedup: DFlash {values[1]/normal:.2f}x | v2 {values[2]/normal:.2f}x | MOBS {values[3]/normal:.2f}x | JUMP {values[4]/normal:.2f}x", font=_font(15, True), fill=TEXT)
        frames.append(im)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=70, loop=0, optimize=True)


def make_architecture_gif(out_path: str | Path) -> None:
    frames = []
    positions = [250, 390, 530, 670]
    for step in range(44):
        im, d = _base("DFlash4-JUMP-MOBS", "Sparse future-token jumps (+2/+4) become anchors; O(BK) local filling; target verifies exactly")
        d.text((42, 112), "Parallel drafter", font=_font(18, True), fill=ORANGE)
        for i, x in enumerate(positions):
            d.rounded_rectangle((x, 100, x + 96, 150), radius=9, outline=ORANGE, width=2)
            d.text((x + 20, 116), f"K @ +{i+1}", font=_font(14, True), fill=ORANGE)

        if step >= 7:
            d.text((42, 190), "Indexed jump head", font=_font(18, True), fill=CYAN)
            d.rounded_rectangle((250, 174, 766, 224), radius=10, outline=CYAN, width=3)
            d.text((355, 190), "context + offset embedding → token logits", font=_font(15, True), fill=CYAN)

        if step >= 13:
            d.text((42, 257), "Sparse anchors", font=_font(18, True), fill=CYAN)
            for idx in (1, 3):
                x = positions[idx]
                d.line((x + 48, 226, x + 48, 280), fill=CYAN, width=4)
                d.rounded_rectangle((x, 280, x + 96, 326), radius=9, fill=CYAN)
                d.text((x + 24, 294), f"jump +{idx+1}", font=_font(13, True), fill=(255,255,255))

        if step >= 22:
            d.text((42, 355), "Gap filling", font=_font(18, True), fill=PURPLE)
            for idx in (0, 2):
                x = positions[idx]
                active = step >= 22 + idx * 3
                color = PURPLE if active else (220, 214, 235)
                d.rounded_rectangle((x, 344, x + 96, 390), radius=9, fill=color)
                d.text((x + 26, 358), f"fill +{idx+1}", font=_font(13, True), fill=(255,255,255))
            d.text((245, 407), "each gap: top-k draft score + adjacent anchor compatibility → O(BK)", font=_font(14, True), fill=MUTED)

        if step >= 32:
            d.rounded_rectangle((250, 438, 766, 486), radius=10, fill=GREEN)
            d.text((354, 452), "TARGET verifies the complete proposal", font=_font(16, True), fill=(255,255,255))
            d.text((260, 503), "Jump predictions can be wrong; verifier preserves exact greedy output.", font=_font(14, True), fill=TEXT)
        frames.append(im)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
