# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///
"""Render assets/demo.gif and assets/social-preview.png.

The commands are run for real and their output is drawn into the images, so the numbers in the
pictures are the numbers the examples print. Run from the repository root:

    uv run assets/make_assets.py

Fonts come from the operating system (Consolas and Segoe UI on Windows, DejaVu or Menlo
elsewhere), so the glyphs differ slightly between machines.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
BG, CHROME, FG = "#0d1117", "#161b22", "#c9d1d9"
GREEN, RED, CYAN, WHITE, GREY = "#7ee787", "#ff7b72", "#79c0ff", "#ffffff", "#8b949e"
ANSI_COLORS = {"1": WHITE, "31": RED, "36": CYAN}
COLS, ROWS, FONT_SIZE, PAD = 96, 16, 17, 22

MONO = [
    "C:/Windows/Fonts/consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
]
SANS_BOLD = [
    "C:/Windows/Fonts/segoeuib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]
SANS = [
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]

Cell = tuple[str, str]  # character, color


def font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont:
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise SystemExit(f"no usable font found, tried: {candidates}")


def run(*argv: str) -> str:
    result = subprocess.run(argv, cwd=ROOT.parent, capture_output=True, text=True, encoding="utf-8")
    return result.stdout


def parse_ansi(line: str) -> list[Cell]:
    cells: list[Cell] = []
    color = FG
    for token in re.split(r"(\x1b\[\d+m)", line):
        match = re.fullmatch(r"\x1b\[(\d+)m", token)
        if match:
            color = ANSI_COLORS.get(match.group(1), FG)
        else:
            cells.extend((ch, color) for ch in token)
    return cells


def wrap(cells: list[Cell]) -> list[list[Cell]]:
    rows = []
    while len(cells) > COLS:
        cut = max((i for i, (ch, _) in enumerate(cells[:COLS]) if ch == " "), default=COLS)
        rows.append(cells[:cut])
        cells = cells[cut:]
        while cells and cells[0][0] == " " and len(rows) > 0 and cut != COLS:
            cells = cells[1:]
    rows.append(cells)
    return rows


def output_rows(text: str) -> list[list[Cell]]:
    return [row for line in text.rstrip("\n").split("\n") for row in wrap(parse_ansi(line))]


def prompt_row(command: str) -> list[Cell]:
    return [("$", GREEN), (" ", FG), *((ch, FG) for ch in command)]


class Terminal:
    def __init__(self) -> None:
        self.font = font(MONO, FONT_SIZE)
        self.cw = round(self.font.getlength("M"))
        self.lh = FONT_SIZE + 8
        self.size = (COLS * self.cw + 2 * PAD, ROWS * self.lh + 2 * PAD + 30)

    def render(self, rows: list[list[Cell]], cursor: bool = False) -> Image.Image:
        img = Image.new("RGB", self.size, BG)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, self.size[0], 30), fill=CHROME)
        for i, color in enumerate(("#ff5f56", "#ffbd2e", "#27c93f")):
            draw.ellipse((14 + i * 22, 9, 26 + i * 22, 21), fill=color)
        for r, row in enumerate(rows):
            y = 30 + PAD + r * self.lh
            for c, (ch, color) in enumerate(row):
                draw.text((PAD + c * self.cw, y), ch, font=self.font, fill=color)
        if cursor and rows:
            x, y = PAD + len(rows[-1]) * self.cw, 30 + PAD + (len(rows) - 1) * self.lh
            draw.rectangle((x, y + 3, x + self.cw - 2, y + self.lh - 3), fill=FG)
        return img


def scene(term: Terminal, command: str, text: str, think_ms: int, hold_ms: int):
    frames: list[tuple[Image.Image, int]] = []
    for n in range(len(command) + 1):
        frames.append((term.render([prompt_row(command[:n])], cursor=True), 38))
    frames[-1] = (frames[-1][0], 450)
    shown = [prompt_row(command)]
    frames.append((term.render(shown), think_ms))
    for row in output_rows(text):
        shown.append(row)
        frames.append((term.render(shown), 70))
    frames[-1] = (frames[-1][0], hold_ms)
    return frames


def save_gif(frames: list[tuple[Image.Image, int]], path: Path) -> None:
    merged: list[tuple[Image.Image, int]] = []
    for image, ms in frames:
        if merged and merged[-1][0].tobytes() == image.tobytes():
            merged[-1] = (merged[-1][0], merged[-1][1] + ms)
        else:
            merged.append((image, ms))
    palette = [im.quantize(colors=24, dither=Image.Dither.NONE) for im, _ in merged]
    palette[0].save(
        path,
        save_all=True,
        append_images=palette[1:],
        duration=[ms for _, ms in merged],
        loop=0,
        optimize=True,
    )


def social_preview(leaky: str, honest: str, path: Path) -> None:
    img = Image.new("RGB", (1280, 640), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 14, 640), fill=GREEN)
    draw.text((80, 60), "trainlint", font=font(SANS_BOLD, 110), fill=WHITE)
    tagline = font(SANS, 38)
    draw.text((84, 212), "Agent skills and an offline checker", font=tagline, fill=GREY)
    draw.text((84, 262), "for ML bugs that do not crash.", font=tagline, fill=GREY)
    mono = font(MONO, 34)
    draw.rounded_rectangle((80, 360, 1200, 540), radius=14, fill=CHROME)
    rows = [
        ("random labels, selection before CV", leaky, RED),
        ("same steps inside a pipeline", honest, GREEN),
    ]
    for i, (label, value, color) in enumerate(rows):
        y = 388 + i * 70
        draw.text((112, y), label, font=mono, fill=FG)
        draw.text((1130 - mono.getlength(value), y), value, font=mono, fill=color)
    img.save(path, optimize=True)


def main() -> None:
    term = Terminal()
    leaky = run("uv", "run", "examples/noise_leak.py")
    report = run("uv", "run", "trainlint", "check", "--color", "always", "examples/noise_leak.py")
    honest = run("uv", "run", "examples/noise_leak_fixed.py")
    frames = (
        scene(term, "uv run examples/noise_leak.py", leaky, think_ms=900, hold_ms=2200)
        + scene(term, "uvx trainlint check examples/noise_leak.py", report, 500, 4500)
        + scene(term, "uv run examples/noise_leak_fixed.py", honest, 900, 3200)
    )
    save_gif(frames, ROOT / "demo.gif")
    percent = re.compile(r"\d+%")
    social_preview(
        percent.search(leaky).group(), percent.search(honest).group(), ROOT / "social-preview.png"
    )


if __name__ == "__main__":
    main()
