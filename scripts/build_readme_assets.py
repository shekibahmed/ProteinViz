"""Draw the README workflow diagrams.

The GIFs and the tour are boxes and arrows. They do not copy scores, sequences,
or residue lists from a result. Pillow writes the GIFs. imageio writes the mp4.
Neither library is a runtime dependency of the app.

    python scripts/build_readme_assets.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 960, 540
BG = (247, 245, 242)
INK = (31, 41, 51)
MUTED = (90, 104, 120)
BLUE = (76, 120, 168)
SOFT = (238, 243, 248)
WHITE = (255, 255, 255)
LINE = (196, 206, 216)

ASSETS = Path(__file__).resolve().parents[1] / "docs" / "assets"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default(size)


def new_frame() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), BG)
    return image, ImageDraw.Draw(image)


def centered(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, face, fill
) -> None:
    left, top, right, bottom = box
    lines = _wrap(draw, text, face, right - left - 24)
    heights = [draw.textbbox((0, 0), line, font=face)[3] for line in lines]
    block = sum(heights) + 8 * (len(lines) - 1)
    y = top + (bottom - top - block) / 2
    for line, height in zip(lines, heights, strict=True):
        bbox = draw.textbbox((0, 0), line, font=face)
        width = bbox[2] - bbox[0]
        draw.text((left + (right - left - width) / 2, y), line, font=face, fill=fill)
        y += height + 8


def _wrap(draw: ImageDraw.ImageDraw, text: str, face, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), trial, font=face)[2] <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def card(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, active: bool
) -> None:
    fill = BLUE if active else WHITE
    ink = WHITE if active else INK
    outline = BLUE if active else LINE
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=3)
    centered(draw, box, text, font(18), ink)


def title(draw: ImageDraw.ImageDraw, text: str) -> None:
    draw.text((48, 36), text, font=font(32), fill=INK)


def caption(draw: ImageDraw.ImageDraw, text: str) -> None:
    draw.text((48, H - 72), text, font=font(20), fill=MUTED)


def disease_frames() -> list[Image.Image]:
    stages = ["Targets", "Shared pathways", "Interactions", "Drugs and leads"]
    frames: list[Image.Image] = []
    for active in range(len(stages)):
        image, draw = new_frame()
        title(draw, "Disease explorer")
        caption(draw, "These four views stay on the disease page.")
        for i, label in enumerate(stages):
            left = 48 + i * 228
            card(draw, (left, 180, left + 210, 300), label, i == active)
        frames.append(image)
    image, draw = new_frame()
    title(draw, "Disease explorer")
    caption(draw, "Next: a pair you choose, with no guessed PDB entry.")
    card(draw, (48, 180, 430, 300), "Disease page", True)
    draw.line((450, 240, 530, 240), fill=BLUE, width=6)
    draw.polygon([(530, 228), (558, 240), (530, 252)], fill=BLUE)
    card(draw, (578, 180, 912, 300), "Structure", False)
    frames.append(image)
    return frames


def structure_frames() -> list[Image.Image]:
    frames: list[Image.Image] = []
    image, draw = new_frame()
    title(draw, "Structure and interfaces")
    caption(draw, "A pair you choose.")
    card(draw, (140, 190, 420, 320), "Protein A", True)
    card(draw, (540, 190, 820, 320), "Protein B", True)
    frames.append(image)

    image, draw = new_frame()
    title(draw, "Structure and interfaces")
    caption(draw, "An experimental complex from the PDB, when one exists.")
    card(draw, (180, 180, 780, 320), "Experimental complex", True)
    frames.append(image)

    image, draw = new_frame()
    title(draw, "Structure and interfaces")
    caption(draw, "Interface residues are geometric contacts from the coordinates.")
    card(draw, (250, 180, 710, 320), "Interface residues", True)
    frames.append(image)

    image, draw = new_frame()
    title(draw, "Structure and interfaces")
    caption(draw, "Not a binding site. A missing contact is reported, not replaced.")
    card(draw, (160, 190, 800, 320), "No deposited contact", False)
    frames.append(image)
    return frames


TOUR = (
    ("Home", "Start from a disease, a protein, or your own gene list."),
    ("Disease explorer", "Targets, shared pathways, interactions, and known drugs."),
    ("Network and pathways", "Optional door: the STRING neighbourhood of one protein."),
    ("Protein", "Function, disease links, and sequence for one protein."),
    ("Structure and interfaces", "An experimental complex and geometric interface residues."),
    ("EGCG showcase", "Evidence-tiered targets, with the assay-interference caveat."),
    ("Your data", "A gene list or interaction table, then the same pathway view."),
    ("About and cite", "Methods, licences, and the source behind each number."),
)


def tour_frames() -> list[Image.Image]:
    frames: list[Image.Image] = []
    for step, (heading, line) in enumerate(TOUR, start=1):
        image, draw = new_frame()
        draw.text((48, 48), f"{step} of {len(TOUR)}", font=font(20), fill=MUTED)
        title(draw, heading)
        draw.rounded_rectangle((48, 160, 912, 360), radius=18, fill=SOFT, outline=BLUE, width=3)
        centered(draw, (80, 190, 880, 330), line, font(24), INK)
        frames.append(image)
    return frames


def save_gif(frames: list[Image.Image], path: Path) -> None:
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=1000,
        loop=0,
        optimize=True,
    )


def save_mp4(frames: list[Image.Image], path: Path) -> None:
    import imageio.v2 as imageio

    with imageio.get_writer(
        path, fps=1, codec="libx264", pixelformat="yuv420p", macro_block_size=1
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame))


def write_assets(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_gif(disease_frames(), out_dir / "disease-path.gif")
    save_gif(structure_frames(), out_dir / "structure-path.gif")
    save_mp4(tour_frames(), out_dir / "tour.mp4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ASSETS)
    args = parser.parse_args()
    write_assets(args.out)


if __name__ == "__main__":
    main()
