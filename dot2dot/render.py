"""Stage 6a: render a Puzzle as SVG (handy for previewing in a browser)."""

from pathlib import Path

from dot2dot.labels import DIGIT_HEIGHT, DIGIT_WIDTH, DOT_RADIUS, GAP, RING_RADIUS
from dot2dot.models import Dot, Puzzle

LINE_COLOR = "#d43c3c"
FILLER_COLOR = "#8cb4e1"  # background filler lines (mystery mode)


def label_center(dot: Dot, font_size: float) -> tuple[float, float]:
    """Where to center a dot's number, matching the box labels.py planned for."""
    half_w = DIGIT_WIDTH * font_size * len(str(dot.number)) / 2
    half_h = DIGIT_HEIGHT * font_size / 2
    reach = abs(dot.label_dx) * half_w + abs(dot.label_dy) * half_h
    radius = RING_RADIUS if dot.starts_stroke else DOT_RADIUS
    distance = (radius + GAP) * font_size + reach
    return dot.x + dot.label_dx * distance, dot.y + dot.label_dy * distance


def to_svg(puzzle: Puzzle, solution: bool = False) -> str:
    """Build an SVG document. With `solution=True` the connecting lines are drawn."""
    font = puzzle.font_size
    radius = DOT_RADIUS * font

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {puzzle.width} {puzzle.height}" '
        f'width="{puzzle.width}" height="{puzzle.height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    if solution:
        for filler, color in ((True, FILLER_COLOR), (False, LINE_COLOR)):
            for a, b in puzzle.segments(filler=filler):
                parts.append(
                    f'<line x1="{a.x:.1f}" y1="{a.y:.1f}" x2="{b.x:.1f}" y2="{b.y:.1f}" '
                    f'stroke="{color}" stroke-width="{radius * 0.8:.2f}" stroke-linecap="round"/>'
                )

    # A ring marks where a new line starts (lift the pencil before this dot).
    for dot in puzzle.stroke_start_dots():
        parts.append(
            f'<circle cx="{dot.x:.1f}" cy="{dot.y:.1f}" r="{RING_RADIUS * font:.2f}" '
            f'fill="none" stroke="black" stroke-width="{radius * 0.5:.2f}"/>'
        )

    for dot in puzzle.dots:
        parts.append(f'<circle cx="{dot.x:.1f}" cy="{dot.y:.1f}" r="{radius:.2f}" fill="black"/>')
        lx, ly = label_center(dot, font)
        weight = "bold" if dot.number == 1 else "normal"
        parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="{font:.1f}" '
            f'font-family="Helvetica, Arial, sans-serif" font-weight="{weight}" '
            f'text-anchor="middle" dominant-baseline="central">{dot.number}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def save_svg(puzzle: Puzzle, path: str | Path, solution: bool = False) -> None:
    Path(path).write_text(to_svg(puzzle, solution), encoding="utf-8")
