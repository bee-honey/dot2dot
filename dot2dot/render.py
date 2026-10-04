"""Stage 6a: render a Puzzle as SVG (handy for previewing in a browser)."""

from pathlib import Path

from dot2dot.models import Puzzle

LINE_COLOR = "#d43c3c"


def to_svg(puzzle: Puzzle, solution: bool = False) -> str:
    """Build an SVG document. With `solution=True` the connecting lines are drawn."""
    size = max(puzzle.width, puzzle.height)
    radius = size / 250
    font_size = size / 70
    label_offset = radius + font_size * 0.7

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {puzzle.width} {puzzle.height}" '
        f'width="{puzzle.width}" height="{puzzle.height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    if solution:
        for a, b in puzzle.segments():
            parts.append(
                f'<line x1="{a.x:.1f}" y1="{a.y:.1f}" x2="{b.x:.1f}" y2="{b.y:.1f}" '
                f'stroke="{LINE_COLOR}" stroke-width="{radius * 0.6:.2f}"/>'
            )

    for dot in puzzle.dots:
        parts.append(f'<circle cx="{dot.x:.1f}" cy="{dot.y:.1f}" r="{radius:.2f}" fill="black"/>')
        lx = dot.x + dot.label_dx * label_offset
        ly = dot.y + dot.label_dy * label_offset
        weight = "bold" if dot.number == 1 else "normal"
        parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="{font_size:.1f}" '
            f'font-family="Helvetica, Arial, sans-serif" font-weight="{weight}" '
            f'text-anchor="middle" dominant-baseline="central">{dot.number}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def save_svg(puzzle: Puzzle, path: str | Path, solution: bool = False) -> None:
    Path(path).write_text(to_svg(puzzle, solution), encoding="utf-8")
