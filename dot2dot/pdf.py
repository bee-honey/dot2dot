"""Stage 6b: write a printable PDF (page 1 = puzzle, page 2 = answer key)."""

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from reportlab.lib.colors import HexColor, black
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas

from dot2dot.labels import DOT_RADIUS, RING_RADIUS
from dot2dot.models import Puzzle
from dot2dot.render import label_center

MARGIN = 36  # points; 72 points = 1 inch
TITLE_SPACE = 40
LEGEND_SPACE = 20
LINE_COLOR = HexColor("#d43c3c")


def pdf_bytes(puzzle: Puzzle, title: str = "Connect the Dots") -> bytes:
    """The PDF as bytes in memory, e.g. to send in an HTTP response."""
    buffer = BytesIO()
    save_pdf(puzzle, buffer, title)
    return buffer.getvalue()


def save_pdf(puzzle: Puzzle, path: str | Path | BinaryIO, title: str = "Connect the Dots") -> None:
    """Write the PDF to a file path or any binary stream."""
    canvas = Canvas(path if hasattr(path, "write") else str(path), pagesize=letter)
    _draw_page(canvas, puzzle, title, solution=False)
    canvas.showPage()
    _draw_page(canvas, puzzle, f"{title} — Answer Key", solution=True)
    canvas.showPage()
    canvas.save()


def _draw_page(canvas: Canvas, puzzle: Puzzle, title: str, solution: bool) -> None:
    page_width, page_height = letter
    has_legend = len(puzzle.strokes) > 1

    canvas.setFont("Helvetica-Bold", 18)
    canvas.drawCentredString(page_width / 2, page_height - MARGIN - 18, title)

    # Fit the puzzle into the area below the title, keeping its aspect ratio.
    bottom = MARGIN + (LEGEND_SPACE if has_legend else 0)
    box_width = page_width - 2 * MARGIN
    box_height = page_height - bottom - MARGIN - TITLE_SPACE
    scale = min(box_width / puzzle.width, box_height / puzzle.height)
    offset_x = MARGIN + (box_width - puzzle.width * scale) / 2
    offset_y = bottom + (box_height - puzzle.height * scale) / 2

    def to_page(x: float, y: float) -> tuple[float, float]:
        # PDF origin is bottom-left with y pointing up; images are top-left
        # with y pointing down, so flip the y axis.
        return offset_x + x * scale, offset_y + (puzzle.height - y) * scale

    font = puzzle.font_size * scale
    radius = DOT_RADIUS * font

    if solution:
        canvas.setStrokeColor(LINE_COLOR)
        canvas.setLineWidth(radius * 0.8)
        canvas.setLineCap(1)  # round
        for a, b in puzzle.segments():
            canvas.line(*to_page(a.x, a.y), *to_page(b.x, b.y))

    canvas.setStrokeColor(black)
    canvas.setLineWidth(radius * 0.5)
    for dot in puzzle.stroke_start_dots():
        canvas.circle(*to_page(dot.x, dot.y), RING_RADIUS * font, stroke=1, fill=0)

    canvas.setFillColor(black)
    for dot in puzzle.dots:
        canvas.circle(*to_page(dot.x, dot.y), radius, stroke=0, fill=1)
        lx, ly = to_page(*label_center(dot, puzzle.font_size))
        canvas.setFont("Helvetica-Bold" if dot.number == 1 else "Helvetica", font)
        # drawCentredString positions the baseline; nudge down to center vertically.
        canvas.drawCentredString(lx, ly - font * 0.35, str(dot.number))

    if has_legend:
        y = MARGIN + 4
        canvas.circle(MARGIN + 6, y + 3, 4, stroke=1, fill=0)
        canvas.setFont("Helvetica", 9)
        canvas.drawString(MARGIN + 16, y, "A ringed dot starts a new line: lift your pencil and continue from there.")
