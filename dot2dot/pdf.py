"""Stage 6b: write a printable PDF (page 1 = puzzle, page 2 = answer key)."""

from pathlib import Path

from reportlab.lib.colors import HexColor, black
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas

from dot2dot.models import Puzzle

MARGIN = 36  # points; 72 points = 1 inch
TITLE_SPACE = 40
DOT_RADIUS = 1.8
FONT_SIZE = 7
LABEL_OFFSET = DOT_RADIUS + FONT_SIZE * 0.75
LINE_COLOR = HexColor("#d43c3c")


def save_pdf(puzzle: Puzzle, path: str | Path, title: str = "Connect the Dots") -> None:
    canvas = Canvas(str(path), pagesize=letter)
    _draw_page(canvas, puzzle, title, solution=False)
    canvas.showPage()
    _draw_page(canvas, puzzle, f"{title} — Answer Key", solution=True)
    canvas.showPage()
    canvas.save()


def _draw_page(canvas: Canvas, puzzle: Puzzle, title: str, solution: bool) -> None:
    page_width, page_height = letter

    canvas.setFont("Helvetica-Bold", 18)
    canvas.drawCentredString(page_width / 2, page_height - MARGIN - 18, title)

    # Fit the puzzle into the area below the title, keeping its aspect ratio.
    box_width = page_width - 2 * MARGIN
    box_height = page_height - 2 * MARGIN - TITLE_SPACE
    scale = min(box_width / puzzle.width, box_height / puzzle.height)
    offset_x = MARGIN + (box_width - puzzle.width * scale) / 2
    offset_y = MARGIN + (box_height - puzzle.height * scale) / 2

    def to_page(x: float, y: float) -> tuple[float, float]:
        # PDF origin is bottom-left with y pointing up; images are top-left
        # with y pointing down, so flip the y axis.
        return offset_x + x * scale, offset_y + (puzzle.height - y) * scale

    if solution:
        canvas.setStrokeColor(LINE_COLOR)
        canvas.setLineWidth(0.8)
        for a, b in puzzle.segments():
            canvas.line(*to_page(a.x, a.y), *to_page(b.x, b.y))

    canvas.setFillColor(black)
    for dot in puzzle.dots:
        px, py = to_page(dot.x, dot.y)
        canvas.circle(px, py, DOT_RADIUS, stroke=0, fill=1)

        # Flip label_dy for the same reason as in to_page.
        lx = px + dot.label_dx * LABEL_OFFSET
        ly = py - dot.label_dy * LABEL_OFFSET
        canvas.setFont("Helvetica-Bold" if dot.number == 1 else "Helvetica", FONT_SIZE)
        # drawCentredString positions the baseline; nudge down to center vertically.
        canvas.drawCentredString(lx, ly - FONT_SIZE * 0.35, str(dot.number))
