"""Local web app: upload an image, see the puzzle and answer key, download the PDF.

Run with `dot2dot-web` (or `uvicorn dot2dot.web.app:app --reload` while developing),
then open http://127.0.0.1:8000.
"""

import base64
import uuid
from collections import OrderedDict
from pathlib import Path
from threading import Lock

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dot2dot.pdf import pdf_bytes
from dot2dot.pipeline import STYLES, build
from dot2dot.preprocess import decode_image
from dot2dot.quality import coverage_image, evaluate
from dot2dot.render import to_svg

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MIN_DOTS, MAX_DOTS = 10, 1000
# Recent puzzles kept in memory so their PDF can be downloaded afterwards.
CACHE_SIZE = 20

app = FastAPI(title="dot2dot")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class QualityOut(BaseModel):
    overall: float
    outline_coverage: float | None
    detail_coverage: float | None
    accuracy: float
    crowded_labels: int


class PuzzleOut(BaseModel):
    id: str
    dots: int
    lines: int
    puzzle_svg: str
    solution_svg: str
    check_png: str  # data: URL, ready to drop into an <img src>
    quality: QualityOut
    pdf_url: str


class _PdfCache:
    """A small thread-safe LRU cache of generated PDFs, keyed by puzzle id."""

    def __init__(self, size: int):
        self._items: OrderedDict[str, tuple[str, bytes]] = OrderedDict()
        self._size = size
        self._lock = Lock()

    def put(self, key: str, filename: str, data: bytes) -> None:
        with self._lock:
            self._items[key] = (filename, data)
            self._items.move_to_end(key)
            while len(self._items) > self._size:
                self._items.popitem(last=False)  # evict the oldest

    def get(self, key: str) -> tuple[str, bytes] | None:
        with self._lock:
            return self._items.get(key)


pdfs = _PdfCache(CACHE_SIZE)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# A plain `def` (not `async def`) on purpose: image processing is CPU-heavy
# and blocking, so FastAPI runs it on a worker thread instead of the event loop.
@app.post("/api/puzzles", response_model=PuzzleOut)
def create_puzzle(
    image: UploadFile = File(...),
    dots: int = Form(150),
    style: str = Form("lineart"),
    max_lines: int | None = Form(None),
    title: str = Form("Connect the Dots"),
) -> PuzzleOut:
    if style not in STYLES:
        raise HTTPException(400, f"style must be one of: {', '.join(STYLES)}")
    if not MIN_DOTS <= dots <= MAX_DOTS:
        raise HTTPException(400, f"dots must be between {MIN_DOTS} and {MAX_DOTS}")

    data = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is too large (max 15 MB)")

    try:
        max_paths = max_lines or (1 if style == "outline" else None)
        result = build(decode_image(data), num_dots=dots, max_paths=max_paths, style=style)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error

    puzzle = result.puzzle
    quality = evaluate(puzzle, result.reference)
    puzzle_id = uuid.uuid4().hex
    filename = f"{Path(image.filename or 'puzzle').stem}.pdf"
    pdfs.put(puzzle_id, filename, pdf_bytes(puzzle, title=title or "Connect the Dots"))

    _, png = cv2.imencode(".png", coverage_image(puzzle, result.reference))
    return PuzzleOut(
        id=puzzle_id,
        dots=len(puzzle.dots),
        lines=len(puzzle.strokes),
        puzzle_svg=to_svg(puzzle),
        solution_svg=to_svg(puzzle, solution=True),
        check_png="data:image/png;base64," + base64.b64encode(png.tobytes()).decode(),
        quality=QualityOut(
            overall=quality.overall,
            outline_coverage=quality.outline_coverage,
            detail_coverage=quality.detail_coverage,
            accuracy=quality.accuracy,
            crowded_labels=quality.crowded_labels,
        ),
        pdf_url=f"/api/puzzles/{puzzle_id}/pdf",
    )


@app.get("/api/puzzles/{puzzle_id}/pdf")
def download_pdf(puzzle_id: str) -> Response:
    cached = pdfs.get(puzzle_id)
    if cached is None:
        raise HTTPException(404, "Puzzle not found (it may have expired; generate it again)")
    filename, data = cached
    return Response(
        data,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def main() -> None:
    import uvicorn

    print("dot2dot running at http://127.0.0.1:8000  (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=8000)
