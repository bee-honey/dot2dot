"""Local web app: upload an image, see the puzzle and answer key, download the PDF.

Run with `dot2dot-web` (or `uvicorn dot2dot.web.app:app --reload` while developing),
then open http://127.0.0.1:8000.
"""

import base64
import importlib.util
import uuid
from collections import OrderedDict
from pathlib import Path
from threading import Lock

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dot2dot import config
from dot2dot.guidance import plan_overlay
from dot2dot.pdf import pdf_bytes
from dot2dot.pipeline import STYLES, build
from dot2dot.judge import OpenAIJudge, judge_images, run_guess_test, tune_mystery
from dot2dot.planner import DEFAULT_OPENAI_MODEL, OpenAIPlanner
from dot2dot.raster import render
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


class PartOut(BaseModel):
    name: str
    coverage: float


class GuessOut(BaseModel):
    label: str
    confidence: float


class GuessTestOut(BaseModel):
    truth: str
    before: GuessOut
    after: GuessOut
    before_correct: bool
    after_correct: bool
    mystery_score: float
    verdict: str  # mystery | too easy | unrecognizable


class CandidateOut(BaseModel):
    level: int
    verdict: str
    mystery_score: float
    before: str
    after: str


class QualityOut(BaseModel):
    overall: float
    outline_coverage: float | None
    detail_coverage: float | None
    accuracy: float
    crowded_labels: int
    spread: float = 0.0
    parts: list[PartOut] = []


class PlanOut(BaseModel):
    subject: str
    image_kind: str
    style: str
    suggested_dots: int
    must_include: list[str]
    ignore: list[str]
    overlay_png: str  # the original image with the plan's boxes drawn on


class PuzzleOut(BaseModel):
    id: str
    dots: int
    lines: int
    style: str  # style actually used ("auto" resolved)
    puzzle_svg: str
    solution_svg: str
    check_png: str  # data: URL, ready to drop into an <img src>
    quality: QualityOut
    pdf_url: str
    plan: PlanOut | None = None
    ai_error: str | None = None  # set when an AI step failed and we carried on without it
    mystery_level: int = 0
    judge: str | None = None  # which judge ran the guess test: clip | openai
    guess_test: GuessTestOut | None = None
    candidates: list[CandidateOut] = []  # what auto-tune tried


class ConfigOut(BaseModel):
    ai_available: bool
    ai_model: str | None
    clip_available: bool  # local CLIP judge installed (free, no API key)


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
_planner: OpenAIPlanner | None = None
_openai_judge: OpenAIJudge | None = None
CLIP_AVAILABLE = importlib.util.find_spec("open_clip") is not None


def get_judge(kind: str = "clip"):
    """The shared judge for guess tests: local CLIP (free) or OpenAI. None if unavailable."""
    global _openai_judge
    if kind == "openai":
        if _openai_judge is None and config.openai_api_key():
            _openai_judge = OpenAIJudge()
        return _openai_judge
    if CLIP_AVAILABLE:
        from dot2dot.clip_judge import default_clip_judge

        return default_clip_judge()  # loaded once, then cached
    return None


def _guess_test_out(test) -> GuessTestOut:
    return GuessTestOut(
        truth=test.truth,
        before=GuessOut(label=test.before.label, confidence=test.before.confidence),
        after=GuessOut(label=test.after.label, confidence=test.after.confidence),
        before_correct=test.before_correct,
        after_correct=test.after_correct,
        mystery_score=test.mystery_score,
        verdict=test.verdict,
    )


def get_planner() -> OpenAIPlanner | None:
    """The shared AI planner (created on first use), or None without an API key."""
    global _planner
    if _planner is None and config.openai_api_key():
        _planner = OpenAIPlanner()
    return _planner


def _png_data_url(image) -> str:
    _, png = cv2.imencode(".png", image)
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config", response_model=ConfigOut)
def get_config() -> ConfigOut:
    available = config.openai_api_key() is not None
    return ConfigOut(
        ai_available=available,
        ai_model=config.openai_model(DEFAULT_OPENAI_MODEL) if available else None,
        clip_available=CLIP_AVAILABLE,
    )


# A plain `def` (not `async def`) on purpose: image processing is CPU-heavy
# and blocking, so FastAPI runs it on a worker thread instead of the event loop.
@app.post("/api/puzzles", response_model=PuzzleOut)
def create_puzzle(
    image: UploadFile = File(...),
    dots: int = Form(150),
    style: str = Form("auto"),
    max_lines: int | None = Form(None),
    title: str = Form("Connect the Dots"),
    ai: bool = Form(False),
    request: str = Form(""),
    mystery: str = Form("0"),  # "0", "1", "2" or "auto" (AI picks the level)
    guess_test: bool = Form(False),
    judge_kind: str = Form("clip", alias="judge"),  # clip (local, free) | openai
) -> PuzzleOut:
    if style not in STYLES:
        raise HTTPException(400, f"style must be one of: {', '.join(STYLES)}")
    if not MIN_DOTS <= dots <= MAX_DOTS:
        raise HTTPException(400, f"dots must be between {MIN_DOTS} and {MAX_DOTS}")
    if mystery not in ("0", "1", "2", "auto"):
        raise HTTPException(400, "mystery must be 0, 1, 2 or auto")

    data = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is too large (max 15 MB)")

    try:
        picture = decode_image(data)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error

    plan, ai_error = None, None
    if ai:
        planner = get_planner()
        if planner is None:
            ai_error = "AI planner unavailable: set OPENAI_API_KEY in .env"
        else:
            try:
                plan = planner.plan(picture, request)
            except Exception as error:  # network, quota, bad response: carry on without AI
                ai_error = f"AI planner failed, made the puzzle without it ({type(error).__name__}: {error})"[:300]

    max_paths = max_lines or (1 if style == "outline" else None)

    def build_at(level: int):
        return build(picture, num_dots=dots, max_paths=max_paths, style=style, plan=plan, mystery_level=level)

    test, candidates = None, []
    if judge_kind not in ("clip", "openai"):
        raise HTTPException(400, "judge must be clip or openai")
    judge = get_judge(judge_kind) if (guess_test or mystery == "auto") else None
    if (guess_test or mystery == "auto") and judge is None:
        ai_error = ("OpenAI judge unavailable: set OPENAI_API_KEY in .env" if judge_kind == "openai"
                    else "Local CLIP judge unavailable: pip install open_clip_torch")
    try:
        if mystery == "auto" and judge is not None:
            try:
                best, tried = tune_mystery(judge, picture, build_at)
                result, level, test = best.result, best.level, best.test
                candidates = [
                    CandidateOut(level=c.level, verdict=c.test.verdict, mystery_score=c.test.mystery_score,
                                 before=c.test.before.label, after=c.test.after.label)
                    for c in tried
                ]
            except Exception as error:  # AI failure: fall back to a fixed level
                ai_error = f"Auto-tune failed, used level 1 ({type(error).__name__}: {error})"[:300]
                result, level = build_at(1), 1
        else:
            level = 1 if mystery == "auto" else int(mystery)
            result = build_at(level)
            if guess_test and judge is not None:
                try:
                    test = run_guess_test(judge, picture, *judge_images(judge, result.puzzle))
                except Exception as error:
                    ai_error = f"Guess test failed ({type(error).__name__}: {error})"[:300]
    except ValueError as error:
        raise HTTPException(422, str(error)) from error

    puzzle = result.puzzle
    quality = evaluate(puzzle, result.reference, plan)
    puzzle_id = uuid.uuid4().hex
    filename = f"{Path(image.filename or 'puzzle').stem}.pdf"
    pdfs.put(puzzle_id, filename, pdf_bytes(puzzle, title=title or "Connect the Dots"))

    plan_out = None
    if plan is not None:
        preview = picture if max(picture.shape[:2]) <= 900 else cv2.resize(
            picture, None, fx=900 / max(picture.shape[:2]), fy=900 / max(picture.shape[:2]), interpolation=cv2.INTER_AREA
        )
        plan_out = PlanOut(
            subject=plan.subject,
            image_kind=plan.image_kind,
            style=plan.style,
            suggested_dots=plan.suggested_dots,
            must_include=[p.name for p in plan.must_include],
            ignore=[p.name for p in plan.ignore],
            overlay_png=_png_data_url(plan_overlay(preview, plan)),
        )

    return PuzzleOut(
        id=puzzle_id,
        dots=len(puzzle.dots),
        lines=len(puzzle.strokes),
        style=result.style,
        puzzle_svg=to_svg(puzzle),
        solution_svg=to_svg(puzzle, solution=True),
        check_png=_png_data_url(coverage_image(puzzle, result.reference)),
        quality=QualityOut(
            overall=quality.overall,
            outline_coverage=quality.outline_coverage,
            detail_coverage=quality.detail_coverage,
            accuracy=quality.accuracy,
            crowded_labels=quality.crowded_labels,
            spread=quality.spread,
            parts=[PartOut(name=name, coverage=c) for name, c in quality.parts],
        ),
        pdf_url=f"/api/puzzles/{puzzle_id}/pdf",
        plan=plan_out,
        ai_error=ai_error,
        mystery_level=level,
        judge=judge_kind if test else None,
        guess_test=_guess_test_out(test) if test else None,
        candidates=candidates,
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
