"""Local web app: upload an image, see the puzzle and answer key, download the PDF.

Run with `dot2dot-web` (or `uvicorn dot2dot.web.app:app --reload` while developing),
then open http://127.0.0.1:8000.
"""

import base64
import json
import math

import numpy as np
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
from dot2dot.guidance import plan_overlay, apply_plan
from dot2dot.feedback import Correction, CorrectionCheck, apply_feedback, check_feedback
from dot2dot.pdf import pdf_bytes
from dot2dot.pipeline import STYLES, build, extract_paths
from dot2dot.models import Path as DrawingPath
from dot2dot.preprocess import resize
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


class EditPath(BaseModel):
    points: list[tuple[float, float]]
    closed: bool = False
    essential: bool = False
    protected: bool = False
    feature: bool = False
    weight: float = 1.0


def parse_edit_paths(raw: str, width: int, height: int) -> list[DrawingPath]:
    try:
        items = json.loads(raw)
        if not isinstance(items, list) or not 1 <= len(items) <= 2000:
            raise ValueError("Provide between 1 and 2000 paths")
        paths = []
        total_points = 0
        for item in items:
            path = EditPath.model_validate(item)
            total_points += len(path.points)
            if not 2 <= len(path.points) <= 20000 or total_points > 100000:
                raise ValueError("Too many or too few path points")
            if not math.isfinite(path.weight) or not 0.1 <= path.weight <= 10:
                raise ValueError("Invalid path weight")
            if any(not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height)
                   for x, y in path.points):
                raise ValueError("Path points must lie inside the image")
            points = np.asarray(path.points, dtype=float)
            if np.linalg.norm(np.diff(points, axis=0), axis=1).sum() < 1:
                raise ValueError("Path must contain distinct points")
            paths.append(DrawingPath(points, path.closed, path.essential, path.weight, protected=path.protected, feature=path.feature))
        return paths
    except (ValueError, TypeError) as error:
        raise HTTPException(400, f"Invalid edited paths: {error}") from error


class PuzzleOut(BaseModel):
    source_png: str
    width: int
    height: int
    edit_paths: list[EditPath]
    warnings: list[str] = []
    corrections: list[Correction] = []
    correction_checks: list[CorrectionCheck] = []
    feedback_ai_used: bool = False
    id: str
    dots: int
    printed_details: int = 0
    lines: int
    style: str  # style actually used ("auto" resolved)
    puzzle_svg: str
    solution_svg: str
    check_png: str  # data: URL, ready to drop into an <img src>
    quality: QualityOut
    pdf_url: str
    plan: PlanOut | None = None
    sketch_png: str | None = None  # raw pencil sketch of the picture (reference)
    lines_png: str | None = None  # cleaned line drawing that Sketch style traces (reference)
    ai_error: str | None = None  # set when an AI step failed and we carried on without it
    mystery_level: int = 0
    judge: str | None = None  # which judge ran the guess test: clip | openai
    guess_test: GuessTestOut | None = None
    candidates: list[CandidateOut] = []  # what auto-tune tried


class ConfigOut(BaseModel):
    ai_available: bool
    ai_model: str | None
    clip_available: bool  # local CLIP judge installed (free, no API key)
    trained_available: bool  # distilled judge (CLIP + trained head) available


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
_trained_judge = None


def trained_available() -> bool:
    from dot2dot.trained_judge import HEAD_FILE

    return CLIP_AVAILABLE and HEAD_FILE.exists()


def get_judge(kind: str = "clip"):
    """The shared judge for guess tests: trained, local CLIP (free) or OpenAI. None if unavailable."""
    global _openai_judge, _trained_judge
    if kind == "trained":
        if _trained_judge is None and trained_available():
            from dot2dot.trained_judge import TrainedJudge

            _trained_judge = TrainedJudge()
        return _trained_judge
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


def _sketch_previews(picture) -> tuple[str | None, str | None]:
    """(pencil sketch, cleaned line drawing) as data URLs; None if they can't be made."""
    from dot2dot import sketch
    from dot2dot.preprocess import resize

    try:
        image = resize(picture, 1000)
        pencil = cv2.cvtColor(sketch.pencil_sketch(image), cv2.COLOR_GRAY2BGR)
        return _png_data_url(pencil), _png_data_url(sketch.preview(image))
    except Exception:  # a reference image must never break puzzle generation
        return None, None


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
        trained_available=trained_available(),
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
    fine_details: bool = Form(False),
    detail_density: str = Form("balanced"),
    edited_paths: str | None = Form(None),
    feedback: str = Form("[]"),
    feedback_ai: bool = Form(False),
    ai: bool = Form(False),
    request: str = Form(""),
    mystery: str = Form("0"),  # "0", "1", "2" or "auto" (AI picks the level)
    guess_test: bool = Form(False),
    judge_kind: str = Form("clip", alias="judge"),  # trained | clip (local, free) | openai
) -> PuzzleOut:
    if style not in STYLES:
        raise HTTPException(400, f"style must be one of: {', '.join(STYLES)}")
    if not MIN_DOTS <= dots <= MAX_DOTS:
        raise HTTPException(400, f"dots must be between {MIN_DOTS} and {MAX_DOTS}")
    if detail_density not in ("balanced", "all"):
        raise HTTPException(400, "detail_density must be balanced or all")
    if mystery not in ("0", "1", "2", "auto"):
        raise HTTPException(400, "mystery must be 0, 1, 2 or auto")

    data = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is too large (max 15 MB)")

    try:
        picture = decode_image(data)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error

    if edited_paths is not None and len(edited_paths) > 8 * 1024 * 1024:
        raise HTTPException(413, "Edited paths are too large")

    try:
        if len(feedback) > 64000:
            raise ValueError("Feedback is too large")
        raw_feedback = json.loads(feedback)
        if not isinstance(raw_feedback, list) or len(raw_feedback) > 20:
            raise ValueError("Use at most 20 correction regions")
        corrections = [Correction.model_validate(item) for item in raw_feedback]
        if len({c.id for c in corrections}) != len(corrections):
            raise ValueError("Correction ids must be unique")
    except (ValueError, TypeError) as error:
        raise HTTPException(400, f"Invalid feedback: {error}") from error

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

    from dot2dot.pipeline import choose_style, WORKING_SIZE

    actual_style = choose_style(picture, plan) if style == "auto" else style
    working = resize(picture, WORKING_SIZE[actual_style])
    override = parse_edit_paths(edited_paths, working.shape[1], working.shape[0]) if edited_paths is not None else None

    affected = {}
    focused_plans = {}
    feedback_ai_used = False
    feedback_notes = []
    if corrections:
        # Every revision starts from the source (or explicit manual edits), then
        # reapplies the complete annotation history before global renumbering.
        try:
            source_paths = override if override is not None else extract_paths(working, actual_style)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        if plan is not None and override is None:
            source_paths = apply_plan(source_paths, plan, working)
        if feedback_ai:
            interpreter = get_planner()
            if interpreter is None:
                feedback_notes.append("AI feedback interpretation unavailable; used the selected local correction actions.")
            else:
                from dataclasses import replace
                from dot2dot.planner import Part
                for correction in corrections:
                    if correction.resolved_action() not in ("restore", "protect"):
                        continue
                    x0,y0,x1,y1 = correction.pixels(working.shape)
                    try:
                        local_plan = interpreter.plan(working[y0:y1,x0:x1],
                            "Correct this marked region: " + correction.note +
                            ". Identify the specific visible parts needed for this correction, not invented geometry.")
                        height,width = working.shape[:2]
                        def global_part(part):
                            a,b,c,d = part.box
                            return Part(part.name, ((x0+a*(x1-x0))/width, (y0+b*(y1-y0))/height,
                                                    (x0+c*(x1-x0))/width, (y0+d*(y1-y0))/height))
                        focused = replace(local_plan, must_include=[global_part(p) for p in local_plan.must_include], ignore=[])
                        focused_plans[correction.id] = focused
                        feedback_ai_used = True
                    except Exception:
                        feedback_notes.append(f"AI interpretation failed for region {correction.id}; used its local action.")
        override = source_paths
        for correction in corrections:
            override, changed = apply_feedback(override, [correction], working)
            affected.update(changed)
            if correction.id in focused_plans:
                override = apply_plan(override, focused_plans[correction.id], working)

    def build_at(level: int):
        return build(picture, num_dots=dots, max_paths=max_paths, style=actual_style, plan=plan,
                     mystery_level=level, reference_override=override, fine_details=fine_details, detail_density=detail_density)

    test, candidates = None, []
    if judge_kind not in ("trained", "clip", "openai"):
        raise HTTPException(400, "judge must be trained, clip or openai")
    judge = get_judge(judge_kind) if (guess_test or mystery == "auto") else None
    if (guess_test or mystery == "auto") and judge is None:
        ai_error = {
            "openai": "OpenAI judge unavailable: set OPENAI_API_KEY in .env",
            "trained": "Trained judge unavailable: run scripts/train_judge.py (needs the CLIP extra)",
        }.get(judge_kind, "Local CLIP judge unavailable: pip install -e '.[clip]'")
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

    sketch_png, lines_png = _sketch_previews(picture)

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

    warnings = list(feedback_notes)
    correction_checks = check_feedback(puzzle, result.reference, corrections, affected)
    if corrections:
        warnings.append("Feedback checks measure traced geometry, not whether the requested anatomy or identity is correct. Review before keeping the revision.")
    if puzzle.details and detail_density == "balanced":
        warnings.append("Balanced detail thins dense short strokes; choose All details to retain every extracted detail.")
    if puzzle.details:
        warnings.append("Fine details are printed on both sheets. This preserves small features but makes the picture easier to guess.")
    picture_dots = sum(s.end - s.start for s in puzzle.strokes if not s.filler)
    if picture_dots > dots:
        raise HTTPException(422, "The puzzle exceeded its picture dot budget; the revision was not accepted")
    if quality.outline_coverage is not None and quality.outline_coverage < 0.8:
        warnings.append("The main outline is incomplete. Do not accept this revision without reviewing the whole shape.")
    if quality.detail_coverage is not None and quality.detail_coverage < 0.65:
        warnings.append("Interior details are missing. Inspect the traced lines, protect important details, or increase picture dots.")
    for name, coverage in quality.parts:
        if coverage < 0.65:
            warnings.append(f"Important part needs attention: {name} ({coverage:.0%} traced).")
    if any(p.protected for p in result.reference):
        warnings.append("Important lines are reserved before other details; tight dot budgets or line limits may still reduce fidelity.")

    return PuzzleOut(
        source_png=_png_data_url(working), width=puzzle.width, height=puzzle.height,
        edit_paths=[EditPath(points=p.points.tolist(), closed=p.closed, essential=p.essential,
                             protected=p.protected, feature=p.feature, weight=p.weight) for p in result.reference],
        warnings=warnings, corrections=corrections, correction_checks=correction_checks,
        feedback_ai_used=feedback_ai_used,
        id=puzzle_id,
        dots=len(puzzle.dots),
        printed_details=len(puzzle.details),
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
        sketch_png=sketch_png,
        lines_png=lines_png,
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
