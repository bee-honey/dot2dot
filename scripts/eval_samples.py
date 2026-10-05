"""Run every sample image through the pipeline and save a comparison grid.

A visual regression suite: after changing the pipeline, run this and compare
the grid (and the printed scores) with the previous run.

    python scripts/eval_samples.py                       # samples/private, all styles
    python scripts/eval_samples.py --styles lineart photo --dots 150
    python scripts/eval_samples.py --out output/eval/after.png
    python scripts/eval_samples.py --styles auto --ai     # also run with the AI planner
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from dot2dot.models import Puzzle
from dot2dot.pipeline import STYLES, build
from dot2dot.guidance import plan_overlay
from dot2dot.planner import OpenAIPlanner
from dot2dot.preprocess import load_image
from dot2dot.quality import evaluate

TILE = 360
CAPTION = 40


def render_solution(puzzle: Puzzle, size: int = TILE) -> np.ndarray:
    """A quick raster preview of the answer key (lines + dots, no numbers)."""
    scale = size / max(puzzle.width, puzzle.height)
    canvas = np.full((size, size, 3), 255, np.uint8)

    def at(dot):
        return round(dot.x * scale), round(dot.y * scale)

    for a, b in puzzle.segments():
        cv2.line(canvas, at(a), at(b), (60, 60, 212), 1, cv2.LINE_AA)
    for dot in puzzle.dots:
        cv2.circle(canvas, at(dot), 2 if not dot.starts_stroke else 3, (0, 0, 0), -1, cv2.LINE_AA)
    return canvas


def tile(image: np.ndarray, caption: str, subcaption: str = "") -> np.ndarray:
    out = np.full((TILE + CAPTION, TILE, 3), 255, np.uint8)
    h, w = image.shape[:2]
    scale = TILE / max(h, w)
    image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
    out[: image.shape[0], : image.shape[1]] = image
    cv2.putText(out, caption[:44], (4, TILE + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(out, subcaption[:52], (4, TILE + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (TILE - 1, TILE + CAPTION - 1), (220, 220, 220), 1)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path, default=Path("samples/private"))
    parser.add_argument("--styles", nargs="+", choices=STYLES, default=list(STYLES))
    parser.add_argument("--dots", type=int, default=150)
    parser.add_argument("--out", type=Path, default=Path("output/eval/grid.png"))
    parser.add_argument("--ai", action="store_true", help="add a column per style using the AI planner")
    args = parser.parse_args(argv)
    planner = OpenAIPlanner() if args.ai else None

    images = sorted(p for p in args.samples.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".heic", ".webp"})
    if not images:
        print(f"No images in {args.samples}", file=sys.stderr)
        return 1

    rows = []
    for path in images:
        image = load_image(path)
        row = [tile(image, path.name, "original")]
        plan = planner.plan(image) if planner else None
        if plan:
            row.append(tile(plan_overlay(image, plan), f"plan: {plan.subject}", f"{plan.image_kind}, {plan.style}, {plan.suggested_dots} dots"))
        runs = [(style, None) for style in args.styles] + ([(style, plan) for style in args.styles] if plan else [])
        for style, run_plan in runs:
            label = style + (" + AI" if run_plan else "")
            started = time.perf_counter()
            try:
                result = build(image, num_dots=args.dots, max_paths=1 if style == "outline" else None, style=style, plan=run_plan)
                quality = evaluate(result.puzzle, result.reference, run_plan)
                seconds = time.perf_counter() - started
                summary = f"{label} ({result.style}): {quality.overall:.0f}/100, {len(result.puzzle.dots)} dots"
                detail = f"{seconds:.1f}s, {len(result.puzzle.strokes)} lines, crowded {quality.crowded_labels}"
                if quality.parts:
                    missing = [name for name, c in quality.parts if c < 0.5]
                    detail += f", missing: {', '.join(missing) or 'none'}"
                row.append(tile(render_solution(result.puzzle), summary, detail))
            except Exception as error:  # keep going: one bad sample shouldn't stop the run
                summary = f"{label}: FAILED"
                row.append(tile(np.full((TILE, TILE, 3), 245, np.uint8), summary, str(error)))
            print(f"{path.name:<40} {summary}")
        rows.append(np.hstack(row))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.vstack(rows))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
