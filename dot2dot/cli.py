"""Command-line entry point: `dot2dot photo.jpg --dots 60`."""

import argparse
import sys
from pathlib import Path

import cv2

from dot2dot.pdf import save_pdf
from dot2dot.pipeline import STYLES, build
from dot2dot.quality import coverage_image, evaluate
from dot2dot.render import save_svg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dot2dot", description="Turn a photo into a connect-the-dots puzzle."
    )
    parser.add_argument("image", type=Path, help="input image (jpg, png, ...)")
    parser.add_argument("--dots", type=int, default=60, help="number of dots (default: 60)")
    parser.add_argument(
        "--style",
        choices=STYLES,
        default="auto",
        help="auto: pick for you (default); lineart: cartoons and coloring pages; "
        "photo: real photos and busy backgrounds; outline: just the border",
    )
    parser.add_argument("--ai", action="store_true", help="use the AI planner (needs OPENAI_API_KEY in .env)")
    parser.add_argument("--request", default="", help='instructions for the AI planner, e.g. "easy, for a 5-year-old"')
    parser.add_argument(
        "--max-lines",
        type=int,
        help="max number of separate lines/shapes to trace (default: 1 for outline, no limit for lineart)",
    )
    parser.add_argument(
        "--out", type=Path, help="output PDF path (default: output/<image name>.pdf)"
    )
    parser.add_argument("--title", default="Connect the Dots", help="title printed on the page")
    parser.add_argument("--svg", action="store_true", help="also write puzzle and solution SVGs")
    parser.add_argument(
        "--check",
        action="store_true",
        help="also write a PNG showing which lines the puzzle covers (gray) and misses (magenta)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out or Path("output") / f"{args.image.stem}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        plan = None
        if args.ai:
            from dot2dot.planner import OpenAIPlanner
            from dot2dot.preprocess import load_image

            plan = OpenAIPlanner().plan(load_image(args.image), args.request)
            print(f"AI plan: {plan.subject} ({plan.image_kind}); suggests {plan.suggested_dots} dots; "
                  f"must include: {', '.join(p.name for p in plan.must_include) or '-'}")
        max_paths = args.max_lines or (1 if args.style == "outline" else None)
        result = build(args.image, num_dots=args.dots, max_paths=max_paths, style=args.style, plan=plan)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    puzzle = result.puzzle
    save_pdf(puzzle, out, title=args.title)
    print(f"Wrote {out} ({len(puzzle.dots)} dots, {len(puzzle.strokes)} lines)")
    print(f"Style: {result.style}")
    print(evaluate(puzzle, result.reference, result.plan).summary())

    if args.svg:
        for name, solution in [("puzzle", False), ("solution", True)]:
            svg_path = out.with_name(f"{out.stem}_{name}.svg")
            save_svg(puzzle, svg_path, solution=solution)
            print(f"Wrote {svg_path}")
    if args.check:
        check_path = out.with_name(f"{out.stem}_check.png")
        cv2.imwrite(str(check_path), coverage_image(puzzle, result.reference))
        print(f"Wrote {check_path}")
    return 0
