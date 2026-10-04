"""Command-line entry point: `dot2dot photo.jpg --dots 60`."""

import argparse
import sys
from pathlib import Path

from dot2dot.pdf import save_pdf
from dot2dot.pipeline import generate
from dot2dot.render import save_svg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dot2dot", description="Turn a photo into a connect-the-dots puzzle."
    )
    parser.add_argument("image", type=Path, help="input image (jpg, png, ...)")
    parser.add_argument("--dots", type=int, default=60, help="number of dots (default: 60)")
    parser.add_argument(
        "--outlines", type=int, default=1, help="max number of separate shapes to trace (default: 1)"
    )
    parser.add_argument(
        "--out", type=Path, help="output PDF path (default: output/<image name>.pdf)"
    )
    parser.add_argument("--title", default="Connect the Dots", help="title printed on the page")
    parser.add_argument("--svg", action="store_true", help="also write puzzle and solution SVGs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out or Path("output") / f"{args.image.stem}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        puzzle = generate(args.image, num_dots=args.dots, max_outlines=args.outlines)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    save_pdf(puzzle, out, title=args.title)
    print(f"Wrote {out} ({len(puzzle.dots)} dots)")

    if args.svg:
        for name, solution in [("puzzle", False), ("solution", True)]:
            svg_path = out.with_name(f"{out.stem}_{name}.svg")
            save_svg(puzzle, svg_path, solution=solution)
            print(f"Wrote {svg_path}")
    return 0
