"""Build a labelled dataset of puzzles from Quick, Draw! doodles.

For each doodle: make puzzles (mystery level 0 and a random 1 or 2), save the
unsolved page and answer key, then (with --label) have the OpenAI judge make
blind guesses. The true answer is the doodle's category, so each puzzle gets
two labels: "guessable before solving" and "recognizable after solving".

Resumable: re-running skips puzzles that are already labelled.

    python scripts/make_dataset.py --categories 100 --per-category 3            # images only (free)
    python scripts/make_dataset.py --categories 100 --per-category 3 --label    # + LLM labels (~$4)
"""

import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

from dot2dot import quickdraw
from dot2dot.judge import Guess, OpenAIJudge, run_guess_test
from dot2dot.pipeline import assemble
from dot2dot.raster import render

# Assumed gpt-4.1 prices (USD per million tokens), only for the cost estimate.
PRICE_IN, PRICE_OUT = 2.0, 8.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--categories", type=int, default=100, help="how many random categories")
    parser.add_argument("--per-category", type=int, default=3, help="doodles per category")
    parser.add_argument("--out", type=Path, default=Path("output/dataset"))
    parser.add_argument("--label", action="store_true", help="label with the OpenAI judge (costs money)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, help="label at most this many puzzles this run")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(quickdraw.categories(), args.categories))
    images_dir = args.out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    index_file = args.out / "puzzles.jsonl"
    existing = {}
    if index_file.exists():
        for line in index_file.read_text().splitlines():
            row = json.loads(line)
            existing[row["id"]] = row

    # 1. Make puzzles (free, local).
    rows = []
    for category in chosen:
        try:
            drawings = quickdraw.load_drawings(category, args.per_category)
        except Exception as error:
            print(f"skip {category}: {error}", file=sys.stderr)
            continue
        for drawing in drawings:
            paths = quickdraw.to_paths(drawing["drawing"])
            for level in (0, rng.choice((1, 2))):
                puzzle_id = f"{drawing['key_id']}_m{level}"
                if puzzle_id in existing:
                    rows.append(existing[puzzle_id])
                    continue
                dots = rng.randint(50, 150)
                try:
                    puzzle = assemble(paths, quickdraw.CANVAS, quickdraw.CANVAS, dots, None, level, drawing["key_id"].encode())
                except ValueError:
                    continue
                for name, image in (
                    ("page", render(puzzle, 768)),
                    ("key", render(puzzle, 768, solution=True)),
                    ("key_plain", render(puzzle, 768, solution=True, numbers=False)),
                ):
                    cv2.imwrite(str(images_dir / f"{puzzle_id}_{name}.png"), image)
                rows.append({"id": puzzle_id, "category": category, "level": level, "dots": len(puzzle.dots),
                             "fillers": sum(s.filler for s in puzzle.strokes)})
    print(f"{len(rows)} puzzles from {len(chosen)} categories in {images_dir}")

    # 2. Label with the LLM judge (blind guesses, graded against the category).
    if args.label:
        judge = OpenAIJudge()
        todo = [r for r in rows if "llm" not in r][: args.limit]
        print(f"Labelling {len(todo)} puzzles with {judge.model} ({args.workers} at a time)...")
        lock = threading.Lock()

        def label(row: dict) -> dict:
            page = cv2.imread(str(images_dir / f"{row['id']}_page.png"))
            key = cv2.imread(str(images_dir / f"{row['id']}_key.png"))
            test = run_guess_test(judge, page, page, key, truth=Guess(row["category"], 1.0, []))
            row["llm"] = test.to_dict()
            return row

        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(label, r) for r in todo]):
                try:
                    future.result()
                except Exception as error:  # keep going; unlabelled rows are retried next run
                    print(f"  failed: {error}", file=sys.stderr)
                with lock:
                    done += 1
                    if done % 50 == 0 or done == len(todo):
                        cost = judge.input_tokens / 1e6 * PRICE_IN + judge.output_tokens / 1e6 * PRICE_OUT
                        print(f"  {done}/{len(todo)} labelled, ~${cost:.2f} so far")
                        index_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    index_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    labelled = [r for r in rows if "llm" in r]
    if labelled:
        verdicts = {}
        for r in labelled:
            verdicts[r["llm"]["verdict"]] = verdicts.get(r["llm"]["verdict"], 0) + 1
        print(f"{len(labelled)} labelled: " + ", ".join(f"{k} {v}" for k, v in sorted(verdicts.items())))
    print(f"Wrote {index_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
