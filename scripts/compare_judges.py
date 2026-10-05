"""Compare the local CLIP judge with the OpenAI judge on the sample puzzles.

For every sample and mystery level, both judges run the blind guess test.
Prints how often they agree, and saves every result (including the LLM's
verdicts) to a JSON file, the first labelled data for training a judge.

    python scripts/compare_judges.py
    python scripts/compare_judges.py --levels 0 2 --dots 200
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dot2dot.clip_judge import default_clip_judge
from dot2dot.judge import OpenAIJudge, judge_images, run_guess_test
from dot2dot.pipeline import build
from dot2dot.preprocess import load_image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path, default=Path("samples/private"))
    parser.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--dots", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("output/eval/judges.json"))
    args = parser.parse_args(argv)

    clip, llm = default_clip_judge(), OpenAIJudge()
    rows = []
    images = sorted(p for p in args.samples.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".heic"})
    for path in images:
        original = load_image(path)
        clip_truth, llm_truth = clip.guess(original), llm.guess(original)
        for level in args.levels:
            result = build(original, num_dots=args.dots, max_paths=None, style="auto", mystery_level=level)
            page, key = judge_images(clip, result.puzzle)

            started = time.perf_counter()
            c = run_guess_test(clip, original, page, key, truth=clip_truth)
            clip_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            o = run_guess_test(llm, original, *judge_images(llm, result.puzzle), truth=llm_truth)
            llm_ms = (time.perf_counter() - started) * 1000

            # CLIP's continuous view: rank of the true label before/after (1 = top guess).
            rank_before, p_before = clip.truth_rank(page, clip_truth.label)
            rank_after, p_after = clip.truth_rank(key, clip_truth.label)
            rows.append({
                "image": path.name, "level": level,
                "clip": c.to_dict() | {"ms": round(clip_ms), "rank_before": rank_before, "rank_after": rank_after,
                                       "p_before": p_before, "p_after": p_after},
                "llm": o.to_dict() | {"ms": round(llm_ms)},
            })
            print(f"{path.name[:22]:<24} L{level}  CLIP {c.verdict:<15} (before #{rank_before:<3} after #{rank_after:<3}) "
                  f"| LLM {o.verdict:<15} | before: {c.before.label!r} vs {o.before.label!r}; after: {c.after.label!r} vs {o.after.label!r}")

    def agree(key: str) -> float:
        return sum(r["clip"][key] == r["llm"][key] for r in rows) / len(rows)

    print(f"\n{len(rows)} puzzles judged.")
    print(f"Agreement with the LLM judge: verdict {agree('verdict'):.0%}, "
          f"'guessable before' {agree('before_correct'):.0%}, 'recognizable after' {agree('after_correct'):.0%}")
    print(f"Average time per guess test: CLIP {sum(r['clip']['ms'] for r in rows) / len(rows):.0f} ms, "
          f"LLM {sum(r['llm']['ms'] for r in rows) / len(rows):.0f} ms")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
