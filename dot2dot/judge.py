"""AI guess test: can someone tell what the picture is before solving it?

A good mystery puzzle is hard to guess from the dots alone and obvious once
the dots are connected. We measure exactly that with a vision model:

1. Blind guesses: the model sees one image at a time with no hints:
   the original picture (ground truth), the unsolved puzzle page, and the
   answer key. For each it names what it sees, with a confidence.
2. Grading: a text-only call decides whether each guess names the same
   thing as the original (synonyms OK; "bird" for a penguin is NOT a match).

mystery score = P(recognized after solving) x P(NOT recognized before solving)
"""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from dot2dot import config
from dot2dot.planner import DEFAULT_OPENAI_MODEL, _data_url


@dataclass(frozen=True)
class Guess:
    label: str  # what the model thinks it is, e.g. "penguin"
    confidence: float  # 0..1
    alternatives: list[str]


@dataclass(frozen=True)
class GuessTest:
    truth: str
    before: Guess  # from the unsolved puzzle page
    after: Guess  # from the answer key
    before_correct: bool
    after_correct: bool

    @property
    def mystery_score(self) -> float:
        """High when the picture only appears once solved (0..1)."""
        p_after = self.after.confidence if self.after_correct else 0.0
        p_before = self.before.confidence if self.before_correct else 0.0
        return p_after * (1 - p_before)

    @property
    def verdict(self) -> str:
        if self.after_correct and not self.before_correct:
            return "mystery"  # the goal
        if self.after_correct:
            return "too easy"  # guessable before solving
        return "unrecognizable"  # not even the solution shows it

    def to_dict(self) -> dict:
        def guess(g: Guess) -> dict:
            return {"label": g.label, "confidence": g.confidence, "alternatives": g.alternatives}

        return {
            "truth": self.truth,
            "before": guess(self.before),
            "after": guess(self.after),
            "before_correct": self.before_correct,
            "after_correct": self.after_correct,
            "mystery_score": self.mystery_score,
            "verdict": self.verdict,
        }


class Judge(Protocol):
    def guess(self, image: np.ndarray) -> Guess: ...

    def same_thing(self, truth: str, guesses: list[str]) -> list[bool]: ...


def judge_images(judge, puzzle) -> tuple[np.ndarray, np.ndarray]:
    """The unsolved page and answer key, rendered the way this judge reads best.

    The unsolved page always shows numbers (that's what a kid sees). Some
    judges (CLIP) misread the answer key when digits are printed all over the
    drawing, so they set `answer_key_numbers = False`.
    """
    from dot2dot.raster import render

    numbers = getattr(judge, "answer_key_numbers", True)
    return render(puzzle), render(puzzle, solution=True, numbers=numbers)


def run_guess_test(
    judge: Judge, original: np.ndarray, puzzle_page: np.ndarray, answer_key: np.ndarray, truth: Guess | None = None
) -> GuessTest:
    """Blind-guess the original, the unsolved page and the answer key, then grade.

    Pass `truth` (a previous guess of the original) to skip guessing it again.
    Judges with their own decision logic (a trained model) provide `judge_puzzle`.
    """
    if hasattr(judge, "judge_puzzle"):
        return judge.judge_puzzle(original, puzzle_page, answer_key, truth)
    images = [puzzle_page, answer_key] + ([] if truth else [original])
    # The image guesses are independent, so ask them in parallel threads.
    with ThreadPoolExecutor(max_workers=3) as pool:
        guesses = list(pool.map(judge.guess, images))
    before, after = guesses[0], guesses[1]
    truth = truth or guesses[2]
    before_ok, after_ok = judge.same_thing(truth.label, [before.label, after.label])
    return GuessTest(truth.label, before, after, before_ok, after_ok)


@dataclass(frozen=True)
class Candidate:
    level: int
    result: object  # pipeline.Result (kept untyped here to avoid a circular import)
    test: GuessTest


def tune_mystery(judge: Judge, original: np.ndarray, build_at_level, levels=(2, 1, 0)) -> tuple[Candidate, list[Candidate]]:
    """Try several mystery levels; keep the best puzzle according to the guess test.

    `build_at_level(level)` makes the puzzle for one level. The winner is the
    highest mystery score; ties go to the earlier (more mysterious) level.
    If nothing is recognizable once solved, the faithful level-0 puzzle wins.
    """
    truth = judge.guess(original)
    candidates = []
    for level in levels:
        result = build_at_level(level)
        test = run_guess_test(judge, original, *judge_images(judge, result.puzzle), truth)
        candidates.append(Candidate(level, result, test))
    recognizable = [c for c in candidates if c.test.after_correct]
    pool = recognizable or [c for c in candidates if c.level == 0] or candidates
    best = max(pool, key=lambda c: c.test.mystery_score)
    return best, candidates


GUESS_INSTRUCTIONS = """\
You will see one image: a photo, a drawing, or a connect-the-dots puzzle
(numbered dots, possibly already connected with lines). Name the SUBJECT the
image depicts, never the medium: don't answer "dot puzzle", "drawing" or
"pattern". Be as specific as you can: a known character's name ("Spider-Man",
"Hulk"), otherwise a short common noun phrase ("penguin", "smiley face").
For unconnected dots, say what the dots look like they form. If you
genuinely can't tell, answer "unknown" with a low confidence.
Give a confidence from 0 to 1 and up to 3 alternative guesses.
"""

GUESS_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "confidence": {"type": "number"},
        "alternatives": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["label", "confidence", "alternatives"],
    "additionalProperties": False,
}

GRADE_INSTRUCTIONS = """\
Decide whether each guess names the same subject as the truth. Synonyms and
more/less specific names of the SAME thing count ("Spider-Man" vs
"superhero in a spider suit": yes). A broader category does NOT count
("bird" for "penguin": no; "animal" for "dog": no). "unknown" never counts.
"""

GRADE_SCHEMA = {
    "type": "object",
    "properties": {"matches": {"type": "array", "items": {"type": "boolean"}}},
    "required": ["matches"],
    "additionalProperties": False,
}


class OpenAIJudge:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        from openai import OpenAI

        api_key = api_key or config.openai_api_key()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (add it to your .env file)")
        self.model = model or config.openai_model(DEFAULT_OPENAI_MODEL)
        self._client = OpenAI(api_key=api_key)
        # Running token totals, to report what a batch of labelling cost.
        self.input_tokens = 0
        self.output_tokens = 0

    def _track(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.input_tokens += usage.input_tokens or 0
            self.output_tokens += usage.output_tokens or 0

    def guess(self, image: np.ndarray) -> Guess:
        response = self._client.responses.create(
            model=self.model,
            instructions=GUESS_INSTRUCTIONS,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What does this picture show?"},
                    {"type": "input_image", "image_url": _data_url(image), "detail": "high"},
                ],
            }],
            text={"format": {"type": "json_schema", "name": "guess", "schema": GUESS_SCHEMA, "strict": True}},
            temperature=0,  # same image, same answer: needed to compare settings fairly
        )
        self._track(response)
        data = json.loads(response.output_text)
        return Guess(data["label"].strip(), float(min(max(data["confidence"], 0), 1)), data["alternatives"][:3])

    def same_thing(self, truth: str, guesses: list[str]) -> list[bool]:
        response = self._client.responses.create(
            model=self.model,
            instructions=GRADE_INSTRUCTIONS,
            input=json.dumps({"truth": truth, "guesses": guesses}),
            text={"format": {"type": "json_schema", "name": "grade", "schema": GRADE_SCHEMA, "strict": True}},
            temperature=0,
        )
        self._track(response)
        matches = json.loads(response.output_text)["matches"]
        return [bool(m) for m in (matches + [False] * len(guesses))[: len(guesses)]]
