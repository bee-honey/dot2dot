"""Mystery mode and the AI guess test. A fake judge stands in for OpenAI."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from dot2dot import mystery
from dot2dot.judge import Guess, run_guess_test, tune_mystery
from dot2dot.models import Path
from dot2dot.pipeline import build
from dot2dot.raster import render
from dot2dot.web import app as web


def test_fillers_stay_clear_of_the_picture():
    angles = np.linspace(0, 2 * np.pi, 600, endpoint=False)
    circle = Path(np.column_stack([300 + 100 * np.cos(angles), 300 + 100 * np.sin(angles)]), closed=True, essential=True)
    fillers = mystery.add_fillers([circle], (600, 600, 3), num_dots=60, level=1, seed=b"x")
    assert fillers and all(f.filler and not f.closed for f in fillers)
    for f in fillers:
        distance = np.linalg.norm(f.points - 300, axis=1)
        assert distance.min() > 100  # level 1: never inside the subject


def test_fillers_are_repeatable_for_the_same_image():
    line = Path(np.array([(x, 300.0) for x in range(100, 500)]), closed=False)
    a = mystery.add_fillers([line], (600, 600, 3), 60, 1, seed=b"same")
    b = mystery.add_fillers([line], (600, 600, 3), 60, 1, seed=b"same")
    assert [len(f.points) for f in a] == [len(f.points) for f in b]


def test_mystery_spreads_dots_more_evenly(square_image):
    plain = build(square_image, num_dots=60, max_paths=None, style="lineart")
    hidden = build(square_image, num_dots=60, max_paths=None, style="lineart", mystery_level=1)
    spread = lambda r: mystery.dot_spread(np.array([(d.x, d.y) for d in r.puzzle.dots]), r.puzzle.width, r.puzzle.height)
    assert any(s.filler for s in hidden.puzzle.strokes)
    assert spread(hidden) > spread(plain)
    # Fillers are drawn but don't count against accuracy.
    assert len(hidden.puzzle.segments(filler=False)) < len(hidden.puzzle.segments())


class FakeJudge:
    """Pretends to be the AI: guesses from image size (puzzle pages are 1024px)."""

    def __init__(self, before="blob", after="square"):
        self.before, self.after = before, after
        self.calls = 0

    def guess(self, image):
        self.calls += 1
        return Guess("square", 0.9, [])

    def same_thing(self, truth, guesses):
        return [g == truth for g in guesses]


class ScriptedJudge(FakeJudge):
    """Returns scripted guesses in order: truth first, then (before, after) pairs."""

    def __init__(self, labels):
        super().__init__()
        self.labels = list(labels)

    def guess(self, image):
        self.calls += 1
        return Guess(self.labels.pop(0), 0.9, [])


def test_guess_test_verdicts():
    img = np.zeros((10, 10, 3), np.uint8)
    hidden = run_guess_test(ScriptedJudge(["blob", "square", "square"]), img, img, img)
    assert hidden.verdict == "mystery" and hidden.mystery_score > 0.8
    easy = run_guess_test(ScriptedJudge(["square", "square", "square"]), img, img, img)
    assert easy.verdict == "too easy"
    lost = run_guess_test(ScriptedJudge(["blob", "blob", "square"]), img, img, img)
    assert lost.verdict == "unrecognizable" and lost.mystery_score == 0


def test_tune_prefers_mystery_but_never_unrecognizable(square_image):
    # truth, then per level (2, 1, 0): (before, after)
    judge = ScriptedJudge(["square", "blob", "cat", "blob", "square", "square", "square"])
    best, tried = tune_mystery(judge, np.zeros((10, 10, 3), np.uint8),
                               lambda level: build(square_image, 40, None, "lineart", mystery_level=level))
    assert [c.test.verdict for c in tried] == ["unrecognizable", "mystery", "too easy"]
    assert best.level == 1


@pytest.fixture
def client():
    return TestClient(web.app)


def upload(client, path, **form):
    with open(path, "rb") as f:
        return client.post("/api/puzzles", files={"image": ("x.png", f, "image/png")}, data=form)


def test_web_mystery_and_guess_test(client, square_image, monkeypatch):
    monkeypatch.setattr(web, "get_judge", lambda: ScriptedJudge(["blob", "square", "square"]))
    body = upload(client, square_image, dots="40", mystery="1", guess_test="true").json()
    assert body["mystery_level"] == 1
    assert body["guess_test"]["verdict"] == "mystery"
    assert body["guess_test"]["before"]["label"] == "blob"


def test_web_auto_mystery_reports_candidates(client, square_image, monkeypatch):
    judge = ScriptedJudge(["square", "blob", "square", "blob", "square", "square", "square"])
    monkeypatch.setattr(web, "get_judge", lambda: judge)
    body = upload(client, square_image, dots="40", mystery="auto").json()
    assert [c["level"] for c in body["candidates"]] == [2, 1, 0]
    assert body["mystery_level"] == 2  # first mystery wins the tie
