"""Planner and plan-guidance tests. No network: a fake planner stands in for OpenAI."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from dot2dot import guidance
from dot2dot.models import Path
from dot2dot.pipeline import build, choose_style
from dot2dot.planner import Part, Plan, parse_plan
from dot2dot.web import app as web


def make_plan(must=(), ignore=(), style="lineart") -> Plan:
    return Plan(
        subject="test subject",
        image_kind="cartoon",
        style=style,
        suggested_dots=80,
        must_include=[Part(name, box) for name, box in must],
        ignore=[Part(name, box) for name, box in ignore],
    )


class FakePlanner:
    """Behaves like OpenAIPlanner without calling the API (like a Mockito mock)."""

    def __init__(self, plan: Plan | None = None, error: Exception | None = None):
        self._plan, self._error = plan, error
        self.calls = 0

    def plan(self, image, request=""):
        self.calls += 1
        if self._error:
            raise self._error
        return self._plan


def test_parse_plan_converts_and_clamps_boxes():
    plan = parse_plan({
        "subject": "penguin", "image_kind": "cartoon", "style": "lineart", "suggested_dots": 5000,
        "must_include": [{"name": "beak", "box": [400, 300, 600, 400]}, {"name": "tiny", "box": [10, 10, 11, 11]}],
        "ignore": [{"name": "caption", "box": [1200, -50, 0, 100]}],
        "notes": "",
    })
    assert plan.suggested_dots == 600  # clamped
    assert [p.name for p in plan.must_include] == ["beak"]  # degenerate box dropped
    x0, y0, x1, y1 = plan.must_include[0].box
    assert x0 < 0.4 and x1 > 0.6 and y0 < 0.3 and y1 > 0.4  # padded
    assert plan.ignore[0].box[0] == 0.0 and plan.ignore[0].box[2] == 1.0  # clamped + sorted


def line(x0, y0, x1, y1) -> Path:
    n = int(max(abs(x1 - x0), abs(y1 - y0)))
    return Path(np.column_stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)]), closed=False)


def test_apply_plan_ignores_weights_and_repairs():
    image = np.full((400, 400, 3), 255, np.uint8)
    cv2.rectangle(image, (300, 300), (360, 360), (0, 0, 0), 3)  # a "foot" the extractor missed
    caption = line(20, 20, 120, 20)
    eye = line(150, 150, 200, 150)
    body = line(50, 250, 250, 250)
    plan = make_plan(
        must=[("eye", (0.35, 0.35, 0.52, 0.40)), ("foot", (0.72, 0.72, 0.93, 0.93))],
        ignore=[("caption", (0.0, 0.0, 0.35, 0.1))],
    )
    paths = guidance.apply_plan([caption, eye, body], plan, image)
    assert caption not in paths  # ignored
    assert any(p.weight == guidance.PART_WEIGHT and p.points[0][0] == 150 for p in paths)  # eye weighted
    repaired = [p for p in paths if p.points[:, 0].min() >= 280]
    assert repaired, "the missing foot should have been repaired from local edges"


def test_large_ignore_boxes_are_skipped():
    plan = make_plan(ignore=[("background", (0.0, 0.0, 1.0, 0.6))])
    body = line(50, 100, 350, 100)
    assert guidance.apply_plan([body], plan, np.full((400, 400, 3), 255, np.uint8)) == [body]


def test_auto_style_uses_photo_for_busy_backgrounds():
    rng = np.random.default_rng(0)
    busy = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
    plain = np.full((300, 300, 3), 255, np.uint8)
    cv2.circle(plain, (150, 150), 80, (0, 0, 0), 4)
    assert choose_style(busy) == "photo"
    assert choose_style(plain) == "lineart"
    assert choose_style(plain, make_plan(style="outline")) == "lineart"  # plan can't pick outline
    assert choose_style(plain, make_plan(style="photo")) == "photo"


def test_build_with_plan_reports_part_coverage(square_image):
    from dot2dot.quality import evaluate

    plan = make_plan(must=[("top edge", (0.2, 0.2, 0.8, 0.3))])
    result = build(square_image, num_dots=30, max_paths=None, style="auto", plan=plan)
    quality = evaluate(result.puzzle, result.reference, plan)
    assert result.plan is plan
    assert quality.parts and quality.parts[0][0] == "top edge"


@pytest.fixture
def client():
    return TestClient(web.app)


def upload(client, path, **form):
    with open(path, "rb") as f:
        return client.post("/api/puzzles", files={"image": ("x.png", f, "image/png")}, data=form)


def test_web_uses_planner_when_asked(client, square_image, monkeypatch):
    fake = FakePlanner(make_plan(must=[("edge", (0.2, 0.2, 0.8, 0.3))]))
    monkeypatch.setattr(web, "get_planner", lambda: fake)
    body = upload(client, square_image, dots="30", ai="true", request="easy").json()
    assert fake.calls == 1
    assert body["plan"]["subject"] == "test subject"
    assert body["plan"]["overlay_png"].startswith("data:image/png")
    assert body["quality"]["parts"][0]["name"] == "edge"


def test_web_survives_planner_failure(client, square_image, monkeypatch):
    monkeypatch.setattr(web, "get_planner", lambda: FakePlanner(error=TimeoutError("slow")))
    response = upload(client, square_image, dots="30", ai="true")
    assert response.status_code == 200
    assert response.json()["plan"] is None
    assert "TimeoutError" in response.json()["ai_error"]


def test_web_skips_planner_unless_asked(client, square_image, monkeypatch):
    fake = FakePlanner(make_plan())
    monkeypatch.setattr(web, "get_planner", lambda: fake)
    upload(client, square_image, dots="30")
    assert fake.calls == 0
