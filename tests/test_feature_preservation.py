"""Regressions for lost facial details and correction/export round trips."""
import json

import cv2
import numpy as np
from fastapi.testclient import TestClient

from dot2dot.lineart import extract_paths, ink_mask
from dot2dot.models import Path
from dot2dot.pipeline import assemble, build
from dot2dot.simplify import select_paths
from dot2dot.web.app import app


def test_thin_gray_eyes_are_extracted():
    image = np.full((400, 400, 3), 255, np.uint8)
    cv2.circle(image, (200, 200), 150, (0, 0, 0), 3)
    for x in (150, 250):
        cv2.ellipse(image, (x, 160), (15, 8), 0, 0, 360, (120, 120, 120), 1)
    assert ink_mask(image)[152, 150] == 255
    paths = extract_paths(image)
    for x in (150, 250):
        assert any(not p.essential and np.linalg.norm(p.points - [x, 160], axis=1).min() < 16 for p in paths)


def test_small_protected_feature_survives_large_outline_budget():
    outline = Path(np.array([[0., 0.], [900, 0], [900, 900], [0, 900]]), True, essential=True)
    eye = Path(np.array([[400., 400.], [410, 400], [405, 405]]), True, protected=True)
    assert eye in select_paths([outline, eye], 25)
    puzzle = assemble([outline, eye], 1000, 1000, 25)
    assert len(puzzle.strokes) == 2
    assert len(puzzle.dots[puzzle.strokes[1].start:puzzle.strokes[1].end]) >= 3


def picture_segments(puzzle):
    return {tuple(sorted(((round(a.x, 5), round(a.y, 5)), (round(b.x, 5), round(b.y, 5)))))
            for a, b in puzzle.segments(filler=False)}


def test_mystery_keeps_exact_completed_picture(square_image):
    plain = build(square_image, 60, None, 'lineart')
    for level in (1, 2):
        hidden = build(square_image, 60, None, 'lineart', mystery_level=level)
        assert picture_segments(hidden.puzzle) == picture_segments(plain.puzzle)
        assert any(s.filler for s in hidden.puzzle.strokes)


def test_corrected_paths_generate_downloadable_pdf(square_image):
    client = TestClient(app)
    with open(square_image, 'rb') as f:
        initial = client.post('/api/puzzles', files={'image': ('square.png', f)},
                              data={'style': 'lineart', 'dots': '30'}).json()
    edits = initial['edit_paths']
    edits.append(dict(points=[[160, 160], [180, 160], [170, 175]], closed=True, protected=True))
    with open(square_image, 'rb') as f:
        response = client.post('/api/puzzles', files={'image': ('square.png', f)},
                               data={'style': 'lineart', 'dots': '30', 'edited_paths': json.dumps(edits)})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['edit_paths'][-1]['protected']
    assert result['lines'] >= 2
    assert client.get(result['pdf_url']).content.startswith(b'%PDF')


def test_rejects_invalid_edit_geometry(square_image):
    client = TestClient(app)
    for points in ([[[-1, 0], [10, 10]]], [[[float('nan'), 0], [10, 10]]], [[[10, 10], [10, 10]]]):
        with open(square_image, 'rb') as f:
            response = client.post('/api/puzzles', files={'image': ('square.png', f)},
                                   data={'style': 'lineart', 'edited_paths': json.dumps([{'points': points[0]}])})
        assert response.status_code == 400


def test_plan_marks_eyes_as_protected():
    from dot2dot.guidance import apply_plan
    from dot2dot.planner import Part, Plan
    eye = Path(np.array([[40., 40.], [45, 40], [42, 44]]), True)
    plan = Plan('face', 'cartoon', 'lineart', 30, [Part('eye', (0.3, 0.3, 0.6, 0.6))])
    paths = apply_plan([eye], plan, np.full((100, 100, 3), 255, np.uint8))
    assert paths[0].protected
    assert paths[0] in select_paths(paths, 10)


def test_auto_mystery_rejects_lost_details(square_image, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from dot2dot import judge as module
    from dot2dot.judge import Guess, GuessTest, tune_mystery
    result = build(square_image, 30, None, 'lineart')
    monkeypatch.setattr(module, 'run_guess_test', lambda *args: GuessTest(
        'square', Guess('unknown', 0.9, []), Guess('square', 0.9, []), False, True))
    monkeypatch.setattr('dot2dot.quality.evaluate', lambda puzzle, *args: SimpleNamespace(
        parts=(('eye', 1.0 if puzzle is result.puzzle else 0.0),), detail_coverage=1.0))
    class Judge:
        def guess(self, image):
            return Guess('square', 0.9, [])
    best, _ = tune_mystery(Judge(), np.zeros((10, 10, 3), np.uint8),
                          lambda level: result if level == 0 else replace(result, puzzle=replace(result.puzzle)))
    assert best.level == 0


def test_hybrid_keeps_omitted_details_on_both_sheets():
    from dot2dot.render import to_svg
    from dot2dot.raster import render
    from dot2dot.pdf import pdf_bytes
    outline = Path(np.array([[10., 10.], [390, 10], [390, 390], [10, 390]]), True, essential=True)
    eye = Path(np.array([[170., 150.], [180, 148], [190, 150], [180, 155]]), True)
    plain = assemble([outline, eye], 400, 400, 25)
    hybrid = assemble([outline, eye], 400, 400, 25, fine_details=True)
    assert eye in hybrid.details
    assert 'class="fine-detail"' in to_svg(hybrid)
    assert 'class="fine-detail"' in to_svg(hybrid, solution=True)
    assert np.any(render(plain) != render(hybrid))
    assert pdf_bytes(hybrid).startswith(b'%PDF')


def test_hybrid_web_reports_printed_details(square_image):
    client = TestClient(app)
    edits = [dict(points=[[100,100],[300,100],[300,300],[100,300]],closed=True,essential=True),
             dict(points=[[160,160],[170,155],[180,160]],closed=False)]
    with open(square_image,'rb') as f:
        response = client.post('/api/puzzles', files={'image': ('square.png',f)},
                               data={'style':'lineart','dots':'25','fine_details':'true','edited_paths':json.dumps(edits)})
    assert response.status_code == 200
    body = response.json()
    assert body['printed_details'] == 1
    assert 'fine-detail' in body['puzzle_svg'] and 'fine-detail' in body['solution_svg']
    assert any('easier to guess' in w for w in body['warnings'])


def test_ironman_small_parts_are_preserved_when_sample_available():
    from pathlib import Path as FilePath
    import pytest
    sample = FilePath('samples/private/ironman.gif')
    if not sample.exists():
        pytest.skip('Private reference image is not committed')
    result = build(sample, 300, None, 'lineart', fine_details=True)
    for x0,y0,x1,y1 in [(350,130,510,320),(0,735,130,945),(375,920,490,998),(740,880,946,980)]:
        assert any(((p.points[:,0]>=x0)&(p.points[:,0]<=x1)&(p.points[:,1]>=y0)&(p.points[:,1]<=y1)).mean()>.5
                   for p in result.puzzle.details)
