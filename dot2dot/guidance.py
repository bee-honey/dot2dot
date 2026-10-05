"""Apply an AI plan to the extracted paths. All deterministic; no AI calls here.

- ignore:       drop paths (and silhouette parts) lying inside "ignore" boxes,
                such as captions, logos and watermarks.
- must_include: lines inside these boxes get extra weight, so parts like
                eyes and beaks earn enough dots to be recognizable.
- repair:       if a must-include box has (almost) no lines at all, the
                extractor missed that part (e.g. orange feet with no dark
                outline). Run a local edge detector in that box and add what it finds.

Must-include boxes win over ignore boxes where they overlap.
"""

from dataclasses import replace

import cv2
import numpy as np
from skimage.morphology import skeletonize

from dot2dot.models import Path
from dot2dot.planner import Plan, to_pixels
from dot2dot.skeleton import trace_skeleton

PART_WEIGHT = 2.5
# A part counts as missing (and gets repaired) only if its box contains less
# than this many pixels of line per pixel of box perimeter: nearly empty.
MIN_PART_LINES = 0.15
# Repair: ignore edge fragments shorter than this (pixels), and add at most
# this much line (as a multiple of the box perimeter), longest pieces first.
MIN_REPAIR_LENGTH = 20
MAX_REPAIR_LINES = 1.5
# A path is "in" a box when at least this share of its points are.
INSIDE_SHARE = 0.5
IGNORE_SHARE = 0.7
# Ignore boxes larger than this share of the image are skipped: they're
# "background" regions that overlap the subject. Small boxes (captions,
# logos, watermarks) are the useful ones.
MAX_IGNORE_AREA = 0.15


def apply_plan(paths: list[Path], plan: Plan, image: np.ndarray) -> list[Path]:
    must = [to_pixels(p.box, image.shape) for p in plan.must_include]
    ignore = [
        to_pixels(p.box, image.shape)
        for p in plan.ignore
        if (p.box[2] - p.box[0]) * (p.box[3] - p.box[1]) <= MAX_IGNORE_AREA
    ]

    paths = [p for p in paths if not _ignored(p, ignore, must)]
    paths = [replace(p, weight=max(p.weight, PART_WEIGHT)) if _share_in(p, must) >= INSIDE_SHARE else p for p in paths]

    for box in must:
        if _line_amount(paths, box) < MIN_PART_LINES * _perimeter(box):
            found = [p for p in repair_lines(image, box) if not _ignored(p, ignore, [])]
            budget = MAX_REPAIR_LINES * _perimeter(box)
            for piece in sorted(found, key=lambda p: p.length(), reverse=True):
                if budget <= 0:
                    break
                paths.append(replace(piece, weight=PART_WEIGHT))
                budget -= piece.length()
    return paths


def repair_lines(image: np.ndarray, box: tuple[int, int, int, int]) -> list[Path]:
    """Edges found by a local edge detector inside `box`, as paths in image coordinates."""
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    gray = cv2.bilateralFilter(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 9, 40, 7)
    # Canny thresholds from the crop's own median brightness ("auto Canny").
    median = float(np.median(gray))
    edges = cv2.Canny(gray, int(max(0, 0.66 * median)), int(min(255, 1.33 * median)))
    # Close tiny gaps so outlines come out as continuous strokes.
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    paths = []
    for path in trace_skeleton(skeletonize(edges > 0)):
        if path.length() >= MIN_REPAIR_LENGTH:
            paths.append(replace(path, points=path.points + (x0, y0)))
    return paths


def _inside(points: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return (points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1)


def _share_in(path: Path, boxes: list[tuple[int, int, int, int]]) -> float:
    if not boxes:
        return 0.0
    inside = np.zeros(len(path.points), dtype=bool)
    for box in boxes:
        inside |= _inside(path.points, box)
    return float(inside.mean())


def _ignored(path: Path, ignore, must) -> bool:
    return _share_in(path, ignore) >= IGNORE_SHARE and _share_in(path, must) < INSIDE_SHARE


def _line_amount(paths: list[Path], box) -> int:
    """Roughly how many pixels of line fall inside `box` (paths are ~1 point per pixel)."""
    return int(sum(_inside(p.points, box).sum() for p in paths))


def _perimeter(box) -> int:
    x0, y0, x1, y1 = box
    return 2 * ((x1 - x0) + (y1 - y0))


def part_coverage(plan: Plan, solution_mask: np.ndarray, reference: list[Path], tolerance: float) -> list[tuple[str, float]]:
    """For each must-include part: the share of its lines the solution traces (0 if no lines found)."""
    to_solution = cv2.distanceTransform(np.where(solution_mask > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 5)
    height, width = solution_mask.shape
    results = []
    for part in plan.must_include:
        box = to_pixels(part.box, solution_mask.shape)
        points = np.concatenate([p.points[_inside(p.points, box)] for p in reference] or [np.zeros((0, 2))])
        if len(points) == 0:
            results.append((part.name, 0.0))
            continue
        xs = np.clip(points[:, 0].astype(int), 0, width - 1)
        ys = np.clip(points[:, 1].astype(int), 0, height - 1)
        results.append((part.name, float((to_solution[ys, xs] <= tolerance).mean())))
    return results


def plan_overlay(image: np.ndarray, plan: Plan) -> np.ndarray:
    """The image with the plan's boxes drawn on: green = must include, red = ignore."""
    out = image.copy()
    thickness = max(2, max(out.shape[:2]) // 300)
    scale = max(out.shape[:2]) / 1200
    for parts, color in ((plan.must_include, (40, 160, 40)), (plan.ignore, (40, 40, 220))):
        for part in parts:
            x0, y0, x1, y1 = to_pixels(part.box, out.shape)
            cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
            cv2.putText(out, part.name[:24], (x0 + 4, y0 + int(22 * scale) + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6 * scale, color, max(1, thickness // 2), cv2.LINE_AA)
    return out
