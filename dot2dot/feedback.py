"""Region feedback in normalized source coordinates, applied before full assembly."""
from dataclasses import replace
import re
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field, model_validator
from skimage.morphology import skeletonize

from dot2dot.models import Path, Puzzle
from dot2dot.guidance import _inside
from dot2dot import lineart
from dot2dot.skeleton import trace_skeleton


class Correction(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    box: tuple[float, float, float, float]
    note: str = Field(min_length=1, max_length=1000)
    action: Literal['auto', 'restore', 'protect', 'remove', 'simplify'] = 'auto'

    @model_validator(mode='after')
    def valid_box(self):
        x0, y0, x1, y1 = self.box
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError('Correction boxes must be ordered fractions between 0 and 1')
        if not self.note.strip():
            raise ValueError('Describe the correction')
        return self

    def pixels(self, shape):
        height, width = shape[:2]
        x0, y0, x1, y1 = self.box
        return (int(x0 * width), int(y0 * height), min(width, max(int(x1 * width), int(x0 * width) + 1)),
                min(height, max(int(y1 * height), int(y0 * height) + 1)))

    def resolved_action(self):
        if self.action != 'auto':
            return self.action
        text = self.note.lower()
        # Conservative: negated removal must never erase a requested feature.
        if re.search(r"\b(don't|do not|never)\b", text):
            return 'protect'
        if re.search(r'\b(restore|missing|wrong|merged|fix|correct|repair)\b', text):
            return 'restore'
        if re.search(r'\b(keep|preserve|protect)\b', text):
            return 'protect'
        if re.search(r'\b(remove|erase|delete|exclude)\b', text):
            return 'remove'
        if re.search(r'\b(simplify|clutter|busy|fewer|less)\b', text):
            return 'simplify'
        return 'restore'


class CorrectionCheck(BaseModel):
    id: str
    action: str
    coverage: float | None
    status: str
    message: str
    affected_paths: int = 0


def local_lines(image, box):
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1]
    ink = lineart.ink_mask(crop)
    # Ink centerlines avoid doubled Canny edges on drawings. If there is no
    # usable dark ink, also look for contrast edges (colored/photo details).
    if np.count_nonzero(ink) < 6:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ink = cv2.Canny(gray, 40, 120)
    paths = trace_skeleton(skeletonize(ink > 0), min_spur=3)
    return [replace(p, points=p.points + (x0, y0), protected=True, weight=2.5)
            for p in paths if p.length() >= 6]


def apply_feedback(paths: list[Path], corrections: list[Correction], image: np.ndarray):
    """Apply all corrections from fresh extraction. Later overlapping marks win."""
    result = list(paths)
    counts = {}
    for correction in corrections:
        box = correction.pixels(image.shape)
        action = correction.resolved_action()
        selected = [p for p in result if _inside(p.points, box).mean() >= 0.5]
        if action in ('remove', 'restore'):
            mask = np.zeros(image.shape[:2], np.uint8)
            x0,y0,x1,y1 = box
            mask[y0:y1,x0:x1] = 255
            # Restore interior tracing while leaving the outer pose intact.
            untouched = [p for p in result if p.essential] if action == 'restore' else []
            clipped = [p for p in result if not p.essential] if action == 'restore' else result
            result = untouched + lineart.clip_paths(clipped, mask)
            if action == 'restore':
                recovered = local_lines(image, box)
                # Keep the longest local strokes first, preventing hundreds of
                # tiny texture fragments from consuming the global dot budget.
                recovered = sorted(recovered, key=lambda p: p.length(), reverse=True)[:12]
                result += recovered
                counts[correction.id] = len(recovered)
            else:
                counts[correction.id] = len(selected)
        elif action == 'protect':
            result = [replace(p, protected=True, weight=max(p.weight, 2.5)) if p in selected else p for p in result]
            if not selected:
                recovered = sorted(local_lines(image, box), key=lambda p: p.length(), reverse=True)[:8]
                result += recovered
                selected = recovered
            counts[correction.id] = len(selected)
        else:
            # Simplify only ordinary interior detail, never protected features.
            eligible = [p for p in selected if not (p.essential or p.protected or p.feature)]
            keep = sorted(eligible, key=lambda p: p.length(), reverse=True)[:2]
            result = [p for p in result if p not in eligible or p in keep]
            counts[correction.id] = len(eligible) - len(keep)
    return result, counts


def check_feedback(puzzle: Puzzle, reference: list[Path], corrections: list[Correction], affected: dict):
    """Geometric checks, not a claim that natural-language intent was satisfied."""
    from dot2dot.quality import _solution_mask, _distance_to
    mask = _solution_mask(puzzle)
    distance = _distance_to(mask)
    checks = []
    for correction in corrections:
        box = correction.pixels(mask.shape)
        action = correction.resolved_action()
        samples = [p.points[_inside(p.points, box)] for p in reference]
        points = np.concatenate(samples) if samples else np.zeros((0,2))
        coverage = None
        if len(points):
            xs = np.clip(points[:,0].astype(int),0,puzzle.width-1)
            ys = np.clip(points[:,1].astype(int),0,puzzle.height-1)
            coverage = float((distance[ys,xs] <= max(2, max(mask.shape)*0.004)).mean())
        count = affected.get(correction.id, 0)
        if action == 'remove':
            x0,y0,x1,y1 = box
            remaining = bool(mask[y0:y1,x0:x1].any())
            status = 'needs_review' if remaining else 'geometry_checked'
            message = 'Lines still cross the marked area; review the revision.' if remaining else 'No rendered lines remain in the marked area.'
        elif action == 'simplify':
            status = 'needs_review'
            message = f'Removed {count} ordinary detail paths. Review whether enough shape remains.'
        else:
            status = 'geometry_checked' if count and coverage is not None and coverage >= 0.8 else 'needs_review'
            message = ('Marked reference lines are represented; visually check the requested feature.' if status == 'geometry_checked'
                       else 'The marked feature is not sufficiently represented. Try a tighter box or manual tracing.')
        checks.append(CorrectionCheck(id=correction.id, action=action, coverage=coverage,
                                      status=status, message=message, affected_paths=count))
    return checks
