"""AI planner: a vision model looks at the picture and plans the puzzle.

The pipeline itself has no idea what a picture *shows*. The planner fills
that gap with a structured plan:

- what the subject is, and what kind of image it is (coloring page, cartoon, photo)
- which style to use and a sensible dot count
- parts that must appear in the puzzle (beak, feet, eyes...), with rough boxes
- things to ignore (captions, watermarks, logos, page frames, scenery), with boxes

The plan only turns knobs the pipeline already has, so a bad or missing plan
never breaks puzzle generation: without a planner everything works as before.

`Planner` is a Protocol, Python's version of a Java interface. Any class with a
matching `plan()` method is a Planner; no `implements` keyword needed.
`OpenAIPlanner` is the implementation; a Claude one could be added beside it.
"""

import base64
import hashlib
import json
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

from dot2dot import config

# Chosen by comparing models on the sample set: fast (~3s) with accurate part
# boxes. gpt-5.5 gives the most precise plans but takes 10-15s.
DEFAULT_OPENAI_MODEL = "gpt-4.1"
# The model sees a downscaled copy; this is plenty to identify parts.
PLANNER_IMAGE_SIZE = 768
# Boxes from vision models are approximate; grow them by this fraction.
BOX_PADDING = 0.15

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 as fractions of width/height


@dataclass(frozen=True)
class Part:
    name: str
    box: Box


@dataclass(frozen=True)
class Plan:
    subject: str
    image_kind: str  # coloring_page | cartoon | photo | render | logo
    style: str  # lineart | photo | outline
    suggested_dots: int
    must_include: list[Part] = field(default_factory=list)
    ignore: list[Part] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "image_kind": self.image_kind,
            "style": self.style,
            "suggested_dots": self.suggested_dots,
            "must_include": [{"name": p.name, "box": list(p.box)} for p in self.must_include],
            "ignore": [{"name": p.name, "box": list(p.box)} for p in self.ignore],
            "notes": self.notes,
        }


class Planner(Protocol):
    def plan(self, image: np.ndarray, request: str = "") -> Plan: ...


INSTRUCTIONS = """\
You plan connect-the-dots puzzles. You'll get a picture and maybe a request
from a parent or teacher. A separate image-processing program turns the
picture into dots; your plan tells it what matters.

Decide:
- subject: a short description, e.g. "cartoon penguin standing on ice".
- image_kind: coloring_page (black lines on white), cartoon (colored drawing
  with outlines), photo (camera picture), render (3D/CGI), or logo.
- style: "lineart" for coloring pages and cartoons on a plain background;
  "photo" for photos, 3D renders, and any picture with a busy or scenic
  background; "outline" only for simple silhouettes/logos.
- suggested_dots: about 40-80 for young kids or very simple shapes, 100-200
  for normal pictures, 200-350 for detailed ones. Follow the request if given.
- must_include: the 3-8 parts OF THE MAIN SUBJECT a person needs to
  recognize it (e.g. eyes, beak, feet, flippers; or face, hands, emblem).
  Never list background objects (flags, vehicles, scenery, other items).
  Box each part tightly. Do not list the whole body.
- ignore: everything that is NOT the subject but would get traced: captions,
  titles, website names, copyright lines, watermarks, logos, page frames or
  borders, and background scenery (ice floes, furniture, trees). Box each.
  Leave it empty if there is nothing.

Boxes are [x0, y0, x1, y1] in thousandths of the image width and height
(0 = left/top edge, 1000 = right/bottom edge).
"""

_BOX_SCHEMA = {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}
_PART_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "box": _BOX_SCHEMA},
    "required": ["name", "box"],
    "additionalProperties": False,
}
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {"type": "string"},
        "image_kind": {"type": "string", "enum": ["coloring_page", "cartoon", "photo", "render", "logo"]},
        "style": {"type": "string", "enum": ["lineart", "photo", "outline"]},
        "suggested_dots": {"type": "integer"},
        "must_include": {"type": "array", "items": _PART_SCHEMA},
        "ignore": {"type": "array", "items": _PART_SCHEMA},
        "notes": {"type": "string"},
    },
    "required": ["subject", "image_kind", "style", "suggested_dots", "must_include", "ignore", "notes"],
    "additionalProperties": False,
}


def parse_plan(data: dict) -> Plan:
    """Turn the model's JSON into a Plan, clamping anything out of range."""

    def part(item: dict) -> Part | None:
        x0, y0, x1, y1 = (min(max(v, 0), 1000) / 1000 for v in item["box"])
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        if x1 - x0 < 0.005 or y1 - y0 < 0.005:
            return None
        return Part(item["name"].strip() or "part", _pad((x0, y0, x1, y1)))

    style = data.get("style") if data.get("style") in ("lineart", "photo", "outline") else "photo"
    return Plan(
        subject=data.get("subject", ""),
        image_kind=data.get("image_kind", ""),
        style=style,
        suggested_dots=int(min(max(data.get("suggested_dots", 150), 20), 600)),
        must_include=[p for p in map(part, data.get("must_include", [])) if p],
        ignore=[p for p in map(part, data.get("ignore", [])) if p],
        notes=data.get("notes", ""),
    )


def _pad(box: Box) -> Box:
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * BOX_PADDING, (y1 - y0) * BOX_PADDING
    return (max(x0 - dx, 0.0), max(y0 - dy, 0.0), min(x1 + dx, 1.0), min(y1 + dy, 1.0))


def to_pixels(box: Box, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """A fractional box as pixel coordinates (x0, y0, x1, y1) for an image of `shape`."""
    height, width = shape[:2]
    x0, y0, x1, y1 = box
    return round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height)


def _data_url(image: np.ndarray) -> str:
    height, width = image.shape[:2]
    scale = PLANNER_IMAGE_SIZE / max(height, width)
    if scale < 1:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("Could not encode image for the planner")
    return "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode()


class OpenAIPlanner:
    """Plans puzzles with an OpenAI vision model (Responses API, JSON schema output)."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        from openai import OpenAI  # imported lazily: only needed when the planner is used

        api_key = api_key or config.openai_api_key()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (add it to your .env file)")
        self.model = model or config.openai_model(DEFAULT_OPENAI_MODEL)
        self._client = OpenAI(api_key=api_key)
        self._cache: dict[str, Plan] = {}

    def plan(self, image: np.ndarray, request: str = "") -> Plan:
        # Same picture + same request = same plan; don't pay for it twice.
        key = hashlib.sha256(image.tobytes() + request.encode()).hexdigest()
        if key not in self._cache:
            self._cache[key] = self._ask(image, request)
        return self._cache[key]

    def _ask(self, image: np.ndarray, request: str) -> Plan:
        prompt = "Plan a connect-the-dots puzzle for this picture."
        if request.strip():
            prompt += f"\nRequest from the user: {request.strip()}"
        response = self._client.responses.create(
            model=self.model,
            instructions=INSTRUCTIONS,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": _data_url(image), "detail": "high"},
                    ],
                }
            ],
            text={"format": {"type": "json_schema", "name": "puzzle_plan", "schema": PLAN_SCHEMA, "strict": True}},
        )
        return parse_plan(json.loads(response.output_text))


def default_planner() -> OpenAIPlanner | None:
    """The configured planner, or None if no API key is set."""
    return OpenAIPlanner() if config.openai_api_key() else None
