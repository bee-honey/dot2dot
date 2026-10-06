"""Google's Quick, Draw! doodles as dot2dot paths (for building datasets).

Quick, Draw! (https://quickdraw.withgoogle.com/data, CC BY 4.0) has 50M
doodles in 345 categories, stored as pen strokes, which is exactly what a
connect-the-dots puzzle is made of. Each category file is tens of MB, so we
fetch only the first few hundred KB with an HTTP Range request.
"""

import json
import urllib.parse
import urllib.request
from pathlib import Path as FilePath

import numpy as np

from dot2dot.models import Path

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"
CACHE_DIR = FilePath.home() / ".cache" / "dot2dot" / "quickdraw"
CATEGORIES_FILE = FilePath(__file__).parent / "data" / "quickdraw_categories.txt"
CANVAS = 1000  # doodles (0-255) are scaled onto a canvas this size
MARGIN = 60


def categories() -> list[str]:
    return [line.strip() for line in CATEGORIES_FILE.read_text().splitlines() if line.strip()]


def load_drawings(category: str, count: int, chunk_bytes: int = 300_000) -> list[dict]:
    """Up to `count` doodles of `category` that Google's own model recognized."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{category}.ndjson"
    if not cache.exists():
        request = urllib.request.Request(
            BASE_URL + urllib.parse.quote(category) + ".ndjson",
            headers={"Range": f"bytes=0-{chunk_bytes - 1}"},  # only the start of the file
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read().decode("utf-8", errors="ignore")
        # The last line is probably cut off by the byte range; drop it.
        cache.write_text("\n".join(data.splitlines()[:-1]))
    drawings = []
    for line in cache.read_text().splitlines():
        record = json.loads(line)
        if record.get("recognized"):
            drawings.append(record)
        if len(drawings) >= count:
            break
    return drawings


def to_paths(drawing: list) -> list[Path]:
    """Convert a doodle's strokes ([[xs], [ys]] in 0-255) to dense paths on the canvas."""
    scale = (CANVAS - 2 * MARGIN) / 255
    paths = []
    for xs, ys in drawing:
        points = np.column_stack([xs, ys]).astype(float) * scale + MARGIN
        dense = _densify(points)
        if len(dense) < 2:
            continue
        closed = len(dense) > 60 and np.linalg.norm(dense[0] - dense[-1]) < 15
        paths.append(Path(dense[:-1] if closed else dense, closed=closed))
    return paths


def _densify(points: np.ndarray) -> np.ndarray:
    """Insert points so consecutive points are ~1 px apart, like a traced outline."""
    pieces = [points[:1]]
    for a, b in zip(points, points[1:]):
        steps = max(int(np.ceil(np.linalg.norm(b - a))), 1)
        pieces.append(a + (b - a) * (np.arange(1, steps + 1) / steps)[:, None])
    return np.concatenate(pieces)
