"""Extract strokes from real photos.

Photos have no ink outlines, so the cartoon approach (trace the dark lines)
picks up hair, shadows and the backdrop. Instead:

1. Background removal: a segmentation model (rembg / U²-Net, runs locally)
   finds the subject. Its border becomes the silhouette. Any part lying on
   the image edge (e.g. a shirt cut off by the frame) is dropped.
2. Sketch lines: XDoG ("extended difference of Gaussians") turns the photo
   into a pencil-sketch-like line drawing inside the subject: eyes, glasses,
   mouth, collar. Those lines are traced with the same code as lineart.
3. Faces: OpenCV's YuNet detector finds faces, and lines inside a face get
   extra weight, so features like eyes get enough dots to stay recognizable.
"""

import urllib.request
from dataclasses import replace
from functools import lru_cache
from pathlib import Path as FilePath

import cv2
import numpy as np

from dot2dot import lineart
from dot2dot.models import Path

MODEL_DIR = FilePath.home() / ".cache" / "dot2dot"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

# Blur applied to the subject mask before tracing, as a fraction of image
# size. Smooths away stray hairs so the silhouette is a clean curve.
MASK_SMOOTHING = 0.006
# XDoG scale (fraction of image size): bigger = bolder, fewer lines.
SKETCH_SIGMA = 0.002
SKETCH_TAU = 0.98
SKETCH_EPSILON = -0.01
# Faces are sketched at a finer scale, since eyes and lips are small.
FACE_SIGMA_SCALE = 0.6
# Line fragments smaller than this many pixels are noise.
MIN_SKETCH_AREA = 40
# Lines inside a detected face count this many times their length.
FACE_WEIGHT = 3.0
FACE_SCORE_THRESHOLD = 0.7
# Silhouette points within this many pixels of the image edge are dropped.
EDGE_CLIP = 4


def extract_paths(image: np.ndarray) -> list[Path]:
    mask = subject_mask(image)
    outline = _largest_contour(mask)
    if outline is None:
        raise ValueError("Couldn't find a subject in the photo")

    outer = Path(outline, closed=True, essential=True)
    faces = detect_faces(image)
    interior = lineart.trace_ink(
        sketch_lines(image, mask, faces),
        band=lineart.silhouette_band(outer, image.shape),
        include_fills=False,  # dark masses (hair, shadows) aren't lines
    )
    interior = [_weight_by_faces(p, faces) for p in interior]
    return _drop_frame_edges(outer, image.shape) + interior


def subject_mask(image: np.ndarray) -> np.ndarray:
    """White where the subject is, black for background."""
    from rembg import remove  # imported here: slow to import, only needed for photos

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = remove(rgb, session=_rembg_session(), only_mask=True)
    # Smooth, then re-threshold: rounds off flyaway hairs and jagged edges.
    sigma = MASK_SMOOTHING * max(image.shape[:2])
    mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def sketch_lines(image: np.ndarray, mask: np.ndarray, faces: list[tuple[int, int, int, int]] = ()) -> np.ndarray:
    """XDoG line drawing of the photo, limited to the subject. White = line.

    Difference of Gaussians: blur the image a little and a bit more, then
    subtract. Flat areas cancel out; edges and thin dark features (eyelids,
    glasses, lips) leave a strong negative response, which we keep as lines.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    # Edge-preserving blur: smooths skin and fabric texture, keeps outlines.
    gray = cv2.bilateralFilter(gray, 9, 0.1, 5)
    sigma = SKETCH_SIGMA * max(image.shape[:2])
    lines = _difference_of_gaussians(gray, sigma)
    # Inside each face, swap in a finer sketch that keeps small features.
    face_lines = _difference_of_gaussians(gray, sigma * FACE_SIGMA_SCALE)
    for x, y, w, h in faces:
        y0, x0 = max(y, 0), max(x, 0)
        lines[y0 : y + h, x0 : x + w] = face_lines[y0 : y + h, x0 : x + w]
    lines[mask == 0] = 0

    # Remove specks: keep only connected line pieces of a reasonable size.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= MIN_SKETCH_AREA
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def _difference_of_gaussians(gray: np.ndarray, sigma: float) -> np.ndarray:
    fine = cv2.GaussianBlur(gray, (0, 0), sigma)
    coarse = cv2.GaussianBlur(gray, (0, 0), sigma * 1.6)
    return np.where(fine - SKETCH_TAU * coarse < SKETCH_EPSILON, 255, 0).astype(np.uint8)


def detect_faces(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Face boxes (x, y, w, h), slightly enlarged to include hairline and chin."""
    height, width = image.shape[:2]
    detector = cv2.FaceDetectorYN.create(str(_yunet_model()), "", (width, height), FACE_SCORE_THRESHOLD)
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    boxes = []
    for face in faces if faces is not None else []:
        x, y, w, h = face[:4]
        boxes.append((int(x - 0.1 * w), int(y - 0.15 * h), int(1.2 * w), int(1.3 * h)))
    return boxes


def _weight_by_faces(path: Path, faces: list[tuple[int, int, int, int]]) -> Path:
    for x, y, w, h in faces:
        inside = (
            (path.points[:, 0] >= x) & (path.points[:, 0] <= x + w)
            & (path.points[:, 1] >= y) & (path.points[:, 1] <= y + h)
        )
        if inside.mean() > 0.5:
            return replace(path, weight=FACE_WEIGHT)
    return path


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < lineart.MIN_SILHOUETTE_FRACTION * mask.size:
        return None
    return largest.reshape(-1, 2).astype(np.float64)


def _drop_frame_edges(outer: Path, shape: tuple[int, ...]) -> list[Path]:
    """Remove silhouette stretches that just run along the image border."""
    border = np.zeros(shape[:2], np.uint8)
    border[:EDGE_CLIP, :] = border[-EDGE_CLIP:, :] = 255
    border[:, :EDGE_CLIP] = border[:, -EDGE_CLIP:] = 255
    return lineart.clip_paths([outer], border)


@lru_cache(maxsize=1)
def _rembg_session():
    """Load the segmentation model once and reuse it (it takes ~1s to load)."""
    from rembg import new_session

    return new_session("u2net")


def _yunet_model() -> FilePath:
    """Path to the YuNet face model, downloading it (~230 KB) on first use."""
    path = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
    if not path.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(YUNET_URL, path)
    return path
