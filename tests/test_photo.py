import cv2
import numpy as np
import pytest

from dot2dot import photo
from dot2dot.models import Path
from dot2dot.preprocess import decode_image


def portrait_like() -> tuple[np.ndarray, np.ndarray]:
    """A gray 'head' with dark eyes and mouth on a blue backdrop, plus its mask."""
    img = np.full((500, 400, 3), (160, 110, 60), np.uint8)
    mask = np.zeros((500, 400), np.uint8)
    cv2.ellipse(img, (200, 230), (110, 140), 0, 0, 360, (150, 170, 200), -1)
    cv2.ellipse(mask, (200, 230), (110, 140), 0, 0, 360, 255, -1)
    for x in (160, 240):
        cv2.ellipse(img, (x, 200), (22, 10), 0, 0, 360, (40, 40, 40), 3)
    cv2.ellipse(img, (200, 290), (45, 18), 0, 0, 180, (40, 40, 90), 4)
    return img, mask


def test_sketch_lines_find_features_inside_subject_only():
    img, mask = portrait_like()
    lines = photo.sketch_lines(img, mask)
    assert lines[mask == 0].max() == 0  # nothing drawn on the backdrop
    assert lines[190:212, 135:185].any()  # left eye
    assert lines[285:312, 150:250].any()  # mouth


def test_face_paths_get_extra_weight():
    inside = Path(np.array([[110.0, 110.0], [150.0, 120.0]]), closed=False)
    outside = Path(np.array([[400.0, 400.0], [450.0, 420.0]]), closed=False)
    faces = [(100, 100, 100, 100)]
    assert photo._weight_by_faces(inside, faces).weight == photo.FACE_WEIGHT
    assert photo._weight_by_faces(outside, faces).weight == 1.0


def test_silhouette_along_image_edge_is_dropped():
    # A subject cut off by the bottom of the frame, like shoulders in a portrait.
    outline = np.array(
        [(x, 100) for x in range(100, 300)] + [(300, y) for y in range(100, 399)]
        + [(x, 399) for x in range(300, 100, -1)] + [(100, y) for y in range(399, 100, -1)],
        dtype=np.float64,
    )
    pieces = photo._drop_frame_edges(Path(outline, closed=True, essential=True), (400, 400, 3))
    assert len(pieces) == 1
    assert pieces[0].essential and not pieces[0].closed
    assert pieces[0].points[:, 1].max() < 396


def test_heic_photos_can_be_decoded():
    from PIL import Image
    from io import BytesIO

    buffer = BytesIO()
    Image.new("RGB", (40, 30), (255, 0, 0)).save(buffer, format="HEIF")
    image = decode_image(buffer.getvalue())
    assert image.shape == (30, 40, 3)
    assert image[15, 20, 2] > 200  # red channel (BGR order)


def test_phone_rotation_is_applied():
    from PIL import Image
    from io import BytesIO

    buffer = BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90° clockwise to display", as phones write it
    Image.new("RGB", (40, 30)).save(buffer, format="JPEG", exif=exif)
    assert decode_image(buffer.getvalue()).shape == (40, 30, 3)  # now portrait


@pytest.mark.skipif(
    not (photo.MODEL_DIR.parent.parent / ".u2net" / "u2net.onnx").exists()
    and not (photo.MODEL_DIR.parent.parent / ".rembg" / "models" / "u2net" / "u2net.onnx").exists(),
    reason="background-removal model not downloaded (it is ~170 MB)",
)
def test_photo_style_end_to_end(tmp_path):
    from dot2dot.pipeline import generate

    img, _ = portrait_like()
    path = tmp_path / "portrait.png"
    cv2.imwrite(str(path), img)
    puzzle = generate(path, num_dots=60, max_paths=None, style="photo")
    assert len(puzzle.dots) >= 50
