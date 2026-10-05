"""Draw simple silhouette images for trying out the pipeline.

Run: python samples/make_samples.py
"""

from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent


def cat() -> np.ndarray:
    img = np.full((600, 600, 3), 255, np.uint8)
    black = (30, 30, 30)
    cv2.ellipse(img, (300, 400), (150, 120), 0, 0, 360, black, -1)  # body
    cv2.circle(img, (300, 220), 95, black, -1)  # head
    ears = np.array([[215, 190], [230, 90], [285, 140], [315, 140], [370, 90], [385, 190]])
    cv2.fillPoly(img, [ears[:3], ears[3:]], black)
    cv2.ellipse(img, (450, 450), (90, 25), -40, 0, 360, black, -1)  # tail
    return img


def star() -> np.ndarray:
    img = np.full((600, 600, 3), 255, np.uint8)
    angles = np.linspace(-np.pi / 2, 3 * np.pi / 2, 10, endpoint=False)
    radii = np.where(np.arange(10) % 2 == 0, 250, 100)
    points = np.column_stack([300 + radii * np.cos(angles), 300 + radii * np.sin(angles)])
    cv2.fillPoly(img, [points.astype(np.int32)], (30, 30, 30))
    return img


def face() -> np.ndarray:
    """An inked cartoon face on a colored fill, for --style lineart."""
    img = np.full((600, 600, 3), (225, 235, 240), np.uint8)
    ink = (20, 20, 20)
    cv2.circle(img, (300, 310), 200, (120, 180, 250), -1)  # skin fill
    cv2.circle(img, (300, 310), 200, ink, 6)  # head
    for x in (225, 375):
        cv2.ellipse(img, (x, 260), (38, 48), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (x, 260), (38, 48), 0, 0, 360, ink, 6)  # eyes
        cv2.circle(img, (x + 8, 270), 14, ink, -1)  # pupils
        cv2.ellipse(img, (x, 190), (45, 18), 0, 200, 340, ink, 6)  # brows
    cv2.ellipse(img, (300, 380), (110, 60), 0, 15, 165, ink, 6)  # smile
    cv2.ellipse(img, (300, 330), (18, 12), 0, 0, 180, ink, 5)  # nose
    for x in (115, 485):
        cv2.ellipse(img, (x, 310), (25, 45), 0, 0, 360, ink, 6)  # ears
    return img


if __name__ == "__main__":
    for name, draw in [("cat", cat), ("star", star), ("face", face)]:
        cv2.imwrite(str(HERE / f"{name}.png"), draw())
        print(f"Wrote samples/{name}.png")
