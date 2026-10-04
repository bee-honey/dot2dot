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


if __name__ == "__main__":
    for name, draw in [("cat", cat), ("star", star)]:
        cv2.imwrite(str(HERE / f"{name}.png"), draw())
        print(f"Wrote samples/{name}.png")
