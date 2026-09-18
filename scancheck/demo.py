"""Generate a deterministic sample photograph for an offline demo."""

from pathlib import Path

import cv2
import numpy as np


def make_demo_image(path: str | Path) -> Path:
    canvas = np.full((700, 1000, 3), (52, 62, 68), dtype=np.uint8)
    document = np.full((520, 720, 3), 242, dtype=np.uint8)
    cv2.putText(document, "SCANCHECK DEMO", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (25, 25, 25), 3)
    cv2.line(document, (60, 105), (660, 105), (60, 60, 60), 2)
    lines = [
        "Computer Vision Mini Project",
        "1. Detect the page contour",
        "2. Correct perspective distortion",
        "3. Improve text readability",
        "4. Measure image quality",
    ]
    for index, line in enumerate(lines):
        cv2.putText(document, line, (65, 165 + index * 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (35, 35, 35), 2)
    cv2.rectangle(document, (60, 440), (660, 480), (100, 100, 100), 2)
    cv2.putText(document, "Runs fully offline", (210, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 40, 40), 2)

    source = np.float32([[0, 0], [719, 0], [719, 519], [0, 519]])
    target = np.float32([[160, 90], [845, 55], [900, 620], [105, 650]])
    transform = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(document, transform, (1000, 700))
    mask = cv2.warpPerspective(np.full((520, 720), 255, np.uint8), transform, (1000, 700))
    canvas[mask > 0] = warped[mask > 0]
    cv2.circle(canvas, (80, 80), 30, (35, 42, 46), -1)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), canvas)
    return destination
