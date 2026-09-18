"""Core computer-vision pipeline for ScanCheck."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


class ScanError(RuntimeError):
    """Raised when an image cannot be loaded or processed."""


@dataclass(frozen=True)
class QualityMetrics:
    width: int
    height: int
    blur_score: float
    brightness: float
    contrast: float
    edge_density: float
    is_blurry: bool
    exposure: str

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return asdict(self)


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ScanError(f"Could not read image: {path}")
    return image


def order_points(points: np.ndarray) -> np.ndarray:
    """Return four points ordered as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = order_points(points)
    width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    if width < 2 or height < 2:
        raise ScanError("Detected document is too small to transform")
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]), destination)
    return cv2.warpPerspective(image, matrix, (width, height))


def detect_document(image: np.ndarray) -> np.ndarray | None:
    """Find the largest plausible four-sided document contour."""
    height, width = image.shape[:2]
    scale = min(1.0, 1000.0 / max(height, width))
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = resized.shape[0] * resized.shape[1] * 0.15
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        if cv2.contourArea(contour) < minimum_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            return polygon.reshape(4, 2).astype(np.float32) / scale
    return None


def enhance_document(image: np.ndarray, color: bool = False) -> np.ndarray:
    """Improve readability of a perspective-corrected document."""
    if color:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, a, b = cv2.split(lab)
        lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
        return cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12
    )


def scan_document(image: np.ndarray, color: bool = False) -> tuple[np.ndarray, np.ndarray]:
    points = detect_document(image)
    if points is None:
        raise ScanError("No four-sided document was detected")
    warped = four_point_transform(image, points)
    return enhance_document(warped, color=color), points


def analyze_quality(image: np.ndarray, blur_threshold: float = 100.0) -> QualityMetrics:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.count_nonzero(edges) / edges.size)
    exposure = "dark" if brightness < 70 else "bright" if brightness > 190 else "balanced"
    return QualityMetrics(
        width=image.shape[1],
        height=image.shape[0],
        blur_score=round(blur, 2),
        brightness=round(brightness, 2),
        contrast=round(contrast, 2),
        edge_density=round(edge_density, 4),
        is_blurry=blur < blur_threshold,
        exposure=exposure,
    )


def save_image(path: str | Path, image: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise ScanError(f"Could not write image: {destination}")
