import cv2
import numpy as np

from scancheck.core import analyze_quality, detect_document, four_point_transform, order_points
from scancheck.demo import make_demo_image


def test_order_points() -> None:
    points = np.array([[90, 90], [10, 10], [10, 90], [90, 10]], dtype=np.float32)
    assert order_points(points).tolist() == [[10, 10], [90, 10], [90, 90], [10, 90]]


def test_transform_returns_expected_shape() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    points = np.array([[10, 20], [140, 20], [140, 100], [10, 100]], dtype=np.float32)
    result = four_point_transform(image, points)
    assert result.shape[:2] == (80, 130)


def test_quality_flags_blurred_image() -> None:
    image = np.full((100, 100, 3), 120, dtype=np.uint8)
    metrics = analyze_quality(image)
    assert metrics.is_blurry is True
    assert metrics.exposure == "balanced"


def test_demo_contains_detectable_document(tmp_path) -> None:
    path = make_demo_image(tmp_path / "demo.png")
    image = cv2.imread(str(path))
    assert detect_document(image) is not None
