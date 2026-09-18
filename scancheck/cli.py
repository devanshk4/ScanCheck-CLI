"""Command-line interface for ScanCheck."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .core import ScanError, analyze_quality, load_image, save_image, scan_document
from .demo import make_demo_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scancheck", description="Scan photographed documents and report image quality."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Detect, straighten, and enhance a document")
    scan.add_argument("input", help="Path to an input photograph")
    scan.add_argument("-o", "--output", default="scanned.png", help="Output image path")
    scan.add_argument("--color", action="store_true", help="Preserve color instead of making a B&W scan")

    analyze = subparsers.add_parser("analyze", help="Measure blur, exposure, contrast, and edges")
    analyze.add_argument("input", help="Path to an image")
    analyze.add_argument("--blur-threshold", type=float, default=100.0)
    analyze.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    demo = subparsers.add_parser("demo", help="Generate and process an offline sample image")
    demo.add_argument("-o", "--output-dir", default="demo_output")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.command == "scan":
        image = load_image(args.input)
        scanned, points = scan_document(image, color=args.color)
        save_image(args.output, scanned)
        print(f"Saved scan: {Path(args.output).resolve()}")
        print(f"Document corners: {points.round(1).tolist()}")
        return 0

    if args.command == "analyze":
        metrics = analyze_quality(load_image(args.input), args.blur_threshold)
        if args.json:
            print(json.dumps(metrics.to_dict(), indent=2))
        else:
            for key, value in metrics.to_dict().items():
                print(f"{key.replace('_', ' ').title()}: {value}")
        return 0

    output_dir = Path(args.output_dir)
    original = make_demo_image(output_dir / "sample_photo.png")
    image = load_image(original)
    scanned, _ = scan_document(image)
    scan_path = output_dir / "sample_scan.png"
    save_image(scan_path, scanned)
    metrics = analyze_quality(image)
    (output_dir / "quality.json").write_text(
        json.dumps(metrics.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Demo completed in: {output_dir.resolve()}")
    return 0


def main() -> None:
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except ScanError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    except cv2.error as error:
        print(f"OpenCV error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
