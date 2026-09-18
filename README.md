# ScanCheck CLI

ScanCheck is a small, fully offline computer-vision project that converts a photograph of a paper document into a flat, readable scan. It also reports blur, brightness, contrast, and edge-density measurements so a user can judge whether the image is suitable for scanning.

## Features

- Detects the largest four-sided document using Canny edges and contours.
- Corrects camera perspective with a homography.
- Produces either a high-contrast black-and-white scan or an enhanced color scan.
- Measures image quality and prints human-readable or JSON output.
- Includes a deterministic offline demo and automated tests.
- Includes a GitHub Actions workflow that runs the tests and demo after every push.
- Runs entirely from the command line; no dataset, GPU, network, or GUI is needed.

## Project structure

```text
scancheck-cli/
|-- scancheck/          # Application source code
|-- tests/              # Automated tests
|-- docs/               # Project report
|-- requirements.txt    # Runtime dependencies
|-- requirements-dev.txt
|-- pyproject.toml
`-- README.md
```

## Requirements

- Python 3.10 or newer
- pip

## Setup

Open a terminal in the repository root. Create an isolated environment:

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No configuration files, credentials, model weights, or environment variables are required.

## Quick start: built-in offline demo

```bash
python -m scancheck.cli demo
```

This creates `demo_output/sample_photo.png`, `demo_output/sample_scan.png`, and `demo_output/quality.json`.

## Scan your own image

Place the paper on a contrasting background and keep all four corners visible.

```bash
python -m scancheck.cli scan path/to/photo.jpg --output scanned.png
```

For an enhanced color result:

```bash
python -m scancheck.cli scan path/to/photo.jpg --output scanned-color.png --color
```

## Analyze image quality

```bash
python -m scancheck.cli analyze path/to/photo.jpg
python -m scancheck.cli analyze path/to/photo.jpg --json
```

The blur score is the variance of the Laplacian. Lower values mean less detail; the default blurry/not-blurry threshold is 100 and can be changed with `--blur-threshold`.

## Command reference

```bash
python -m scancheck.cli --help
python -m scancheck.cli scan --help
python -m scancheck.cli analyze --help
python -m scancheck.cli demo --help
```

Successful commands exit with code 0. Invalid or unreadable images and failed document detection exit with code 2 and a concise error message.

## Run the tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

The tests cover point ordering, perspective transformation, quality classification, and end-to-end detection on the generated sample.

## Method

1. Resize the input temporarily for efficient contour detection.
2. Convert to grayscale, blur noise, and calculate Canny edges.
3. Close small gaps morphologically and rank contours by area.
4. Approximate contours and select the largest convex quadrilateral.
5. Order its corners and compute a perspective transform.
6. Enhance the corrected image using adaptive thresholding or CLAHE.
7. Measure focus, brightness, contrast, and edge density.

## Limitations

- All four document edges should be visible against a reasonably contrasting background.
- Strong shadows, folds, or another larger quadrilateral may confuse contour detection.
- The blur threshold is a useful heuristic, not a universal camera-quality standard.
- Handwriting or text is enhanced but not recognized; OCR is outside this project's scope.

## Reproducibility

The demo is generated from fixed geometry and contains no randomness. Dependencies are pinned. The project makes no network calls at runtime.

## License

MIT
