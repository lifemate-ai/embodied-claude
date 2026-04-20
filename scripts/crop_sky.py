#!/usr/bin/env python3
"""Crop an image to keep only the sky (top) region, removing identifying features below."""

import argparse
import sys
from pathlib import Path

from PIL import Image


def crop_sky(
    input_path: Path,
    top_ratio: float = 0.3,
    left_ratio: float = 0.0,
    right_ratio: float = 1.0,
    output_path: Path | None = None,
) -> Path:
    if not (0.0 < top_ratio <= 1.0):
        raise ValueError(f"top_ratio must be in (0, 1], got {top_ratio}")
    if not (0.0 <= left_ratio < right_ratio <= 1.0):
        raise ValueError(
            f"require 0 <= left_ratio ({left_ratio}) < right_ratio ({right_ratio}) <= 1"
        )

    img = Image.open(input_path)
    width, height = img.size

    left = int(width * left_ratio)
    right = int(width * right_ratio)
    top = 0
    bottom = int(height * top_ratio)

    cropped = img.crop((left, top, right, bottom))

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_sky{input_path.suffix}")

    cropped.save(output_path, quality=90)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Crop sky region from an image.")
    parser.add_argument("input", type=Path, help="Input image path")
    parser.add_argument(
        "--top",
        type=float,
        default=0.3,
        help="Keep top N fraction (0-1, default 0.3)",
    )
    parser.add_argument(
        "--left",
        type=float,
        default=0.0,
        help="Left edge ratio (0-1, default 0.0)",
    )
    parser.add_argument(
        "--right",
        type=float,
        default=1.0,
        help="Right edge ratio (0-1, default 1.0)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path (default: <input>_sky.jpg)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 1

    output = crop_sky(
        args.input,
        top_ratio=args.top,
        left_ratio=args.left,
        right_ratio=args.right,
        output_path=args.output,
    )
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
