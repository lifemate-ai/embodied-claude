#!/usr/bin/env python3
"""Run the hardware-free EFPF integrity benchmark."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from individual_kernel_mcp.benchmark import run_candidate_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "latest",
    )
    parser.add_argument("--db-path", type=Path)
    args = parser.parse_args()
    db_path = args.db_path or Path(tempfile.mkdtemp(prefix="efpf-benchmark-")) / "field.db"
    result = run_candidate_benchmark(output_dir=args.output_dir, db_path=db_path)
    print(args.output_dir / "indicator-profile.json")
    print(result["profile"]["profile_name"])


if __name__ == "__main__":
    main()
