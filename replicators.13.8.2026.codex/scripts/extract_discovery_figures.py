#!/usr/bin/env python3
"""Extract only the three plastic-heredity figures embedded in the Distill HTML."""

from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path


FIGURE_NAMES = {
    3: "reference_plastic_heredity_processes.png",
    4: "reference_rank_transfer.png",
    5: "reference_calibration.png",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("html", type=Path, nargs="?", default=Path("PRE_PRINT_DISTILL.PUB.html"))
    parser.add_argument("--output", type=Path, default=Path("reference"))
    arguments = parser.parse_args()
    source = arguments.html.read_text(encoding="utf-8")
    encoded = re.findall(r'<img src="data:image/png;base64,([^"]+)"', source)
    if len(encoded) != 5:
        raise SystemExit(f"expected five embedded PNGs, found {len(encoded)}")
    arguments.output.mkdir(parents=True, exist_ok=True)
    for one_based_index, filename in FIGURE_NAMES.items():
        (arguments.output / filename).write_bytes(
            base64.b64decode(encoded[one_based_index - 1], validate=True)
        )
        print(arguments.output / filename)


if __name__ == "__main__":
    main()

