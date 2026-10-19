"""Build docs/views/index.html from the same seeded runs as the README.

Run `uv run python scripts/build_views.py`. The page is self-contained: this script puts the chart data into
`docs/views/template.html` as JSON, and the page draws the charts from it.
"""

import argparse
import json
from pathlib import Path

VIEWS = Path(__file__).resolve().parents[1] / "docs" / "views"
PLACEHOLDER = "/*VIEW_DATA*/null"


def build() -> dict:
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=VIEWS / "index.html")
    args = parser.parse_args()
    template = (VIEWS / "template.html").read_text()
    if PLACEHOLDER not in template:
        raise SystemExit(f"{PLACEHOLDER} is missing from the template")
    args.out.write_text(template.replace(PLACEHOLDER, json.dumps(build())))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
