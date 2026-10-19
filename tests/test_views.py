import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_views.py"


@pytest.fixture(scope="module")
def page(tmp_path_factory) -> str:
    """One fresh build of the page, shared by every test here."""
    out = tmp_path_factory.mktemp("views") / "index.html"
    subprocess.run([sys.executable, str(SCRIPT), "--out", str(out)], check=True, capture_output=True)
    return out.read_text()


def test_committed_page_matches_a_fresh_build(page):
    committed = SCRIPT.parents[1] / "docs" / "views" / "index.html"
    assert committed.read_text() == page, "run `uv run python scripts/build_views.py` and commit the page"
