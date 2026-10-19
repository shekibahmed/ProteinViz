"""README diagrams are generated files, not app runtime."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "build_readme_assets.py"


def _builder():
    spec = importlib.util.spec_from_file_location("build_readme_assets", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_asset_script_writes_gif_and_mp4(tmp_path):
    _builder().write_assets(tmp_path)
    gif = tmp_path / "disease-path.gif"
    structure = tmp_path / "structure-path.gif"
    tour = tmp_path / "tour.mp4"
    assert gif.read_bytes()[:3] == b"GIF"
    assert structure.read_bytes()[:3] == b"GIF"
    assert b"ftyp" in tour.read_bytes()[:32]
    assert gif.stat().st_size < 1_000_000
    assert structure.stat().st_size < 1_000_000
    assert tour.stat().st_size < 5_000_000


def test_readme_links_committed_assets_and_app_does_not():
    readme = (ROOT / "README.md").read_text()
    for name in ("disease-path.gif", "structure-path.gif", "tour.mp4"):
        assert f"docs/assets/{name}" in readme
        assert (ROOT / "docs" / "assets" / name).is_file()
    source = "\n".join(path.read_text() for path in (ROOT / "src").rglob("*.py"))
    assert "tour.mp4" not in source
    assert "import PIL" not in source
    assert "import imageio" not in source
    assert "from PIL" not in source
    assert "from imageio" not in source
