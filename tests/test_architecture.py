"""Guards for project-wide guarantees."""

import re
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parents[1] / "src" / "proteinviz"


def test_core_does_not_import_streamlit():
    code = (
        "import sys, pkgutil, importlib, proteinviz\n"
        "for m in pkgutil.walk_packages(proteinviz.__path__, 'proteinviz.'):\n"
        "    if not m.name.startswith('proteinviz.app'):\n"
        "        importlib.import_module(m.name)\n"
        "assert 'streamlit' not in sys.modules, 'core pulled in streamlit'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_no_fabricated_data_generators_in_package():
    """The package must never synthesise values: no random generators outside tests/scripts."""
    pattern = re.compile(r"\bnp\.random\b|\bimport random\b|\btorch\.randn?\b|from random import")
    offenders = [
        str(p.relative_to(SRC)) for p in SRC.rglob("*.py") if pattern.search(p.read_text())
    ]
    assert offenders == []


def test_cli_version():
    out = subprocess.run(
        [sys.executable, "-m", "proteinviz.cli", "version"],
        capture_output=True,
        text=True,
        check=True,
    )
    from proteinviz import __version__

    assert out.stdout.strip() == __version__
