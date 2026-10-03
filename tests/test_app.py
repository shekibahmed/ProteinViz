"""Smoke tests: every page renders without exceptions on recorded data."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parents[1]
ENTRY = str(ROOT / "src" / "proteinviz" / "app" / "streamlit_app.py")
CLOUD_ENTRY = str(ROOT / "streamlit_app.py")  # used by Streamlit Community Cloud

PAGES = [
    ("views/home.py", {}),
    ("views/disease.py", {"q": "Alzheimer disease", "id": "MONDO_0004975"}),
    ("views/network.py", {"q": "TP53", "n": "20", "score": "700", "type": "functional"}),
    ("views/protein.py", {"q": "TP53"}),
    ("views/structure.py", {"a": "TP53", "b": "MDM2"}),
    ("views/egcg.py", {}),
    ("views/upload.py", {}),
    ("views/about.py", {}),
]


@pytest.mark.parametrize(("page", "params"), PAGES, ids=[p for p, _ in PAGES])
def test_page_renders(page, params):
    at = AppTest.from_file(ENTRY, default_timeout=120)
    for k, v in params.items():
        at.query_params[k] = v
    at.run()
    at.switch_page(page)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_structure_page_shows_interface_table():
    at = AppTest.from_file(ENTRY, default_timeout=120)
    at.query_params.update({"a": "TP53", "b": "MDM2"})
    at.run()
    at.switch_page("views/structure.py")
    at.run()
    assert not at.exception
    assert any("contain both" in s.value for s in at.success)
    tables = [df.value for df in at.dataframe]
    iface = next(t for t in tables if "Min dist (Å)" in t.columns)
    assert {"PHE19", "TRP23", "LEU26"} <= set(iface["Residue"])


def test_upload_gene_list_enrichment():
    at = AppTest.from_file(ENTRY, default_timeout=120)
    at.run()
    at.switch_page("views/upload.py")
    at.run()
    at.text_area[0].input("SNCA\nLRRK2\nPRKN\nPINK1\nPARK7").run()
    assert not at.exception
    assert any("Pathway enrichment for 5" in s.value for s in at.subheader)


@pytest.mark.parametrize("entry", [ENTRY, CLOUD_ENTRY], ids=["package", "cloud"])
def test_home_links_resolve_from_any_entry_point(entry):
    """page_link targets must resolve whichever script launched the app."""
    at = AppTest.from_file(entry, default_timeout=120)
    at.run()
    # Home is the default page and renders st.page_link for every section.
    assert not at.exception, [e.value for e in at.exception]
    assert any("ProteinViz" in t.value for t in at.title)
