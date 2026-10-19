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


def _structure(params: dict[str, str]) -> AppTest:
    at = AppTest.from_file(ENTRY, default_timeout=120)
    for key, value in params.items():
        at.query_params[key] = value
    at.run()
    at.switch_page("views/structure.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _text(at: AppTest, label: str) -> str:
    box = next(widget for widget in at.text_input if widget.label == label)
    return box.value or ""


def _page_links(at: AppTest):
    return list(at.main.get("page_link")) + list(at.sidebar.get("page_link"))


def test_structure_page_shows_interface_table():
    at = _structure({"a": "TP53", "b": "MDM2"})
    assert any("contain both" in s.value for s in at.success)
    tables = [df.value for df in at.dataframe]
    iface = next(t for t in tables if "Min dist (Å)" in t.columns)
    assert {"PHE19", "TRP23", "LEU26"} <= set(iface["Residue"])


def test_structure_demo_defaults_when_no_query():
    at = _structure({})
    assert _text(at, "Protein A (gene or UniProt)") == "TP53"
    assert _text(at, "Protein B (optional, finds complexes)") == "MDM2"


def test_structure_omitted_partner_stays_empty():
    at = _structure({"a": "TP53"})
    assert _text(at, "Protein A (gene or UniProt)") == "TP53"
    assert _text(at, "Protein B (optional, finds complexes)") == ""


def test_structure_pdb_without_pair_contact_does_not_substitute():
    """P53 is an alias of TP53, so 1YCR has no contact between two partners."""
    at = _structure({"a": "TP53", "b": "P53", "pdb": "1YCR"})
    assert any("no deposited contact" in w.value for w in at.warning)
    assert not any("Min dist (Å)" in df.value.columns for df in at.dataframe)


def test_home_alzheimer_link_keeps_id_and_query():
    at = AppTest.from_file(ENTRY, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    queries = [link.proto.query_string for link in _page_links(at)]
    assert any("MONDO_0004975" in q and "Alzheimer" in q for q in queries)
    labels = [link.proto.label for link in _page_links(at)]
    assert "Your data" in labels


def test_disease_structure_hop_names_two_genes_and_no_pdb():
    at = AppTest.from_file(ENTRY, default_timeout=120)
    at.query_params.update({"q": "Alzheimer disease", "id": "MONDO_0004975"})
    at.run()
    at.switch_page("views/disease.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    hops = [
        link.proto.query_string for link in _page_links(at) if link.proto.label.startswith("Open ")
    ]
    assert hops
    assert all("pdb" not in q for q in hops)
    assert all("a=" in q and "b=" in q for q in hops)


def test_protein_structure_link_passes_only_protein_a():
    at = AppTest.from_file(ENTRY, default_timeout=120)
    at.query_params["q"] = "TP53"
    at.run()
    at.switch_page("views/protein.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    structure = [link for link in _page_links(at) if "a=TP53" in link.proto.query_string]
    assert len(structure) == 1
    assert "b=" not in structure[0].proto.query_string


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
