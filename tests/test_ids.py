import pytest

from proteinviz.errors import InvalidIdentifier
from proteinviz.ids import IdKind, classify


@pytest.mark.parametrize(
    ("query", "kind"),
    [
        ("P04637", IdKind.UNIPROT),
        ("p04637", IdKind.UNIPROT),
        ("Q00987", IdKind.UNIPROT),
        ("A0A024RBG1", IdKind.UNIPROT),
        ("P04637-2", IdKind.UNIPROT),
        ("1YCR", IdKind.PDB),
        ("4hfz", IdKind.PDB),
        ("TP53", IdKind.GENE),
        ("P53", IdKind.GENE),
        ("HLA-A", IdKind.GENE),
        (" egfr ", IdKind.GENE),
    ],
)
def test_classify(query, kind):
    assert classify(query) is kind


@pytest.mark.parametrize("query", ["", "   ", "not a gene!", "a" * 40])
def test_classify_rejects(query):
    with pytest.raises(InvalidIdentifier):
        classify(query)
