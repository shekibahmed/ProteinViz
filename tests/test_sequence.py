import pytest

from proteinviz.sequence import properties
from proteinviz.sources import uniprot


def test_tp53_properties_match_uniprot_mass():
    rec = uniprot.get_entry("P04637")
    props = properties(rec.sequence)
    # UniProt lists 43,653 Da for P04637.
    assert props.molecular_weight_da == pytest.approx(43653, abs=5)
    assert 6.0 < props.isoelectric_point < 7.0
    assert props.length == 393


def test_nonstandard_residues_dropped_and_empty_rejected():
    assert properties("ACDXU").length == 5
    with pytest.raises(ValueError):
        properties("XXXX")
