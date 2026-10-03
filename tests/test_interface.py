"""Interface computation validated against a textbook complex.

In p53–MDM2 (PDB 1YCR) the p53 transactivation peptide inserts Phe19, Trp23
and Leu26 into MDM2's hydrophobic cleft (Kussie et al., Science 1996).
"""

import pytest

from proteinviz.errors import NotFound
from proteinviz.sources import rcsb
from proteinviz.structure.interface import compute_interface, contacting_chain_pairs


@pytest.fixture(scope="module")
def cif_1ycr():
    return rcsb.download_mmcif("1YCR")


def test_p53_triad_in_interface(cif_1ycr):
    iface = compute_interface(cif_1ycr, "A", "B", pdb_id="1YCR")
    p53 = {(r.residue_name, r.residue_number) for r in iface.residues_b}
    assert {("PHE", 19), ("TRP", 23), ("LEU", 26)} <= p53
    assert all(r.min_distance <= 5.0 for r in iface.residues_a + iface.residues_b)
    assert all(r.partner_chain == "A" for r in iface.residues_b)
    assert 15 <= len(iface.residues_a) <= 35


def test_tighter_cutoff_gives_subset(cif_1ycr):
    loose = compute_interface(cif_1ycr, "A", "B", pdb_id="1YCR", cutoff=5.0)
    tight = compute_interface(cif_1ycr, "A", "B", pdb_id="1YCR", cutoff=3.5)
    key = lambda rs: {(r.chain, r.residue_number) for r in rs}  # noqa: E731
    assert key(tight.residues_a) <= key(loose.residues_a)
    assert len(tight.residues_b) < len(loose.residues_b)


def test_symmetry_mates_and_self_contacts_excluded(cif_1ycr):
    pairs = contacting_chain_pairs(cif_1ycr)
    assert [(a, b) for a, b, _ in pairs] == [("A", "B")]


def test_missing_chain(cif_1ycr):
    with pytest.raises(NotFound):
        compute_interface(cif_1ycr, "A", "Z", pdb_id="1YCR")


def test_same_chain_rejected(cif_1ycr):
    with pytest.raises(ValueError):
        compute_interface(cif_1ycr, "A", "A", pdb_id="1YCR")
