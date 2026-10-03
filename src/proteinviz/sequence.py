"""Sequence-derived physicochemical properties (Biopython ProtParam)."""

from __future__ import annotations

from Bio.SeqUtils.ProtParam import ProteinAnalysis

from proteinviz.models import SequenceProperties

_STANDARD = set("ACDEFGHIKLMNPQRSTVWY")


def properties(sequence: str) -> SequenceProperties:
    """Compute properties for a protein sequence.

    Non-standard residues (X, U, B, Z …) are dropped before analysis, since
    ProtParam has no parameters for them; this slightly changes mass and pI for
    sequences that contain them.
    """
    seq = "".join(c for c in sequence.upper() if c in _STANDARD)
    if not seq:
        raise ValueError("Sequence contains no standard amino acids.")
    pa = ProteinAnalysis(seq)
    return SequenceProperties(
        length=len(sequence),
        molecular_weight_da=round(pa.molecular_weight(), 1),
        isoelectric_point=round(pa.isoelectric_point(), 2),
        gravy=round(pa.gravy(), 3),
        aromaticity=round(pa.aromaticity(), 3),
        instability_index=round(pa.instability_index(), 2),
    )
