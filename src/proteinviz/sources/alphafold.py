"""AlphaFold Protein Structure Database client (https://alphafold.ebi.ac.uk/api-docs).

Uses the post-June-2026 field names (``modelEntityId``, ``sequenceStart`` …)
and never hardcodes a model version: file URLs come from the API response.
"""

from __future__ import annotations

import numpy as np

from proteinviz.errors import NotFound
from proteinviz.http import WEEK, fetch
from proteinviz.ids import normalize
from proteinviz.models import AlphaFoldModel, Provenance

API = "https://alphafold.ebi.ac.uk/api/prediction"
SOURCE = "AlphaFold DB"

# Standard AlphaFold DB pLDDT confidence bands: (lower bound, label, colour).
PLDDT_BANDS = [
    (90.0, "Very high (pLDDT > 90)", "#0053D6"),
    (70.0, "Confident (90 > pLDDT > 70)", "#65CBF3"),
    (50.0, "Low (70 > pLDDT > 50)", "#FFDB13"),
    (0.0, "Very low (pLDDT < 50)", "#FF7D45"),
]


def get_model(accession: str) -> AlphaFoldModel:
    """The canonical (F1, single-chain) AlphaFold DB model for a UniProt accession."""
    acc = normalize(accession)
    resp = fetch(f"{API}/{acc}", ttl=WEEK, source=SOURCE)
    entries = resp.json()
    if not entries:
        raise NotFound(f"AlphaFold DB has no model for {acc}.")
    preferred = f"AF-{acc}-F1"
    entry = next((e for e in entries if e.get("modelEntityId") == preferred), None)
    if entry is None:
        entry = next((e for e in entries if not e.get("isComplex")), entries[0])
    return AlphaFoldModel(
        model_entity_id=entry["modelEntityId"],
        accession=entry.get("uniprotAccession", acc),
        gene=entry.get("gene"),
        organism=entry.get("organismScientificName"),
        sequence_start=entry["sequenceStart"],
        sequence_end=entry["sequenceEnd"],
        model_version=int(entry["latestVersion"]),
        mean_plddt=entry.get("globalMetricValue"),
        cif_url=entry["cifUrl"],
        pdb_url=entry.get("pdbUrl"),
        pae_url=entry.get("paeDocUrl"),
        plddt_url=entry.get("plddtDocUrl"),
        provenance=Provenance.at(
            SOURCE,
            f"https://alphafold.ebi.ac.uk/entry/{acc}",
            version=f"model v{entry['latestVersion']}",
            fetched_at=resp.fetched_at,
        ),
    )


def fetch_plddt(model: AlphaFoldModel) -> tuple[list[int], list[float]]:
    """Per-residue (residue numbers, pLDDT) from the model's confidence JSON."""
    if not model.plddt_url:
        raise NotFound(f"No pLDDT file listed for {model.model_entity_id}.")
    data = fetch(model.plddt_url, ttl=WEEK, source=SOURCE).json()
    return [int(n) for n in data["residueNumber"]], [float(s) for s in data["confidenceScore"]]


def fetch_pae(model: AlphaFoldModel) -> np.ndarray:
    """Predicted aligned error matrix (Å) from the model's PAE JSON."""
    if not model.pae_url:
        raise NotFound(f"No PAE file listed for {model.model_entity_id}.")
    data = fetch(model.pae_url, ttl=WEEK, source=SOURCE).json()
    record = data[0] if isinstance(data, list) else data
    return np.asarray(record["predicted_aligned_error"], dtype=float)


def plddt_band(score: float) -> tuple[str, str]:
    """(label, colour) of the AlphaFold DB confidence band for a pLDDT score."""
    for lower, label, colour in PLDDT_BANDS:
        if score > lower or lower == 0.0:
            return label, colour
    raise AssertionError("unreachable")


def plddt_segments(residues: list[int], scores: list[float]) -> list[tuple[int, int, str]]:
    """Collapse per-residue pLDDT into contiguous (start, end, colour) segments."""
    segments: list[tuple[int, int, str]] = []
    for num, score in zip(residues, scores, strict=True):
        _, colour = plddt_band(score)
        if segments and segments[-1][2] == colour and segments[-1][1] == num - 1:
            start, _, _ = segments[-1]
            segments[-1] = (start, num, colour)
        else:
            segments.append((num, num, colour))
    return segments
