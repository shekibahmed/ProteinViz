"""Identifier classification for user input."""

from __future__ import annotations

import re
from enum import StrEnum

from proteinviz.errors import InvalidIdentifier

# Official UniProt accession pattern (https://www.uniprot.org/help/accession_numbers),
# optionally with an isoform suffix such as P04637-2.
UNIPROT_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-\d+)?$"
)
PDB_RE = re.compile(r"^[0-9][A-Z0-9]{3}$")
# Extended PDB IDs (pdb_0000xxxx) announced by wwPDB.
PDB_EXT_RE = re.compile(r"^PDB_[0-9]{4}[0-9][A-Z0-9]{3}$")
GENE_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,24}$")


class IdKind(StrEnum):
    UNIPROT = "uniprot"
    PDB = "pdb"
    GENE = "gene"


def normalize(query: str) -> str:
    return query.strip().upper()


def classify(query: str) -> IdKind:
    """Classify a user query as a UniProt accession, PDB ID or gene symbol.

    PDB IDs win over gene names for 4-character tokens starting with a digit
    (e.g. ``1YCR``); UniProt accessions are matched with the official regex.
    """
    q = normalize(query)
    if not q:
        raise InvalidIdentifier("Empty identifier.")
    if UNIPROT_RE.match(q):
        return IdKind.UNIPROT
    if PDB_RE.match(q) or PDB_EXT_RE.match(q):
        return IdKind.PDB
    if GENE_RE.match(q):
        return IdKind.GENE
    raise InvalidIdentifier(
        f"'{query}' is not a UniProt accession (e.g. P04637), PDB ID (e.g. 1YCR) or gene symbol (e.g. TP53)."
    )
