"""ChEMBL web-services client (https://www.ebi.ac.uk/chembl/api/data/docs)."""

from __future__ import annotations

from collections.abc import Iterator

from proteinviz.http import WEEK, fetch

BASE = "https://www.ebi.ac.uk/chembl/api/data"
SOURCE = "ChEMBL"
PAGE = 1000


def status() -> dict:
    """Database status, including ``chembl_db_version`` and ``chembl_release_date``."""
    return fetch(f"{BASE}/status.json", ttl=WEEK, source=SOURCE).json()


def _paginate(endpoint: str, key: str, params: dict) -> Iterator[dict]:
    offset = 0
    while True:
        page = fetch(
            f"{BASE}/{endpoint}.json",
            params={**params, "limit": PAGE, "offset": offset},
            ttl=WEEK,
            source=SOURCE,
        ).json()
        yield from page[key]
        if not page["page_meta"].get("next"):
            return
        offset += PAGE


def activities(molecule_chembl_id: str) -> list[dict]:
    """All bioactivity records for a molecule."""
    return list(_paginate("activity", "activities", {"molecule_chembl_id": molecule_chembl_id}))


def molecule(molecule_chembl_id: str) -> dict:
    return fetch(f"{BASE}/molecule/{molecule_chembl_id}.json", ttl=WEEK, source=SOURCE).json()


def targets(target_ids: list[str]) -> dict[str, dict]:
    """Target records keyed by ChEMBL target ID (batched ``__in`` queries)."""
    out: dict[str, dict] = {}
    ids = sorted(set(target_ids))
    for i in range(0, len(ids), 50):
        chunk = ids[i : i + 50]
        for t in _paginate("target", "targets", {"target_chembl_id__in": ",".join(chunk)}):
            out[t["target_chembl_id"]] = t
    return out


def documents(document_ids: list[str]) -> dict[str, dict]:
    """Document records (DOI, PubMed ID, year, journal) keyed by ChEMBL document ID."""
    out: dict[str, dict] = {}
    ids = sorted(set(document_ids))
    for i in range(0, len(ids), 50):
        chunk = ids[i : i + 50]
        for d in _paginate("document", "documents", {"document_chembl_id__in": ",".join(chunk)}):
            out[d["document_chembl_id"]] = d
    return out


def uniprot_accessions(target: dict) -> list[str]:
    return sorted(
        {
            c["accession"]
            for c in target.get("target_components", [])
            if c.get("accession") and c.get("component_type") == "PROTEIN"
        }
    )
