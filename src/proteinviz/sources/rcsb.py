"""RCSB PDB clients: Search API v2, Data API (GraphQL) and file downloads."""

from __future__ import annotations

from proteinviz.errors import NotFound
from proteinviz.http import DAY, WEEK, fetch
from proteinviz.ids import PDB_RE, normalize
from proteinviz.models import PdbEntry, PolymerEntity, Provenance

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"
FILES_URL = "https://files.rcsb.org/download"
SOURCE = "RCSB PDB"

_ACC_ATTR = (
    "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession"
)
_DB_ATTR = "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name"


def _uniprot_node(accession: str) -> dict:
    return {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": _ACC_ATTR,
                    "operator": "exact_match",
                    "value": accession,
                },
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": _DB_ATTR,
                    "operator": "exact_match",
                    "value": "UniProt",
                },
            },
        ],
    }


def search_entries(*accessions: str, rows: int = 25) -> tuple[list[str], int]:
    """PDB entries that contain *all* the given UniProt accessions.

    Returns ``(pdb_ids, total_count)``; IDs are sorted best resolution first.
    Pass two accessions to find experimentally solved complexes of a pair.
    """
    if not accessions:
        raise ValueError("At least one accession is required.")
    nodes = [_uniprot_node(normalize(a)) for a in accessions]
    query = (
        nodes[0]
        if len(nodes) == 1
        else {"type": "group", "logical_operator": "and", "nodes": nodes}
    )
    body = {
        "query": query,
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": rows},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
        },
    }
    resp = fetch(SEARCH_URL, method="POST", json_body=body, ttl=WEEK, source=SOURCE)
    # The search API answers 204 No Content when nothing matches.
    if resp.status == 204 or not resp.content:
        return [], 0
    data = resp.json()
    return [r["identifier"] for r in data.get("result_set", [])], int(data.get("total_count", 0))


_ENTRY_QUERY = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    struct { title }
    exptl { method }
    rcsb_entry_info { resolution_combined }
    rcsb_accession_info { initial_release_date }
    polymer_entities {
      rcsb_id
      rcsb_polymer_entity { pdbx_description }
      rcsb_polymer_entity_container_identifiers {
        entity_id
        auth_asym_ids
        reference_sequence_identifiers { database_name database_accession }
      }
      rcsb_entity_source_organism { ncbi_scientific_name }
    }
  }
}
"""


def get_entries(pdb_ids: list[str]) -> list[PdbEntry]:
    """Metadata (title, method, resolution, entities→chains→UniProt) for PDB entries."""
    ids = [normalize(i) for i in pdb_ids]
    if not ids:
        return []
    resp = fetch(
        GRAPHQL_URL,
        method="POST",
        json_body={"query": _ENTRY_QUERY, "variables": {"ids": ids}},
        ttl=WEEK,
        source=SOURCE,
    )
    entries = (resp.json().get("data") or {}).get("entries") or []
    out = []
    for e in entries:
        entities = []
        for pe in e.get("polymer_entities") or []:
            ids_ = pe.get("rcsb_polymer_entity_container_identifiers") or {}
            refs = ids_.get("reference_sequence_identifiers") or []
            orgs = pe.get("rcsb_entity_source_organism") or []
            entities.append(
                PolymerEntity(
                    entity_id=str(ids_.get("entity_id", "")),
                    description=(pe.get("rcsb_polymer_entity") or {}).get("pdbx_description"),
                    chains=ids_.get("auth_asym_ids") or [],
                    uniprot_accessions=sorted(
                        {
                            r["database_accession"]
                            for r in refs
                            if r.get("database_name") == "UniProt"
                        }
                    ),
                    organism=orgs[0].get("ncbi_scientific_name") if orgs else None,
                )
            )
        res = (e.get("rcsb_entry_info") or {}).get("resolution_combined") or [None]
        release = (e.get("rcsb_accession_info") or {}).get("initial_release_date") or ""
        out.append(
            PdbEntry(
                pdb_id=e["rcsb_id"],
                title=(e.get("struct") or {}).get("title"),
                method=((e.get("exptl") or [{}])[0]).get("method"),
                resolution=res[0],
                release_date=release[:10] or None,
                entities=entities,
                provenance=Provenance.at(
                    SOURCE,
                    f"https://www.rcsb.org/structure/{e['rcsb_id']}",
                    fetched_at=resp.fetched_at,
                ),
            )
        )
    order = {pid: i for i, pid in enumerate(ids)}
    return sorted(out, key=lambda x: order.get(x.pdb_id, len(order)))


def get_entry(pdb_id: str) -> PdbEntry:
    pid = normalize(pdb_id)
    if not PDB_RE.match(pid):
        raise NotFound(f"'{pdb_id}' is not a 4-character PDB ID.")
    entries = get_entries([pid])
    if not entries:
        raise NotFound(f"PDB entry {pid} does not exist.")
    return entries[0]


def mmcif_url(pdb_id: str) -> str:
    return f"{FILES_URL}/{normalize(pdb_id)}.cif"


def download_mmcif(pdb_id: str) -> str:
    """Download the mmCIF file text for a PDB entry."""
    return fetch(mmcif_url(pdb_id), ttl=30 * DAY, source=SOURCE).text
