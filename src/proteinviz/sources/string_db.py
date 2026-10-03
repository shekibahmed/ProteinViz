"""STRING API client (https://string-db.org/help/api/).

STRING asks API users to identify themselves and to use bulk downloads for
large jobs; proteinviz only issues small per-protein queries and caches them.
"""

from __future__ import annotations

from proteinviz.errors import NotFound, ProteinVizError
from proteinviz.http import WEEK, fetch
from proteinviz.models import (
    Enrichment,
    EnrichmentTerm,
    Provenance,
    StringInteraction,
    StringNetwork,
)

BASE = "https://string-db.org/api/json"
SOURCE = "STRING"
CALLER = "proteinviz"

NETWORK_TYPES = ("functional", "physical")


def version() -> str:
    resp = fetch(f"{BASE}/version", ttl=WEEK, source=SOURCE)
    return resp.json()[0]["string_version"]


def network(
    identifier: str,
    species: int = 9606,
    limit: int = 20,
    required_score: int = 400,
    network_type: str = "functional",
) -> StringNetwork:
    """Return the STRING neighbourhood of ``identifier`` (gene symbol or UniProt accession).

    ``required_score`` is STRING's 0–1000 combined-score threshold
    (400 = medium, 700 = high, 900 = highest confidence).
    """
    if network_type not in NETWORK_TYPES:
        raise ValueError(f"network_type must be one of {NETWORK_TYPES}")
    params = {
        "identifiers": identifier,
        "species": species,
        "add_nodes": limit,
        "required_score": required_score,
        "network_type": network_type,
        "caller_identity": CALLER,
    }
    resp = fetch(f"{BASE}/network", params=params, ttl=WEEK, source=SOURCE)
    rows = resp.json()
    if isinstance(rows, dict) and "Error" in rows:
        raise NotFound(f"STRING: {rows.get('ErrorMessage', rows['Error'])}")
    if not rows:
        raise NotFound(
            f"STRING has no {network_type} interactions for '{identifier}' "
            f"(species {species}, score ≥ {required_score})."
        )
    interactions = [_parse_row(r) for r in rows]
    try:
        ver = version()
    except ProteinVizError:
        ver = None
    return StringNetwork(
        query=identifier,
        species=species,
        network_type=network_type,
        required_score=required_score,
        interactions=interactions,
        provenance=Provenance.at(
            SOURCE,
            f"https://string-db.org/network/{species}.{identifier}"
            if "\r" not in identifier
            else "https://string-db.org/cgi/input?input_page_active_form=multiple_identifiers",
            version=f"v{ver}" if ver else None,
            fetched_at=resp.fetched_at,
        ),
    )


def pair_evidence(a: str, b: str, species: int = 9606) -> StringInteraction | None:
    """STRING evidence for a specific pair, or ``None`` if STRING has no edge."""
    params = {
        "identifiers": f"{a}\r{b}",
        "species": species,
        "required_score": 0,
        "caller_identity": CALLER,
    }
    resp = fetch(f"{BASE}/network", params=params, ttl=WEEK, source=SOURCE)
    rows = resp.json()
    if not isinstance(rows, list) or not rows:
        return None
    return _parse_row(rows[0])


def _parse_row(r: dict) -> StringInteraction:
    return StringInteraction(
        string_id_a=r["stringId_A"],
        string_id_b=r["stringId_B"],
        name_a=r["preferredName_A"],
        name_b=r["preferredName_B"],
        score=float(r["score"]),
        experimental=float(r.get("escore", 0)),
        database=float(r.get("dscore", 0)),
        textmining=float(r.get("tscore", 0)),
        coexpression=float(r.get("ascore", 0)),
        neighborhood=float(r.get("nscore", 0)),
        fusion=float(r.get("fscore", 0)),
        cooccurrence=float(r.get("pscore", 0)),
    )


def enrichment(genes: list[str], species: int = 9606, max_fdr: float = 0.05) -> Enrichment:
    """Functional enrichment (GO, KEGG, Reactome, WikiPathways, DISEASES …) for a gene set.

    STRING tests the set against the whole genome background and reports
    Benjamini–Hochberg FDR. Only terms with ``fdr <= max_fdr`` are returned.
    """
    unique = list(dict.fromkeys(g for g in genes if g))
    if len(unique) < 2:
        raise ValueError("Enrichment needs at least two genes.")
    params = {"identifiers": "\r".join(unique), "species": species, "caller_identity": CALLER}
    resp = fetch(f"{BASE}/enrichment", params=params, ttl=WEEK, source=SOURCE)
    rows = resp.json()
    if isinstance(rows, dict) and "Error" in rows:
        raise NotFound(f"STRING: {rows.get('ErrorMessage', rows['Error'])}")
    terms = [
        EnrichmentTerm(
            category=r["category"],
            term=r["term"],
            description=r["description"],
            n_genes=int(r["number_of_genes"]),
            n_background=int(r["number_of_genes_in_background"]),
            p_value=float(r["p_value"]),
            fdr=float(r["fdr"]),
            genes=list(r.get("preferredNames") or []),
        )
        for r in rows
        if float(r["fdr"]) <= max_fdr
    ]
    terms.sort(key=lambda t: (t.fdr, -t.n_genes))
    return Enrichment(
        genes=unique,
        species=species,
        terms=terms,
        provenance=Provenance.at(
            SOURCE,
            "https://string-db.org/cgi/input?input_page_active_form=multiple_identifiers",
            fetched_at=resp.fetched_at,
        ),
    )
