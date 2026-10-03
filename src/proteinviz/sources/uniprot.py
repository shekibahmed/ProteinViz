"""UniProtKB REST client (https://rest.uniprot.org)."""

from __future__ import annotations

from typing import Any

from proteinviz.errors import InvalidIdentifier, NotFound
from proteinviz.http import WEEK, fetch
from proteinviz.ids import IdKind, classify, normalize
from proteinviz.models import Domain, ProteinRecord, Provenance

BASE = "https://rest.uniprot.org/uniprotkb"
SOURCE = "UniProt"
HUMAN = 9606


def get_entry(accession: str) -> ProteinRecord:
    """Fetch a UniProtKB entry by accession (isoform suffixes allowed)."""
    acc = normalize(accession)
    resp = fetch(f"{BASE}/{acc}.json", ttl=WEEK, source=SOURCE)
    data = resp.json()
    if not data.get("primaryAccession"):
        # Inactive/merged entries come back as stubs without a sequence.
        raise NotFound(f"UniProt entry {acc} is inactive or obsolete.")
    return parse_entry(data, fetched_at=resp.fetched_at)


def search_gene(gene: str, taxon_id: int = HUMAN) -> str:
    """Resolve a gene symbol or protein name to the best UniProtKB accession.

    Prefers reviewed (Swiss-Prot) entries with an exact gene-name match in the
    given organism, then falls back to a broader name search.
    """
    g = normalize(gene)
    queries = [
        f"(gene_exact:{g}) AND (organism_id:{taxon_id}) AND (reviewed:true)",
        f"(gene_exact:{g}) AND (organism_id:{taxon_id})",
        f"({g}) AND (organism_id:{taxon_id}) AND (reviewed:true)",
    ]
    for q in queries:
        resp = fetch(
            f"{BASE}/search",
            params={"query": q, "fields": "accession", "format": "json", "size": 1},
            ttl=WEEK,
            source=SOURCE,
        )
        results = resp.json().get("results", [])
        if results:
            return results[0]["primaryAccession"]
    raise NotFound(f"No UniProt entry found for '{gene}' in taxon {taxon_id}.")


def resolve(query: str, taxon_id: int = HUMAN) -> ProteinRecord:
    """Resolve a UniProt accession or gene symbol to a :class:`ProteinRecord`."""
    kind = classify(query)
    if kind is IdKind.PDB:
        raise InvalidIdentifier(
            f"'{query}' looks like a PDB ID; use the structure view for PDB entries."
        )
    acc = normalize(query) if kind is IdKind.UNIPROT else search_gene(query, taxon_id)
    return get_entry(acc)


def parse_entry(entry: dict[str, Any], fetched_at: float | None = None) -> ProteinRecord:
    acc = entry["primaryAccession"]
    desc = entry.get("proteinDescription", {})
    rec = desc.get("recommendedName") or (desc.get("submissionNames") or [{}])[0]
    protein_name = rec.get("fullName", {}).get("value")

    genes = [g["geneName"]["value"] for g in entry.get("genes", []) if "geneName" in g]

    function = None
    locations: list[str] = []
    for c in entry.get("comments", []):
        if c.get("commentType") == "FUNCTION" and c.get("texts") and function is None:
            function = c["texts"][0].get("value")
        elif c.get("commentType") == "SUBCELLULAR LOCATION":
            for loc in c.get("subcellularLocations", []):
                if "location" in loc:
                    locations.append(loc["location"]["value"])

    domains = [
        Domain(
            description=f.get("description", ""),
            start=f["location"]["start"]["value"],
            end=f["location"]["end"]["value"],
        )
        for f in entry.get("features", [])
        if f.get("type") == "Domain"
        and isinstance(f.get("location", {}).get("start", {}).get("value"), int)
        and isinstance(f.get("location", {}).get("end", {}).get("value"), int)
    ]

    pdb_ids = sorted(
        {x["id"] for x in entry.get("uniProtKBCrossReferences", []) if x.get("database") == "PDB"}
    )

    seq = entry.get("sequence", {})
    organism = entry.get("organism", {})
    version = entry.get("entryAudit", {}).get("entryVersion")
    return ProteinRecord(
        accession=acc,
        entry_name=entry.get("uniProtkbId"),
        protein_name=protein_name,
        gene_names=genes,
        organism=organism.get("scientificName"),
        taxon_id=organism.get("taxonId"),
        reviewed="reviewed" in entry.get("entryType", "").lower()
        and "unreviewed" not in entry.get("entryType", "").lower(),
        sequence=seq.get("value", ""),
        length=seq.get("length", 0),
        function=function,
        subcellular_locations=list(dict.fromkeys(locations)),
        keywords=[k.get("name", "") for k in entry.get("keywords", [])],
        domains=domains,
        pdb_ids=pdb_ids,
        provenance=Provenance.at(
            SOURCE,
            f"https://www.uniprot.org/uniprotkb/{acc}/entry",
            version=f"entry v{version}" if version else None,
            fetched_at=fetched_at,
        ),
    )
