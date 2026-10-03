"""Open Targets Platform GraphQL client (https://platform.opentargets.org/api).

Open Targets integrates genetic, somatic, literature, pathway and drug evidence
into disease–target association scores. Data are CC0.
"""

from __future__ import annotations

from typing import Any

from proteinviz.errors import NotFound, SourceUnavailable
from proteinviz.http import WEEK, fetch
from proteinviz.models import (
    DiseaseAssociation,
    DiseaseHit,
    DiseaseProfile,
    DrugCandidate,
    Pathway,
    Provenance,
    TargetAssociation,
    TargetProfile,
)

URL = "https://api.platform.opentargets.org/api/v4/graphql"
SOURCE = "Open Targets Platform"

# Clinical stages ordered from most to least advanced, for sorting drug candidates.
STAGE_ORDER = [
    "APPROVAL",
    "PHASE_4",
    "PHASE_3",
    "PHASE_2_3",
    "PHASE_2",
    "PHASE_1_2",
    "PHASE_1",
    "EARLY_PHASE_1",
    "PRECLINICAL",
    "UNKNOWN",
]


def _query(query: str, variables: dict[str, Any]) -> tuple[dict, float]:
    resp = fetch(
        URL,
        method="POST",
        json_body={"query": query, "variables": variables},
        ttl=WEEK,
        source=SOURCE,
    )
    payload = resp.json()
    if payload.get("errors"):
        raise SourceUnavailable(f"{SOURCE}: {payload['errors'][0].get('message')}")
    return payload.get("data") or {}, resp.fetched_at


def data_version() -> str | None:
    data, _ = _query("{ meta { dataVersion { year month } } }", {})
    v = (data.get("meta") or {}).get("dataVersion") or {}
    return f"{v['year']}.{v['month']}" if v.get("year") else None


def search_diseases(text: str, size: int = 10) -> list[DiseaseHit]:
    q = """query($q: String!, $n: Int!) {
      search(queryString: $q, entityNames: ["disease"], page: {index: 0, size: $n}) { hits { id name } }
    }"""
    data, _ = _query(q, {"q": text, "n": size})
    return [DiseaseHit(disease_id=h["id"], name=h["name"]) for h in data["search"]["hits"]]


def ensembl_for(symbol_or_accession: str) -> str:
    """Resolve a gene symbol or UniProt accession to an Ensembl gene ID."""
    q = """query($q: String!) {
      search(queryString: $q, entityNames: ["target"], page: {index: 0, size: 1}) { hits { id } }
    }"""
    data, _ = _query(q, {"q": symbol_or_accession})
    hits = data["search"]["hits"]
    if not hits:
        raise NotFound(f"Open Targets has no target matching '{symbol_or_accession}'.")
    return hits[0]["id"]


_DISEASE_Q = """query($id: String!, $n: Int!) {
  disease(efoId: $id) {
    id name description
    therapeuticAreas { name }
    associatedTargets(page: {index: 0, size: $n}) {
      count
      rows { score target { id approvedSymbol approvedName proteinIds { id source } } }
    }
    drugAndClinicalCandidates {
      count
      rows {
        maxClinicalStage
        drug { id name drugType
               mechanismsOfAction { rows { mechanismOfAction targets { approvedSymbol } } } }
      }
    }
  }
}"""


def disease_profile(disease_id: str, n_targets: int = 25) -> DiseaseProfile:
    """Top associated targets and known drug candidates for a disease (EFO/MONDO ID)."""
    data, fetched_at = _query(_DISEASE_Q, {"id": disease_id, "n": n_targets})
    d = data.get("disease")
    if not d:
        raise NotFound(f"Open Targets has no disease '{disease_id}'.")
    targets = []
    for row in d["associatedTargets"]["rows"]:
        t = row["target"]
        swissprot = [
            p["id"] for p in t.get("proteinIds") or [] if p["source"] == "uniprot_swissprot"
        ]
        targets.append(
            TargetAssociation(
                ensembl_id=t["id"],
                symbol=t["approvedSymbol"],
                name=t.get("approvedName"),
                uniprot=swissprot[0] if swissprot else None,
                score=round(row["score"], 4),
            )
        )
    drugs: dict[str, DrugCandidate] = {}
    for row in d["drugAndClinicalCandidates"]["rows"]:
        drug = row.get("drug") or {}
        if not drug.get("id"):
            continue
        moa = ((drug.get("mechanismsOfAction") or {}).get("rows")) or []
        cand = DrugCandidate(
            chembl_id=drug["id"],
            name=drug.get("name"),
            drug_type=drug.get("drugType"),
            max_stage=row.get("maxClinicalStage") or "UNKNOWN",
            mechanisms=sorted({m["mechanismOfAction"] for m in moa if m.get("mechanismOfAction")}),
            targets=sorted({t["approvedSymbol"] for m in moa for t in (m.get("targets") or [])}),
        )
        prev = drugs.get(cand.chembl_id)
        if prev is None or _stage_rank(cand.max_stage) < _stage_rank(prev.max_stage):
            drugs[cand.chembl_id] = cand
    ranked = sorted(drugs.values(), key=lambda c: (_stage_rank(c.max_stage), c.name or ""))
    return DiseaseProfile(
        disease_id=d["id"],
        name=d["name"],
        description=d.get("description"),
        therapeutic_areas=[a["name"] for a in d.get("therapeuticAreas") or []],
        n_associated_targets=d["associatedTargets"]["count"],
        targets=targets,
        n_drug_candidates=d["drugAndClinicalCandidates"]["count"],
        drugs=ranked,
        provenance=Provenance.at(
            SOURCE, f"https://platform.opentargets.org/disease/{d['id']}", fetched_at=fetched_at
        ),
    )


_TARGET_Q = """query($id: String!, $n: Int!) {
  target(ensemblId: $id) {
    id approvedSymbol
    associatedDiseases(page: {index: 0, size: $n}) { count rows { score disease { id name } } }
    pathways { pathwayId pathway topLevelTerm }
  }
}"""


def target_profile(symbol_or_accession: str, n_diseases: int = 15) -> TargetProfile:
    """Top associated diseases and Reactome pathways for a target."""
    ensembl = ensembl_for(symbol_or_accession)
    data, fetched_at = _query(_TARGET_Q, {"id": ensembl, "n": n_diseases})
    t = data.get("target")
    if not t:
        raise NotFound(f"Open Targets has no target '{ensembl}'.")
    seen: set[str] = set()
    pathways = []
    for p in t.get("pathways") or []:
        if p["pathwayId"] not in seen:
            seen.add(p["pathwayId"])
            pathways.append(
                Pathway(
                    pathway_id=p["pathwayId"], name=p["pathway"], top_level=p.get("topLevelTerm")
                )
            )
    return TargetProfile(
        ensembl_id=t["id"],
        symbol=t["approvedSymbol"],
        n_associated_diseases=t["associatedDiseases"]["count"],
        diseases=[
            DiseaseAssociation(
                disease_id=r["disease"]["id"], name=r["disease"]["name"], score=round(r["score"], 4)
            )
            for r in t["associatedDiseases"]["rows"]
        ],
        pathways=pathways,
        provenance=Provenance.at(
            SOURCE, f"https://platform.opentargets.org/target/{t['id']}", fetched_at=fetched_at
        ),
    )


def _stage_rank(stage: str) -> int:
    return STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER)
