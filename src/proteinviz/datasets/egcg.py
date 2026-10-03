"""EGCG (epigallocatechin gallate) compound–target activity showcase dataset.

Built from ChEMBL by ``scripts/build_egcg_dataset.py``. EGCG is a catechol-rich
polyphenol that is a well-known promiscuous / assay-interference compound
(colloidal aggregation, redox cycling, covalent reactivity). Many reported
in-vitro "targets" therefore need orthogonal confirmation, so every row carries
an evidence tier and interference-relevant caveats.
"""

from __future__ import annotations

import json
from importlib import resources

import pandas as pd

EGCG_CHEMBL_ID = "CHEMBL297453"
POTENT_PCHEMBL = 6.0  # ≤ 1 µM

COLUMNS = [
    "activity_id",
    "molecule_chembl_id",
    "assay_chembl_id",
    "assay_type",
    "assay_description",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "data_validity_comment",
    "potential_duplicate",
    "target_chembl_id",
    "target_pref_name",
    "target_type",
    "target_organism",
    "target_tax_id",
    "uniprot_accessions",
    "document_chembl_id",
    "document_year",
    "journal",
    "doi",
    "pubmed_id",
    "evidence_tier",
    "caveats",
]

ASSAY_TYPES = {
    "B": "Binding",
    "F": "Functional",
    "A": "ADME",
    "T": "Toxicity",
    "P": "Physicochemical",
    "U": "Unclassified",
}

PROTEIN_TARGET_TYPES = {
    "SINGLE PROTEIN",
    "PROTEIN COMPLEX",
    "PROTEIN FAMILY",
    "PROTEIN COMPLEX GROUP",
    "SELECTIVITY GROUP",
    "PROTEIN-PROTEIN INTERACTION",
    "CHIMERIC PROTEIN",
    "PROTEIN NUCLEIC-ACID COMPLEX",
}

TIER_MULTI = "Potent in ≥2 publications"
TIER_SINGLE = "Potent in 1 publication"
TIER_WEAK = "Weak or non-quantitative"

INTERFERENCE_NOTE = (
    "EGCG is a catechol/pyrogallol polyphenol and a known pan-assay interference compound "
    "(PAINS): it forms colloidal aggregates, generates H₂O₂ by redox cycling and reacts "
    "covalently with proteins. Biochemical activity alone is weak evidence of a specific "
    "target; prefer targets potent across independent publications and confirmed in "
    "orthogonal or cellular assays."
)


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``evidence_tier`` and ``caveats`` columns (pure function of the activity table)."""
    out = df.copy()
    pchembl = pd.to_numeric(out["pchembl_value"], errors="coerce")
    potent = pchembl >= POTENT_PCHEMBL
    docs_per_target = (
        out[potent].groupby("target_chembl_id")["document_chembl_id"].nunique()
        if potent.any()
        else pd.Series(dtype=int)
    )
    n_docs = out["target_chembl_id"].map(docs_per_target).fillna(0).astype(int)
    out["evidence_tier"] = TIER_WEAK
    out.loc[potent & (n_docs == 1), "evidence_tier"] = TIER_SINGLE
    out.loc[potent & (n_docs >= 2), "evidence_tier"] = TIER_MULTI

    def caveats(row) -> str:
        f = []
        if isinstance(row.get("data_validity_comment"), str) and row["data_validity_comment"]:
            f.append(f"ChEMBL validity: {row['data_validity_comment']}")
        if row.get("potential_duplicate") in (1, True, "1", "True"):
            f.append("potential duplicate")
        if row.get("target_type") and row["target_type"] not in PROTEIN_TARGET_TYPES:
            f.append(f"non-protein target ({str(row['target_type']).lower()})")
        if row.get("standard_relation") in (">", ">=", "<", "<="):
            f.append(f"censored value ({row['standard_relation']})")
        return "; ".join(f)

    out["caveats"] = out.apply(caveats, axis=1)
    return out


def load() -> pd.DataFrame:
    """Load the bundled EGCG activity table."""
    path = resources.files("proteinviz.datasets") / "egcg" / "egcg_chembl_activities.csv"
    with resources.as_file(path) as p:
        df = pd.read_csv(p, dtype={"uniprot_accessions": "string", "doi": "string"})
    return df


def metadata() -> dict:
    path = resources.files("proteinviz.datasets") / "egcg" / "metadata.json"
    return json.loads(path.read_text())


def target_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per protein target: best pChEMBL, publications, assays and evidence tier."""
    prot = df[df["target_type"].isin(PROTEIN_TARGET_TYPES)].copy()
    prot["pchembl_value"] = pd.to_numeric(prot["pchembl_value"], errors="coerce")
    tier_rank = {TIER_MULTI: 0, TIER_SINGLE: 1, TIER_WEAK: 2}
    g = prot.groupby(["target_chembl_id", "target_pref_name", "target_organism"], dropna=False)
    summary = g.agg(
        uniprot_accessions=("uniprot_accessions", "first"),
        best_pchembl=("pchembl_value", "max"),
        n_activities=("activity_id", "count"),
        n_publications=("document_chembl_id", "nunique"),
        evidence_tier=("evidence_tier", lambda s: min(s, key=lambda t: tier_rank.get(t, 3))),
    ).reset_index()
    summary["_rank"] = summary["evidence_tier"].map(tier_rank)
    return (
        summary.sort_values(["_rank", "best_pchembl"], ascending=[True, False], na_position="last")
        .drop(columns="_rank")
        .reset_index(drop=True)
    )
