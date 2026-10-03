"""Rebuild the EGCG compound–target activity dataset from ChEMBL.

Usage:
    python scripts/build_egcg_dataset.py [--out src/proteinviz/datasets/egcg]

Every row is a ChEMBL activity record with its assay, target (UniProt
accessions) and source document (DOI / PubMed ID), so each value can be traced
back to the publication it came from. Re-running the script against a newer
ChEMBL release regenerates the files; the release is recorded in metadata.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from proteinviz.datasets.egcg import COLUMNS, EGCG_CHEMBL_ID, annotate
from proteinviz.sources import chembl

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "src" / "proteinviz" / "datasets" / "egcg"


def build() -> tuple[pd.DataFrame, dict]:
    status = chembl.status()
    mol = chembl.molecule(EGCG_CHEMBL_ID)
    acts = chembl.activities(EGCG_CHEMBL_ID)
    tgts = chembl.targets([a["target_chembl_id"] for a in acts if a.get("target_chembl_id")])
    docs = chembl.documents([a["document_chembl_id"] for a in acts if a.get("document_chembl_id")])

    rows = []
    for a in acts:
        t = tgts.get(a.get("target_chembl_id"), {})
        d = docs.get(a.get("document_chembl_id"), {})
        rows.append(
            {
                "activity_id": a["activity_id"],
                "molecule_chembl_id": a["molecule_chembl_id"],
                "assay_chembl_id": a["assay_chembl_id"],
                "assay_type": a.get("assay_type"),
                "assay_description": a.get("assay_description"),
                "standard_type": a.get("standard_type"),
                "standard_relation": a.get("standard_relation"),
                "standard_value": a.get("standard_value"),
                "standard_units": a.get("standard_units"),
                "pchembl_value": a.get("pchembl_value"),
                "data_validity_comment": a.get("data_validity_comment"),
                "potential_duplicate": a.get("potential_duplicate"),
                "target_chembl_id": a.get("target_chembl_id"),
                "target_pref_name": a.get("target_pref_name"),
                "target_type": t.get("target_type"),
                "target_organism": a.get("target_organism"),
                "target_tax_id": a.get("target_tax_id"),
                "uniprot_accessions": ";".join(chembl.uniprot_accessions(t)),
                "document_chembl_id": a.get("document_chembl_id"),
                "document_year": d.get("year") or a.get("document_year"),
                "journal": d.get("journal"),
                "doi": d.get("doi"),
                "pubmed_id": d.get("pubmed_id"),
            }
        )
    df = pd.DataFrame(rows)
    for col in ("standard_value", "pchembl_value"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("target_tax_id", "pubmed_id", "document_year"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = annotate(df)[COLUMNS].sort_values(
        ["target_pref_name", "pchembl_value"], na_position="last"
    )

    structures = mol.get("molecule_structures") or {}
    meta = {
        "compound": mol.get("pref_name"),
        "molecule_chembl_id": EGCG_CHEMBL_ID,
        "canonical_smiles": structures.get("canonical_smiles"),
        "standard_inchi_key": structures.get("standard_inchi_key"),
        "chembl_release": status.get("chembl_db_version"),
        "chembl_release_date": status.get("chembl_release_date"),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_activities": len(df),
        "n_with_pchembl": int(df["pchembl_value"].notna().sum()),
        "n_targets": int(df["target_chembl_id"].nunique()),
        "license": "CC BY-SA 3.0 (derived from ChEMBL)",
        "source_url": f"https://www.ebi.ac.uk/chembl/explore/compound/{EGCG_CHEMBL_ID}",
    }
    return df, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    df, meta = build()
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "egcg_chembl_activities.csv", index=False)
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
