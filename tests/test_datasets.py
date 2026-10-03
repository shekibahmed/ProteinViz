import pandas as pd

from proteinviz.datasets import egcg
from proteinviz.datasets.schema import validate


def test_bundled_egcg_dataset_schema_and_provenance():
    df = egcg.load()
    meta = egcg.metadata()
    assert list(df.columns) == egcg.COLUMNS
    assert len(df) == meta["n_activities"] > 500
    assert (df["molecule_chembl_id"] == egcg.EGCG_CHEMBL_ID).all()
    assert df["activity_id"].is_unique
    for col in ("assay_chembl_id", "target_chembl_id", "document_chembl_id", "evidence_tier"):
        assert df[col].notna().all(), col
    # Nearly every record is traceable to a DOI or PubMed ID.
    traceable = df["doi"].notna() | df["pubmed_id"].notna()
    assert traceable.mean() > 0.95
    assert meta["chembl_release"].startswith("ChEMBL_")
    assert "CC BY-SA" in meta["license"]


def test_egcg_target_summary_ranks_by_evidence():
    s = egcg.target_summary(egcg.load())
    tiers = list(s["evidence_tier"])
    order = [egcg.TIER_MULTI, egcg.TIER_SINGLE, egcg.TIER_WEAK]
    assert [order.index(t) for t in tiers] == sorted(order.index(t) for t in tiers)
    assert (s["n_publications"] >= 1).all()


def test_annotate_tiers_and_caveats():
    df = pd.DataFrame(
        {
            "target_chembl_id": ["T1", "T1", "T2", "T3"],
            "document_chembl_id": ["D1", "D2", "D1", "D3"],
            "pchembl_value": [6.5, 7.0, 6.1, 4.0],
            "data_validity_comment": [None, None, "Outside typical range", None],
            "potential_duplicate": [0, 0, 1, 0],
            "target_type": ["SINGLE PROTEIN", "SINGLE PROTEIN", "SINGLE PROTEIN", "ORGANISM"],
            "standard_relation": ["=", "=", "=", ">"],
        }
    )
    out = egcg.annotate(df)
    assert list(out["evidence_tier"]) == [
        egcg.TIER_MULTI,
        egcg.TIER_MULTI,
        egcg.TIER_SINGLE,
        egcg.TIER_WEAK,
    ]
    assert "Outside typical range" in out.loc[2, "caveats"]
    assert "potential duplicate" in out.loc[2, "caveats"]
    assert "non-protein target" in out.loc[3, "caveats"]
    assert "censored" in out.loc[3, "caveats"]
    assert out.loc[0, "caveats"] == ""


def test_validate_ppi_table():
    df = pd.DataFrame(
        {
            " Protein_A ": ["TP53", "", "SNCA"],
            "protein_b": ["MDM2", "X", "SNCA"],
            "score": ["0.9", "1", "high"],
        }
    )
    res = validate(df)
    assert res.ok and res.kind == "ppi"
    assert len(res.frame) == 2
    assert any("empty" in w for w in res.warnings)
    assert any("non-numeric" in w for w in res.warnings)
    assert any("self-interaction" in w for w in res.warnings)


def test_validate_compound_target_and_rejects_unknown():
    assert (
        validate(pd.DataFrame({"compound": ["EGCG"], "target": ["DYRK1A"]})).kind
        == "compound_target"
    )
    bad = validate(pd.DataFrame({"foo": [1]}))
    assert not bad.ok and "Missing required columns" in bad.errors[0]
    assert not validate(pd.DataFrame({"protein_a": [], "protein_b": []})).ok
