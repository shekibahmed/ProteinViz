"""Source clients against recorded responses (or live with PROTEINVIZ_LIVE=1)."""

import numpy as np
import pytest

from proteinviz.errors import InvalidIdentifier, NotFound
from proteinviz.sources import alphafold, opentargets, rcsb, string_db, uniprot


def test_uniprot_resolve_gene_and_accession():
    by_gene = uniprot.resolve("TP53")
    assert by_gene.accession == "P04637"
    assert by_gene.primary_gene == "TP53"
    assert by_gene.reviewed
    assert by_gene.length == len(by_gene.sequence) == 393
    assert "1YCR" in by_gene.pdb_ids
    assert by_gene.provenance.url.endswith("P04637/entry")
    assert uniprot.resolve("P04637").accession == "P04637"


def test_uniprot_alias_falls_back_to_name_search():
    assert uniprot.resolve("P53").accession == "P04637"


def test_uniprot_unknown_gene_raises():
    with pytest.raises(NotFound):
        uniprot.resolve("NOTAREALGENEXYZ")


def test_uniprot_rejects_pdb_id():
    with pytest.raises(InvalidIdentifier):
        uniprot.resolve("1YCR")


def test_string_network_tp53():
    net = string_db.network("TP53", limit=10, required_score=700)
    assert "TP53" in net.nodes
    assert len(net.nodes) == 11  # query + 10 partners
    # Many TP53 partners tie at 0.999, so check thresholds rather than specific names.
    assert all(i.score >= 0.7 for i in net.interactions if "TP53" in (i.name_a, i.name_b))
    assert net.provenance.version and net.provenance.version.startswith("v12")


def test_string_pair_evidence():
    ev = string_db.pair_evidence("TP53", "MDM2")
    assert ev is not None and ev.score > 0.9 and ev.experimental > 0.5


def test_string_enrichment_finds_p53_pathway():
    enr = string_db.enrichment(["TP53", "MDM2", "CDKN1A", "ATM", "CHEK2", "BAX"])
    kegg = {t.description for t in enr.by_category("KEGG")}
    assert "p53 signaling pathway" in kegg
    assert all(t.fdr <= 0.05 for t in enr.terms)


def test_string_enrichment_needs_two_genes():
    with pytest.raises(ValueError):
        string_db.enrichment(["TP53"])


def test_rcsb_pair_search_finds_1ycr():
    ids, total = rcsb.search_entries("P04637", "Q00987")
    assert "1YCR" in ids and total >= 1


def test_rcsb_entry_metadata_maps_chains():
    entry = rcsb.get_entry("1YCR")
    assert entry.method == "X-RAY DIFFRACTION"
    assert entry.resolution == pytest.approx(2.6)
    assert entry.chains_for("P04637") == ["B"]
    assert entry.chains_for("Q00987") == ["A"]


def test_rcsb_unknown_entry():
    with pytest.raises(NotFound):
        rcsb.get_entry("0ZZZ")


def test_alphafold_model_uses_api_urls():
    model = alphafold.get_model("P04637")
    assert model.model_entity_id == "AF-P04637-F1"
    assert model.model_version >= 4
    assert f"model_v{model.model_version}" in model.cif_url
    residues, scores = alphafold.fetch_plddt(model)
    assert len(residues) == len(scores) == model.sequence_end - model.sequence_start + 1
    pae = alphafold.fetch_pae(model)
    assert pae.shape == (len(residues), len(residues))
    assert np.all(pae >= 0)


def test_plddt_segments_and_bands():
    segs = alphafold.plddt_segments([1, 2, 3, 4, 5], [95, 92, 60, 61, 30])
    assert segs == [(1, 2, "#0053D6"), (3, 4, "#FFDB13"), (5, 5, "#FF7D45")]
    assert alphafold.plddt_band(90.0)[1] == "#65CBF3"


def test_opentargets_disease_profile():
    hits = opentargets.search_diseases("Alzheimer disease", 3)
    assert hits[0].disease_id == "MONDO_0004975"
    prof = opentargets.disease_profile("MONDO_0004975", 10)
    symbols = [t.symbol for t in prof.targets]
    assert "APP" in symbols and "PSEN1" in symbols
    assert prof.n_drug_candidates > 0
    stages = [d.max_stage for d in prof.drugs]
    ranks = [opentargets._stage_rank(s) for s in stages]
    assert ranks == sorted(ranks)


def test_opentargets_target_profile():
    tp = opentargets.target_profile("P04637")
    assert tp.symbol == "TP53"
    assert any("Li-Fraumeni" in d.name for d in tp.diseases)
    assert all(p.pathway_id.startswith("R-HSA-") for p in tp.pathways)
