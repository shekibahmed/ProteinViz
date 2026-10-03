import streamlit as st

from proteinviz.app import common as c
from proteinviz.app.navigation import page

st.title("🧬 ProteinViz")
st.markdown(
    "#### Explore disease pathways from genes to proteins, interactions, 3D structures and compounds."
)
st.markdown(
    """
ProteinViz connects public, peer-reviewed resources so you can go from **a disease** to the
**proteins and pathways behind it**, see **how those proteins interact** and **what their
interfaces look like**, and check **which compounds and drugs act on them**. Everything comes
from live databases and is traced back to its source. Nothing is simulated.
"""
)

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("1 · Start from a disease")
    st.markdown("Top genetic and clinical targets, enriched pathways, and drugs already in trials.")
    st.page_link(page("disease"), label="Disease explorer", icon="🩺")
    st.caption("e.g. Alzheimer disease, Parkinson disease, type 2 diabetes")
with c2:
    st.subheader("2 · Map the network")
    st.markdown("STRING interaction partners, evidence channels and KEGG/Reactome enrichment.")
    st.page_link(page("network"), label="Network & pathways", icon="🕸️")
    st.caption("e.g. TP53, APP, SNCA")
with c3:
    st.subheader("3 · Look at the structure")
    st.markdown("AlphaFold confidence, solved complexes and residue-level interfaces in Mol*.")
    st.page_link(page("structure"), label="Structure & interfaces", icon="🔬")
    st.caption("e.g. TP53 + MDM2 → PDB 1YCR")

st.divider()
st.subheader("Try a worked example")
e1, e2, e3 = st.columns(3)
e1.page_link(
    page("disease"),
    label="Alzheimer disease → targets & pathways",
    icon="➡️",
    query_params={"id": "MONDO_0004975"},
)
e2.page_link(
    page("structure"),
    label="p53 · MDM2 interface (1YCR)",
    icon="➡️",
    query_params={"a": "TP53", "b": "MDM2", "pdb": "1YCR"},
)
e3.page_link(page("egcg"), label="Green-tea EGCG: evidence-tiered targets", icon="➡️")

st.divider()
st.markdown(
    f"""
**Data sources:** [Open Targets](https://platform.opentargets.org) ·
[STRING](https://string-db.org) · [UniProt](https://www.uniprot.org) ·
[RCSB PDB](https://www.rcsb.org) · [AlphaFold DB](https://alphafold.ebi.ac.uk) ·
[ChEMBL](https://www.ebi.ac.uk/chembl/) · [Reactome](https://reactome.org)

**Related research paper:** the EGCG showcase follows on from a published study of Assam tea
that the author of ProteinViz co-authored. {c.TEA_PAPER_CITATION}
See **About & cite** for a summary.

ProteinViz is a hypothesis-generation tool for researchers. Associations and pathway enrichments
are evidence to follow up, not conclusions. See **About & cite** for methods and limitations.
"""
)
