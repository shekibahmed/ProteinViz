import streamlit as st

from proteinviz.app import common as c
from proteinviz.datasets import egcg
from proteinviz.viz.network import network_figure
from proteinviz.viz.plots import enrichment_bar

meta = egcg.metadata()
c.page_header(
    "🍵 EGCG showcase: from a natural compound to disease pathways",
    "Epigallocatechin gallate (EGCG), the main catechin in green tea, is studied for "
    "neurodegeneration, cancer and metabolic disease. This page builds a "
    "**traceable target list from ChEMBL**, sorts it by strength of evidence, and asks which "
    "pathways and diseases the well-supported targets point to.",
)
st.warning(egcg.INTERFERENCE_NOTE, icon="⚠️")


@st.cache_data
def data():
    df = egcg.load()
    return df, egcg.target_summary(df)


acts, targets = data()
m = st.columns(4)
m[0].metric("Activity records", f"{meta['n_activities']:,}")
m[1].metric("With pChEMBL", f"{meta['n_with_pchembl']:,}")
m[2].metric("Protein targets", f"{len(targets):,}")
m[3].metric("Potent in ≥2 papers", int((targets["evidence_tier"] == egcg.TIER_MULTI).sum()))
st.caption(
    f"Built from [{meta['chembl_release']}]({meta['source_url']}) ({meta['chembl_release_date']}) by "
    "`scripts/build_egcg_dataset.py`. Data © EMBL-EBI, CC BY-SA 3.0. 'Potent' = pChEMBL ≥ 6 (≤ 1 µM)."
)

st.subheader("1 · Targets ranked by evidence")
f1, f2 = st.columns(2)
tiers = f1.multiselect(
    "Evidence tier",
    [egcg.TIER_MULTI, egcg.TIER_SINGLE, egcg.TIER_WEAK],
    default=[egcg.TIER_MULTI, egcg.TIER_SINGLE],
)
organisms = sorted(targets["target_organism"].dropna().unique())
org = f2.selectbox(
    "Organism", ["Homo sapiens", "All"] + [o for o in organisms if o != "Homo sapiens"]
)
view = targets[targets["evidence_tier"].isin(tiers)]
if org != "All":
    view = view[view["target_organism"] == org]
st.dataframe(
    view.assign(ChEMBL="https://www.ebi.ac.uk/chembl/explore/target/" + view["target_chembl_id"]),
    hide_index=True,
    use_container_width=True,
    column_config={
        "target_pref_name": "Target",
        "target_organism": "Organism",
        "uniprot_accessions": "UniProt",
        "n_activities": "Activities",
        "n_publications": "Publications",
        "evidence_tier": "Evidence tier",
        "best_pchembl": st.column_config.NumberColumn("Best pChEMBL", format="%.2f"),
        "ChEMBL": st.column_config.LinkColumn(display_text=r"target/(.*)"),
        "target_chembl_id": None,
    },
)

human = view[view["target_organism"] == "Homo sapiens"]
accs = sorted({a for s in human["uniprot_accessions"].dropna() for a in str(s).split(";") if a})

st.subheader("2 · Which pathways do these targets share?")
st.markdown(
    f"STRING enrichment of the **{len(accs)} human targets** selected above, compared with the "
    "genome. This generates hypotheses about disease pathways EGCG *could* modulate. It does not "
    "show that it does in vivo."
)
if len(accs) < 2:
    st.info("Select tiers that include at least two human targets.")
else:
    enr = c.attempt(c.enrichment, accs)
    if enr is not None:
        cats = {
            "KEGG": "KEGG",
            "Reactome": "RCTM",
            "Diseases": "DISEASES",
            "WikiPathways": "WikiPathways",
        }
        pick = st.multiselect("Categories", list(cats), default=["KEGG", "Reactome", "Diseases"])
        terms = enr.by_category(*[cats[p] for p in pick])
        if terms:
            st.plotly_chart(enrichment_bar(terms), use_container_width=True)
        else:
            st.info("No terms pass FDR ≤ 0.05. Try including more tiers.")
        c.provenance_caption(enr.provenance)

    st.subheader("3 · How do the targets connect?")
    net = c.attempt(c.string_network, "\r".join(accs), 9606, 0, 400)
    if net is not None:
        st.plotly_chart(network_figure(net), use_container_width=True)
        c.provenance_caption(net.provenance)

st.subheader("4 · Every activity, with its source")
cols = [
    "target_pref_name",
    "target_organism",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "assay_type",
    "assay_description",
    "evidence_tier",
    "caveats",
    "document_year",
    "doi",
    "pubmed_id",
]
show = acts[acts["evidence_tier"].isin(tiers)][cols]
st.dataframe(
    show.assign(doi="https://doi.org/" + show["doi"].fillna("")).replace(
        {"doi": {"https://doi.org/": None}}
    ),
    hide_index=True,
    use_container_width=True,
    column_config={
        "doi": st.column_config.LinkColumn("DOI", display_text=r"doi.org/(.*)"),
        "pubmed_id": st.column_config.NumberColumn("PMID", format="%d"),
    },
)
st.download_button(
    "Download full dataset (CSV)",
    acts.to_csv(index=False),
    "egcg_chembl_activities.csv",
    "text/csv",
)
