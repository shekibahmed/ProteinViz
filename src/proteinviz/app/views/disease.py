import pandas as pd
import streamlit as st

from proteinviz.app import common as c
from proteinviz.viz.network import network_figure
from proteinviz.viz.plots import enrichment_bar

c.page_header(
    "🩺 Disease explorer",
    "Find the proteins most strongly linked to a disease, the pathways they share, how they "
    "interact, and which drugs already target them. Associations come from the "
    "[Open Targets Platform](https://platform.opentargets.org) (genetics, somatic mutations, "
    "known drugs, pathways, literature); pathway enrichment and networks come from STRING.",
)

query = st.text_input(
    "Search a disease",
    value=c.param("q", "Alzheimer disease"),
    placeholder="e.g. Parkinson disease, asthma, breast carcinoma",
)
disease_id = c.param("id")
hits = c.attempt(c.disease_search, query, 10) if query else []
if hits:
    ids = [h.disease_id for h in hits]
    idx = ids.index(disease_id) if disease_id in ids else 0
    choice = st.selectbox(
        "Disease", hits, index=idx, format_func=lambda h: f"{h.name} ({h.disease_id})"
    )
    disease_id = choice.disease_id
elif query:
    st.info("No matching disease. Try a broader term.")
c.set_params(q=query, id=disease_id)
if not disease_id:
    st.stop()

n_targets = st.slider("Number of top targets", 10, 50, 25, step=5)
profile = c.attempt(c.disease_profile, disease_id, n_targets)
if profile is None:
    st.stop()

st.subheader(profile.name)
if profile.therapeutic_areas:
    st.caption("Therapeutic areas: " + ", ".join(profile.therapeutic_areas))
if profile.description:
    st.markdown(profile.description)
m1, m2, m3 = st.columns(3)
m1.metric("Associated targets", f"{profile.n_associated_targets:,}")
m2.metric("Drug / clinical candidates", f"{profile.n_drug_candidates:,}")
approved = sum(1 for d in profile.drugs if d.max_stage == "APPROVAL")
m3.metric("Approved drugs", approved)
c.provenance_caption(profile.provenance)

tab_t, tab_p, tab_n, tab_d = st.tabs(
    ["Top targets", "Shared pathways", "Interaction network", "Drugs"]
)

symbols = [t.symbol for t in profile.targets]
with tab_t:
    df = pd.DataFrame(
        [
            {
                "Gene": t.symbol,
                "Protein": t.name,
                "UniProt": t.uniprot,
                "Association score": t.score,
                "Explore": f"./protein?q={t.symbol}",
            }
            for t in profile.targets
        ]
    )
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Association score": st.column_config.ProgressColumn(
                min_value=0, max_value=1, format="%.3f"
            ),
            "Explore": st.column_config.LinkColumn(display_text="open ↗"),
        },
    )
    st.caption("Open Targets overall association score (0–1), integrating all evidence types.")

with tab_p:
    st.markdown(
        "Pathways over-represented among the top targets compared with the whole genome "
        "(STRING functional enrichment, Benjamini–Hochberg FDR ≤ 0.05)."
    )
    enr = c.attempt(c.enrichment, symbols)
    if enr is not None:
        cats = {
            "KEGG": "KEGG",
            "Reactome": "RCTM",
            "WikiPathways": "WikiPathways",
            "GO process": "Process",
            "Diseases": "DISEASES",
        }
        pick = st.multiselect("Categories", list(cats), default=["KEGG", "Reactome"])
        terms = enr.by_category(*[cats[p] for p in pick])
        if terms:
            st.plotly_chart(enrichment_bar(terms), use_container_width=True)
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Category": t.category,
                            "Term": t.term,
                            "Description": t.description,
                            "Genes": ", ".join(t.genes),
                            "FDR": t.fdr,
                        }
                        for t in terms
                    ]
                ),
                hide_index=True,
                use_container_width=True,
                column_config={"FDR": st.column_config.NumberColumn(format="%.2e")},
            )
        else:
            st.info("No enriched terms in the selected categories.")
        c.provenance_caption(enr.provenance)

with tab_n:
    st.markdown(
        "STRING interactions **among** the top targets. Densely connected groups hint at shared "
        "mechanisms; hubs are candidates for follow-up."
    )
    score = st.select_slider(
        "Minimum STRING confidence",
        options=[150, 400, 700, 900],
        value=400,
        format_func=lambda s: {150: "low", 400: "medium", 700: "high", 900: "highest"}[s],
    )
    net = c.attempt(c.string_network, "\r".join(symbols), 9606, 0, score)
    if net is not None:
        st.plotly_chart(network_figure(net), use_container_width=True)
        c.provenance_caption(net.provenance)

with tab_d:
    if not profile.drugs:
        st.info("No drug or clinical candidates recorded for this disease.")
    else:
        stages = sorted({d.max_stage for d in profile.drugs})
        keep = st.multiselect(
            "Clinical stage",
            stages,
            default=[s for s in stages if s in ("APPROVAL", "PHASE_3", "PHASE_4")] or stages,
        )
        rows = [d for d in profile.drugs if d.max_stage in keep]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Drug": d.name,
                        "ChEMBL": f"https://www.ebi.ac.uk/chembl/explore/compound/{d.chembl_id}",
                        "Type": d.drug_type,
                        "Max stage": d.max_stage,
                        "Mechanism": "; ".join(d.mechanisms),
                        "Targets": ", ".join(d.targets),
                    }
                    for d in rows
                ]
            ),
            hide_index=True,
            use_container_width=True,
            column_config={"ChEMBL": st.column_config.LinkColumn(display_text=r"compound/(.*)")},
        )
        targeted = {t for d in profile.drugs for t in d.targets}
        untargeted = [s for s in symbols if s not in targeted]
        if untargeted:
            st.markdown(
                "**Top-associated targets with no drug for this disease in Open Targets:** "
                + ", ".join(c.protein_link(s) for s in untargeted[:20])
            )
            st.caption(
                "A starting point for discovery, not a recommendation. Check tractability "
                "and safety in Open Targets before prioritising."
            )
