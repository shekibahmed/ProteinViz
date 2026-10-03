import pandas as pd
import streamlit as st

from proteinviz.app import common as c
from proteinviz.viz.network import CHANNELS, network_figure
from proteinviz.viz.plots import enrichment_bar

c.page_header(
    "🕸️ Network & pathways",
    "The interaction neighbourhood of a protein from [STRING](https://string-db.org), coloured by "
    "the strongest evidence type, plus the pathways and diseases over-represented in that "
    "neighbourhood.",
)

col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
q = col1.text_input("Gene symbol or UniProt accession", value=c.param("q", "TP53"))
n = col2.slider("Partners", 5, 50, int(c.param("n", "20") or 20), step=5)
score = col3.select_slider(
    "Confidence",
    options=[150, 400, 700, 900],
    value=int(c.param("score", "700") or 700),
    format_func=lambda s: {150: "low", 400: "medium", 700: "high", 900: "highest"}[s],
)
ntype = col4.radio(
    "Network",
    ["functional", "physical"],
    horizontal=False,
    index=0 if c.param("type", "functional") == "functional" else 1,
    help="Functional = any association; physical = evidence of direct binding / complex membership.",
)
c.set_params(q=q, n=str(n), score=str(score), type=ntype)
if not q:
    st.stop()

net = c.attempt(c.string_network, q.strip(), 9606, n, score, ntype)
if net is None:
    st.stop()

left, right = st.columns([3, 2])
with left:
    st.plotly_chart(network_figure(net, highlight=q.strip()), use_container_width=True)
    c.provenance_caption(net.provenance)
with right:
    st.markdown(f"**{len(net.nodes)} proteins · {len(net.interactions)} interactions**")
    edges = pd.DataFrame(
        [
            {
                "A": i.name_a,
                "B": i.name_b,
                "Combined": i.score,
                **{
                    CHANNELS[ch][0]: getattr(i, ch)
                    for ch in ("experimental", "database", "textmining", "coexpression")
                },
            }
            for i in sorted(net.interactions, key=lambda i: -i.score)
        ]
    )
    st.dataframe(edges, hide_index=True, use_container_width=True, height=420)
    st.download_button(
        "Download edges (CSV)", edges.to_csv(index=False), f"{q}_string_network.csv", "text/csv"
    )

st.subheader("What pathways does this neighbourhood point to?")
enr = c.attempt(c.enrichment, net.nodes)
if enr is not None:
    cats = {
        "KEGG": "KEGG",
        "Reactome": "RCTM",
        "WikiPathways": "WikiPathways",
        "Diseases (DISEASES)": "DISEASES",
        "GO process": "Process",
        "Hallmark": "Hallmark",
    }
    pick = st.multiselect(
        "Categories", list(cats), default=["KEGG", "Reactome", "Diseases (DISEASES)"]
    )
    terms = enr.by_category(*[cats[p] for p in pick])
    if terms:
        st.plotly_chart(enrichment_bar(terms, top=20), use_container_width=True)
        table = pd.DataFrame(
            [
                {
                    "Category": t.category,
                    "Term": t.term,
                    "Description": t.description,
                    "Genes in set": t.n_genes,
                    "FDR": t.fdr,
                    "Genes": ", ".join(t.genes),
                }
                for t in terms
            ]
        )
        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
            column_config={"FDR": st.column_config.NumberColumn(format="%.2e")},
        )
        st.download_button(
            "Download enrichment (CSV)",
            table.to_csv(index=False),
            f"{q}_enrichment.csv",
            "text/csv",
        )
    else:
        st.info("No terms pass FDR ≤ 0.05 in the selected categories.")
    st.caption(
        "Caveat: the network was seeded from one protein, so its neighbourhood is biased toward that "
        "protein's known biology. Treat enrichment as a description of the neighbourhood, not "
        "independent evidence."
    )
    c.provenance_caption(enr.provenance)
