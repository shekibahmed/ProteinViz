import io

import pandas as pd
import streamlit as st

from proteinviz.app import common as c
from proteinviz.datasets.schema import KINDS, validate
from proteinviz.viz.plots import enrichment_bar

c.page_header(
    "📤 Your data",
    "Upload your own gene list, interaction table or compound–target table and run the same "
    "pathway analysis on it. Files are processed in memory for this session only and never stored.",
)

with st.expander("Accepted formats"):
    st.markdown(
        """
* **PPI edge list:** columns `protein_a`, `protein_b`; optional `score`, `source`.
* **Compound–target table:** columns `compound`, `target`; optional `value`, `units`, `reference`.

Use gene symbols or UniProt accessions (human). CSV or TSV.
"""
    )
    st.download_button(
        "Example PPI CSV",
        "protein_a,protein_b,score,source\nTP53,MDM2,0.99,PDB 1YCR\n"
        "SNCA,LRRK2,0.7,literature\nAPP,PSEN1,0.9,literature\n",
        "example_ppi.csv",
    )

genes_text = st.text_area(
    "…or paste a gene list (one per line)", placeholder="SNCA\nLRRK2\nPRKN\nPINK1\nGBA1"
)
up = st.file_uploader("Upload CSV/TSV", type=["csv", "tsv", "txt"])

genes: list[str] = []
if up is not None:
    raw = up.getvalue().decode("utf-8", errors="replace")
    sep = "\t" if up.name.endswith((".tsv", ".txt")) or raw.count("\t") > raw.count(",") else ","
    try:
        df = pd.read_csv(io.StringIO(raw), sep=sep)
    except Exception as exc:  # pandas raises several parser error types
        st.error(f"Could not parse the file: {exc}")
        st.stop()
    result = validate(df)
    for w in result.warnings:
        st.warning(w)
    if not result.ok:
        for e in result.errors:
            st.error(e)
        st.stop()
    st.success(
        f"Detected a **{result.kind.replace('_', '–')}** table with {len(result.frame)} rows."
    )
    st.dataframe(result.frame.head(200), use_container_width=True, hide_index=True)
    cols = KINDS[result.kind]["required"]
    genes = (
        (list(result.frame[cols[0]]) + list(result.frame[cols[1]]))
        if result.kind == "ppi"
        else list(result.frame["target"])
    )
elif genes_text.strip():
    genes = [g.strip() for g in genes_text.replace(",", "\n").splitlines() if g.strip()]

genes = list(dict.fromkeys(genes))
if len(genes) < 2:
    st.info("Provide at least two genes or proteins to run pathway enrichment.")
    st.stop()

st.subheader(f"Pathway enrichment for {len(genes)} genes/proteins")
enr = c.attempt(c.enrichment, genes)
if enr is not None:
    cats = {
        "KEGG": "KEGG",
        "Reactome": "RCTM",
        "Diseases": "DISEASES",
        "GO process": "Process",
        "WikiPathways": "WikiPathways",
    }
    pick = st.multiselect("Categories", list(cats), default=["KEGG", "Reactome", "Diseases"])
    terms = enr.by_category(*[cats[p] for p in pick])
    if terms:
        st.plotly_chart(enrichment_bar(terms, top=20), use_container_width=True)
        table = pd.DataFrame(
            [
                {
                    "Category": t.category,
                    "Term": t.term,
                    "Description": t.description,
                    "FDR": t.fdr,
                    "Genes": ", ".join(t.genes),
                }
                for t in terms
            ]
        )
        st.download_button("Download enrichment (CSV)", table.to_csv(index=False), "enrichment.csv")
    else:
        st.info("No enriched terms at FDR ≤ 0.05.")
    c.provenance_caption(enr.provenance)
