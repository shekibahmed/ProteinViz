import pandas as pd
import streamlit as st

from proteinviz.app import common as c
from proteinviz.sequence import properties

c.page_header("🧬 Protein", "Function, disease links, pathways and structures for one protein.")

q = st.text_input("Gene symbol or UniProt accession (human by default)", value=c.param("q", "TP53"))
c.set_params(q=q)
if not q:
    st.stop()

rec = c.attempt(c.protein, q)
if rec is None:
    st.stop()

st.subheader(f"{rec.primary_gene or rec.accession} · {rec.protein_name or ''}")
st.caption(
    f"UniProt [{rec.accession}](https://www.uniprot.org/uniprotkb/{rec.accession}/entry) · "
    f"{rec.organism} · {'reviewed (Swiss-Prot)' if rec.reviewed else 'unreviewed (TrEMBL)'}"
)

props = properties(rec.sequence)
m = st.columns(5)
m[0].metric("Length", f"{props.length} aa")
m[1].metric("Mass", f"{props.molecular_weight_da / 1000:.1f} kDa")
m[2].metric("pI", f"{props.isoelectric_point:.2f}")
m[3].metric("GRAVY", f"{props.gravy:+.2f}", help="Grand average of hydropathy (Kyte–Doolittle)")
m[4].metric("PDB entries", len(rec.pdb_ids))
st.caption(f"Computed from the UniProt sequence with {props.method}.")

g = rec.primary_gene or rec.accession
st.markdown(
    f"**Go to:** [Network & pathways](./network?q={g}) · [Structure](./structure?a={g}) · "
    f"[Open Targets](https://platform.opentargets.org/search?q={g}) · "
    f"[STRING](https://string-db.org/network/9606.{g})"
)

tab_f, tab_d, tab_p, tab_s = st.tabs(["Function", "Diseases", "Pathways", "Sequence & domains"])
with tab_f:
    st.markdown(rec.function or "_No function annotation in UniProt._")
    if rec.subcellular_locations:
        st.markdown("**Subcellular location:** " + ", ".join(rec.subcellular_locations))
    if rec.keywords:
        st.markdown("**Keywords:** " + ", ".join(rec.keywords[:20]))

tp = None
if rec.taxon_id == 9606:
    tp = c.attempt(c.target_profile, rec.accession)
with tab_d:
    if tp is None:
        st.info("Disease associations are shown for human proteins (Open Targets).")
    else:
        st.markdown(f"Top of **{tp.n_associated_diseases:,}** associated diseases (Open Targets):")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Disease": d.name,
                        "Score": d.score,
                        "Explore": f"./disease?id={d.disease_id}&q={d.name}",
                    }
                    for d in tp.diseases
                ]
            ),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
                "Explore": st.column_config.LinkColumn(display_text="open ↗"),
            },
        )
        c.provenance_caption(tp.provenance)
with tab_p:
    if tp is None or not tp.pathways:
        st.info("No Reactome pathway annotations available.")
    else:
        df = pd.DataFrame(
            [
                {"Top-level area": p.top_level, "Pathway": p.name, "Reactome": p.url}
                for p in tp.pathways
            ]
        ).sort_values(["Top-level area", "Pathway"])
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={"Reactome": st.column_config.LinkColumn(display_text=r"detail/(.*)")},
        )
with tab_s:
    if rec.domains:
        st.dataframe(pd.DataFrame([d.model_dump() for d in rec.domains]), hide_index=True)
    seq = rec.sequence
    st.code("\n".join(seq[i : i + 60] for i in range(0, len(seq), 60)), language=None)
    st.download_button(
        "Download FASTA",
        f">{rec.accession}|{rec.entry_name or ''} {rec.protein_name or ''}\n{seq}\n",
        f"{rec.accession}.fasta",
    )

c.provenance_caption(rec.provenance)
