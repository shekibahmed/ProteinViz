import pandas as pd
import streamlit as st
from molviewspec import molstar_streamlit

from proteinviz.app import common as c
from proteinviz.sources import alphafold, rcsb
from proteinviz.structure.interface import compute_interface, contacting_chain_pairs
from proteinviz.viz.molstar import alphafold_scene, complex_scene
from proteinviz.viz.plots import pae_heatmap

c.page_header(
    "🔬 Structure & interfaces",
    "Predicted structure confidence from AlphaFold DB, experimentally solved complexes from the "
    "PDB, and **interface residues computed from the real coordinates** (heavy atoms within the "
    "cutoff of the partner chain).",
)

c1, c2, c3 = st.columns([2, 2, 1])
a_q = c1.text_input("Protein A (gene or UniProt)", value=c.param("a", "TP53"))
b_q = c2.text_input("Protein B (optional, finds complexes)", value=c.param("b", "MDM2"))
pdb_q = c3.text_input("PDB ID (optional)", value=c.param("pdb", ""))
c.set_params(a=a_q, b=b_q, pdb=pdb_q)


@st.cache_data(ttl=c.TTL, show_spinner="Computing interface…")
def interface_for(pdb_id: str, chain_a: str, chain_b: str, cutoff: float):
    return compute_interface(c.pdb_mmcif(pdb_id), chain_a, chain_b, pdb_id=pdb_id, cutoff=cutoff)


@st.cache_data(ttl=c.TTL, show_spinner=False)
def chain_pairs(pdb_id: str):
    return contacting_chain_pairs(c.pdb_mmcif(pdb_id))


rec_a = c.attempt(c.protein, a_q) if a_q else None
rec_b = c.attempt(c.protein, b_q) if b_q else None

tab_cx, tab_af = st.tabs(["Complexes & interface", "AlphaFold model (A)"])


def complexes_tab():
    pdb_ids: list[str] = []
    if pdb_q:
        pdb_ids = [pdb_q.strip().upper()]
    elif rec_a and rec_b:
        found = c.attempt(c.pdb_search, rec_a.accession, rec_b.accession, rows=25)
        if found:
            pdb_ids, total = found
            st.success(
                f"{total} experimental PDB entr{'y' if total == 1 else 'ies'} contain both "
                f"{rec_a.primary_gene} and {rec_b.primary_gene} (best resolution first)."
            )
        if rec_a and rec_b:
            ev = c.attempt(
                c.pair_evidence,
                rec_a.primary_gene or rec_a.accession,
                rec_b.primary_gene or rec_b.accession,
            )
            if ev:
                st.caption(
                    f"STRING combined score {ev.score:.3f} (experimental {ev.experimental:.3f}, "
                    f"curated databases {ev.database:.3f}, text mining {ev.textmining:.3f})."
                )
    elif rec_a:
        found = c.attempt(c.pdb_search, rec_a.accession, rows=25)
        if found:
            pdb_ids, total = found
            st.info(
                f"{total} experimental PDB entries contain {rec_a.primary_gene}. Add protein B to find complexes."
            )

    if not pdb_ids:
        if rec_a and rec_b:
            st.warning(
                "No experimentally solved complex of this pair is in the PDB. You can predict one "
                "for free with [AlphaFold Server](https://alphafoldserver.com) (non-commercial) or "
                "Boltz-2 on a free Colab GPU. Import of predicted complexes is on the roadmap."
            )
        return

    entries = c.attempt(c.pdb_entries, pdb_ids)
    if not entries:
        return
    table = pd.DataFrame(
        [
            {
                "PDB": e.pdb_id,
                "Title": e.title,
                "Method": e.method,
                "Resolution (Å)": e.resolution,
                "Released": e.release_date,
            }
            for e in entries
        ]
    )
    st.dataframe(
        table, hide_index=True, use_container_width=True, height=min(38 * (len(table) + 1), 280)
    )
    entry = st.selectbox(
        "Entry", entries, format_func=lambda e: f"{e.pdb_id} · {e.title or ''}"[:110]
    )

    # Which chains to compare: by protein identity when both are given, else largest contact.
    chains_a = entry.chains_for(rec_a.accession) if rec_a else []
    chains_b = entry.chains_for(rec_b.accession) if rec_b else []
    pairs = c.attempt(chain_pairs, entry.pdb_id) or []
    candidate = [
        (x, y)
        for x, y, _ in pairs
        if (x in chains_a and y in chains_b) or (y in chains_a and x in chains_b)
    ]
    options = candidate or [(x, y) for x, y, _ in pairs]
    if not options:
        st.info("No contacting protein chains in this entry.")
        return

    def label(p):
        names = {ch: e.description or "" for e in entry.entities for ch in e.chains}
        return f"{p[0]} ({names.get(p[0], '')[:30]}) ↔ {p[1]} ({names.get(p[1], '')[:30]})"

    k1, k2 = st.columns([3, 1])
    pair = k1.selectbox("Chain pair", options, format_func=label)
    cutoff = k2.slider("Cutoff (Å)", 3.5, 8.0, 5.0, 0.5)
    ca, cb = pair
    if candidate and rec_a and ca not in chains_a:
        ca, cb = cb, ca

    iface = c.attempt(interface_for, entry.pdb_id, ca, cb, cutoff)
    if iface is None:
        return
    v, t = st.columns([3, 2])
    with v:
        scene = complex_scene(
            rcsb.mmcif_url(entry.pdb_id), [ca, cb], iface.residues_a + iface.residues_b
        )
        molstar_streamlit(scene, height=560)
        st.caption(
            f"Chain {ca} blue, chain {cb} orange, interface residues red sticks. "
            "Other chains grey. Drag to rotate, scroll to zoom."
        )
    with t:
        st.markdown(
            f"**{len(iface.residues_a)} residues on {ca} · {len(iface.residues_b)} on {cb}** "
            f"(≤ {cutoff:g} Å)"
        )
        rows = [
            {
                "Chain": r.chain,
                "Residue": f"{r.residue_name}{r.residue_number}{r.insertion_code}",
                "Min dist (Å)": r.min_distance,
                "Contacts": r.n_contacts,
            }
            for r in iface.residues_a + iface.residues_b
        ]
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True, height=440)
        st.download_button(
            "Download interface (CSV)",
            df.to_csv(index=False),
            f"{entry.pdb_id}_{ca}{cb}_interface.csv",
            "text/csv",
        )
    st.caption(
        iface.method + ". Crystal-packing contacts with symmetry mates are excluded, but "
        "interfaces in the deposited asymmetric unit can still include non-biological contacts."
    )
    c.provenance_caption(entry.provenance)


def alphafold_tab():
    if rec_a is None:
        st.info("Enter protein A.")
        return
    model = c.attempt(c.af_model, rec_a.accession)
    if model:
        res = c.attempt(c.af_plddt, model)
        m1, m2, m3 = st.columns(3)
        m1.metric("Model", model.model_entity_id)
        m2.metric("Mean pLDDT", f"{model.mean_plddt:.1f}" if model.mean_plddt else "–")
        m3.metric("Residues", f"{model.sequence_start}–{model.sequence_end}")
        left, right = st.columns([3, 2])
        with left:
            if res:
                segs = alphafold.plddt_segments(*res)
                molstar_streamlit(alphafold_scene(model.cif_url, segs), height=520)
                legend = " · ".join(
                    f"<span style='color:{col}'>■</span> {lab}"
                    for _, lab, col in alphafold.PLDDT_BANDS
                )
                st.markdown(legend, unsafe_allow_html=True)
        with right:
            pae = c.attempt(c.af_pae, model)
            if pae is not None:
                st.plotly_chart(pae_heatmap(pae), use_container_width=True)
                st.caption(
                    "Low PAE between two regions means their relative position is "
                    "confidently predicted. Use it to judge domain arrangements."
                )
        c.provenance_caption(model.provenance)


with tab_cx:
    complexes_tab()
with tab_af:
    alphafold_tab()
