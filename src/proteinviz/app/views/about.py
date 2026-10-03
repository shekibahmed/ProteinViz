import streamlit as st

from proteinviz import __version__

st.title("📖 About & cite")
st.markdown(
    f"""
**ProteinViz v{__version__}** is an open-source (MIT) tool that helps researchers and the wider
community go from a disease to the proteins, interactions, pathways, structures and compounds
behind it, using only public data that is traceable to its source.

### Principles
* **No fabricated output.** If a source fails or has no record, ProteinViz says so. It never
  substitutes simulated values.
* **Provenance everywhere.** Each table links to the source record and states when it was retrieved.
* **Hypotheses, not conclusions.** Associations, enrichments and in-vitro activities are leads to
  test experimentally.

### Methods
| Feature | Method / source |
|---|---|
| Disease–target associations, known drugs | Open Targets Platform GraphQL API (overall association score) |
| Interaction networks | STRING v12.5 API, functional or physical network, combined score thresholds |
| Pathway / disease enrichment | STRING enrichment (genome background, Benjamini–Hochberg FDR ≤ 0.05) |
| Protein annotation | UniProtKB REST; properties via Biopython ProtParam |
| Predicted structures | AlphaFold DB API (latest model version, pLDDT and PAE from the confidence files) |
| Experimental complexes | RCSB PDB Search API v2 + Data API (GraphQL); mmCIF from files.rcsb.org |
| Interface residues | gemmi heavy-atom contact search (default 5 Å), first model, symmetry mates excluded |
| EGCG activities | ChEMBL web services; evidence tier = number of publications with pChEMBL ≥ 6 |

### Limitations
* Database coverage is uneven: well-studied proteins have far more evidence (knowledge bias).
* Enrichment of a network seeded from one protein describes that neighbourhood. It is not
  independent evidence.
* Contacts in a crystal's asymmetric unit are not always biologically relevant interfaces.
* EGCG and other polyphenols often interfere with assays, so treat single biochemical hits with caution.
* Interaction prediction for unseen pairs is planned (v0.3) with a leakage-free benchmark. Until then
  ProteinViz shows only observed evidence.

### Citing
Please cite ProteinViz (see `CITATION.cff` in the repository) **and the underlying resources**:
Open Targets (Ochoa et al.), STRING (Szklarczyk et al.), UniProt (UniProt Consortium),
RCSB PDB (Burley et al.), AlphaFold DB (Varadi et al.), ChEMBL (Zdrazil et al.),
Mol*/MolViewSpec (Sehnal et al.; Bittrich et al.).

### Data licences
UniProt, STRING and AlphaFold DB: CC BY 4.0 · PDB: CC0 · Open Targets: CC0 · ChEMBL: CC BY-SA 3.0.

### AI use
The original prototype was generated with an AI coding agent on Replit. Version 0.2 was rebuilt
with AI assistance (Claude Code) under human direction. Fabricated components were removed, and
every data path was checked against live sources. See `AI_USAGE.md`.

[GitHub repository](https://github.com/shekibahmed/ProteinViz) · [live app](https://proteinviz.streamlit.app) · issues and contributions welcome.
"""
)
