import streamlit as st

from proteinviz import __version__
from proteinviz.app import common as c

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

### Related research paper
The EGCG showcase follows on from a peer-reviewed study co-authored by ProteinViz's author:

> {c.TEA_PAPER_CITATION}

**What the study did.** It followed how the phytometabolite content of Assam tea
(*Camellia sinensis* var. *assamica*, the main tea variety of Northeast India) changes with the
season. Buds and dormant leaves (*banjhi*) were harvested in the monsoon, autumn and winter, and
an LC-MS/MS method was developed and validated to quantify the metabolites. The aim was to help
growers choose harvest times.

**What it found.** EGCG was the most abundant catechin, and its level depended strongly on the
harvest season:

| Metabolite | Sample | Monsoon (µg/mg) | Later season (µg/mg) |
|---|---|---|---|
| EGCG | Bud | 78.01 ± 13.31 to 125.21 ± 5.65 | 17.86 ± 1.22 to 51.11 ± 1.21 (winter) |
| EGCG | Banjhi leaf | 203.86 ± 21.06 to 312.58 ± 28.23 | 74.43 ± 15.22 to 97.52 ± 8.47 (autumn) |
| Epicatechin gallate | Bud | 17.04 ± 2.57 to 24.78 ± 1.15 | 5.36 ± 0.39 to 11.16 ± 1.74 (winter) |
| Epicatechin gallate | Banjhi leaf | 63.61 ± 1.60 to 72.72 ± 1.23 | 12.73 ± 2.07 to 15.35 ± 1.33 (autumn) |

Catechin hydrate, caffeine, gallic acid, theanine and theaflavin also varied by season. The
authors conclude that harvest timing matters for tea quality and for cultivation and processing
practices under a changing environment.

**How it relates to ProteinViz.** The paper measures *how much* EGCG and related metabolites the
tea plant contains. The EGCG showcase asks the next question: *which proteins and disease
pathways* is EGCG reported to act on? The values above are quoted from the paper's abstract.
ProteinViz does not bundle or re-analyse the paper's data, and the showcase's activity records
come from ChEMBL.

### Citing
If you use the EGCG showcase, please also cite the paper above. Please cite ProteinViz (see `CITATION.cff` in the repository) **and the underlying resources**:
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
