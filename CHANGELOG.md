# Changelog

## Unreleased
- Docs: the related research paper (Pulimamidi et al., *J. Food Compos. Anal.* 2024,
  [doi:10.1016/j.jfca.2024.106546](https://doi.org/10.1016/j.jfca.2024.106546)) is now described
  and cited in the README, on the app's Home, EGCG and About pages, and in `CITATION.cff`.
- Fix: interface calculations now also reject contacts with copies shifted by whole unit cells,
  which gemmi reports with the identity operator. A test builds such a copy and checks it
  contacts nothing. Real interfaces (e.g. 1YCR) are unchanged.
- Deploy target switched to **Streamlit Community Cloud**, because Hugging Face Docker Spaces now
  need a paid plan. Added the repo-root `streamlit_app.py`, plus
  `docs/DEPLOY.md`. Removed the Hugging Face deploy workflow.
- Fix: the web-app libraries (streamlit, molviewspec, plotly, openpyxl) are now base dependencies.
  Community Cloud installs from `uv.lock`, which skipped the `[app]` extra, so pages using
  Mol\* or Plotly failed on the live demo. `proteinviz[app]` remains as an alias.
- Fix: links on the Home page now resolve whichever script launches the app (pages defined once in
  `proteinviz.app.navigation`).

## 0.2.0 (2026-10-03): honest relaunch

This release rebuilds ProteinViz around one rule: **never show fabricated output.**

### Removed
- All machine-learning models (Random Forest, SVM, GNN, Graph Transformer, EGNN). They were trained
  on random synthetic data, and their predictions and "interface residues" were not meaningful.
- Mock sequence generation, random-walk "coordinates" and random fallback scores.
- The unsourced EGCG CSV files, Replit configuration and committed model binaries.

### Added
- `proteinviz` Python package (src layout) with typed, provenance-carrying clients for UniProt,
  STRING v12.5, RCSB PDB (Search v2 + GraphQL), AlphaFold DB (post-2026 API fields), Open Targets
  and ChEMBL, behind an on-disk HTTP cache with offline mode.
- **Disease explorer:** associated targets, shared pathways, target network, drugs by clinical stage.
- **Network & pathways:** STRING neighbourhood with evidence channels and FDR-controlled enrichment.
- **Structure & interfaces:** Mol\* (MolViewSpec) viewer, pLDDT colouring, PAE heatmap,
  experimental complexes for protein pairs, interface residues from coordinates (gemmi).
- **EGCG showcase** rebuilt from ChEMBL 37 (1,044 activities, DOI/PMID for >95%), with evidence tiers
  and assay-interference caveats; reproducible via `scripts/build_egcg_dataset.py`.
- **Your data:** gene-list / table upload with validation and enrichment.
- Shareable permalinks, `proteinviz app` CLI, Dockerfile for self-hosting.
- Offline-reproducible test suite (recorded API responses), weekly live-API checks, CI on
  Python 3.11–3.14, citation and data-licence files.
