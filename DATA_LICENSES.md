# Data sources, licences and references

ProteinViz **code** is MIT-licensed. Data are retrieved live from the sources below, under their own
terms. Only the EGCG dataset is redistributed in this repository.

| Source | Used for | Licence | Reference |
|---|---|---|---|
| [Open Targets Platform](https://platform.opentargets.org) | Disease–target associations, drugs, Reactome pathway annotations | CC0 1.0 | Buniello et al., *Nucleic Acids Res.* 2025 |
| [STRING](https://string-db.org) | Interaction networks, functional enrichment | CC BY 4.0 | Szklarczyk et al., *Nucleic Acids Res.* 2025 |
| [UniProtKB](https://www.uniprot.org) | Protein annotation and sequence | CC BY 4.0 | The UniProt Consortium, *Nucleic Acids Res.* 2025 |
| [RCSB PDB](https://www.rcsb.org) | Experimental structures, search, metadata | CC0 1.0 | Burley et al., *Nucleic Acids Res.* 2025 |
| [AlphaFold DB](https://alphafold.ebi.ac.uk) | Predicted structures, pLDDT, PAE | CC BY 4.0 | Varadi et al., *Nucleic Acids Res.* 2024 |
| [ChEMBL](https://www.ebi.ac.uk/chembl/) | EGCG bioactivities (**bundled**) | CC BY-SA 3.0 | Zdrazil et al., *Nucleic Acids Res.* 2024 |
| [Reactome](https://reactome.org) | Pathway pages (linked) | CC BY 4.0 | Milacic et al., *Nucleic Acids Res.* 2024 |
| [Mol\*](https://molstar.org) / [MolViewSpec](https://molstar.org/mol-view-spec/) | 3D visualisation | MIT | Sehnal et al., *Nucleic Acids Res.* 2021; Bittrich et al., *Nucleic Acids Res.* 2025 |
| [gemmi](https://gemmi.readthedocs.io) | Structure parsing, contact search | MPL 2.0 | Wojdyr, *J. Open Source Softw.* 2022 |

## Bundled data

`src/proteinviz/datasets/egcg/` is derived from ChEMBL and is distributed under
**CC BY-SA 3.0** (see the `LICENSE` file there). If you redistribute modified versions, keep the
same licence and attribute ChEMBL.
