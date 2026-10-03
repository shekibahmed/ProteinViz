# EGCG compound–target activity dataset

Bioactivity records for **epigallocatechin gallate (EGCG, ChEMBL ID CHEMBL297453)** from ChEMBL,
built by `scripts/build_egcg_dataset.py`. `metadata.json` records the ChEMBL release and build time.

**Licence:** CC BY-SA 3.0, derived from ChEMBL (© EMBL-EBI). See `LICENSE`.

## Columns
| Column | Meaning |
|---|---|
| `activity_id`, `assay_chembl_id`, `document_chembl_id`, `target_chembl_id` | ChEMBL identifiers (stable, linkable) |
| `assay_type` | B binding · F functional · A ADME · T toxicity · P physicochemical · U unclassified |
| `assay_description` | ChEMBL assay description |
| `standard_type`, `standard_relation`, `standard_value`, `standard_units` | Standardised activity (e.g. IC50 = 1200 nM) |
| `pchembl_value` | −log10(molar) potency where comparable (IC50, Ki, Kd, EC50…); 6 ≙ 1 µM |
| `data_validity_comment`, `potential_duplicate` | ChEMBL curation flags |
| `target_pref_name`, `target_type`, `target_organism`, `target_tax_id` | Target annotation |
| `uniprot_accessions` | `;`-separated UniProt accessions of the target's protein components |
| `document_year`, `journal`, `doi`, `pubmed_id` | Source publication |
| `evidence_tier` | **Potent in ≥2 publications** / **Potent in 1 publication** / **Weak or non-quantitative** (potent = pChEMBL ≥ 6, counted per target across distinct documents) |
| `caveats` | Validity comments, potential duplicates, non-protein targets, censored values |

## Interpretation
EGCG is a catechol/pyrogallol polyphenol and a known assay-interference compound (aggregation,
redox cycling, covalent reactivity). Treat single biochemical hits with caution, and prefer targets
in the top tier that are also confirmed in cellular or orthogonal assays.
