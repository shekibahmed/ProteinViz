---
title: Structure page query defaults fill a missing partner with the demo protein and fall through to an unrelated chain pair
date: 2026-10-19
category: logic-errors
module: web-app
problem_type: logic_error
component: frontend
symptoms:
  - A structure permalink that names only one protein fills the other input with the demo partner (MDM2 or TP53)
  - A PDB id whose entry has no deposited contact between the two queried proteins shows an unrelated chain pair
  - "a=TP53, b=P53, pdb=1YCR shows a chain pair even though P53 is a UniProt alias of TP53 and 1YCR has no contact between two partners"
root_cause: logic_error
resolution_type: code_fix
severity: high
tags: [streamlit, query-params, structure-page, pdb, chain-pairs, permalink]
---

# Structure page query defaults fill a missing partner with the demo protein and fall through to an unrelated chain pair

## Problem

Structure-page permalinks filled in a protein the query never named, and when a named PDB entry had no deposited contact between the two resolved proteins they showed some other chain pair from that entry.

## Symptoms

- A visit with no `a`, `b`, or `pdb` in the query is the TP53–MDM2 demo. A query that named only one side used to keep the other side as the demo partner, so `a=TP53` still presented MDM2.
- A query that named a PDB id and two proteins with no deposited contact between them used to render a Min dist table for a different chain pair in that entry.
- The real pair query `a=TP53`, `b=MDM2` still shows success text containing "contain both" and interface residues PHE19, TRP23, and LEU26.

## What Didn't Work

The session record does not include a failed attempt. The previous page applied the demo on every visit with `c.param("a", "TP53")` and `c.param("b", "MDM2")`, and after building candidate chain pairs it kept `options = candidate or [(x, y) for x, y, _ in pairs]`, so an empty candidate list showed other contacts from the same PDB entry.

## Solution

The change is on branch `feat/guided-demo-readme`, commit subject "Guide visitors from a disease to a protein pair they choose." Merge is pending as of this writing.

`c.param` returns the query value when the key is present and the supplied default otherwise (`src/proteinviz/app/common.py:63-65`). The structure page now uses the TP53 and MDM2 defaults only when none of `a`, `b`, or `pdb` is in `st.query_params`; if any of those keys is present, a missing side stays empty (`src/proteinviz/app/views/structure.py:22-27`):

```python
explicit = any(key in st.query_params for key in ("a", "b", "pdb"))
a_q = c1.text_input("Protein A (gene or UniProt)", value=c.param("a", "" if explicit else "TP53"))
b_q = c2.text_input(
    "Protein B (optional, finds complexes)", value=c.param("b", "" if explicit else "MDM2")
)
pdb_q = c3.text_input("PDB ID (optional)", value=c.param("pdb", ""))
```

Candidate pairs are the contacting pairs whose chains belong to the two resolved accessions (`src/proteinviz/app/views/structure.py:113-117`). When `pdb_q` is set, both proteins resolve, and that list is empty, the page warns and returns (`src/proteinviz/app/views/structure.py:118-125`):

```python
if pdb_q and rec_a and rec_b and not candidate:
    name_a = rec_a.primary_gene or rec_a.accession
    name_b = rec_b.primary_gene or rec_b.accession
    st.warning(
        f"{entry.pdb_id} has no deposited contact between {name_a} and {name_b}. "
        "Another chain pair from this entry is not shown in its place."
    )
    return
```

`options = candidate or [(x, y) for x, y, _ in pairs]` is still the next line (`src/proteinviz/app/views/structure.py:126`). The return above is what keeps that fallthrough from running for a named PDB with no contact for the resolved pair.

The disease page's structure hop passes only `a` and `b` (`src/proteinviz/app/views/disease.py:67-71`). The protein page's structure link passes only `a` (`src/proteinviz/app/views/protein.py:38`).

## Why This Works

A missing query key is the only case where `param` may insert a demo value. Unconditional defaults treated "this permalink named one protein" as "show TP53 and MDM2," so the omitted partner was invented. Requiring all three keys to be absent before applying those defaults leaves a bare visit on the demo and leaves an omitted side empty, which is what an `a`-only link from the protein page needs.

An empty `candidate` means this entry has no deposited contact between the two resolved proteins. The old `candidate or all pairs` expression treated that absence as a reason to show some other contacting pair from the same entry. That other pair can be a real contact in the asymmetric unit; it is a different chain pair from the one the permalink named. The warning reports the missing contact, and the return stops the page from selecting a substitute. `test_structure_pdb_without_pair_contact_does_not_substitute` locks that return: `b=P53` with `a=TP53` and `pdb=1YCR` resolves both labels to one protein (P53 is a recorded UniProt alias of TP53, accession P04637, in `tests/test_sources.py:22`), so the entry has no contact between two partners.

## Prevention

Keep the TP53–MDM2 strings behind the bare-visit check, and return before `options = candidate or all pairs` when a named PDB entry has no deposited contact for the two resolved proteins. A missing pair is reported as missing. These tests in `tests/test_app.py` already lock the behavior:

- `test_structure_demo_defaults_when_no_query` (lines 64-67) calls the structure page with no query params and asserts Protein A is TP53 and Protein B is MDM2.
- `test_structure_omitted_partner_stays_empty` (lines 70-73) uses `a=TP53` and asserts Protein B is empty.
- `test_structure_pdb_without_pair_contact_does_not_substitute` (lines 76-80) uses `a=TP53`, `b=P53`, `pdb=1YCR`, asserts a warning contains "no deposited contact", and asserts no dataframe has a "Min dist (Å)" column.
- `test_structure_page_shows_interface_table` (lines 56-61) uses `a=TP53`, `b=MDM2`, asserts a success message contains "contain both", and asserts the Min dist table's Residue values include PHE19, TRP23, and LEU26.
- `test_disease_structure_hop_names_two_genes_and_no_pdb` (lines 93-105) asserts the disease page's "Open …" links all contain `a=` and `b=` and that none of those query strings contain `pdb`.
- `test_protein_structure_link_passes_only_protein_a` (lines 108-117) asserts exactly one link whose query contains `a=TP53`, and that this query string does not contain `b=`.

## Related Issues

- [Do not fabricate output](../conventions/no-fabricated-output-guardrails.md) is the broader rule. This note is the permalink case: a demo partner and an unrelated chain pair are stand-ins for a result the query did not ask for.
- [Streamlit page links](../ui-bugs/streamlit-navigation-pages-dir-and-page-link-paths.md) covers how `/structure?a=TP53&b=MDM2` is routed. Both partners are explicit in that URL.
- [Crystal symmetry contacts](../logic-errors/interface-residues-include-crystal-symmetry-contacts.md) defines which contacts are deposited. This note uses that result only to refuse a substitute chain pair.
