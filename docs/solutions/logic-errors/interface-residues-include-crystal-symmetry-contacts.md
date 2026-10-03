---
title: Interface residues from gemmi ContactSearch silently include crystal symmetry mates
date: 2026-10-03
category: logic-errors
module: structure-analysis
problem_type: logic_error
component: service_layer
symptoms:
  - "contacting_chain_pairs reports a chain in contact with itself (A–A, B–B) despite Ignore.SameChain"
  - "Interface residue counts are inflated on one side of a complex (MDM2 side of 1YCR — 29 instead of 24)"
  - "Results still look plausible: the textbook p53 Phe19/Trp23/Leu26 triad is present either way"
root_cause: wrong_api
resolution_type: code_fix
severity: high
framework_version: gemmi 0.7.5
tags: [gemmi, contact-search, crystal-symmetry, protein-interface, pdb, mmcif, structural-biology]
---

# Interface residues from gemmi ContactSearch silently include crystal symmetry mates

## Problem
`compute_interface` in `src/proteinviz/structure/interface.py` reports a residue as part of a
protein–protein interface when any heavy atom is within the cutoff of the partner chain. Built on
`gemmi.NeighborSearch` + `gemmi.ContactSearch` with the unit cell, it also counted contacts with
**crystallographic symmetry copies** of the chains. Those copies are lattice packing, not the
deposited complex. That inflates interfaces, which matters for a tool whose promise is "interfaces
computed from real coordinates".

## Symptoms
- `contacting_chain_pairs` on 1YCR returned `[('A','B',511), ('A','A',340), ('B','B',5)]`.
  A chain "contacting itself" with `Ignore.SameChain` set is the giveaway: those are contacts
  with symmetry images of the same chain.
- The MDM2 (chain A) side of the 1YCR interface listed 29 residues. After the fix it lists 24.
  The p53 (chain B) side was unchanged at 11.
- Nothing crashed, and the scientifically famous residues were present, so a spot check against
  the literature passed with the bug in place.

## What Didn't Work
- **Validating only against known hot-spot residues.** Checking that p53 Phe19, Trp23 and Leu26
  appear (Kussie et al., 1996) passed both before and after the fix. Presence checks can't catch
  *extra* residues.
- **`ContactSearch.Ignore.SameChain`.** It only suppresses contacts within the same chain *copy*.
  It does not stop contacts between a chain and its own symmetry image, or with the partner's
  image.

## Solution
Keep only contacts whose partner is in the deposited asymmetric unit (`image_idx == 0`), in both
functions (`src/proteinviz/structure/interface.py:67` and `:116`):

```python
cs = gemmi.ContactSearch(cutoff)
cs.ignore = gemmi.ContactSearch.Ignore.SameChain
for c in cs.find_contacts(ns):
    p1, p2 = c.partner1, c.partner2
    # Skip contacts with crystallographic symmetry mates: only the deposited coordinates count.
    if c.image_idx != 0 or {p1.chain.name, p2.chain.name} != pair:
        continue
    ...
```

and in `contacting_chain_pairs`:

```python
if c.image_idx != 0 or a == b:
    continue
```

After the fix, 1YCR gives exactly one contacting pair, `('A','B',461)`, and the p53 triad is
still on the interface.

## Why This Works
`NeighborSearch(model, st.cell, radius)` indexes atoms *together with their symmetry images*
generated from the unit cell. Each contact carries `image_idx`, the index of the symmetry
operator that produced the second partner, where 0 is the identity operator. Filtering to
`image_idx == 0` drops contacts with rotated and translated symmetry copies, so the result
approximates "the interface in this PDB entry", meaning the deposited complex.

**Known limit.** The identity operator combined with a whole-cell lattice translation (the same
molecule in a neighbouring unit cell) may also report `image_idx == 0`. gemmi's stubs don't
document this, so it is not ruled out. On 1YCR, all 461 retained contacts were checked: each
distance equals the direct distance between the deposited coordinates. Small unit cells,
where that kind of neighbour can come within the cutoff, have not been verified.

## Prevention
- Test for **absence** as well as presence. `tests/test_interface.py` asserts that 1YCR has only
  the A–B pair (`test_symmetry_mates_and_self_contacts_excluded`), checks that a tighter cutoff
  gives a subset (`test_tighter_cutoff_gives_subset`), and checks the p53 triad plus a bound on the
  MDM2-side count, `15 <= len(iface.residues_a) <= 35` (`test_p53_triad_in_interface`).
- To close the known limit above, also require each contact's reported distance to equal the
  direct distance between the two deposited atom positions, so any lattice-translated partner is
  rejected. Add a small-unit-cell entry to the tests.
- Treat any "chain contacts itself" result as a red flag for symmetry leakage.
- Remaining caveat, stated in the UI: the asymmetric unit itself can contain non-biological crystal
  contacts. A biological-assembly-aware version would read the PDB's assembly definitions
  (or PISA) instead.

## Related Issues
- PR #2 (the v0.2 relaunch, which introduced the interface module including this fix)
