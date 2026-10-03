---
title: Interface residues from gemmi ContactSearch silently include crystal symmetry mates
date: 2026-10-03
last_updated: 2026-10-03
category: logic-errors
module: structure-analysis
problem_type: logic_error
component: service_layer
symptoms:
  - "contacting_chain_pairs reports a chain in contact with itself (A–A, B–B) despite Ignore.SameChain"
  - "Interface residue counts are inflated on one side of a complex (MDM2 side of 1YCR — 29 instead of 24)"
  - "Results still look plausible: the textbook p53 Phe19/Trp23/Leu26 triad is present either way"
  - "A chain copy shifted by whole unit cells still reports contacts with image_idx 0"
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
Accept a contact only when both partners are the **deposited atoms**. The check lives in
`is_deposited_contact` (`src/proteinviz/structure/interface.py`) and has two parts:

```python
_DIRECT_TOL = 1e-3  # Å


def is_deposited_contact(c: gemmi.ContactSearch.Result) -> bool:
    if c.image_idx != 0:  # rotated/translated symmetry copy
        return False
    # Same operator but shifted by whole unit cells: the reported distance then differs
    # from the direct distance between the two deposited positions.
    return abs(c.partner1.atom.pos.dist(c.partner2.atom.pos) - c.dist) <= _DIRECT_TOL
```

`compute_interface` and `contacting_chain_pairs` both skip any contact that fails this check.

The fix came in two steps. The v0.2 relaunch (PR #2) added only the `image_idx != 0` filter,
which fixed 1YCR: one contacting pair, `('A','B',461)`, with the p53 triad still on the interface.
The direct-distance condition was added afterwards (PR #8), once the lattice-translation leak
below had been demonstrated.

## Why This Works
`NeighborSearch(model, st.cell, radius)` indexes atoms *together with their symmetry images*
generated from the unit cell. Each contact carries `image_idx`, the index of the symmetry
**operator** that produced the second partner, where 0 is the identity.

`image_idx` alone is not enough. A copy produced by the identity operator plus a whole-cell
lattice translation also reports `image_idx == 0`. This was demonstrated with 1YCR plus a copy of
chain A shifted by three cells along *a* (a 130 Å shift; its nearest atoms are 108 Å from
chain A). Under the `image_idx`-only filter, 16,693 phantom contacts were kept between the copy
and chain A. The direct-distance check keeps none. The same experiment on crambin,
with a copy shifted by 2a + b (84 Å), produced a 46 + 46-residue "interface" from 7,943 phantom
contacts between chains more than 70 Å apart.

For a deposited partner, the contact distance is exactly the distance between the two stored
positions. For any shifted image it is not, so comparing the two rejects every non-deposited
partner, whatever operator produced it.

## Prevention
- Test for **absence** as well as presence. `tests/test_interface.py` asserts that:
  - 1YCR has only the A–B pair (`test_symmetry_mates_and_self_contacts_excluded`);
  - a tighter cutoff gives a subset (`test_tighter_cutoff_gives_subset`);
  - the p53 triad is present and the MDM2-side count stays within a bound
    (`test_p53_triad_in_interface`);
  - a lattice-translated chain copy contacts nothing while the real interface is unchanged
    (`test_lattice_translated_copies_are_not_contacts`, built from the recorded 1YCR file, so it
    runs offline). This test fails with the `image_idx`-only filter.
- Treat any "chain contacts itself", or any contact between chains far apart in the deposited
  coordinates, as a red flag for symmetry leakage.
- Remaining caveat, stated in the UI: the asymmetric unit itself can contain non-biological crystal
  contacts. A biological-assembly-aware version would read the PDB's assembly definitions
  (or PISA) instead.

## Related Issues
- PR #2 (the v0.2 relaunch: interface module with the `image_idx` filter)
- PR #8 (the direct-distance check for lattice-translated copies)
