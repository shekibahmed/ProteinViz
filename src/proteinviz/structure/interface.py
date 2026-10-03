"""Interface residues computed from real coordinates.

Residues are reported as interface residues when any of their heavy atoms lies
within ``cutoff`` Å of a heavy atom in the partner chain. Only the first model
is used; waters, ligands and hydrogens are removed first, and contacts with
crystallographic symmetry mates are ignored (only deposited coordinates count).
"""

from __future__ import annotations

from collections import defaultdict

import gemmi

from proteinviz.errors import NotFound
from proteinviz.models import Interface, InterfaceResidue, Provenance

DEFAULT_CUTOFF = 5.0


def read_structure(cif_text: str) -> gemmi.Structure:
    doc = gemmi.cif.read_string(cif_text)
    st = gemmi.make_structure_from_block(doc.sole_block())
    st.setup_entities()
    st.remove_hydrogens()
    st.remove_ligands_and_waters()
    st.remove_empty_chains()
    return st


def chain_names(st: gemmi.Structure) -> list[str]:
    return [ch.name for ch in st[0]]


def compute_interface(
    cif_text: str,
    chain_a: str,
    chain_b: str,
    *,
    pdb_id: str,
    cutoff: float = DEFAULT_CUTOFF,
    provenance: Provenance | None = None,
) -> Interface:
    """Interface residues between ``chain_a`` and ``chain_b`` (author chain IDs)."""
    if chain_a == chain_b:
        raise ValueError("chain_a and chain_b must differ.")
    st = read_structure(cif_text)
    model = st[0]
    present = set(chain_names(st))
    for ch in (chain_a, chain_b):
        if ch not in present:
            raise NotFound(
                f"Chain {ch} not found in {pdb_id} (chains: {', '.join(sorted(present))})."
            )

    ns = gemmi.NeighborSearch(model, st.cell, max(cutoff, 5.0)).populate()
    cs = gemmi.ContactSearch(cutoff)
    cs.ignore = gemmi.ContactSearch.Ignore.SameChain
    contacts = cs.find_contacts(ns)

    # (chain, seqnum, icode) -> [resname, partner, min_dist, n_contacts]
    stats: dict[tuple[str, int, str], list] = defaultdict(lambda: [None, None, float("inf"), 0])
    pair = {chain_a, chain_b}
    for c in contacts:
        p1, p2 = c.partner1, c.partner2
        # Skip contacts with crystallographic symmetry mates: only the deposited coordinates count.
        if c.image_idx != 0 or {p1.chain.name, p2.chain.name} != pair:
            continue
        for mine, other in ((p1, p2), (p2, p1)):
            key = (mine.chain.name, mine.residue.seqid.num, mine.residue.seqid.icode.strip())
            s = stats[key]
            s[0] = mine.residue.name
            s[1] = other.chain.name
            s[2] = min(s[2], c.dist)
            s[3] += 1

    def residues_for(chain: str) -> list[InterfaceResidue]:
        rows = [
            InterfaceResidue(
                chain=ch,
                residue_number=num,
                insertion_code=icode,
                residue_name=v[0],
                partner_chain=v[1],
                min_distance=round(v[2], 2),
                n_contacts=v[3],
            )
            for (ch, num, icode), v in stats.items()
            if ch == chain
        ]
        return sorted(rows, key=lambda r: (r.residue_number, r.insertion_code))

    return Interface(
        pdb_id=pdb_id,
        chain_a=chain_a,
        chain_b=chain_b,
        cutoff=cutoff,
        residues_a=residues_for(chain_a),
        residues_b=residues_for(chain_b),
        provenance=provenance
        or Provenance(source="RCSB PDB", url=f"https://www.rcsb.org/structure/{pdb_id}"),
    )


def contacting_chain_pairs(
    cif_text: str, cutoff: float = DEFAULT_CUTOFF
) -> list[tuple[str, str, int]]:
    """All chain pairs in contact, with the number of heavy-atom contacts, largest first."""
    st = read_structure(cif_text)
    ns = gemmi.NeighborSearch(st[0], st.cell, max(cutoff, 5.0)).populate()
    cs = gemmi.ContactSearch(cutoff)
    cs.ignore = gemmi.ContactSearch.Ignore.SameChain
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for c in cs.find_contacts(ns):
        a, b = sorted((c.partner1.chain.name, c.partner2.chain.name))
        if c.image_idx != 0 or a == b:
            continue
        counts[(a, b)] += 1
    return sorted(((a, b, n) for (a, b), n in counts.items()), key=lambda t: -t[2])
