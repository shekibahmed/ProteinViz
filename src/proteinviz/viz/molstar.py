"""Mol* scenes described with MolViewSpec (https://molstar.org/mol-view-spec/)."""

from __future__ import annotations

from collections.abc import Iterable

from proteinviz.models import InterfaceResidue

CHAIN_COLOURS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#9D755D", "#72B7B2"]
INTERFACE_COLOUR = "#E45756"


def _require():
    try:
        import molviewspec as mvs
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("Install the app extra: pip install 'proteinviz[app]'") from exc
    return mvs


def _ranges(nums: Iterable[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for n in sorted(set(nums)):
        if out and out[-1][1] == n - 1:
            out[-1] = (out[-1][0], n)
        else:
            out.append((n, n))
    return out


def alphafold_scene(cif_url: str, plddt_segments: list[tuple[int, int, str]], title: str = ""):
    """AlphaFold model coloured by the standard pLDDT confidence bands."""
    mvs = _require()
    builder = mvs.create_builder()
    structure = builder.download(url=cif_url).parse(format="mmcif").model_structure()
    rep = structure.component(selector="polymer").representation(type="cartoon")
    for start, end, colour in plddt_segments:
        rep.color(
            color=colour,
            selector=mvs.ComponentExpression(beg_auth_seq_id=start, end_auth_seq_id=end),
        )
    if title:
        builder.canvas(background_color="white")
    return builder


def complex_scene(
    cif_url: str,
    chains: list[str],
    interface: Iterable[InterfaceResidue] = (),
    focus_interface: bool = True,
):
    """Experimental structure: chosen chains coloured distinctly, interface residues as sticks.

    Chains not listed are shown as faint grey cartoon for context.
    """
    mvs = _require()
    builder = mvs.create_builder()
    structure = builder.download(url=cif_url).parse(format="mmcif").model_structure()
    polymer = structure.component(selector="polymer").representation(type="cartoon")
    polymer.color(color="#D0D0D0")
    for i, chain in enumerate(chains):
        polymer.color(
            color=CHAIN_COLOURS[i % len(CHAIN_COLOURS)],
            selector=mvs.ComponentExpression(auth_asym_id=chain),
        )
    structure.component(selector="ligand").representation(type="ball_and_stick").color(
        color="#7F7F7F"
    )

    residues = list(interface)
    if residues:
        selectors = [
            mvs.ComponentExpression(auth_asym_id=chain, beg_auth_seq_id=a, end_auth_seq_id=b)
            for chain in sorted({r.chain for r in residues})
            for a, b in _ranges(r.residue_number for r in residues if r.chain == chain)
        ]
        site = structure.component(selector=selectors)
        site.representation(type="ball_and_stick").color(color=INTERFACE_COLOUR)
        site.tooltip(text="Interface residue (heavy atom within cutoff of partner chain)")
        if focus_interface:
            site.focus()
    return builder
