"""Plotly figures for confidence and enrichment results."""

from __future__ import annotations

import math

import numpy as np

from proteinviz.models import EnrichmentTerm

CATEGORY_LABELS = {
    "KEGG": "KEGG pathway",
    "RCTM": "Reactome pathway",
    "WikiPathways": "WikiPathways",
    "DISEASES": "Disease (DISEASES)",
    "Process": "GO biological process",
    "Function": "GO molecular function",
    "COMPARTMENTS": "Compartment",
    "HPO": "Human phenotype",
    "GWAS": "GWAS trait",
    "Hallmark": "MSigDB hallmark",
}


def pae_heatmap(pae: np.ndarray, height: int = 420):
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Heatmap(
            z=pae,
            colorscale="Greens_r",
            zmin=0,
            zmax=float(np.nanmax(pae)) if pae.size else 30,
            colorbar={"title": "Expected<br>error (Å)"},
            hovertemplate="scored residue %{y}<br>aligned residue %{x}<br>PAE %{z:.1f} Å<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={"title": "Aligned residue"},
        yaxis={"title": "Scored residue", "autorange": "reversed"},
        title="Predicted aligned error",
    )
    return fig


def enrichment_bar(terms: list[EnrichmentTerm], top: int = 15, height: int | None = None):
    import plotly.graph_objects as go

    rows = sorted(terms, key=lambda t: t.fdr)[:top][::-1]
    fig = go.Figure(
        go.Bar(
            x=[-math.log10(max(t.fdr, 1e-300)) for t in rows],
            y=[
                t.description if len(t.description) <= 60 else t.description[:57] + "…"
                for t in rows
            ],
            orientation="h",
            marker={"color": "#4C78A8"},
            customdata=[
                [CATEGORY_LABELS.get(t.category, t.category), t.term, t.n_genes, ", ".join(t.genes)]
                for t in rows
            ],
            hovertemplate="%{y}<br>%{customdata[0]} %{customdata[1]}<br>FDR 10^-%{x:.1f}"
            "<br>%{customdata[2]} genes: %{customdata[3]}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height or 120 + 26 * len(rows),
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={"title": "−log₁₀ FDR"},
        plot_bgcolor="white",
    )
    return fig
