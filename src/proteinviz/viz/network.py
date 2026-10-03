"""Interactive STRING network figures (Plotly)."""

from __future__ import annotations

import networkx as nx

from proteinviz.models import StringNetwork

CHANNELS = {
    "experimental": ("Experiments", "#D62728"),
    "database": ("Curated databases", "#1F77B4"),
    "textmining": ("Text mining", "#BCBD22"),
    "coexpression": ("Co-expression", "#17BECF"),
    "neighborhood": ("Gene neighbourhood", "#2CA02C"),
    "fusion": ("Gene fusion", "#9467BD"),
    "cooccurrence": ("Co-occurrence", "#8C564B"),
}


def dominant_channel(inter) -> str:
    return max(CHANNELS, key=lambda c: getattr(inter, c))


def layout(net: StringNetwork, seed: int = 7) -> dict[str, tuple[float, float]]:
    g = nx.Graph()
    for i in net.interactions:
        g.add_edge(i.name_a, i.name_b, weight=i.score)
    return nx.spring_layout(g, seed=seed, weight="weight", k=1.2 / max(len(g), 1) ** 0.5)


def network_figure(net: StringNetwork, highlight: str | None = None, height: int = 620):
    """Nodes = proteins, edge width = combined score, edge colour = strongest evidence channel."""
    import plotly.graph_objects as go

    pos = layout(net)
    traces = []
    seen_channels: set[str] = set()
    for inter in sorted(net.interactions, key=lambda i: i.score):
        ch = dominant_channel(inter)
        label, colour = CHANNELS[ch]
        (x0, y0), (x1, y1) = pos[inter.name_a], pos[inter.name_b]
        traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line={"width": 0.5 + 4 * inter.score, "color": colour},
                opacity=0.35 + 0.6 * inter.score,
                hoverinfo="text",
                text=f"{inter.name_a} – {inter.name_b}<br>combined {inter.score:.3f}<br>strongest: {label}",
                name=label,
                legendgroup=ch,
                showlegend=ch not in seen_channels,
            )
        )
        seen_channels.add(ch)
    degree: dict[str, int] = {}
    for i in net.interactions:
        degree[i.name_a] = degree.get(i.name_a, 0) + 1
        degree[i.name_b] = degree.get(i.name_b, 0) + 1
    names = list(pos)
    hl = (highlight or "").upper()
    traces.append(
        go.Scatter(
            x=[pos[n][0] for n in names],
            y=[pos[n][1] for n in names],
            mode="markers+text",
            text=names,
            textposition="top center",
            marker={
                "size": [14 + 2 * degree.get(n, 0) for n in names],
                "color": ["#E45756" if n.upper() == hl else "#4C78A8" for n in names],
                "line": {"width": 1, "color": "white"},
            },
            hovertext=[f"{n}: {degree.get(n, 0)} partners" for n in names],
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig = go.Figure(traces)
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={"visible": False},
        yaxis={"visible": False},
        legend={"title": "Strongest evidence", "orientation": "h", "y": -0.02},
        plot_bgcolor="white",
        hovermode="closest",
    )
    return fig
