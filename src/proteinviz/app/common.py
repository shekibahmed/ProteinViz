"""Shared Streamlit helpers: cached core calls, error display, permalinks, provenance."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import streamlit as st

from proteinviz.errors import InvalidIdentifier, NotFound, OfflineCacheMiss, ProteinVizError
from proteinviz.models import Provenance
from proteinviz.sources import alphafold, opentargets, rcsb, string_db, uniprot

T = TypeVar("T")
TTL = 24 * 3600

# The tea phytometabolite study that the EGCG showcase follows on from.
TEA_PAPER_DOI = "10.1016/j.jfca.2024.106546"
TEA_PAPER_URL = f"https://doi.org/{TEA_PAPER_DOI}"
TEA_PAPER_CITATION = (
    "Pulimamidi SS, Naik DD, Yadav M, Suryawanshi KG, Marathe SS, Jorvekar SB, Ponneganti S, "
    "Ahmed S, Hazarika A, Borkar RM. Seasonal dynamics of phytometabolites content in Assam tea, "
    "*Camellia sinensis* var. *assamica* by LC-MS/MS: Implications for quality. "
    f"*Journal of Food Composition and Analysis* 2024; 134: 106546. [doi:{TEA_PAPER_DOI}]({TEA_PAPER_URL})"
)


def attempt(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T | None:
    """Run a core call; show a clear message instead of a traceback on known failures."""
    try:
        return fn(*args, **kwargs)
    except InvalidIdentifier as exc:
        st.warning(str(exc))
    except NotFound as exc:
        st.info(f"Not found: {exc}")
    except OfflineCacheMiss as exc:
        st.error(f"Offline mode: {exc}")
    except ProteinVizError as exc:
        st.error(f"A data source is unavailable right now. Please retry shortly. ({exc})")
    return None


# Cached wrappers. The core already caches HTTP on disk; these avoid re-parsing on reruns.
protein = st.cache_data(ttl=TTL, show_spinner="Querying UniProt…")(uniprot.resolve)
string_network = st.cache_data(ttl=TTL, show_spinner="Querying STRING…")(string_db.network)
enrichment = st.cache_data(ttl=TTL, show_spinner="Running STRING enrichment…")(string_db.enrichment)
pair_evidence = st.cache_data(ttl=TTL, show_spinner=False)(string_db.pair_evidence)
af_model = st.cache_data(ttl=TTL, show_spinner="Querying AlphaFold DB…")(alphafold.get_model)
af_plddt = st.cache_data(ttl=TTL, show_spinner=False)(alphafold.fetch_plddt)
af_pae = st.cache_data(ttl=TTL, show_spinner=False)(alphafold.fetch_pae)
pdb_search = st.cache_data(ttl=TTL, show_spinner="Searching the PDB…")(rcsb.search_entries)
pdb_entries = st.cache_data(ttl=TTL, show_spinner=False)(rcsb.get_entries)
pdb_mmcif = st.cache_data(ttl=TTL, show_spinner="Downloading coordinates…")(rcsb.download_mmcif)
disease_search = st.cache_data(ttl=TTL, show_spinner=False)(opentargets.search_diseases)
disease_profile = st.cache_data(ttl=TTL, show_spinner="Querying Open Targets…")(
    opentargets.disease_profile
)
target_profile = st.cache_data(ttl=TTL, show_spinner="Querying Open Targets…")(
    opentargets.target_profile
)


def param(name: str, default: str = "") -> str:
    """Read a permalink query parameter."""
    return st.query_params.get(name, default)


def set_params(**values: str | None) -> None:
    """Write permalink query parameters (empty values are removed)."""
    for key, value in values.items():
        if value:
            st.query_params[key] = value
        elif key in st.query_params:
            del st.query_params[key]


def provenance_caption(*provs: Provenance | None) -> None:
    parts = []
    for p in provs:
        if p is None:
            continue
        ver = f" {p.version}" if p.version else ""
        parts.append(f"[{p.source}{ver}]({p.url}) · retrieved {p.retrieved_at:%Y-%m-%d %H:%M} UTC")
    if parts:
        st.caption("Source: " + " | ".join(parts))


def protein_link(symbol: str, label: str | None = None) -> str:
    return f"[{label or symbol}](./protein?q={symbol})"


def page_header(title: str, intro: str) -> None:
    st.title(title)
    st.markdown(intro)
