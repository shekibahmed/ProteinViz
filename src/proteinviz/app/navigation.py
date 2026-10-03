"""Single source of truth for the app's pages.

Links use these ``st.Page`` objects instead of file paths, so navigation works no
matter which script launched the app (``proteinviz app`` or the repo-root
``streamlit_app.py`` used by Streamlit Community Cloud).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

VIEWS = Path(__file__).parent / "views"

# name -> (title, icon, url_path, section); the first entry is the default page.
PAGES: dict[str, tuple[str, str, str | None, str]] = {
    "home": ("Home", "🏠", None, "Discover"),
    "disease": ("Disease explorer", "🩺", "disease", "Discover"),
    "network": ("Network & pathways", "🕸️", "network", "Discover"),
    "protein": ("Protein", "🧬", "protein", "Inspect"),
    "structure": ("Structure & interfaces", "🔬", "structure", "Inspect"),
    "egcg": ("EGCG showcase", "🍵", "egcg", "Compounds & data"),
    "upload": ("Your data", "📤", "upload", "Compounds & data"),
    "about": ("About & cite", "📖", "about", "About"),
}


def page(name: str) -> st.Page:
    title, icon, url_path, _ = PAGES[name]
    if url_path is None:
        return st.Page(VIEWS / f"{name}.py", title=title, icon=icon, default=True)
    return st.Page(VIEWS / f"{name}.py", title=title, icon=icon, url_path=url_path)


def sections() -> dict[str, list[st.Page]]:
    out: dict[str, list[st.Page]] = {}
    for name, (*_, section) in PAGES.items():
        out.setdefault(section, []).append(page(name))
    return out
