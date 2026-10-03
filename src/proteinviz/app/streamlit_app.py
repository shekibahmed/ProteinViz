"""ProteinViz Streamlit entry point (multipage via st.navigation)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from proteinviz import __version__

HERE = Path(__file__).parent
PAGES = HERE / "views"

st.set_page_config(page_title="ProteinViz", page_icon="🧬", layout="wide")

pages = {
    "Discover": [
        st.Page(PAGES / "home.py", title="Home", icon="🏠", default=True),
        st.Page(PAGES / "disease.py", title="Disease explorer", icon="🩺", url_path="disease"),
        st.Page(PAGES / "network.py", title="Network & pathways", icon="🕸️", url_path="network"),
    ],
    "Inspect": [
        st.Page(PAGES / "protein.py", title="Protein", icon="🧬", url_path="protein"),
        st.Page(
            PAGES / "structure.py", title="Structure & interfaces", icon="🔬", url_path="structure"
        ),
    ],
    "Compounds & data": [
        st.Page(PAGES / "egcg.py", title="EGCG showcase", icon="🍵", url_path="egcg"),
        st.Page(PAGES / "upload.py", title="Your data", icon="📤", url_path="upload"),
    ],
    "About": [
        st.Page(PAGES / "about.py", title="About & cite", icon="📖", url_path="about"),
    ],
}

nav = st.navigation(pages)
with st.sidebar:
    st.caption(
        f"ProteinViz v{__version__} · open source (MIT) · "
        "[GitHub](https://github.com/shekibahmed/ProteinViz)"
    )
    st.caption(
        "For research and hypothesis generation only. Not medical advice. "
        "Every value links to its source database."
    )
nav.run()
