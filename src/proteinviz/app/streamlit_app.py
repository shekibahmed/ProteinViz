"""ProteinViz Streamlit entry point (multipage via st.navigation)."""

from __future__ import annotations

import streamlit as st

from proteinviz import __version__
from proteinviz.app.navigation import sections

st.set_page_config(page_title="ProteinViz", page_icon="🧬", layout="wide")

nav = st.navigation(sections())
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
