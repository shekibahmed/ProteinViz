"""Entry point for Streamlit Community Cloud (and `streamlit run streamlit_app.py`).

Runs the packaged app on every rerun. Locally you can also use `proteinviz app`.
"""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "src" / "proteinviz" / "app" / "streamlit_app.py"))
