"""Command-line interface: ``proteinviz app`` launches the web UI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

from proteinviz import __version__

app = typer.Typer(
    help="ProteinViz: disease pathways, protein interactions and structures.", no_args_is_help=True
)


@app.command("app")
def run_app(
    port: int = typer.Option(8501, help="Port to serve on."),
    host: str = typer.Option("localhost", help="Address to bind (use 0.0.0.0 in containers)."),
) -> None:
    """Launch the Streamlit web app."""
    import importlib.util

    if importlib.util.find_spec("streamlit") is None:
        typer.echo("Streamlit is missing; reinstall with: pip install proteinviz", err=True)
        raise typer.Exit(1)
    entry = Path(__file__).parent / "app" / "streamlit_app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(entry),
        "--server.port",
        str(port),
        "--server.address",
        host,
        "--client.toolbarMode",
        "viewer",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    raise typer.Exit(subprocess.call(cmd))


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
