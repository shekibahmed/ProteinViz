"""Test configuration.

By default tests run **offline** against recorded HTTP responses in
``tests/fixtures/http`` (the same on-disk format as the runtime cache).

* ``PROTEINVIZ_RECORD=1``: allow network access and record missing responses.
* ``PROTEINVIZ_LIVE=1``: use a fresh temporary cache and hit the real APIs
  (the weekly CI job does this to catch upstream format changes).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "http"

LIVE = os.environ.get("PROTEINVIZ_LIVE") == "1"
RECORD = os.environ.get("PROTEINVIZ_RECORD") == "1"

if LIVE:
    os.environ["PROTEINVIZ_CACHE_DIR"] = tempfile.mkdtemp(prefix="proteinviz-live-")
    os.environ.pop("PROTEINVIZ_OFFLINE", None)
else:
    os.environ["PROTEINVIZ_CACHE_DIR"] = str(FIXTURES)
    if RECORD:
        os.environ.pop("PROTEINVIZ_OFFLINE", None)
    else:
        os.environ["PROTEINVIZ_OFFLINE"] = "1"


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """An empty, isolated cache directory (online unless a test sets offline)."""
    monkeypatch.setenv("PROTEINVIZ_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("PROTEINVIZ_OFFLINE", raising=False)
    return tmp_path
