"""HTTP access with an on-disk cache, retries and an offline mode.

Every remote call in proteinviz goes through :func:`fetch`. Responses are cached
on disk keyed by method, URL and body, with a per-call TTL. The cache doubles as
the recorded-fixture store for the test-suite:

* ``PROTEINVIZ_CACHE_DIR`` overrides the cache location.
* ``PROTEINVIZ_OFFLINE=1`` never touches the network and raises
  :class:`~proteinviz.errors.OfflineCacheMiss` for anything not cached
  (TTL is ignored in offline mode).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from platformdirs import user_cache_dir

from proteinviz import __version__
from proteinviz.errors import NotFound, OfflineCacheMiss, SourceUnavailable

USER_AGENT = f"proteinviz/{__version__} (+https://github.com/shekibahmed/ProteinViz)"

HOUR = 3600
DAY = 24 * HOUR
WEEK = 7 * DAY

_RETRY_STATUSES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class Response:
    """A minimal, cacheable HTTP response."""

    url: str
    status: int
    content: bytes
    fetched_at: float

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")

    def json(self) -> Any:
        return json.loads(self.content)


def cache_dir() -> Path:
    path = Path(os.environ.get("PROTEINVIZ_CACHE_DIR") or user_cache_dir("proteinviz"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_offline() -> bool:
    return os.environ.get("PROTEINVIZ_OFFLINE", "").lower() in {"1", "true", "yes"}


def cache_key(method: str, url: str, params: dict | None, body: Any) -> str:
    payload = json.dumps(
        {"m": method.upper(), "u": url, "p": params or {}, "b": body},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _cache_path(key: str) -> Path:
    return cache_dir() / f"{key}.json.gz"


def _read_cache(key: str, ttl: float | None) -> Response | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        entry = json.load(fh)
    if ttl is not None and not is_offline() and time.time() - entry["fetched_at"] > ttl:
        return None
    return Response(
        url=entry["url"],
        status=entry["status"],
        content=entry["content"].encode("utf-8"),
        fetched_at=entry["fetched_at"],
    )


def _write_cache(key: str, resp: Response) -> None:
    entry = {
        "url": resp.url,
        "status": resp.status,
        "content": resp.text,
        "fetched_at": resp.fetched_at,
    }
    tmp = _cache_path(key).with_suffix(".tmp")
    # mtime=0 keeps the gzip bytes stable so re-recording fixtures gives clean diffs.
    with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as fh:
        fh.write(json.dumps(entry, sort_keys=True).encode("utf-8"))
    tmp.replace(_cache_path(key))


_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
        )
    return _client


def fetch(
    url: str,
    *,
    method: str = "GET",
    params: dict | None = None,
    json_body: Any = None,
    ttl: float | None = DAY,
    source: str = "remote source",
    retries: int = 3,
) -> Response:
    """Fetch ``url`` through the cache.

    Raises :class:`NotFound` on 404 (and caches it, so repeated misses are cheap)
    and :class:`SourceUnavailable` on network failures or other error statuses.
    """
    key = cache_key(method, url, params, json_body)
    cached = _read_cache(key, ttl)
    if cached is not None:
        return _check(cached, source)
    if is_offline():
        raise OfflineCacheMiss(f"{source}: offline and not cached: {method} {url} {params or ''}")

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            r = _get_client().request(method, url, params=params, json=json_body)
        except httpx.HTTPError as exc:
            last_error = exc
        else:
            if r.status_code in _RETRY_STATUSES and attempt < retries - 1:
                last_error = SourceUnavailable(f"{source} returned HTTP {r.status_code}")
            else:
                resp = Response(str(r.url), r.status_code, r.content, time.time())
                if r.status_code < 400 or r.status_code == 404:
                    _write_cache(key, resp)
                return _check(resp, source)
        time.sleep(min(2**attempt, 8))
    raise SourceUnavailable(f"{source} is unreachable: {last_error}") from last_error


def _check(resp: Response, source: str) -> Response:
    if resp.status == 404:
        raise NotFound(f"{source}: no record at {resp.url}")
    if resp.status >= 400:
        raise SourceUnavailable(f"{source} returned HTTP {resp.status} for {resp.url}")
    return resp
