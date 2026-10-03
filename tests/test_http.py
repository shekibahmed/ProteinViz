import httpx
import pytest

from proteinviz import http
from proteinviz.errors import NotFound, OfflineCacheMiss, SourceUnavailable


@pytest.fixture
def mock_transport(monkeypatch):
    calls = []

    def install(handler):
        def wrapped(request):
            calls.append(request)
            return handler(request)

        client = httpx.Client(transport=httpx.MockTransport(wrapped))
        monkeypatch.setattr(http, "_client", client)
        monkeypatch.setattr(http.time, "sleep", lambda s: None)
        return calls

    return install


def test_cache_key_is_stable_and_order_independent():
    a = http.cache_key("GET", "https://x", {"a": 1, "b": 2}, None)
    b = http.cache_key("get", "https://x", {"b": 2, "a": 1}, None)
    assert a == b
    assert a != http.cache_key("POST", "https://x", {"a": 1, "b": 2}, None)


def test_fetch_caches_and_reuses(tmp_cache, mock_transport):
    calls = mock_transport(lambda r: httpx.Response(200, json={"ok": True}))
    assert http.fetch("https://example.org/a").json() == {"ok": True}
    assert http.fetch("https://example.org/a").json() == {"ok": True}
    assert len(calls) == 1


def test_404_raises_not_found_and_is_cached(tmp_cache, mock_transport):
    calls = mock_transport(lambda r: httpx.Response(404, text="nope"))
    for _ in range(2):
        with pytest.raises(NotFound):
            http.fetch("https://example.org/missing")
    assert len(calls) == 1


def test_retries_then_raises_source_unavailable(tmp_cache, mock_transport):
    calls = mock_transport(lambda r: httpx.Response(503))
    with pytest.raises(SourceUnavailable):
        http.fetch("https://example.org/down", retries=3)
    assert len(calls) == 3


def test_errors_are_not_cached(tmp_cache, mock_transport):
    responses = iter([httpx.Response(500), httpx.Response(200, json=[1])])
    mock_transport(lambda r: next(responses))
    with pytest.raises(SourceUnavailable):
        http.fetch("https://example.org/flaky", retries=1)
    assert http.fetch("https://example.org/flaky", retries=1).json() == [1]


def test_offline_miss(tmp_cache, monkeypatch):
    monkeypatch.setenv("PROTEINVIZ_OFFLINE", "1")
    with pytest.raises(OfflineCacheMiss):
        http.fetch("https://example.org/never")


def test_ttl_expiry(tmp_cache, mock_transport, monkeypatch):
    calls = mock_transport(lambda r: httpx.Response(200, text="v"))
    http.fetch("https://example.org/t", ttl=10)
    real = http.time.time
    monkeypatch.setattr(http.time, "time", lambda: real() + 100)
    http.fetch("https://example.org/t", ttl=10)
    assert len(calls) == 2
