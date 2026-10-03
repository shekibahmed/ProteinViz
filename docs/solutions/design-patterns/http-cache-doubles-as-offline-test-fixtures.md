---
title: One on-disk HTTP cache serves as runtime cache, offline test fixtures and live-drift check
date: 2026-10-03
category: design-patterns
module: http
problem_type: design_pattern
component: testing_framework
severity: medium
applies_when:
  - "A library or app wraps several third-party web APIs and needs deterministic tests"
  - "CI must run offline and fast, yet upstream response formats change over time"
  - "Tests must exercise real response payloads, not hand-written mocks"
tags: [http-cache, test-fixtures, offline-testing, vcr, api-drift, pytest]
---

# One on-disk HTTP cache serves as runtime cache, offline test fixtures and live-drift check

## Context
ProteinViz talks to six public services: UniProt, STRING, RCSB PDB (search, GraphQL and file
downloads), AlphaFold DB, Open Targets and ChEMBL. Tests needed real payloads (an interface test is meaningless on a mocked
mmCIF), had to run offline in CI in seconds, and still had to notice upstream changes. Two such
changes landed during 2026, according to the EBI announcements found during this work:
AlphaFold DB renamed its API fields, and the old PDBe API paths were retired. A VCR-style library plus a separate runtime cache would have meant two storage formats
and two code paths. Instead, the runtime cache *is* the fixture store.

## Guidance
Route every request through one `fetch()` that caches responses on disk, keyed by a stable hash of
method, URL, params and body (`src/proteinviz/http.py`). The module reads two environment
switches, and two more rules govern what is stored:

- `PROTEINVIZ_CACHE_DIR` chooses the cache directory (`http.py:57`).
- `PROTEINVIZ_OFFLINE=1` never touches the network, ignores TTLs, and raises a typed
  `OfflineCacheMiss` for anything not cached (`http.py:62-63`, `:143`).
- Successful responses **and 404s** are cached, but other errors never are (`http.py:156-157`), so
  "not found" stays testable and transient failures don't get frozen.
- Entries are gzipped JSON written with `mtime=0` (`http.py:103-104`), so re-recording an
  unchanged response gives a byte-identical file and a clean git diff.

`tests/conftest.py` reads two test-only switches, `PROTEINVIZ_LIVE` and `PROTEINVIZ_RECORD`, and
selects one of three modes at module level, before tests are collected:

```python
LIVE = os.environ.get("PROTEINVIZ_LIVE") == "1"
RECORD = os.environ.get("PROTEINVIZ_RECORD") == "1"
if LIVE:                                   # fresh temp cache, real APIs
    os.environ["PROTEINVIZ_CACHE_DIR"] = tempfile.mkdtemp(prefix="proteinviz-live-")
    os.environ.pop("PROTEINVIZ_OFFLINE", None)
else:                                      # committed fixtures
    os.environ["PROTEINVIZ_CACHE_DIR"] = str(FIXTURES)
    if RECORD:
        os.environ.pop("PROTEINVIZ_OFFLINE", None)   # fill in missing responses
    else:
        os.environ["PROTEINVIZ_OFFLINE"] = "1"       # default: deterministic, offline
```

A weekly workflow runs the same suite with `PROTEINVIZ_LIVE=1`
(`.github/workflows/live-api.yml`, cron `0 6 * * 1`), so format drift fails a scheduled job
instead of surprising users.

## Why This Matters
- **One code path.** Tests exercise the exact request-building and parsing code that production
  runs. No mock layer can drift from reality.
- **Fast and hermetic.** The whole suite, including Streamlit page smoke tests, ran offline in
  roughly 4–7 s in this session's runs, from 33 small gzip files (under 0.5 MB).
- **Drift is caught on a schedule.** Recorded fixtures would otherwise hide upstream format
  changes indefinitely.
- **Missing fixtures are loud.** An offline miss raises with the method, URL and params (the
  POST body is not printed), so a new test tells you which response to record.

## When to Apply
- A project wraps several HTTP APIs whose responses are text (JSON, CSV, mmCIF/PDB). The cache
  stores bodies as UTF-8 text, so binary payloads would need base64.
- You want real-payload tests without the dependency and indirection of VCR-style libraries.
- Not a fit when requests carry secrets or user data in URLs or bodies: the hash input would
  need scrubbing, and fixtures would need review before committing.

## Examples
- **Adding a source client:** write the test, then run `PROTEINVIZ_RECORD=1 uv run pytest`, commit
  the new files under `tests/fixtures/http/`, and re-run plain `uv run pytest` to confirm it
  passes offline.
- **Write assertions that survive live mode.** The same test runs against recorded and live data,
  so assert on invariants, not volatile specifics. Many TP53 STRING partners tie at score 0.999,
  and which tied partners STRING returns can vary. `test_string_network_tp53` therefore checks
  the node count and the score threshold instead of expecting `MDM2` by name
  (`tests/test_sources.py`, comment "Many TP53 partners tie at 0.999…").
- **Isolated unit tests of the cache itself** use a per-test empty cache directory (the
  `tmp_cache` fixture) and an `httpx.MockTransport`. They cover key stability, caching, 404 caching,
  retry then failure, errors not cached, offline misses and TTL expiry (`tests/test_http.py`).

## Related
- `docs/solutions/integration-issues/streamlit-community-cloud-skips-extras-from-uv-lock.md`:
  the same principle applied to deployment (a CI job that mirrors the real environment)
