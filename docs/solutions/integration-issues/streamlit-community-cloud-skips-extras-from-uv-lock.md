---
title: Streamlit Community Cloud installs uv.lock without extras, so app-only dependencies go missing
date: 2026-10-03
category: integration-issues
module: deployment
problem_type: integration_issue
component: infrastructure
symptoms:
  - "Live app raises ModuleNotFoundError for molviewspec on the Structure page"
  - "Disease explorer page crashes on the live demo while Home renders fine"
  - "Everything works locally with pip install -e .[app] and in CI"
root_cause: config_error
resolution_type: dependency_update
severity: high
framework_version: streamlit 1.65 on Streamlit Community Cloud (October 2026)
tags: [streamlit-community-cloud, uv-lock, optional-dependencies, extras, deployment, requirements-txt]
---

# Streamlit Community Cloud installs uv.lock without extras, so app-only dependencies go missing

## Problem
The web-app libraries (`streamlit`, `molviewspec`, `plotly`, `openpyxl`) lived in an optional
`[app]` extra, and a `requirements.txt` containing `.[app]` was added for Community Cloud. The
repo also commits `uv.lock`. On https://proteinviz.streamlit.app, every page importing
`molviewspec` or `plotly` failed, even though the app worked locally and in CI.

## Symptoms
- `ModuleNotFoundError` (the message is redacted on Cloud) with a traceback ending at
  `from molviewspec import molstar_streamlit` in the Structure view.
- The Disease explorer also crashed. Home rendered, because it imports neither library.
- The core package imported fine, which showed that *something* was installed, just not the extras.

## What Didn't Work
- **Adding `requirements.txt` with `.[app]`.** Community Cloud never read it. Its documented
  precedence is `uv.lock` > `Pipfile` > `environment.yml` > `requirements.txt` > `pyproject.toml`,
  and only the first file found is used
  ([docs](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)).
- **Testing the root entry point locally with `pip install -r requirements.txt`.** That passed
  (PR #5), but it exercised the file Cloud ignores. The local check did not mirror Cloud's actual
  install path.

## Solution
Make the app libraries **base dependencies**, so any lockfile sync installs them, and keep `[app]`
as an empty alias so documented commands still work (`pyproject.toml:44-54`, merged in PR #6):

```toml
# before
dependencies = [ "httpx>=0.27", ... ]
[project.optional-dependencies]
app = ["streamlit>=1.45", "molviewspec>=1.8", "plotly>=5.24", "openpyxl>=3.1"]

# after
dependencies = [
    "httpx>=0.27", ...,
    # Web app. Regular dependencies, not an extra, so lockfile-based hosts (Streamlit
    # Community Cloud runs `uv.lock`) install them. Core modules still never import them.
    "streamlit>=1.45",
    "molviewspec>=1.8",
    "plotly>=5.24",
    "openpyxl>=3.1",
]
[project.optional-dependencies]
app = []
```

Then re-run `uv lock` and delete the now-misleading `requirements.txt`.

Before merging, verify the way Cloud installs: copy the repo to a fresh directory, run
`uv sync --frozen` with **no extras**, run `streamlit run streamlit_app.py`, and load every page in
a headless browser.

## Why This Works
Community Cloud chooses its dependency file by precedence, and a committed `uv.lock` wins. Syncing
a lockfile installs the project's base dependencies (plus default groups) but **no optional
extras**. Moving the app libraries into `[project].dependencies` puts them in the set every install
path resolves: lockfile sync, `pip install .`, and `pip install -r` alike.

Making Streamlit a hard dependency does not weaken the "core library has no UI dependency"
architecture. That rule is about *imports*, and it is still enforced by ruff's `banned-api` rule
and the `test_core_does_not_import_streamlit` test in `tests/test_architecture.py`.

## Prevention
- **A CI job that installs exactly like the host** (`.github/workflows/ci.yml:46-62`,
  `cloud-install`):
  ```yaml
  - run: uv sync --frozen --no-dev
  - run: |
      uv run --no-sync python -c "import streamlit, molviewspec, plotly, openpyxl, proteinviz"
      # then render streamlit_app.py with streamlit.testing.v1.AppTest and assert no exception
  ```
- When a repo commits `uv.lock`, treat it as the deployment contract. Any dependency a hosted app
  needs at runtime must be reachable without extras.
- Keep exactly one dependency file meaningful for the host. Leftover `requirements.txt` files
  mislead the next person into thinking they are used.
- After deploying, load the pages that import heavy optional libraries on the live URL, not
  just the landing page. Home rendering proved nothing here.

## Related Issues
- PR #5: introduced the Community Cloud entry point and the ineffective `requirements.txt`
- PR #6: the fix described here
- `docs/DEPLOY.md`: deployment steps, including the `uv.lock` note
