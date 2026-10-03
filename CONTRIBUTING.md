# Contributing to ProteinViz

Thanks for helping. Contributions from biologists (feedback on scientific correctness, use cases,
caveats) are as valuable as code.

## Ways to help
- **Biology review:** use the "Biology review" issue template to flag misleading defaults,
  missing caveats or better data sources.
- **Bugs and features:** open an issue first for anything large.
- **Code:** pick an issue labelled `good first issue`.

## Development setup
```bash
uv sync --extra dev          # or: pip install -e ".[dev]"
uv run pre-commit install
uv run pytest                # offline, deterministic
```

## Ground rules
1. **Never fabricate data.** No random or placeholder values in `src/proteinviz`; a test enforces
   this. Raise `NotFound` or `SourceUnavailable` and let the UI explain.
2. **Keep the core UI-free.** `src/proteinviz/**` (except `app/`) must not import Streamlit
   (enforced by ruff `TID251` and a test).
3. **Carry provenance.** New source clients return pydantic models with a `Provenance`.
4. **Tests use recorded responses.** When you add a client, record fixtures with
   `PROTEINVIZ_RECORD=1 uv run pytest` and commit the new files in `tests/fixtures/http/`
   (keep them small). The weekly `live-api` workflow catches upstream changes.
5. Run `uv run ruff check . && uv run ruff format .` before pushing.

## Pull requests
Describe what changed and why, how you tested it, and any scientific assumptions. Screenshots
help for UI changes.
