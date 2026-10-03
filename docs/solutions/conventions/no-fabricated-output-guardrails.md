---
title: No fabricated output - fail loudly with typed errors and enforce it with tests
date: 2026-10-03
category: conventions
module: core
problem_type: convention
component: service_layer
severity: critical
applies_when:
  - "Adding or changing a data-source client, model or computed result in src/proteinviz"
  - "Tempted to add a fallback value so a page renders when an API fails or returns nothing"
  - "Adding any ML or prediction feature (planned for v0.3)"
symptoms:
  - "The Replit prototype showed confident predictions from models trained on random data"
  - "Mock sequences and random interface residues were indistinguishable from real results in the UI"
root_cause: logic_error
resolution_type: code_fix
tags: [scientific-integrity, provenance, fallbacks, typed-errors, guardrail-tests, honesty]
---

# No fabricated output - fail loudly with typed errors and enforce it with tests

## Context
The original prototype tried hard to never show an empty screen. When UniProt failed it generated
a mock sequence. All five "ML models" were trained on randomly generated data
(`np.random` / `torch.randn`). Interface residues were
random letters and positions, and any exception produced a random confidence score. Every
fallback looked exactly like a real result. For a research audience, one discovered fabrication
discredits the whole tool. The v0.2 rebuild (PR #2) removed all of it and made "never fabricate"
a rule enforced by code and tests, not just good intentions.

## Guidance
1. **The core never substitutes data.** When a source has no record or is unreachable, raise a
   typed error from `src/proteinviz/errors.py`:
   `ProteinVizError` > `InvalidIdentifier`, `NotFound`, `SourceUnavailable` > `OfflineCacheMiss`.
2. **Front-ends translate errors into honest messages.** The Streamlit layer wraps core calls in
   `attempt()` (`src/proteinviz/app/common.py:18`), which turns `InvalidIdentifier` into a warning,
   `NotFound` into "Not found: …", `OfflineCacheMiss` into "Offline mode: …", and any other
   `ProteinVizError` into "A data source is unavailable right now". The page then stops or skips
   that section.
3. **Every top-level result carries provenance.** Top-level records (protein, network, PDB entry,
   AlphaFold model, interface, enrichment, disease and target profiles) require a `Provenance`
   (source, URL, optional version, retrieval time; `src/proteinviz/models.py:14`). Nested items
   inherit their parent's, and the UI prints it under each result.
4. **Enforce with tests, not reviews.** `tests/test_architecture.py`
   (`test_no_fabricated_data_generators_in_package`) fails if `np.random`, `import random`,
   `from random import` or `torch.randn` appears anywhere in `src/proteinviz`:
   ```python
   pattern = re.compile(r"\bnp\.random\b|\bimport random\b|\btorch\.randn?\b|from random import")
   offenders = [str(p.relative_to(SRC)) for p in SRC.rglob("*.py") if pattern.search(p.read_text())]
   assert offenders == []
   ```
   Source tests also assert the failure path, for example `test_uniprot_unknown_gene_raises` and
   `test_rcsb_unknown_entry` (`tests/test_sources.py`) expect `NotFound`.
5. **Absent features are hidden, not faked.** Interaction prediction was removed outright until a
   benchmarked model exists (roadmap v0.3), and there is no "coming soon" page with sample numbers.

## Why This Matters
Plausible fake output is worse than an error: users cannot tell it apart, it propagates into notes
and slides, and finding it destroys trust in everything else the tool shows. A typed error costs
one message on screen. The guardrail tests make the rule survive future contributors (and AI
coding agents) who reach for a quick fallback.

## When to Apply
- Any new source client: raise `NotFound` or `SourceUnavailable`. Never return an empty-but-valid
  object that the UI would render as "no interactions".
- Any new computed result: compute it from fetched coordinates or data, and attach provenance.
- Any future model: train on real benchmark data with published metrics. Random-number generation
  for training belongs in `scripts/` or `tests/`, which the guard deliberately skips.

## Examples
Before (prototype):
```python
except Exception as e:
    np.random.seed(hash(protein_a + protein_b) % 2**32)
    confidence = np.random.uniform(0.1, 0.9)        # looks like a real prediction
    return {"confidence": confidence, "model_used": f"{model_type} (fallback)"}  # abridged
```
After (v0.2): the core raises, the page explains.
```python
rec = c.attempt(c.protein, q)   # shows "Not found: …" or "source unavailable" in the UI
if rec is None:
    st.stop()
```

## Related
- `docs/solutions/design-patterns/http-cache-doubles-as-offline-test-fixtures.md`: offline misses
  surface as the typed `OfflineCacheMiss`
- `docs/solutions/logic-errors/interface-residues-include-crystal-symmetry-contacts.md`: a reminder
  that wrong-but-plausible output also arises from genuine computations, so test for absence too
- `AI_USAGE.md`, `CONTRIBUTING.md` (ground rule 1)
