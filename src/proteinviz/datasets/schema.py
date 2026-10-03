"""Validation for user-uploaded interaction tables.

Two table kinds are accepted:

* **PPI edge list**: columns ``protein_a``, ``protein_b`` (gene symbols or
  UniProt accessions), optional numeric ``score`` and free-text ``source``.
* **Compound–target activity**: columns ``compound``, ``target``, optional
  numeric ``value`` with ``units``, and ``reference`` (DOI/PMID).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

KINDS = {
    "ppi": {"required": ["protein_a", "protein_b"], "numeric": ["score"]},
    "compound_target": {"required": ["compound", "target"], "numeric": ["value"]},
}


@dataclass
class ValidationResult:
    kind: str | None
    frame: pd.DataFrame | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and self.frame is not None


def detect_kind(columns: list[str]) -> str | None:
    cols = {c.strip().lower() for c in columns}
    for kind, spec in KINDS.items():
        if set(spec["required"]) <= cols:
            return kind
    return None


def validate(df: pd.DataFrame, max_rows: int = 5000) -> ValidationResult:
    """Normalise column names, check required columns and types, drop unusable rows."""
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    kind = detect_kind(list(df.columns))
    if kind is None:
        expected = " or ".join(f"({', '.join(s['required'])})" for s in KINDS.values())
        return ValidationResult(
            None, None, errors=[f"Missing required columns: expected {expected}."]
        )
    spec = KINDS[kind]
    res = ValidationResult(kind, None)
    if len(df) == 0:
        res.errors.append("The table has no rows.")
        return res
    if len(df) > max_rows:
        res.warnings.append(f"Only the first {max_rows} of {len(df)} rows are used.")
        df = df.head(max_rows)
    for col in spec["required"]:
        df[col] = df[col].astype("string").str.strip()
    blank = df[spec["required"]].isna().any(axis=1) | (df[spec["required"]] == "").any(axis=1)
    if blank.any():
        res.warnings.append(
            f"Dropped {int(blank.sum())} row(s) with empty {' / '.join(spec['required'])}."
        )
        df = df[~blank]
    for col in spec["numeric"]:
        if col in df.columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            bad = converted.isna() & df[col].notna()
            if bad.any():
                res.warnings.append(
                    f"{int(bad.sum())} non-numeric value(s) in '{col}' set to empty."
                )
            df[col] = converted
    if kind == "ppi":
        a, b = df["protein_a"].str.upper(), df["protein_b"].str.upper()
        self_loops = a == b
        if self_loops.any():
            res.warnings.append(f"{int(self_loops.sum())} self-interaction row(s) kept.")
    if df.empty:
        res.errors.append("No usable rows remain after validation.")
        return res
    res.frame = df.reset_index(drop=True)
    return res
