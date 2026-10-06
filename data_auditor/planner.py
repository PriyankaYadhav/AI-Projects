"""Step 2: what the LLM is allowed to propose, and how we check it.

The LLM can only pick from a fixed vocabulary of actions. Every proposal is
validated against the real data before anything is trusted.
"""
from __future__ import annotations

from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

Action = Literal[
    "drop_duplicate_rows",
    "strip_whitespace",
    "placeholders_to_null",
    "convert_numeric",
    "parse_dates",
    "standardize_values",
    "fill_missing",
    "nullify_values",
    "drop_column",
    "no_action",
]

# Only these are mechanical and lossless enough to auto-approve. Decided by code, not by the LLM.
SAFE_ACTIONS = {"drop_duplicate_rows", "strip_whitespace", "placeholders_to_null", "convert_numeric"}


class Replacement(BaseModel):
    from_value: str = Field(description="Exact value as it appears in the data")
    to_value: str = Field(description="The standardized value to use instead")


class Fix(BaseModel):
    issue_ids: list[str] = Field(description="Issue ids from the scan that this fix addresses, e.g. ['I04']")
    action: Action
    column: Optional[str] = Field(default=None, description="Target column. Null only for drop_duplicate_rows")
    replacements: Optional[list[Replacement]] = Field(default=None, description="Only for standardize_values")
    rule: Optional[Literal["negative", "outlier"]] = Field(default=None, description="Only for nullify_values")
    fill: Optional[Literal["median", "mode", "constant"]] = Field(default=None, description="Only for fill_missing")
    constant: Optional[str] = Field(default=None, description="Only when fill is 'constant'")
    rationale: str = Field(description="One plain-language sentence explaining why this fix is right")
    risk: Literal["safe", "needs_review"] = Field(description="Your honest estimate; code may override it")


class FixPlan(BaseModel):
    summary: str = Field(description="Two sentences: overall state of the data and the main things to fix")
    fixes: list[Fix]


SYSTEM_PROMPT = """You are a careful data-quality engineer. You are given issues already detected in a dataset
by a scanner, plus the distinct values of its text columns. Propose a fix for EVERY issue.

Available actions (use nothing else):
- drop_duplicate_rows: remove exact duplicate rows (no column).
- strip_whitespace: trim leading/trailing spaces in a column.
- placeholders_to_null: turn placeholders like 'N/A' or 'not recorded' into true nulls.
- convert_numeric: strip currency symbols/commas and convert a text column to numbers.
- parse_dates: convert a text column to dates; unparseable values become null.
- standardize_values: map spelling variants to one canonical value (give exact from/to pairs).
- fill_missing: fill nulls with median, mode, or a constant. Only when imputing is defensible.
- nullify_values: set negative values or outliers to null (rule = negative | outlier).
- drop_column: remove a column that carries no information.
- no_action: leave it alone and explain why (use when a human must decide).

Rules:
- Look at the distinct values of every text column. If the same real-world thing is written several
  ways (different spellings, languages, abbreviations, casing), fix it with standardize_values even if
  the scanner did not flag it. Use ONLY from_values that literally appear in the data.
- Never invent data. Prefer nulling a bad value over guessing a replacement.
- Order matters: strip whitespace and placeholders first, then convert/standardize.
- Cover every issue id. A fix may address several ids.
- Keep rationales short and plain; the reader is not a data scientist.
"""


def column_context(df: pd.DataFrame, max_distinct: int = 40) -> str:
    """Give the LLM the actual values it needs to spot variants like 'bangalore' vs 'Bengaluru'."""
    lines = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            lines.append(f"- {col}: numeric, min {s.min():,.4g}, max {s.max():,.4g}")
            continue
        vc = s.dropna().astype(str).value_counts()
        if len(vc) <= max_distinct:
            lines.append(f"- {col}: " + ", ".join(f"{v!r} x{n}" for v, n in vc.items()))
        else:
            lines.append(f"- {col}: {len(vc)} distinct values, e.g. {list(vc.index[:5])!r}")
    return "\n".join(lines)


def validate_plan(plan: FixPlan, df: pd.DataFrame, issue_ids: set[str]) -> list[str]:
    """Return a list of problems. Empty list means the plan is trustworthy enough to show a human."""
    errors: list[str] = []
    covered: set[str] = set()

    for n, f in enumerate(plan.fixes, 1):
        tag = f"Fix {n} ({f.action} on {f.column or 'table'})"
        unknown = [i for i in f.issue_ids if i not in issue_ids]
        if unknown:
            errors.append(f"{tag}: refers to issue ids that do not exist: {unknown}")
        covered.update(i for i in f.issue_ids if i in issue_ids)

        if f.action != "drop_duplicate_rows" and not f.column:
            errors.append(f"{tag}: needs a column")
            continue
        if f.column and f.column not in df.columns:
            errors.append(f"{tag}: column '{f.column}' does not exist. Columns are {list(df.columns)}")
            continue

        if f.action == "standardize_values":
            if not f.replacements:
                errors.append(f"{tag}: needs a non-empty replacements list")
            else:
                present = set(df[f.column].dropna().astype(str))
                missing = [r.from_value for r in f.replacements if r.from_value not in present]
                if missing:
                    errors.append(f"{tag}: these from_values never appear in '{f.column}': {missing}")
        if f.action == "nullify_values" and not f.rule:
            errors.append(f"{tag}: needs rule = negative or outlier")
        if f.action == "fill_missing":
            if not f.fill:
                errors.append(f"{tag}: needs fill = median, mode or constant")
            elif f.fill == "constant" and f.constant is None:
                errors.append(f"{tag}: fill is constant but no constant was given")

    uncovered = sorted(issue_ids - covered)
    if uncovered:
        errors.append(f"No fix covers these issues: {uncovered}. Every issue needs a fix (use no_action if a human must decide)")
    return errors


def enforce_risk(plan: FixPlan) -> FixPlan:
    """Code, not the LLM, decides what may be auto-approved."""
    for f in plan.fixes:
        if f.action not in SAFE_ACTIONS:
            f.risk = "needs_review"
    return plan
