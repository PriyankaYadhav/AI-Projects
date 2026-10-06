"""Step 1: deterministic data-quality scan.

Finds problems with plain pandas and returns structured Issue objects.
No LLM involved: later steps reason about these *detected* issues.
"""
from __future__ import annotations

import re
import sys
import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

PLACEHOLDERS = {"", "n/a", "na", "null", "none", "nan", "-", "--", "unknown", "?", "not recorded"}
NON_NEGATIVE_HINTS = ("qty", "quantity", "price", "amount", "count", "age", "units", "total")


@dataclass
class Issue:
    id: str
    kind: str
    column: str | None
    severity: str  # high | medium | low
    detail: str
    affected: int
    examples: list


def load(path: str) -> pd.DataFrame:
    """Load a file so we see the data as written: only truly empty cells become null.
    (pandas would otherwise silently turn 'N/A', 'null', 'none' into NaN and hide them.)"""
    p = path.lower()
    if p.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, keep_default_na=False, na_values=[""])
    elif p.endswith(".json"):
        df = pd.read_json(path)
    elif p.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _sev(share: float, high: float = 0.2, med: float = 0.03) -> str:
    return "high" if share >= high else "medium" if share >= med else "low"


def _text_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]


def _normalize(v: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def scan(df: pd.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    n = len(df)

    def add(kind, column, severity, detail, affected, examples=()):
        issues.append(Issue(f"I{len(issues) + 1:02d}", kind, column, severity, detail, int(affected), [e.item() if hasattr(e, "item") else e for e in list(examples)[:5]]))

    # --- whole-table: duplicate rows ---
    dups = int(df.duplicated().sum())
    if dups:
        add("duplicate_rows", None, _sev(dups / n, 0.1, 0.01), f"{dups} rows are exact copies of an earlier row", dups)

    for col in df.columns:
        s = df[col]
        nonnull = s.dropna()

        # --- constant column ---
        if s.nunique(dropna=True) <= 1 and n > 1:
            add("constant_column", col, "low", f"Only one distinct value ({nonnull.iloc[0] if len(nonnull) else 'none'}); carries no information", n)

        # --- missing values (true nulls) ---
        nulls = int(s.isna().sum())
        if nulls:
            add("missing", col, _sev(nulls / n), f"{nulls} empty values ({nulls / n:.1%})", nulls)

        if col not in _text_cols(df):
            # --- numeric: outliers & impossible negatives ---
            if pd.api.types.is_numeric_dtype(s) and len(nonnull) > 20:
                q1, q3 = nonnull.quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    bad = nonnull[(nonnull < q1 - 3 * iqr) | (nonnull > q3 + 3 * iqr)]
                    if len(bad):
                        add("outliers", col, _sev(len(bad) / n, 0.05, 0.005), f"{len(bad)} values far outside the normal range ({q1 - 3 * iqr:,.4g} to {q3 + 3 * iqr:,.4g})", len(bad), sorted(bad.unique())[:5])
                if any(h in col.lower() for h in NON_NEGATIVE_HINTS):
                    neg = nonnull[nonnull < 0]
                    if len(neg):
                        add("negative_values", col, "medium", f"{len(neg)} negative values in a column that should not go below zero", len(neg), neg.unique())
            continue

        # ---- text columns from here on ----
        txt = nonnull.astype(str)

        # --- placeholder strings pretending to be data ---
        ph = txt[txt.str.strip().str.lower().isin(PLACEHOLDERS)]
        if len(ph):
            add("placeholder_missing", col, _sev(len(ph) / n), f"{len(ph)} placeholder values that mean 'missing' (e.g. {ph.iloc[0]!r})", len(ph), ph.unique())
        real = txt.drop(ph.index)

        # --- stray whitespace ---
        ws = real[real != real.str.strip()]
        if len(ws):
            add("whitespace", col, "low", f"{len(ws)} values have leading/trailing spaces", len(ws), [repr(v) for v in ws.unique()])

        if real.empty:
            continue

        # --- numbers stored as text ---
        stripped = real.str.replace(r"[,\s$€£₹¥%]", "", regex=True)
        as_num = pd.to_numeric(stripped, errors="coerce")
        if as_num.notna().mean() >= 0.6:
            plain = pd.to_numeric(real, errors="coerce").notna()
            decorated = real[~plain & as_num.notna()]
            broken = real[as_num.isna()]
            if len(decorated):
                add("numeric_formatting", col, "medium", f"{len(decorated)} numbers carry symbols or separators (e.g. {decorated.iloc[0]!r}), so the column is stored as text", len(decorated), decorated.unique())
            if len(broken):
                add("non_numeric_in_numeric", col, "high", f"{len(broken)} values in a mostly-numeric column cannot be read as numbers", len(broken), broken.unique())
            continue

        # --- dates stored as text ---
        if real.str.contains(r"\d").mean() > 0.8:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(real, errors="coerce", format="mixed", dayfirst=False)
            if parsed.notna().mean() >= 0.7:
                bad = real[parsed.isna()]
                if len(bad):
                    add("invalid_dates", col, _sev(len(bad) / n, 0.05, 0.005), f"{len(bad)} values cannot be parsed as dates", len(bad), bad.unique())
                continue
        # also catch date columns where junk text dominates the failures
        if col.lower().endswith("date") or "date" in col.lower():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(real, errors="coerce", format="mixed")
            bad = real[parsed.isna()]
            if len(bad):
                add("invalid_dates", col, _sev(len(bad) / n, 0.05, 0.005), f"{len(bad)} values cannot be parsed as dates", len(bad), bad.unique())
            continue

        # --- inconsistent category spellings ---
        if real.nunique() <= 60:
            groups: dict[str, list[str]] = {}
            for v in real.str.strip().unique():
                groups.setdefault(_normalize(v), []).append(v)
            variants = {k: v for k, v in groups.items() if len(v) > 1}
            if variants:
                counts = real.str.strip().value_counts()
                affected = sum(int(counts[v].sum()) for v in variants.values())
                ex = [" / ".join(f"{x!r}" for x in v) for v in variants.values()]
                add("category_variants", col, "medium", f"{len(variants)} categories are spelled several ways", affected, ex)

    order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda i: (order[i.severity], -i.affected))
    for i, issue in enumerate(issues, 1):
        issue.id = f"I{i:02d}"
    return issues


def report(issues: list[Issue], rows: int, cols: int) -> str:
    out = [f"Scanned {rows:,} rows x {cols} columns: {len(issues)} issues found\n"]
    for i in issues:
        where = i.column or "(whole table)"
        out.append(f"[{i.id}] {i.severity.upper():6} {i.kind:24} {where}\n       {i.detail}")
        if i.examples:
            out.append(f"       examples: {i.examples}")
    return "\n".join(out)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "messy_orders.csv"
    data = load(path)
    found = scan(data)
    print(report(found, *data.shape))
