"""Step 4: apply the approved fixes, save a cleaned COPY, then re-scan to prove it worked.

No LLM and no API key needed: every action is plain pandas.
Your original file is never modified.

Usage:  python apply.py                      (reads approved_plan.json from step 3)
        python apply.py other_plan.json
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import checks

# Run order is decided HERE, not by the LLM: normalise text first, convert types next,
# remove bad values, and only then de-duplicate (so near-duplicates become exact ones).
ORDER = [
    "strip_whitespace", "placeholders_to_null", "standardize_values", "convert_numeric",
    "parse_dates", "nullify_values", "fill_missing", "drop_duplicate_rows", "drop_column", "no_action",
]
RANK = {a: i for i, a in enumerate(ORDER)}


def _is_text(s: pd.Series) -> bool:
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def _changed(old: pd.Series, new: pd.Series) -> int:
    same = (old == new) | (old.isna() & new.isna())
    return int((~same).sum())


# ---- column-level actions: each takes (series, fix) and returns (new_series, note) ----

def strip_whitespace(s, f):
    if not _is_text(s):
        return s, "not a text column; nothing to do"
    return s.map(lambda v: v.strip() if isinstance(v, str) else v), ""


def placeholders_to_null(s, f):
    mask = s.map(lambda v: isinstance(v, str) and v.strip().lower() in checks.PLACEHOLDERS)
    return s.mask(mask, np.nan), ""


def standardize_values(s, f):
    mapping = {r["from_value"].strip(): r["to_value"] for r in (f.get("replacements") or [])}
    return s.map(lambda v: mapping.get(v.strip(), v) if isinstance(v, str) else v), ""


def convert_numeric(s, f):
    if pd.api.types.is_numeric_dtype(s):
        return s, "already numeric"
    cleaned = s.astype("object").map(lambda v: re.sub(r"[,\s$€£₹¥%]", "", v) if isinstance(v, str) else v)
    new = pd.to_numeric(cleaned, errors="coerce")
    lost = int((new.isna() & s.notna()).sum())
    return new, (f"{lost} values could not be read as numbers and became empty" if lost else "")


def parse_dates(s, f):
    txt = s.map(lambda v: v.strip() if isinstance(v, str) else v)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new = pd.to_datetime(txt, format="%Y-%m-%d", errors="coerce")
        other = txt[new.isna() & txt.notna()]
        guess = pd.to_datetime(other, format="mixed", errors="coerce")
    new.loc[guess.index] = guess
    notes = []
    guessed = int(guess.notna().sum())
    if guessed:
        notes.append(f"WARNING: {guessed} dates were not in YYYY-MM-DD form and were read as month/day/year; check they are right")
    lost = int((new.isna() & s.notna()).sum())
    if lost:
        notes.append(f"{lost} values were not valid dates and became empty")
    return new, "; ".join(notes)


def nullify_values(s, f):
    if not pd.api.types.is_numeric_dtype(s):
        return s, "column is not numeric (convert it first); skipped"
    if f.get("rule") == "negative":
        bad = s < 0
    else:
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        bad = ((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)) if iqr > 0 else pd.Series(False, index=s.index)
    return s.mask(bad, np.nan), ""


def fill_missing(s, f):
    how = f.get("fill")
    if how == "median":
        if not pd.api.types.is_numeric_dtype(s):
            return s, "median needs a numeric column; skipped"
        return s.fillna(s.median()), ""
    if how == "mode":
        m = s.mode(dropna=True)
        return (s.fillna(m.iloc[0]), "") if len(m) else (s, "no most-common value; skipped")
    if how == "constant":
        val = f.get("constant")
        if pd.api.types.is_numeric_dtype(s):
            try:
                val = float(val)
            except (TypeError, ValueError):
                return s, f"{val!r} is not a number; skipped"
        return s.fillna(val), ""
    return s, "no fill method given; skipped"


UNITS = {
    "strip_whitespace": "values trimmed", "placeholders_to_null": "placeholders set to empty",
    "standardize_values": "values renamed", "convert_numeric": "values converted to numbers",
    "parse_dates": "values converted to dates", "nullify_values": "values set to empty",
    "fill_missing": "empty cells filled",
}

HANDLERS = {
    "strip_whitespace": strip_whitespace, "placeholders_to_null": placeholders_to_null,
    "standardize_values": standardize_values, "convert_numeric": convert_numeric,
    "parse_dates": parse_dates, "nullify_values": nullify_values, "fill_missing": fill_missing,
}


def apply_plan(df: pd.DataFrame, fixes: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    work, log = df.copy(), []
    for n, f in sorted(enumerate(fixes, 1), key=lambda t: (RANK.get(t[1]["action"], 99), t[0])):
        action, col = f["action"], f.get("column")
        rec = {"n": n, "action": action, "column": col, "changed": 0, "unit": UNITS.get(action, "values changed"), "note": ""}
        try:
            if action == "drop_duplicate_rows":
                before = len(work)
                work = work.drop_duplicates().reset_index(drop=True)
                rec.update(changed=before - len(work), unit="rows removed")
            elif action == "drop_column":
                if col not in work.columns:
                    raise KeyError(f"column '{col}' not found")
                work = work.drop(columns=[col])
                rec.update(changed=1, unit="column removed")
            elif action == "no_action":
                rec["note"] = "left as is"
            elif action in HANDLERS:
                if col not in work.columns:
                    raise KeyError(f"column '{col}' not found (already dropped?)")
                new, note = HANDLERS[action](work[col], f)
                rec.update(changed=_changed(work[col], new), note=note)
                work[col] = new
            else:
                raise ValueError(f"unknown action '{action}'")
        except Exception as e:  # one bad fix must never ruin the whole run
            rec["note"] = f"SKIPPED: {e.args[0] if e.args else e}"
        log.append(rec)
    return work, sorted(log, key=lambda r: r["n"])


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Whole-number columns that gained blanks would print as 3.0; keep them as 3."""
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_float_dtype(s) and s.notna().any() and (s.dropna() % 1 == 0).all():
            df[c] = s.astype("Int64")
    return df


def compare(before: list, after: list) -> list[str]:
    b = {(i.kind, i.column): i.affected for i in before}
    a = {(i.kind, i.column): i.affected for i in after}
    lines = []
    for key, n in b.items():
        kind, col = key
        label = f"{kind} on {col or 'whole table'}"
        if key not in a:
            lines.append(f"  [ok] resolved   {label} ({n})")
        elif a[key] < n:
            lines.append(f"  [~]  reduced    {label}: {n} -> {a[key]}")
        else:
            lines.append(f"  [!]  still open {label}: {n} -> {a[key]}")
    for key, n in a.items():
        if key not in b:
            kind, col = key
            lines.append(f"  [!]  new        {kind} on {col or 'whole table'}: {n}")
    return lines


def main(plan_path: str = "approved_plan.json") -> None:
    if not Path(plan_path).exists():
        sys.exit(f"Cannot find {plan_path}. Run 'python audit.py' first to create it.")
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    src = Path(plan["source"])
    if not src.exists():
        sys.exit(f"Cannot find the source data file {src}. Run this from the folder that contains it.")
    out = src.with_name(src.stem + "_clean.csv")

    raw = checks.load(str(src))
    issues_before = checks.scan(raw)
    cleaned, log = apply_plan(raw, plan["approved"])
    cleaned = tidy(cleaned)
    cleaned.to_csv(out, index=False)
    issues_after = checks.scan(checks.load(str(out)))  # re-read from disk: verify what a user would actually get

    lines = [f"CLEANING REPORT   {src.name} -> {out.name}", ""]
    lines.append(f"Rows: {len(raw):,} -> {len(cleaned):,}    Columns: {raw.shape[1]} -> {cleaned.shape[1]}")
    lines += ["", "WHAT WAS DONE (in the order it ran)"]
    for r in sorted(log, key=lambda r: RANK.get(r["action"], 99)):
        what = f"{r['changed']:,} {r['unit']}" if r["changed"] else "no change"
        lines.append(f"  Fix {r['n']:>2}  {r['action']:<20} {(r['column'] or '(table)'):<14} {what}" + (f"  | {r['note']}" if r["note"] else ""))
    lines += ["", f"BEFORE -> AFTER   ({len(issues_before)} issues before, {len(issues_after)} after; re-scanned from the saved file)"]
    lines += compare(issues_before, issues_after)
    if any(l.startswith("  [!]  new") for l in lines):
        lines += ["", "Note: 'new' issues are usually blanks created on purpose (a placeholder or an invalid value turned into a true empty cell).",
                  "They are honest gaps in the data. Decide whether to leave them, fill them, or collect the missing data."]
    report = "\n".join(lines)
    print(report)
    Path("cleaning_report.txt").write_text(report, encoding="utf-8")
    print(f"\nSaved {out.name} and cleaning_report.txt (original file untouched)")


if __name__ == "__main__":
    main(*sys.argv[1:2])