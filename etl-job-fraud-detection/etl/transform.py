"""
Transform stage: reads the raw Kaggle "Real or Fake Job Postings" CSV and
engineers structured fraud-signal features from the text fields.

Usage:
    python transform.py ../data/raw/fake_job_postings.csv
"""

import re
import sys
from pathlib import Path

import pandas as pd

URGENCY_PATTERNS = re.compile(
    r"\b(urgent|immediate(ly)? start|act now|apply immediately|limited spots|"
    r"hiring fast|don't wait|start today)\b",
    re.IGNORECASE,
)

PERSONAL_INFO_PATTERNS = re.compile(
    r"\b(ssn|social security|bank account|routing number|credit card|"
    r"passport number|date of birth|wire transfer)\b",
    re.IGNORECASE,
)

OFF_PLATFORM_CONTACT = re.compile(
    r"\b(gmail\.com|yahoo\.com|hotmail\.com|whatsapp|telegram|text me at|"
    r"call me at|\+\d{1,3}[\s-]?\d{6,})\b",
    re.IGNORECASE,
)

VAGUE_TITLE_PATTERNS = re.compile(
    r"\b(work from home|earn \$?\d+|no experience needed|easy money|"
    r"be your own boss)\b",
    re.IGNORECASE,
)


def word_count(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())


def has_pattern(text: str, pattern: re.Pattern) -> bool:
    if not isinstance(text, str):
        return False
    return bool(pattern.search(text))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    text_fields = ["description", "requirements", "benefits", "company_profile"]
    for col in text_fields:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    combined_text = (
        df["description"] + " " + df["requirements"] + " " + df["benefits"]
    )

    df["flag_no_salary"] = df.get("salary_range", pd.Series([None] * len(df))).isna()
    df["flag_no_company_logo"] = df.get("has_company_logo", 0).fillna(0) == 0
    df["flag_no_company_profile"] = df["company_profile"].str.strip() == ""
    df["flag_urgency_language"] = combined_text.apply(lambda t: has_pattern(t, URGENCY_PATTERNS))
    df["flag_requests_personal_info"] = combined_text.apply(lambda t: has_pattern(t, PERSONAL_INFO_PATTERNS))
    df["flag_off_platform_contact"] = combined_text.apply(lambda t: has_pattern(t, OFF_PLATFORM_CONTACT))
    df["flag_vague_title_language"] = df["title"].fillna("").apply(lambda t: has_pattern(t, VAGUE_TITLE_PATTERNS))
    df["description_word_count"] = df["description"].apply(word_count)
    df["flag_short_description"] = df["description_word_count"] < 30

    df["fraud_signal_count"] = df[[
        "flag_no_salary", "flag_no_company_logo", "flag_no_company_profile",
        "flag_urgency_language", "flag_requests_personal_info",
        "flag_off_platform_contact", "flag_vague_title_language",
        "flag_short_description",
    ]].sum(axis=1)

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "job_id", "title", "location", "department", "employment_type",
        "required_experience", "required_education", "industry", "function",
        "description_word_count", "fraud_signal_count",
        "flag_no_salary", "flag_no_company_logo", "flag_no_company_profile",
        "flag_urgency_language", "flag_requests_personal_info",
        "flag_off_platform_contact", "flag_vague_title_language",
        "flag_short_description",
    ]
    if "fraudulent" in df.columns:
        keep_cols.append("fraudulent")

    for col in keep_cols:
        if col not in df.columns:
            df[col] = None

    out = df[keep_cols].copy()
    out = out.drop_duplicates(subset=["job_id"])
    return out


def main():
    if len(sys.argv) != 2:
        print("Usage: python transform.py <path_to_raw_csv>")
        sys.exit(1)

    raw_path = sys.argv[1]
    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df)} raw rows from {raw_path}")

    df = engineer_features(df)
    clean_df = clean(df)

    out_dir = Path("../data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "job_postings_clean.csv"
    clean_df.to_csv(out_path, index=False)
    print(f"Wrote {len(clean_df)} clean rows to {out_path}")

    if "fraudulent" in clean_df.columns:
        rate = clean_df["fraudulent"].mean()
        print(f"Fraud rate in dataset: {rate:.2%}")


if __name__ == "__main__":
    main()