"""
Trains a classifier on the engineered fraud-signal features and evaluates it.

Usage:
    python train_model.py ../data/processed/job_postings_clean.csv
"""

import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

FEATURE_COLS = [
    "flag_no_salary", "flag_no_company_logo", "flag_no_company_profile",
    "flag_urgency_language", "flag_requests_personal_info",
    "flag_off_platform_contact", "flag_vague_title_language",
    "flag_short_description", "description_word_count",
]


def main():
    if len(sys.argv) != 2:
        print("Usage: python train_model.py <path_to_clean_csv>")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])

    if "fraudulent" not in df.columns or df["fraudulent"].isna().all():
        print("No 'fraudulent' labels found - can't train. Use the labeled Kaggle CSV.")
        sys.exit(1)

    X = df[FEATURE_COLS].astype(float)
    y = df["fraudulent"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("=== Classification report ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("=== Confusion matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Feature importance (coefficients) ===")
    for feat, coef in sorted(zip(FEATURE_COLS, model.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"{feat:35s} {coef:+.3f}")


if __name__ == "__main__":
    main()