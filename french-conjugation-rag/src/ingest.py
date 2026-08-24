"""
ingest.py
Extracts verb conjugation data from data/raw/french-verbs.pdf
into a structured CSV at data/processed/verbs.csv.
"""

import re
import pdfplumber
import pandas as pd

PDF_PATH = "data/raw/french-verbs.pdf"
OUTPUT_PATH = "data/processed/verbs.csv"

# Pronouns used in the 6-row conjugation grids (Présent, Passé composé,
# Imparfait, Plus-que-parfait, Futur simple, Futur antérieur,
# Conditionnel présent, Conditionnel passé)
GRID_PRONOUNS = ["Ils/elles", "Il/elle", "Nous", "Vous", "Tu", "Je", "J'"]
GRID_PATTERN = "|".join(re.escape(p) for p in sorted(GRID_PRONOUNS, key=len, reverse=True))
GRID_REGEX = re.compile(rf"({GRID_PATTERN})\s*([^A-Zé]*(?:[a-zéèêàûîôç]+\s*)*)")

# Subjonctif / Impératif lines are shaped differently
SUBJ_REGEX = re.compile(r"(Que je|Que tu|Qu'il/elle|Que nous|Que vous|Qu'ils/elles)\s+(\w+)")
IMPV_REGEX = re.compile(r"(\w+)\s*!\s*\((Tu|Nous|Vous)\)")

# Known tense header lines, in the order they appear on each verb page
TENSE_HEADERS = {
    "Présent Passé composé Imparfait Plus-que-parfait":
        ["présent", "passé composé", "imparfait", "plus-que-parfait"],
    "Futur simple Futur antérieur":
        ["futur simple", "futur antérieur"],
    "Conditionnel présent Conditionnel passé":
        ["conditionnel présent", "conditionnel passé"],
}


def parse_grid_line(line, tense_names):
    """Parse one pronoun row into (pronoun, tense, form) tuples."""
    matches = [(m.group(1), m.group(2).strip()) for m in GRID_REGEX.finditer(line)]
    results = []
    for (pronoun, form), tense in zip(matches, tense_names):
        if form:
            results.append((pronoun, tense, form))
    return results


def parse_verb_page(text, page_num):
    """Parse a single page of text into a list of conjugation records."""
    lines = [l for l in text.split("\n") if l.strip()]
    records = []

    # Identify the verb + meaning line, e.g. "• Être To be"
    verb_match = re.search(r"•\s*(\S+(?:\s*\(?s[e\']?\)?)?)\s+To\s+(.+)", text)
    if not verb_match:
        return records  # not a verb page (e.g. table of contents, section divider)

    verb = verb_match.group(1).strip()
    meaning = verb_match.group(2).strip()

    current_tenses = None
    for line in lines:
        if line in TENSE_HEADERS:
            current_tenses = TENSE_HEADERS[line]
            continue

        if line.startswith(("Que je", "Que tu", "Qu'il", "Que nous", "Que vous", "Qu'ils")):
            subj = SUBJ_REGEX.findall(line)
            for pronoun, form in subj:
                records.append({
                    "verb": verb, "meaning": meaning, "tense": "subjonctif présent",
                    "pronoun": pronoun, "form": form, "page": page_num
                })
            impv = IMPV_REGEX.findall(line)
            for form, pronoun in impv:
                records.append({
                    "verb": verb, "meaning": meaning, "tense": "impératif présent",
                    "pronoun": pronoun, "form": form, "page": page_num
                })
            continue

        if current_tenses:
            for pronoun, tense, form in parse_grid_line(line, current_tenses):
                records.append({
                    "verb": verb, "meaning": meaning, "tense": tense,
                    "pronoun": pronoun, "form": form, "page": page_num
                })

    return records


def main():
    all_records = []
    with pdfplumber.open(PDF_PATH) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            page_records = parse_verb_page(text, i + 1)
            all_records.extend(page_records)
            if page_records:
                verb_name = page_records[0]["verb"]
                print(f"Page {i+1}: parsed {len(page_records)} forms for '{verb_name}'")

    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\nDone. {len(df)} total rows written to {OUTPUT_PATH}")
    print(f"Unique verbs found: {df['verb'].nunique()}")


if __name__ == "__main__":
    main()