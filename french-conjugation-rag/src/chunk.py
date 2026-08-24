"""
chunk.py
Groups the row-level conjugation data (verbs.csv) into retrievable chunks:
one chunk per (verb, tense), bundling all pronoun forms together.
Writes the result to data/processed/chunks.json.
"""

import json
import pandas as pd

INPUT_PATH = "data/processed/verbs.csv"
OUTPUT_PATH = "data/processed/chunks.json"

# Order pronouns should appear in within a chunk, for readability
PRONOUN_ORDER = ["Je", "J'", "Tu", "Il/elle", "Nous", "Vous", "Ils/elles",
                  "Que je", "Que tu", "Qu'il/elle", "Que nous", "Que vous", "Qu'ils/elles"]


def build_chunk_text(verb, meaning, tense, rows):
    """Turn a group of pronoun/form rows into one readable text block."""
    rows_sorted = sorted(
        rows,
        key=lambda r: PRONOUN_ORDER.index(r["pronoun"]) if r["pronoun"] in PRONOUN_ORDER else 99
    )
    lines = [f"Verb: {verb} ({meaning})", f"Tense: {tense}"]
    for r in rows_sorted:
        lines.append(f"{r['pronoun']}: {r['form']}")
    return "\n".join(lines)


def main():
    df = pd.read_csv(INPUT_PATH)

    chunks = []
    grouped = df.groupby(["verb", "tense"])

    for (verb, tense), group in grouped:
        meaning = group["meaning"].iloc[0]
        page = int(group["page"].iloc[0])
        rows = group[["pronoun", "form"]].to_dict(orient="records")

        chunk_id = f"{verb.lower()}_{tense.replace(' ', '_')}"
        chunk_text = build_chunk_text(verb, meaning, tense, rows)

        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "metadata": {
                "verb": verb,
                "verb_lower": verb.lower(),
                "meaning": meaning,
                "tense": tense,
                "page": page,
                "num_forms": len(rows),
            }
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Done. {len(chunks)} chunks written to {OUTPUT_PATH}")
    print(f"Covering {df['verb'].nunique()} verbs x up to {df['tense'].nunique()} tenses")
    print("\n--- Sample chunk ---")
    print(json.dumps(chunks[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()