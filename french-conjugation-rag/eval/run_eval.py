"""
run_eval.py
Evaluates retrieval accuracy against a hand-built test set of known
verb/tense pairs. Measures whether the retriever's #1 result is correct.
"""

import csv
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chromadb
from sentence_transformers import SentenceTransformer
from retrieve import retrieve, load_known_verbs

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "french_verbs"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
EVAL_SET_PATH = "data/eval/eval_set.csv"


def main():
    print("Loading model and connecting to Chroma...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    known_verbs = load_known_verbs(collection)

    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        eval_rows = list(csv.DictReader(f))

    print(f"Running {len(eval_rows)} evaluation questions...\n")

    correct = 0
    results_log = []

    for row in eval_rows:
        question = row["question"]
        expected_verb = row["expected_verb"]
        expected_tense = row["expected_tense"]

        results, verb_filter, tense_filter = retrieve(question, model, collection, known_verbs, k=1)
        top_meta = results["metadatas"][0][0] if results["metadatas"][0] else {}

        actual_verb = top_meta.get("verb_lower", "")
        actual_tense = top_meta.get("tense", "")

        is_correct = (actual_verb == expected_verb) and (actual_tense == expected_tense)
        correct += is_correct

        status = "PASS" if is_correct else "FAIL"
        results_log.append((status, question, expected_verb, expected_tense, actual_verb, actual_tense))
        print(f"[{status}] {question}")
        if not is_correct:
            print(f"       expected: {expected_verb} / {expected_tense}")
            print(f"       got:      {actual_verb} / {actual_tense}")

    accuracy = correct / len(eval_rows) * 100
    print(f"\n{'='*50}")
    print(f"Retrieval accuracy: {correct}/{len(eval_rows)} ({accuracy:.1f}%)")
    print(f"{'='*50}")

    failed = [r for r in results_log if r[0] == "FAIL"]
    if failed:
        print(f"\n{len(failed)} failure(s) to review:")
        for _, q, ev, et, av, at in failed:
            print(f"  - {q!r}: expected ({ev}, {et}), got ({av}, {at})")


if __name__ == "__main__":
    main()
