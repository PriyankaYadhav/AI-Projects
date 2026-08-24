"""
retrieve.py
Retrieves relevant verb conjugation chunks for a query.
Combines semantic search with exact verb-name filtering to avoid
embedding confusion between similar-sounding verbs (e.g. aller/avoir).
"""

import re
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "french_verbs"
MODEL_NAME = "intfloat/multilingual-e5-small"


def load_known_verbs(collection):
    """Pull the full list of verb names straight from the vector DB's metadata."""
    all_items = collection.get(include=["metadatas"])
    return sorted(set(m["verb_lower"] for m in all_items["metadatas"]))


def detect_verb(query, known_verbs):
    """Check if the query explicitly names one of our known verbs."""
    query_lower = query.lower()
    # Sort longest-first so e.g. "aller" doesn't wrongly match inside a longer word
    for v in sorted(known_verbs, key=len, reverse=True):
        if re.search(rf"\b{re.escape(v)}\b", query_lower):
            return v
    return None

TENSE_KEYWORDS = [
    ("plus-que-parfait", "plus-que-parfait"),
    ("pluperfect", "plus-que-parfait"),
    ("passé composé", "passé composé"),
    ("passe compose", "passé composé"),
    ("compound past", "passé composé"),
    ("futur antérieur", "futur antérieur"),
    ("future perfect", "futur antérieur"),
    ("futur simple", "futur simple"),
    ("simple future", "futur simple"),
    ("conditionnel passé", "conditionnel passé"),
    ("conditional perfect", "conditionnel passé"),
    ("conditional past", "conditionnel passé"),
    ("conditionnel présent", "conditionnel présent"),
    ("conditional", "conditionnel présent"),
    ("subjonctif", "subjonctif présent"),
    ("subjunctive", "subjonctif présent"),
    ("impératif", "impératif présent"),
    ("imperative", "impératif présent"),
    ("imparfait", "imparfait"),
    ("imperfect", "imparfait"),
    ("présent", "présent"),
    ("present", "présent"),
    ("past", "passé composé"),
]


def detect_tense(query):
    """Check if the query names a specific tense (English or French)."""
    q = query.lower()
    for keyword, tense in TENSE_KEYWORDS:
        if keyword in q:
            return tense
    return None

def retrieve(query, model, collection, known_verbs, k=5):
    """Retrieve the top-k most relevant chunks for a query."""
    verb_filter = detect_verb(query, known_verbs)
    tense_filter = detect_tense(query)

    query_emb = model.encode([f"query: {query}"], normalize_embeddings=True)

    conditions = []
    if verb_filter:
        conditions.append({"verb_lower": verb_filter})
    if tense_filter:
        conditions.append({"tense": tense_filter})

    if len(conditions) == 2:
        where = {"$and": conditions}
    elif len(conditions) == 1:
        where = conditions[0]
    else:
        where = None

    results = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=k,
        where=where,
    )
    return results, verb_filter ,tense_filter


def main():
    print("Loading model and connecting to Chroma...")
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    known_verbs = load_known_verbs(collection)
    print(f"Ready. {len(known_verbs)} verbs indexed.\n")

    # A few test queries to confirm retrieval quality
    test_queries = [
        "conjugate aller in present tense",
        "what's the passé composé of avoir?",
        "show me imparfait for être",
    ]

    for q in test_queries:
        results, verb_filter = retrieve(q, model, collection, known_verbs)
        print(f"Query: {q!r}")
        print(f"Detected verb filter: {verb_filter}")
        for doc_id, doc in zip(results["ids"][0], results["documents"][0]):
            print(f"  [{doc_id}] {doc[:60]}...")
        print()


if __name__ == "__main__":
    main()