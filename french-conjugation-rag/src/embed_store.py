"""
embed_store.py
Embeds each verb chunk and stores it in a local Chroma vector database.
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/processed/chunks.json"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "french_verbs"
MODEL_NAME = "intfloat/multilingual-e5-small"


def main():
    print("Loading chunks...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")

    print(f"Loading embedding model ({MODEL_NAME})... this may take a few minutes the first time.")
    model = SentenceTransformer(MODEL_NAME)

    # E5 models expect a "passage: " prefix on stored text (and "query: " on
    # search queries later) — this is part of how the model was trained and
    # meaningfully improves retrieval quality if followed.
    texts_to_embed = [f"passage: {c['text']}" for c in chunks]

    print("Generating embeddings...")
    embeddings = model.encode(
        texts_to_embed,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=16,
    )

    print("Connecting to Chroma and indexing...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # Fresh start each run, to avoid duplicate/stale entries during development
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings.tolist(),
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )

    print(f"\nDone. {collection.count()} chunks indexed in Chroma at {CHROMA_PATH}")

    # Quick sanity check: does a semantic search actually return something sensible?
    print("\n--- Sanity check: searching for 'aller present tense' ---")
    query_emb = model.encode([f"query: aller present tense"], normalize_embeddings=True)
    results = collection.query(query_embeddings=query_emb.tolist(), n_results=3)
    for doc_id, doc in zip(results["ids"][0], results["documents"][0]):
        print(f"\n[{doc_id}]")
        print(doc[:100] + "...")


if __name__ == "__main__":
    main()