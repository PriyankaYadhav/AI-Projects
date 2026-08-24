"""
generate.py
Generates grounded conjugation answers using a local LLM (via Ollama),
based on retrieved chunks from the vector database.
"""

import ollama
import chromadb
from sentence_transformers import SentenceTransformer
from retrieve import retrieve, load_known_verbs

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "french_verbs"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
LLM_MODEL = "llama3.2:1b"

SYSTEM_PROMPT = """You are a French conjugation assistant. Answer ONLY using
the provided conjugation data below. Output ONLY the pronoun and its exact
conjugated form from the data, with no extra words added.

Example of correct output format:
Je: vais
Tu: vas
Il/elle: va

Do not add infinitives, explanations, or extra verbs unless the data itself
contains them. If the exact verb or tense isn't in the context, say so
clearly rather than guessing."""

EXPLANATION_TRIGGERS = ["why", "explain", "difference", "when do i use", "when should i use", "how does"]


def generate_answer(query, model, collection, known_verbs):
    results, verb_filter, tense_filter = retrieve(query, model, collection, known_verbs, k=3)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    is_explanatory = any(trigger in query.lower() for trigger in EXPLANATION_TRIGGERS)

    # Direct lookup path: if we know the verb and this isn't an explanatory
    # question, trust retrieval and return the top chunk's data verbatim —
    # never let the LLM retype conjugation data (it can garble/hallucinate).
    if verb_filter and not is_explanatory and metadatas:
        top_meta = metadatas[0]
        lines = documents[0].split("\n")
        form_lines = [l for l in lines if not l.startswith(("Verb:", "Tense:"))]
        answer = f"({top_meta['tense']})\n" + "\n".join(form_lines)
        sources = [f"{top_meta['verb']} ({top_meta['tense']}), p.{top_meta['page']}"]
        return answer, sources

    # Explanatory fallback: use the LLM, but only for conceptual questions
    context = "\n\n---\n\n".join(documents)
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    answer = response["message"]["content"]
    sources = [f"{m['verb']} ({m['tense']}), p.{m['page']}" for m in metadatas]
    return answer, sources

def main():
    print("Loading embedding model and connecting to Chroma...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    known_verbs = load_known_verbs(collection)
    print("Ready.\n")

    query = "Conjugate aller in the present tense"
    print(f"Question: {query}\n")

    answer, sources = generate_answer(query, embed_model, collection, known_verbs)
    print("Answer:")
    print(answer)
    print("\nSources:")
    for s in sources:
        print(f"  - {s}")


if __name__ == "__main__":
    main()