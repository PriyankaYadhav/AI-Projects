# 🇫🇷 French Conjugation RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers French verb conjugation questions, grounding every direct lookup in a structured knowledge base extracted from a reference PDF — rather than relying on an LLM's memory, which often gets irregular conjugations wrong.

Built as a fully local, free pipeline: no paid APIs required.


## Demo

Ask things like:
- *"Conjugate aller in the present tense"*
- *"What's the passé composé of avoir?"*
- *"Show me imparfait for être"*
- *"How is imparfait different from passé composé?"*

Direct conjugation lookups are answered **verbatim from retrieved source data** (zero LLM rewriting, zero hallucination risk). Conceptual grammar questions are routed to the LLM separately, clearly labeled as general knowledge rather than sourced from the dataset.

## Evaluation

Retrieval accuracy was measured against a hand-built 20-question test set spanning 12 verbs and 10 tenses, phrased in a mix of English and French terminology (e.g. "past tense" vs. "passé composé") to test robustness to phrasing variation.

```
Retrieval accuracy: 20/20 (100.0%)
```

Since direct lookups return retrieved data verbatim (no LLM rewriting), this 100% retrieval accuracy translates directly into 100% generation accuracy for in-scope conjugation queries. See `eval/run_eval.py` to reproduce, and the full per-question breakdown in the project completion report.

---

## Why RAG (and not just prompting an LLM)

Small LLMs — and even large ones — can confidently produce incorrect irregular conjugations. This project solves that by:
1. Extracting real conjugation tables from a reference PDF into structured data
2. Retrieving the exact matching verb + tense for a query using semantic search **combined with exact metadata filtering**
3. Returning that data directly for lookups, only involving the LLM for genuinely conceptual questions

---

## Architecture

```
french-verbs.pdf
     │
     ▼
[1] Ingestion (ingest.py)        — regex-based extraction of conjugation
     │                              grids from unstructured PDF text
     ▼
[2] Chunking (chunk.py)          — group rows into one chunk per (verb, tense)
     │
     ▼
[3] Embeddings (embed_store.py)  — multilingual-e5-small, stored in ChromaDB
     │
     ▼
[4] Retrieval (retrieve.py)      — semantic search + exact verb-name and
     │                              tense-name metadata filtering
     ▼
[5] Generation (generate.py)     — direct lookup for conjugation queries
     │                              (no LLM involved); local LLM (Llama 3.2)
     │                              only for conceptual/explanatory questions
     ▼
[6] Chat UI (app.py)             — Streamlit interface
```

---

## Tech Stack

| Component | Tool | Notes |
|---|---|---|
| PDF extraction | `pdfplumber` | Source PDF has no real table gridlines — custom regex parsing was required |
| Data wrangling | `pandas` | Cleaned conjugation rows into structured CSV |
| Embeddings | `sentence-transformers` (`intfloat/multilingual-e5-small`) | Local, free, multilingual |
| Vector database | `ChromaDB` | Local, persistent, metadata filtering support |
| LLM | `Ollama` running `llama3.2:1b` | Fully local and free — no API costs |
| Interface | `Streamlit` | Chat-style UI |

---

## Dataset

- **Source:** a 500-verb French conjugation reference PDF
- **Extracted:** 65 verbs × up to 10 tenses = **649 retrievable chunks**, **3,687** individual conjugated forms
- Tenses covered: présent, passé composé, imparfait, plus-que-parfait, futur simple, futur antérieur, conditionnel présent, conditionnel passé, subjonctif présent, impératif présent

---

## Project Structure

```
french-conjugation-rag/
├── data/
│   ├── raw/                  # source PDFs
│   └── processed/            # verbs.csv, chunks.json
├── src/
│   ├── ingest.py              # PDF -> structured CSV
│   ├── chunk.py                # CSV -> chunks.json
│   ├── embed_store.py          # embeddings + Chroma indexing
│   ├── retrieve.py             # verb/tense-filtered retrieval
│   ├── generate.py             # grounded answer generation
│   └── app.py                  # Streamlit chat interface
├── chroma_db/                  # persisted vector store (generated, gitignored)
├── requirements.txt
├── .env                          # API keys if extended (gitignored)
└── README.md
```

---

## Setup

```bash
# Create and activate a virtual environment
uv venv --python 3.12
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS/Linux

# Install dependencies
uv pip install -r requirements.txt

# Install Ollama and pull the local model (see ollama.com/download)
ollama pull llama3.2:1b

# Run the pipeline, in order (only needed once — outputs are cached)
python src/ingest.py
python src/chunk.py
python src/embed_store.py

# Launch the app
streamlit run src/app.py
```

---

## Key Design Decisions

- **No visual gridlines in the source PDF** meant standard table-extraction tools (`camelot`, `pdfplumber.extract_tables()`) returned nothing. A custom regex parser was built and validated against the actual document structure instead.
- **Embedding similarity alone isn't precise enough** for this domain — semantically similar-sounding verbs (e.g. *aller* / *avoir*) and cross-lingual tense names (English "present" vs. French "présent") can confuse pure vector search. This was solved with exact metadata filtering on both verb name and tense, detected via keyword matching before the vector search runs.
- **The LLM never rewrites conjugation data for direct lookups** — retrieved chunks are returned verbatim. Early testing showed even correct retrieval could be undermined by the LLM garbling or hallucinating text during generation (e.g. producing "fus, fus, fut..." — passé simple of *être* — instead of the actual retrieved Aller data). Routing lookups around the LLM entirely eliminated this failure mode.
- **A smaller embedding and LLM model were chosen deliberately** (`multilingual-e5-small`, `llama3.2:1b`) to run reliably on modest hardware (8GB RAM, no dedicated GPU) — demonstrating the pipeline works end-to-end without expensive infrastructure.

---

## Known Limitations

- The dataset covers 65 verbs, not the full French verb inventory — unrecognized verbs currently fall through to an ungrounded LLM response rather than a clean "not found" message (planned improvement).
- "Past tense" defaults to passé composé; French has multiple past tenses depending on context.
- Conceptual grammar explanations come from the LLM's general knowledge, not the source PDF, since the dataset only contains conjugation tables.

---

## Possible Next Steps

- Formal evaluation with RAGAS (faithfulness, context precision) against a hand-built test set
- Expand the "verb not found" handling with a clear fallback message
- Swap in a larger local model (e.g. Phi-3 Mini) or the Claude API for higher-quality explanatory answers
- Deploy publicly (Streamlit Community Cloud or similar)
