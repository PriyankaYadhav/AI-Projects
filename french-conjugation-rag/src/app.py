"""
app.py
Streamlit chat interface for the French Conjugation RAG Tutor.
"""

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from retrieve import load_known_verbs
from generate import generate_answer

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "french_verbs"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"

st.set_page_config(page_title="French Conjugation Tutor", page_icon="🇫🇷")


@st.cache_resource
def load_pipeline():
    """Load the model and vector DB once, cached across reruns."""
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    known_verbs = load_known_verbs(collection)
    return embed_model, collection, known_verbs


st.title("🇫🇷 French Conjugation Tutor")
st.caption("Ask about any of the 65 verbs in the reference dataset — answers are grounded in the source PDF, with page citations.")

embed_model, collection, known_verbs = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("e.g. Conjugate aller in the present tense")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Looking it up..."):
            answer, sources = generate_answer(query, embed_model, collection, known_verbs)
            sources_text = "\n".join(f"- {s}" for s in sources)
            full_response = f"{answer}\n\n**Sources:**\n{sources_text}"
            st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})