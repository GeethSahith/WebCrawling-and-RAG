# Reused RAG functions from rag_api.py for UI
import streamlit as st
import os
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# Loaded the environment variables
load_dotenv()

# Configuration of the vector base, llm model(openAI)
MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "vector_store/index.faiss"
METADATA_PATH = "vector_store/metadata.pkl"
DEFAULT_TOP_K = 5
LLM_MODEL = "gpt-4o-mini"

# Loaded Faiss Index so that the retreival would be easier
def load_index(index_path: str):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    return faiss.read_index(index_path)

# Loaded the pickle file which is saved in vector_store
def load_metadata(metadata_path: str):
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    with open(metadata_path, 'rb') as f:
        return pickle.load(f)

# Converted user query to embedding vector for similarity search
def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    embedding = model.encode([query], normalize_embeddings=True)
    return embedding

# Retrieved top k relevant chunks for the user query(Embedded).
def retrieve_top_k(query: str, top_k: int = DEFAULT_TOP_K):
    model = SentenceTransformer(MODEL_NAME)
    index = load_index(INDEX_PATH)
    metadata_list = load_metadata(METADATA_PATH)

    query_vector = embed_query(query, model)
    distances, indices = index.search(query_vector, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        results.append({
            "rank": rank + 1,
            "score": float(distances[0][rank]),
            "metadata": metadata_list[idx]
        })

    return results

# Built the prompt so that it would take the information from the retrieved chunks correctly
def build_prompt(query: str, retrieved_chunks: list, total_cards: int = None) -> str:
    context = "\n\n".join([
        f"Card: {c['metadata']['card_name']}\nCategory: {c['metadata']['category']}"
        for c in retrieved_chunks
    ])
    
    card_count_info = f"\n\nTotal cards in database: {total_cards}" if total_cards else ""
    
    prompt = f"""Answer the question using the information below. 
            If the answer is not in the context, provide general information.
            Context:
            {context}{card_count_info}
            Question: {query}
            Answer:"""
    return prompt

# Generated the response with OpenAILLM from prompt with context
def generate_answer(query: str, retrieved_chunks: list, total_cards: int = None) -> str:
    openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = build_prompt(query, retrieved_chunks, total_cards)
    
    response = openai.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content


def rag(query: str, top_k: int = DEFAULT_TOP_K) -> dict:
    metadata_list = load_metadata(METADATA_PATH)
    unique_cards = set(m["card_name"] for m in metadata_list)
    total_cards = len(unique_cards)
    
    retrieved_chunks = retrieve_top_k(query, top_k)
    answer = generate_answer(query, retrieved_chunks, total_cards)
    sources = [f"doc{i+1}" for i in range(len(retrieved_chunks))]
    
    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": [
            {
                "card_name": c["metadata"]["card_name"],
                "category": c["metadata"]["category"]
            }
            for c in retrieved_chunks
        ]
    }


# UI is done using Streamlit
st.set_page_config(page_title="Credit Card RAG", layout="wide")
if "query" not in st.session_state:
    st.session_state.query = ""
st.title("Credit Card RAG Pipeline")
st.markdown("---")
col1, col2 = st.columns([4, 1])
with col1:
    user_query = st.text_input("Ask a question about credit cards:", value=st.session_state.query, placeholder="e.g., What travel rewards do Bank of America cards offer?")
    st.session_state.query = user_query
with col2:
    st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
    submit_button = st.button("Search", key="submit", use_container_width=True)
# Query Processing is done 
if submit_button and user_query:
    with st.spinner("Searching and generating answer..."):
        try:
            result = rag(user_query)
            
            # Created three columns for the output 
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("User Query")
                st.write(f'"{user_query}"')
            
            with col2:
                st.subheader("RAG Answer")
                st.write(result["answer"])
            
            with col3:
                st.subheader("Sources")
                sources_text = ", ".join(result["sources"])
                st.write(sources_text)
            
            st.markdown("---")
            # Retrieved documents
            st.subheader("Retrieved Credit Cards")
            for i, chunk in enumerate(result["retrieved_chunks"], 1):
                with st.expander(f"(doc{i}) {chunk['card_name']} - {chunk['category']}"):
                    st.write(f"**Card Name:** {chunk['card_name']}")
                    st.write(f"**Category:** {chunk['category']}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Make sure the vector store is built. Run: `python index.py`")

elif submit_button and not user_query:
    st.warning("Please enter a question!")
# Example Queries
st.markdown("---")
st.subheader("Example Queries")
col1, col2 = st.columns(2)

if st.button("What travel rewards do Bank of America cards offer?", key="example1"):
    st.session_state.query = "What travel rewards do Bank of America cards offer?"
    st.rerun()
if st.button("What does Air France KLM Visa Signature card offer?", key="example2"):
    st.session_state.query = "What does Air France KLM Visa Signature card offer?"
    st.rerun()
