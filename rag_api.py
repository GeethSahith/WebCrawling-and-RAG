# It is CLI based
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# Loaded OPENAI_API_KEY variable from .env file
load_dotenv()


MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "vector_store/index.faiss"
METADATA_PATH = "vector_store/metadata.pkl"
DEFAULT_TOP_K = 5
LLM_MODEL = "gpt-4o-mini"


def load_index(index_path: str):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    return faiss.read_index(index_path)


def load_metadata(metadata_path: str):
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    import pickle
    with open(metadata_path, 'rb') as f:
        return pickle.load(f)


def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    embedding = model.encode([query], normalize_embeddings=True)
    return embedding
def retrieve_top_k(query: str, top_k: int = DEFAULT_TOP_K):
    model = SentenceTransformer(MODEL_NAME)
    index = load_index(INDEX_PATH)
    metadata_list = load_metadata(METADATA_PATH)
    query_vector = embed_query(query, model)
    # Searched FAISS index
    distances, indices = index.search(query_vector, top_k)
    # Collected the results of the search
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
def build_prompt(query: str, retrieved_chunks: list, total_cards: int = None) -> str:
    context = "\n\n".join([
        f"Card: {c['metadata']['card_name']}\nCategory: {c['metadata']['category']}"
        for c in retrieved_chunks
    ])
    card_count_info = f"\n\nTotal cards in database: {total_cards}" if total_cards else ""
    
    prompt = f"""Answer the question using the information below. If the answer is not in the context, provide general information.
            Context:
            {context}{card_count_info}
            Question: {query}
            Answer:"""
    return prompt

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