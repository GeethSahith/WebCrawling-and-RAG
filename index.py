# Converted text chunks into embeddings and stores them in FAISS vector database.
import json
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
# Loaded chunks from JSON
with open("data/chunks/card_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks")
# Extracted texts and metadata the cards(json)
texts = [chunk["text"] for chunk in chunks]
metadata = [chunk["metadata"] for chunk in chunks]
print(f"Extracted {len(texts)} texts")
# Loaded embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
print(f"Generated embeddings with shape: {embeddings.shape}")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
index.add(embeddings)
print(f"FAISS index created with {index.ntotal} vectors")

# Created output directory and saved index.faiss in vector_store
os.makedirs("vector_store", exist_ok=True)
faiss.write_index(index, "vector_store/index.faiss")
with open("vector_store/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)
print(f"Total vectors: {index.ntotal}")
print(f"Embedding dimension: {dimension}")

