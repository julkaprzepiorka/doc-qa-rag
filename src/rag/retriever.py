from typing import List, Dict
import numpy as np
from src.ingest.embeddings import EmbeddingClient
from src.vectordb.faiss_store import FAISSStore


class Retriever:
    """
    Retrieve relevant chunks from FAISS index given a query.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384):
        self.embed_client = EmbeddingClient(model_name)
        self.store = FAISSStore(dim)
        self.store.load()

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Encode query and return top_k most relevant chunks.
        """
        q_vec = self.embed_client.encode([query])[0]  # single query
        results = self.store.search(q_vec, top_k=top_k)
        return results


if __name__ == "__main__":
    retriever = Retriever()
    query = "What is this document about?"
    results = retriever.retrieve(query, top_k=3)

    for r in results:
        print(f"[{r['file']}] score={r['score']:.4f}")
        print(r["preview"][:120], "...\n")
