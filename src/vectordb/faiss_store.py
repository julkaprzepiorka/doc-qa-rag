import faiss
import numpy as np
from typing import List, Dict, Any
import json
import os


class FAISSStore:
    """
    Simple FAISS index wrapper for dense vector search.
    """

    def __init__(self, dim: int, index_path: str = ".faiss/index.faiss", meta_path: str = ".faiss/meta.jsonl"):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity if vectors normalized
        self.index_path = index_path
        self.meta_path = meta_path
        self.metadata: List[Dict[str, Any]] = []

        # ensure dir exists
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Add vectors and their metadata to the index.
        """
        if vectors.ndim == 1:
            vectors = np.expand_dims(vectors, axis=0)

        self.index.add(vectors.astype("float32"))
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search top_k nearest vectors for the query.
        """
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        distances, indices = self.index.search(query_vector.astype("float32"), top_k)
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx].copy()
            meta["score"] = float(score)
            results.append(meta)
        return results

    def save(self):
        """
        Save index and metadata to disk.
        """
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for m in self.metadata:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def load(self):
        """
        Load index and metadata from disk.
        """
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = [json.loads(line) for line in f]


if __name__ == "__main__":
    # demo usage
    dim = 384
    store = FAISSStore(dim)

    # random vectors for demo
    vecs = np.random.rand(3, dim).astype("float32")
    meta = [{"id": i, "text": f"chunk {i}"} for i in range(3)]

    store.add(vecs, meta)
    q = np.random.rand(dim).astype("float32")

    results = store.search(q, top_k=2)
    print("Search results:", results)
