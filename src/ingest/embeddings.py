from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingClient:
    """
    Wrapper around sentence-transformers embedding model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode list of texts into embeddings.
        """
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


if __name__ == "__main__":
    client = EmbeddingClient()
    sample_texts = ["Hello world", "Machine learning is fun"]
    vectors = client.encode(sample_texts)

    print("Shape:", vectors.shape)
    print("First vector:", vectors[0][:10])  # preview first 10 dims
