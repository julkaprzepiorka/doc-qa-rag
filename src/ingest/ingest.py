import argparse
import os
import pathlib
from typing import List
from pypdf import PdfReader
import docx
from src.ingest.chunker import split_text
from src.ingest.embeddings import EmbeddingClient
from src.vectordb.faiss_store import FAISSStore


def load_text_from_file(file_path: str) -> str:
    """
    Load text from txt, pdf, or docx file.
    """
    ext = pathlib.Path(file_path).suffix.lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join([page.extract_text() or "" for page in reader.pages])

    elif ext == ".docx":
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def ingest(path: str, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, overlap: int = 200):
    """
    Ingest documents: load → chunk → embed → index.
    """
    embed_client = EmbeddingClient(model_name)
    store = FAISSStore(dim=384)  # 384 for MiniLM

    files = [os.path.join(path, f) for f in os.listdir(path)]
    for file in files:
        print(f"Ingesting {file} ...")
        text = load_text_from_file(file)
        chunks = split_text(text, max_chars=chunk_size, overlap=overlap)

        vectors = embed_client.encode([c["content"] for c in chunks])
        metadata = [
            {
                "file": os.path.basename(file),
                "char_start": c["char_start"],
                "char_end": c["char_end"],
                "preview": c["content"][:200]
            }
            for c in chunks
        ]
        store.add(vectors, metadata)

    store.save()
    print("Ingestion complete. Index and metadata saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into FAISS index")
    parser.add_argument("--path", type=str, required=True, help="Path to folder with documents")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Embedding model")
    args = parser.parse_args()

    ingest(args.path, args.model)
