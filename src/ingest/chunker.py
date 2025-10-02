from typing import List, Dict


def split_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[Dict]:
    """
    Split text into overlapping chunks.

    Args:
        text: Full input text.
        max_chars: Maximum length of a chunk.
        overlap: Number of characters shared between chunks.

    Returns:
        List of dicts with 'content', 'char_start', and 'char_end'.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + max_chars, text_length)
        chunk_text = text[start:end]

        chunks.append({
            "content": chunk_text.strip(),
            "char_start": start,
            "char_end": end
        })

        # move forward with overlap
        start += max_chars - overlap

    return chunks


if __name__ == "__main__":
    sample_text = "This is a sample text. " * 100
    chunks = split_text(sample_text, max_chars=200, overlap=50)

    for i, ch in enumerate(chunks):
        print(f"Chunk {i}: {ch['char_start']}–{ch['char_end']}")
        print(ch['content'][:80], "...\n")
