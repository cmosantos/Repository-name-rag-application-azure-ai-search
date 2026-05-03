from pathlib import Path


def split_text(text: str, max_chars: int = 1200, overlap_chars: int = 200) -> list[str]:
    """
    Splits long text into overlapping chunks.

    This is a simple chunking strategy for a learning project.
    In production, you might use token-based chunking and preserve document structure.
    """
    clean_text = " ".join(text.replace("\r", "\n").split())

    if not clean_text:
        return []

    if len(clean_text) <= max_chars:
        return [clean_text]

    chunks = []
    start = 0
    text_length = len(clean_text)

    while start < text_length:
        end = min(start + max_chars, text_length)

        if end < text_length:
            sentence_break = clean_text.rfind(". ", start, end)
            if sentence_break > start + int(max_chars * 0.6):
                end = sentence_break + 1

        chunk = clean_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - overlap_chars, 0)

        if start >= end:
            start = end

    return chunks


def load_text_documents(data_dir: Path) -> list[dict]:
    """
    Loads all .txt files from the data folder.
    """
    documents = []

    for file_path in sorted(data_dir.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8")

        documents.append(
            {
                "source": file_path.name,
                "title": file_path.stem.replace("-", " ").replace("_", " ").title(),
                "content": content,
            }
        )

    return documents
