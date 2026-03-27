from pathlib import Path

from app.core.document_loader import load_document
from app.core.graph_builder import build_lexical_graph

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def process_file(file_path: Path) -> dict:
    """Load → chunk → build Hierarchical Lexical Graph in Neo4j."""
    chunks = load_document(str(file_path))
    if not chunks:
        raise ValueError(
            f"Could not extract text from '{file_path.name}'. "
            "File may be corrupted or image-based PDF."
        )
    stats = build_lexical_graph(chunks, filename=file_path.name)
    return stats
