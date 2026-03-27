import time
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def load_document(file_path: str) -> list[Document]:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(file_path)
    elif suffix in (".doc", ".docx"):
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    t0 = time.perf_counter()
    docs = loader.load()
    t_load = time.perf_counter() - t0
    print(f"  [load]  {path.name} — {len(docs)} pages in {t_load:.3f}s")

    t0 = time.perf_counter()
    chunks = _splitter.split_documents(docs)
    t_chunk = time.perf_counter() - t0
    print(f"  [chunk] {len(chunks)} chunks in {t_chunk:.3f}s  (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    for i, chunk in enumerate(chunks):
        chunk.metadata["source_file"] = path.name
        chunk.metadata["chunk_index"] = i

    return chunks
