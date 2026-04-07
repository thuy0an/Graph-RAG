import time
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Kích thước mỗi chunk (số ký tự) và phần overlap giữa các chunk liền kề
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Khởi tạo splitter một lần duy nhất, tái sử dụng cho mọi file
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def load_document(file_path: str) -> list[Document]:
    """Đọc file PDF/DOCX và chia nhỏ thành các chunk văn bản.

    Quy trình:
      1. Load toàn bộ nội dung file theo định dạng tương ứng
      2. Chia nhỏ thành các chunk với kích thước và overlap đã cấu hình
      3. Gắn metadata (tên file, index chunk) vào từng chunk

    Trả về danh sách Document (chunk) sẵn sàng để embedding và lưu vào đồ thị.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    file_size_kb = path.stat().st_size / 1024

    print(f"\n{'='*60}")
    print(f"  DOCUMENT LOADER")
    print(f"{'='*60}")
    print(f"  File     : {path.name}")
    print(f"  Type     : {suffix.upper()}")
    print(f"  Size     : {file_size_kb:.1f} KB")

    # Chọn loader phù hợp theo định dạng file
    if suffix == ".pdf":
        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(file_path)
    elif suffix in (".doc", ".docx"):
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    # Bước 1: Load nội dung file
    t0 = time.perf_counter()
    docs = loader.load()
    t_load = time.perf_counter() - t0

    total_chars = sum(len(d.page_content) for d in docs)
    print(f"\n  [1/2] Load")
    print(f"        Pages    : {len(docs)}")
    print(f"        Chars    : {total_chars:,}")
    print(f"        Time     : {t_load:.3f}s")

    # Bước 2: Chia nhỏ thành các chunk
    t0 = time.perf_counter()
    chunks = _splitter.split_documents(docs)
    t_chunk = time.perf_counter() - t0

    avg_chunk_len = sum(len(c.page_content) for c in chunks) / max(len(chunks), 1)
    print(f"\n  [2/2] Chunk")
    print(f"        Chunks   : {len(chunks)}")
    print(f"        Avg size : {avg_chunk_len:.0f} chars")
    print(f"        Settings : size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    print(f"        Time     : {t_chunk:.3f}s")

    # Gắn metadata vào từng chunk để truy vết nguồn gốc
    for i, chunk in enumerate(chunks):
        chunk.metadata["source_file"] = path.name
        chunk.metadata["chunk_index"] = i

    print(f"\n  Total load+chunk time: {t_load + t_chunk:.3f}s")
    print(f"{'='*60}")

    return chunks
