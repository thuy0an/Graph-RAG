from pathlib import Path

from app.core.document_loader import load_document
from app.core.graph_builder import build_lexical_graph

# Thư mục lưu file upload, tự tạo nếu chưa tồn tại
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def process_file(file_path: Path) -> dict:
    """Pipeline xử lý file hoàn chỉnh: Load → Chunk → Xây dựng đồ thị Neo4j.

    Bước 1: Đọc và chia nhỏ file thành các chunk văn bản
    Bước 2: Xây dựng Hierarchical Lexical Graph trong Neo4j
            (embedding, trích xuất entity, quan hệ, tóm tắt)

    Trả về dict thống kê: số chunk, section, entity, relation, thời gian xử lý.
    Raise ValueError nếu file không trích xuất được văn bản (PDF ảnh, file lỗi).
    """
    chunks = load_document(str(file_path))
    if not chunks:
        raise ValueError(
            f"Could not extract text from '{file_path.name}'. "
            "File may be corrupted or image-based PDF."
        )
    stats = build_lexical_graph(chunks, filename=file_path.name)
    return stats
