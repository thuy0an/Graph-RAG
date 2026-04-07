from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Request gửi câu hỏi đến hệ thống RAG."""
    question: str
    filenames: list[str] = []  # Danh sách file cần query; rỗng = query tất cả files


class QueryResponse(BaseModel):
    """Response trả về câu trả lời và danh sách nguồn tài liệu."""
    answer: str
    sources: list[str] = []  # Tên các file được dùng để trả lời


class UploadResponse(BaseModel):
    """Response sau khi upload và xử lý file thành công."""
    message: str
    filename: str
    chunks: int    # Số chunk được tạo ra
    entities: int  # Số entity được trích xuất


class FileInfo(BaseModel):
    """Thông tin cơ bản của một file đã upload."""
    filename: str
    size_kb: float  # Kích thước file tính bằng KB


class FileListResponse(BaseModel):
    """Response danh sách tất cả file đã upload."""
    files: list[FileInfo]
    total: int  # Tổng số file
