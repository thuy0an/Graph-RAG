import asyncio
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.core.pipeline import UPLOAD_DIR, process_file
from app.core.rag_chain import query as rag_query
from app.core.neo4j_store import list_documents, delete_document
from app.models.schemas import (
    FileInfo,
    FileListResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)

router = APIRouter()

# Các định dạng file được phép upload
ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx"}


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload file và xử lý vào đồ thị Neo4j.

    Kiểm tra định dạng → lưu file vào disk → chạy pipeline xử lý
    (load, chunk, embed, trích xuất entity, lưu đồ thị).
    """
    # Kiểm tra định dạng file hợp lệ
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    # Lưu file vào thư mục uploads
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Chạy pipeline trong thread pool để không block event loop của FastAPI
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(None, process_file, file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return UploadResponse(
        message="File processed successfully",
        filename=file.filename,
        chunks=stats["chunks"],
        entities=stats["entities"],
    )


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Nhận câu hỏi và trả về câu trả lời từ hệ thống Graph RAG.

    Chạy toàn bộ query pipeline trong thread pool:
    embed câu hỏi → vector search → graph traversal → sinh câu trả lời bằng LLM.
    """
    try:
        # Chạy trong thread pool vì rag_query là blocking (gọi LLM, Neo4j)
        loop = asyncio.get_event_loop()
        answer, sources = await loop.run_in_executor(
            None, rag_query, request.question, request.filenames or None
        )
        return QueryResponse(answer=answer, sources=sources)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files", response_model=FileListResponse)
def list_files():
    """Lấy danh sách tất cả file đã upload kèm kích thước."""
    docs = list_documents()
    files = []
    for d in docs:
        fp = UPLOAD_DIR / d["filename"]
        # Tính kích thước file thực tế trên disk, trả về 0 nếu file không còn tồn tại
        size_kb = round(fp.stat().st_size / 1024, 2) if fp.exists() else 0.0
        files.append(FileInfo(filename=d["filename"], size_kb=size_kb))
    return FileListResponse(files=files, total=len(files))


@router.delete("/files/{filename}")
def delete_file(filename: str):
    """Xóa file khỏi Neo4j và disk.

    Tính doc_id từ MD5 của filename (khớp với cách tạo trong graph_builder),
    xóa toàn bộ node liên quan trong Neo4j, sau đó xóa file vật lý.
    """
    import hashlib
    # doc_id được tạo bằng MD5 của filename (xem _uid() trong graph_builder.py)
    doc_id = hashlib.md5(filename.encode()).hexdigest()

    # Xóa document và các node con khỏi Neo4j
    delete_document(doc_id)

    # Xóa file vật lý khỏi disk
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        file_path.unlink()

    return {"message": f"{filename} deleted"}
