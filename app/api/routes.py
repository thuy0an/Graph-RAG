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

ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx"}


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Run in thread pool — tránh block event loop khi file lớn
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
    try:
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
    docs = list_documents()
    files = []
    for d in docs:
        fp = UPLOAD_DIR / d["filename"]
        size_kb = round(fp.stat().st_size / 1024, 2) if fp.exists() else 0.0
        files.append(FileInfo(filename=d["filename"], size_kb=size_kb))
    return FileListResponse(files=files, total=len(files))


@router.delete("/files/{filename}")
def delete_file(filename: str):
    import hashlib
    doc_id = hashlib.md5(filename.encode()).hexdigest()

    # Remove from Neo4j
    delete_document(doc_id)

    # Remove physical file
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        file_path.unlink()

    return {"message": f"{filename} deleted"}
