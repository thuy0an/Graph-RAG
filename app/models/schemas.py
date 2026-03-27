from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str] = []


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks: int
    entities: int


class FileInfo(BaseModel):
    filename: str
    size_kb: float


class FileListResponse(BaseModel):
    files: list[FileInfo]
    total: int
