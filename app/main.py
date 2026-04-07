from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle hook của FastAPI: chạy khi app khởi động và khi tắt.

    Khi khởi động:
      - Kiểm tra kết nối Ollama (nếu đang dùng Ollama làm LLM hoặc embedding)
      - Tạo các constraint và vector index trong Neo4j
    """
    # Kiểm tra kết nối Ollama nếu đang dùng provider này
    if settings.LLM_PROVIDER == "ollama" or settings.EMBED_PROVIDER == "ollama":
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
                resp.raise_for_status()
                models = [m["name"] for m in resp.json().get("models", [])]
                print(f"[Ollama] Connected. Models: {models}")
        except Exception as e:
            print(f"[Ollama] WARNING: {e}")

    # Tạo indexes và constraints trong Neo4j (bỏ qua nếu đã tồn tại)
    try:
        from app.core.neo4j_store import setup_indexes
        setup_indexes()
        print("[Neo4j] Indexes ready.")
    except Exception as e:
        print(f"[Neo4j] WARNING: Could not setup indexes — {e}")

    yield  # App chạy bình thường từ đây


# Khởi tạo FastAPI app
app = FastAPI(title="Graph RAG — Hierarchical Lexical Graph", version="2.0.0", lifespan=lifespan)

# Cho phép tất cả origin truy cập API (phù hợp cho dev, cần giới hạn khi production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký tất cả route API với prefix /api
app.include_router(router, prefix="/api")


@app.get("/health")
def health():
    """Health check endpoint: kiểm tra trạng thái kết nối Neo4j và cấu hình provider."""
    from app.core.neo4j_store import run
    try:
        run("RETURN 1")
        neo4j_status = "connected"
    except Exception as e:
        neo4j_status = f"error: {e}"

    return {
        "status": "ok",
        "llm_provider": settings.LLM_PROVIDER,
        "embed_provider": settings.EMBED_PROVIDER,
        "neo4j": neo4j_status,
    }
