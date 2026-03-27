# Graph RAG v2 — Hierarchical Lexical Graph

Hệ thống hỏi đáp tài liệu dựa trên **Hierarchical Lexical Graph** lưu trong **Neo4j**, kết hợp vector search, graph traversal và hierarchical summarization.

---

## Kiến trúc

```
┌─────────────────────────────────────────────────────────────────┐
│                        NEO4J GRAPH MODEL                        │
│                                                                 │
│  (:Document {summary})                                          │
│       │ HAS_SECTION                                             │
│       ▼                                                         │
│  (:Section {title, summary})  ──NEXT_SECTION──►  (:Section)    │
│       │ HAS_CHUNK                                               │
│       ▼                                                         │
│  (:Chunk {text, embedding})   ──NEXT_CHUNK────►  (:Chunk)      │
│       │ MENTIONS                                                │
│       ▼                                                         │
│  (:Entity {name, type})  ──RELATED_TO {relation}──► (:Entity)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Cấu trúc dự án

```
ProjectGraphRAG/
├── app/
│   ├── main.py                  # FastAPI app, startup: Neo4j index setup
│   ├── api/routes.py            # /upload, /query, /files endpoints
│   ├── core/
│   │   ├── config.py            # Pydantic settings (.env)
│   │   ├── document_loader.py   # PDF/DOCX → chunks
│   │   ├── graph_builder.py     # Hierarchical Lexical Graph builder
│   │   ├── neo4j_store.py       # Neo4j CRUD + vector search + graph traversal
│   │   ├── rag_chain.py         # Query pipeline
│   │   └── providers.py         # LLM + Embeddings factory
│   └── models/schemas.py
├── uploads/
├── .env
├── requirements.txt
└── ui.html
```

---

## Upload Pipeline

```
File (PDF/DOCX)
    │
    ▼
document_loader.py  →  chunks (800 tokens, overlap 100)
    │
    ▼
graph_builder.build_lexical_graph()
    │
    ├─ Group chunks → Sections (4 chunks/section)
    │       └─ LLM: generate title + summary per section
    │
    ├─ LLM: extract entities + relations per chunk
    │       └─ Normalize entity names (lowercase)
    │
    ├─ LLM: generate document-level summary
    │
    └─ Persist to Neo4j:
           Document → Sections → Chunks (with embeddings)
           Chunks → Entities → RELATED_TO edges
           NEXT_SECTION / NEXT_CHUNK links
```

---

## Query Pipeline

```
Question
    │
    ├─ Embed question  →  Neo4j vector search  →  top-K Chunks
    │
    ├─ LLM extract entities from question
    │       └─ Cypher BFS traversal  →  Entity relation facts
    │
    ├─ Section summaries (từ chunks tìm được)  →  hierarchical context
    │
    ├─ Document summary  →  global context
    │
    └─ LLM answer với 4-tầng context:
           [Document Overview]
           [Section Summaries]
           [Knowledge Graph facts]
           [Retrieved Passages]
```

---

## Kiến thức cần biết

### Hierarchical Lexical Graph
Thay vì flat chunks, tài liệu được tổ chức thành 3 tầng:
- **Document** — summary toàn bộ tài liệu
- **Section** — nhóm 4 chunks liên tiếp, có title + summary
- **Chunk** — đoạn text gốc + embedding vector

Khi query, LLM nhận context từ cả 3 tầng → trả lời chính xác hơn với câu hỏi tổng quan lẫn câu hỏi chi tiết.

### Neo4j Vector Index
Neo4j 5.11+ hỗ trợ native vector index (`CREATE VECTOR INDEX`). Thay thế hoàn toàn FAISS — vừa lưu graph vừa làm vector search trong cùng một database.

### Graph Traversal (Cypher)
Thay vì NetworkX BFS trên file JSON, giờ dùng Cypher query trực tiếp trên Neo4j:
```cypher
MATCH (e:Entity)-[r1:RELATED_TO]->(e2:Entity)
WHERE toLower(e.name) CONTAINS toLower($name)
OPTIONAL MATCH (e2)-[r2:RELATED_TO]->(e3:Entity)
RETURN ...
```

### Entity Resolution
Normalize tên entity về lowercase + collapse whitespace trước khi lưu vào Neo4j. Cùng entity sẽ MERGE vào cùng 1 node thay vì tạo duplicate.

---

## Cài đặt

### 1. Neo4j (miễn phí)

**Option A — Docker (khuyến nghị):**
```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.20
```

**Option B — Neo4j Desktop:**
Tải tại https://neo4j.com/download/ → tạo database mới → set password

### 2. Python dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

### 3. Cấu hình `.env`

```env
# LLM
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2

# Embeddings
EMBED_PROVIDER=ollama
OLLAMA_EMBED_MODEL=nomic-embed-text   # output dim: 768

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

> Nếu dùng embedding model khác, cập nhật `vector.dimensions` trong `neo4j_store.setup_indexes()`.

### 4. Chạy server

```bash
.venv\Scripts\uvicorn.exe app.main:app --reload --host 127.0.0.1 --port 8000
```

Mở `ui.html` trong browser. API docs: `http://127.0.0.1:8000/docs`

---

## API Endpoints

| Method | Path | Mô tả |
|--------|------|-------|
| `POST` | `/api/upload` | Upload + index tài liệu vào Neo4j |
| `POST` | `/api/query` | Hỏi đáp |
| `GET` | `/api/files` | Danh sách tài liệu |
| `DELETE` | `/api/files/{filename}` | Xóa tài liệu khỏi Neo4j + disk |
| `GET` | `/health` | Status server + Neo4j connection |

---

## So sánh v1 vs v2

| | v1 (FAISS + NetworkX) | v2 (Neo4j) |
|---|---|---|
| Vector store | FAISS (file) | Neo4j vector index |
| Graph store | NetworkX JSON | Neo4j DiGraph |
| Hierarchy | Không | Document → Section → Chunk |
| Section summary | Không | Có (LLM generated) |
| Document summary | Không | Có (LLM generated) |
| Delete document | Xóa file, vector còn | Xóa hoàn toàn |
| Graph traversal | BFS trên JSON | Cypher query |

---

## Giới hạn hiện tại

- Entity resolution chỉ dùng normalize cơ bản (chưa có fuzzy/embedding-based dedup)
- Không có community detection (Leiden/Louvain)
- Upload lớn sẽ chậm vì mỗi chunk gọi LLM để extract entities
- Neo4j vector index yêu cầu Neo4j 5.11+
