# Graph RAG v2 — Hierarchical Lexical Graph

> Hệ thống hỏi đáp tài liệu thông minh dựa trên **Hierarchical Lexical Graph** lưu trong **Neo4j**, kết hợp vector search, graph traversal và hierarchical summarization.

![Pipeline](figures/01_pipeline.png)

---

## Tại sao Graph RAG?

RAG thông thường chỉ tìm chunks tương tự theo vector — không hiểu quan hệ giữa các thực thể, không trả lời được câu hỏi multi-hop, và mất ngữ cảnh tổng quan khi chia nhỏ tài liệu.

Graph RAG giải quyết điều đó bằng cách xây dựng Knowledge Graph từ tài liệu, cho phép:
- Hiểu quan hệ giữa entities (người, tổ chức, khái niệm)
- Multi-hop reasoning qua graph traversal
- Hierarchical context từ document → section → chunk

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

Khi query, LLM nhận context từ cả 4 tầng → trả lời chính xác hơn với câu hỏi tổng quan lẫn câu hỏi chi tiết.

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

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI (Python) |
| Graph DB | Neo4j 5.20 |
| LLM | Ollama / Groq / OpenAI / Anthropic |
| Embedding | nomic-embed-text (768 chiều) |
| Frontend | Vanilla HTML/JS |
| Orchestration | LangChain |

Toàn bộ stack có thể chạy **hoàn toàn local và miễn phí** với Ollama.

---

## Cấu trúc dự án

```
ProjectGraphRAG/
├── app/
│   ├── main.py                  # FastAPI app
│   ├── api/routes.py            # /upload, /query, /files endpoints
│   ├── core/
│   │   ├── config.py            # Pydantic settings (.env)
│   │   ├── document_loader.py   # PDF/DOCX → chunks
│   │   ├── graph_builder.py     # Hierarchical Lexical Graph builder
│   │   ├── neo4j_store.py       # Neo4j CRUD + vector search + graph traversal
│   │   ├── rag_chain.py         # Query pipeline
│   │   └── providers.py         # LLM + Embeddings factory
│   └── models/schemas.py
├── research/                    # Benchmark & visualization scripts
├── figures/                     # Generated charts & diagrams
├── .env
├── requirements.txt
└── ui.html                      # Chat UI
```

---

## Cài đặt

### 1. Neo4j

**Docker (khuyến nghị):**
```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.20
```

**Neo4j Desktop:** Tải tại https://neo4j.com/download/ → tạo database mới → set password.

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
OLLAMA_EMBED_MODEL=nomic-embed-text

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

Hỗ trợ đổi provider chỉ bằng cách sửa `.env` — không cần thay đổi code.

### 4. Chạy server

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Mở `ui.html` trong browser. API docs: http://127.0.0.1:8000/docs

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

## Pipeline

### Upload

```
PDF/DOCX → Chunks (800 tokens, overlap 100)
         → Embed mỗi chunk (nomic-embed-text)
         → LLM extract entities + relations per chunk
         → Nhóm 6 chunks → LLM generate section title + summary
         → LLM generate document summary
         → Persist toàn bộ vào Neo4j
```

Chi phí LLM calls cho 1 file ~30 chunks: **~36 calls** (30 entity extraction + 5 section summary + 1 doc summary).

### Query

```
Question → Embed → Neo4j vector search (top-8 chunks)
         → LLM extract entities từ question
         → Cypher BFS traversal → entity relation facts
         → Lấy section summaries + document summary
         → Build 4-layer context → LLM generate answer
```

---

## Benchmark & Visualization

```bash
cd research
python 01_pipeline_diagram.py      # Sơ đồ pipeline
python 02_timing_benchmark.py      # Benchmark thời gian
python 03_graph_visualization.py   # Visualize knowledge graph
python 05_architecture_comparison.py  # So sánh kiến trúc
```

![Timing](figures/02_timing.png)
![Graph](figures/03_graph.png)
![Comparison](figures/05_comparison.png)

---

## Giới hạn hiện tại

- Entity resolution chỉ dùng normalize cơ bản (chưa có fuzzy/embedding-based dedup)
- Không có community detection (Leiden/Louvain) như Microsoft GraphRAG gốc
- Upload lớn sẽ chậm khi dùng Ollama local — dùng Groq/OpenAI để nhanh hơn
- Neo4j vector index yêu cầu Neo4j 5.11+

---

## Tham khảo

- [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130) — Microsoft Research, 2024
- [awslabs/graphrag-toolkit](https://github.com/awslabs/graphrag-toolkit) — Hierarchical Lexical Graph inspiration
- [microsoft/graphrag](https://github.com/microsoft/graphrag) — Official Microsoft GraphRAG

---

## License

MIT
