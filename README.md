# Graph RAG — Hierarchical Lexical Graph

Hệ thống hỏi đáp tài liệu thông minh kết hợp **vector search**, **knowledge graph traversal** và **hierarchical summarization** trên nền Neo4j.

![Pipeline](figures/01_pipeline.png)

---

## Tại sao Graph RAG?

RAG thông thường chỉ tìm chunks tương tự theo vector — không hiểu quan hệ giữa các thực thể, không trả lời được câu hỏi multi-hop, và mất ngữ cảnh tổng quan khi chia nhỏ tài liệu.

Graph RAG giải quyết bằng cách xây dựng Knowledge Graph từ tài liệu:
- Hiểu quan hệ giữa entities (người, tổ chức, khái niệm, kỹ năng)
- Multi-hop reasoning qua graph traversal
- Hierarchical context: document → section → chunk

![Comparison](figures/05_comparison.png)

---

## Kiến trúc

```
(:Document {summary})
      │ HAS_SECTION
      ▼
(:Section {title, summary})  ──NEXT_SECTION──►  (:Section)
      │ HAS_CHUNK
      ▼
(:Chunk {text, embedding})   ──NEXT_CHUNK────►  (:Chunk)
      │ MENTIONS
      ▼
(:Entity {name, type})  ──RELATED_TO {relation}──► (:Entity)
```

---

## Benchmark thực tế

Đo trên file PDF 1,446 KB (Java Interview Guide):

| Bước | Thời gian |
|------|-----------|
| Chunking | 1.1s |
| Embedding batch (936 chunks, dim=768) | 17.8s |
| LLM extract entities+relations (156 sections) | ~6,220s* |
| Doc summary (1 LLM call) | 6.6s |
| Neo4j write (est.) | 11.2s |

*Dùng Ollama local. Dùng Groq nhanh hơn ~15-20x.

- Chunks: 936 · Sections: 156 · LLM calls: 157
- Avg entities/section: 10.0 · Avg relations/section: 9.2
- Tokens ước tính: ~117,750

![Timing](figures/02_timing.png)

---

## Knowledge Graph

![Graph](figures/03_graph.png)

---

## Embedding Space

![Similarity](figures/04_similarity.png)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| Graph DB | Neo4j 5.11+ |
| LLM | Ollama / Groq / OpenAI / Anthropic |
| Embedding | nomic-embed-text (dim=768) |
| Orchestration | LangChain |
| Frontend | Vanilla HTML/JS |

---

## Cài đặt

**1. Neo4j (Docker):**
```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.20
```

**2. Python:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**3. `.env`:**
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2

EMBED_PROVIDER=ollama
OLLAMA_EMBED_MODEL=nomic-embed-text

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

Đổi LLM provider chỉ cần sửa `.env`, không cần thay code. Dùng Groq để nhanh hơn nhiều với file lớn:
```env
LLM_PROVIDER=groq
GROQ_MODEL=llama-3.1-8b-instant
```

**4. Chạy:**
```bash
.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000
```

Mở `ui.html` trong browser. API docs: http://127.0.0.1:8000/docs

---

## API

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/api/upload` | Upload + index tài liệu |
| `POST` | `/api/query` | Hỏi đáp |
| `GET` | `/api/files` | Danh sách tài liệu |
| `DELETE` | `/api/files/{filename}` | Xóa tài liệu |
| `GET` | `/health` | Trạng thái server |

---

## Cấu trúc dự án

```
├── app/
│   ├── main.py                  # FastAPI app
│   ├── api/routes.py            # Endpoints
│   └── core/
│       ├── config.py            # Settings (.env)
│       ├── document_loader.py   # PDF/DOCX → chunks
│       ├── graph_builder.py     # Build Hierarchical Lexical Graph
│       ├── neo4j_store.py       # Neo4j CRUD + vector search
│       ├── rag_chain.py         # Query pipeline
│       └── providers.py         # LLM + Embeddings factory
├── research/                    # Benchmark & visualization scripts
├── figures/                     # Generated charts
├── requirements.txt
└── ui.html
```

---

## Research Scripts

```bash
python research/run_all.py
```

Sinh toàn bộ figures trong `figures/`:
- `01_pipeline.png` — sơ đồ pipeline
- `02_timing.png` — benchmark thời gian xử lý
- `03_graph.png` — visualize knowledge graph từ Neo4j
- `04_similarity.png` — embedding space analysis
- `05_comparison.png` — so sánh Naive RAG vs Graph RAG

---

## Giới hạn

- Upload file lớn với Ollama local rất chậm (~40s/section) — dùng Groq/OpenAI để nhanh hơn
- Entity resolution dùng normalize cơ bản, chưa có fuzzy matching
- Chưa có community detection (Leiden/Louvain) như Microsoft GraphRAG gốc
- Neo4j vector index yêu cầu Neo4j 5.11+

---

## Tham khảo

- [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130) — Microsoft Research, 2024
- [awslabs/graphrag-toolkit](https://github.com/awslabs/graphrag-toolkit)
- [microsoft/graphrag](https://github.com/microsoft/graphrag)

---

MIT License
