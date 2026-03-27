# Graph RAG — Hierarchical Lexical Graph

Hệ thống hỏi đáp tài liệu thông minh — bạn upload file PDF/DOCX, đặt câu hỏi bằng ngôn ngữ tự nhiên, hệ thống tìm kiếm và trả lời dựa trên nội dung tài liệu đó.

Điểm khác biệt so với chatbot thông thường: hệ thống không chỉ tìm đoạn văn giống câu hỏi, mà còn **hiểu quan hệ giữa các khái niệm** trong tài liệu thông qua Knowledge Graph.

![Pipeline](figures/01_pipeline.png)

**(a) Indexing — khi upload file:**
Tài liệu được đọc và cắt thành các đoạn nhỏ (chunks, ~800 ký tự mỗi đoạn). Mỗi đoạn được chuyển thành vector số (embedding) để máy tính có thể so sánh độ tương đồng. Sau đó, AI đọc từng nhóm đoạn và tự động rút trích ra các thực thể (người, tổ chức, khái niệm, kỹ năng...) cùng quan hệ giữa chúng. Toàn bộ được lưu vào Neo4j dưới dạng đồ thị phân cấp: Document -> Section -> Chunk -> Entity.

**(b) Querying — khi đặt câu hỏi:**
Câu hỏi được chuyển thành vector rồi tìm các đoạn văn tương tự nhất (vector search). Đồng thời, AI nhận diện các thực thể trong câu hỏi và duyệt đồ thị để tìm các quan hệ liên quan (graph traversal). Kết quả từ cả hai nguồn được gộp lại thành context 4 tầng rồi đưa cho LLM sinh câu trả lời cuối cùng.

---

## Tại sao cần Graph RAG?

Hãy tưởng tượng bạn có một tài liệu kỹ thuật dày 200 trang. RAG thông thường chỉ tìm các đoạn có chứa từ "Java" hoặc "Spring Boot" — nhưng không biết rằng Spring Boot **phụ thuộc vào** Java, hay Java **là nền tảng của** Spring Boot.

Graph RAG giải quyết:

| Loại câu hỏi | RAG thường | Graph RAG |
|---|---|---|
| "X là gì?" | Tốt | Tốt |
| "X và Y liên quan thế nào?" | Kém | Tốt (graph traversal) |
| "Tóm tắt toàn bộ tài liệu" | Kém | Tốt (document summary) |
| "Những khái niệm nào liên quan đến X?" | Kém | Tốt (multi-hop) |

![Comparison](figures/05_comparison.png)

**(Trái) Naive RAG** — quy trình đơn giản: chia chunks -> embed -> tìm chunks giống câu hỏi -> đưa cho LLM. Không có graph, không có hierarchy, chỉ có flat retrieval.

**(Giữa) Graph RAG (dự án này)** — bổ sung thêm: entity extraction (AI tự rút trích thực thể), section summarization (tóm tắt từng nhóm đoạn), Neo4j Hierarchical Graph (lưu cả vector lẫn graph trong 1 database), hybrid retrieval (kết hợp vector search + Cypher graph traversal).

**(Phải) Feature table** — so sánh trực tiếp 13 tiêu chí. Các ô xanh là tính năng Graph RAG có mà Naive RAG không có: entity relationships, knowledge graph, graph traversal, section summaries, document summary, hierarchical context, multi-hop reasoning, relationship-aware retrieval. Context layers: 1 (Naive) vs 4 (Graph RAG).

---

## Kiến trúc đồ thị

Tài liệu được tổ chức thành 4 tầng trong Neo4j:

```
(:Document {summary})              <- tóm tắt toàn bộ tài liệu
      | HAS_SECTION
      v
(:Section {title, summary})        <- nhóm 6 chunks, có tiêu đề + tóm tắt
      --NEXT_SECTION--> (:Section)  <- liên kết tuần tự giữa các section
      | HAS_CHUNK
      v
(:Chunk {text, embedding})         <- đoạn văn gốc + vector 768 chiều
      --NEXT_CHUNK----> (:Chunk)    <- liên kết tuần tự giữa các chunk
      | MENTIONS
      v
(:Entity {name, type})             <- thực thể: PERSON/ORG/CONCEPT/SKILL/PLACE
      --RELATED_TO {relation}--> (:Entity)  <- quan hệ có nhãn giữa các thực thể
```

Khi trả lời câu hỏi, LLM nhận context từ cả 4 tầng cùng lúc — nên trả lời tốt cả câu hỏi tổng quan lẫn câu hỏi chi tiết.

---

## Benchmark thực tế

Đo trên file PDF 1,446 KB — Java Interview Guide (250+ câu hỏi phỏng vấn):

| Bước | Thời gian | Ghi chú |
|------|-----------|---------|
| Chunking | 1.1s | Cắt 936 chunks, avg 628 ký tự/chunk |
| Embedding batch | 17.8s | 936 chunks x 768 chiều, 1 lần gọi duy nhất |
| LLM extract (156 sections) | ~6,220s | ~40s/section với Ollama local |
| Doc summary | 6.6s | 1 LLM call tổng hợp toàn bộ |
| Neo4j write | ~11.2s | Ghi nodes + edges + vector index |
| **Tổng** | **~104 phút** | Ollama local. Groq: ~5-7 phút |

Kết quả graph sau khi index:
- Chunks: 936 · Sections: 156 · LLM calls: 157
- Entities trung bình: 10.0/section · Relations: 9.2/section
- Tokens xử lý ước tính: ~117,750


![Timing](figures/02_timing.png)

**(a) Bar chart thời gian từng bước** — thấy rõ LLM extraction là bottleneck chính. Embedding và chunking gần như tức thì so với LLM.

**(b) Pie chart phân bổ thời gian** — trực quan hóa tỉ lệ: LLM chiếm gần như toàn bộ bánh.

**(c) LLM time per section** — thời gian từng LLM call trong sample 5 sections. Mỗi call xử lý ~3,000 ký tự text và trả về JSON chứa entities + relations + title + summary. Trung bình ~40s/section với Ollama, ~2-3s với Groq.

**(d) Chunk length distribution** — histogram độ dài các chunks. Mean/Median/P10/P90 cho thấy chunks khá đồng đều, ít outlier — chunking strategy hoạt động tốt.

**(e) Entities & relations per section** — số thực thể và quan hệ LLM rút trích được từ mỗi section trong sample. Avg 10 entities và 9.2 relations/section, ước tính ~1,560 entities và ~1,435 relations cho toàn bộ tài liệu.

**(f) Summary table** — bảng tổng hợp toàn bộ số liệu pipeline: file size, chunks, sections, LLM calls, tokens, timing từng bước.

---

## Knowledge Graph

![Graph](figures/03_graph.png)

**(a) Full entity graph** — toàn bộ entities và relations được extract từ tài liệu. Node size tỉ lệ với degree (số kết nối) — node to hơn = thực thể quan trọng hơn, xuất hiện nhiều hơn. Màu sắc phân biệt loại entity: xanh dương = PERSON, xanh lá = ORG, tím = CONCEPT, vàng = SKILL, đỏ = PLACE, xám = OTHER. Chỉ hiển thị label cho top-20 nodes có degree cao nhất để tránh rối.

**(b) Ego graph** — zoom vào entity trung tâm nhất (degree cao nhất), hiển thị tất cả nodes trong bán kính 2 bước. Các cạnh có nhãn quan hệ cụ thể (ví dụ: "is_used_in", "extends", "implements"). Đây là cách hệ thống "nhìn thấy" quan hệ khi trả lời câu hỏi về entity đó.

**(c) Entity type distribution** — số lượng từng loại entity. Tài liệu kỹ thuật thường có nhiều CONCEPT và SKILL hơn PERSON hay PLACE.

**(d) Degree distribution** — phân bổ số kết nối của các nodes. Dạng power-law (đuôi dài bên phải) là bình thường với knowledge graph: phần lớn entities ít kết nối, một số ít hub nodes kết nối rất nhiều.

**(e) Stats table** — tổng hợp: số documents/sections/chunks/entities/relations, avg degree, graph density, top 5 relation types, top 5 entities theo degree.

---

## Embedding Space

Embedding là cách biểu diễn text thành vector số (768 chiều) để máy tính có thể đo độ tương đồng. Hai đoạn văn có nghĩa gần nhau → vector gần nhau → cosine similarity cao (gần 1.0). Hai đoạn hoàn toàn khác chủ đề → cosine similarity thấp (gần 0).

![Similarity](figures/04_similarity.png)

**(a) Cosine similarity matrix** — heatmap màu sắc thể hiện độ tương đồng giữa tất cả cặp chunks (0 = hoàn toàn khác, 1 = giống hệt). Đường chéo = 1.0 (chunk so với chính nó). Off-diagonal mean ~0.56 cho thấy các chunks có độ đa dạng tốt. Nếu mean quá cao (>0.9) nghĩa là tài liệu lặp lại nhiều; quá thấp (<0.3) nghĩa là tài liệu rất đa dạng chủ đề.

**(b) Query-chunk similarity** — điểm cosine của 3 câu hỏi mẫu với từng chunk. Đường đứt đỏ (0.7) là ngưỡng "relevant", đường đứt xám (0.5) là ngưỡng "có thể liên quan". Chunks vượt ngưỡng 0.7 sẽ được ưu tiên đưa vào context khi trả lời.

**(c) Top-K retrieved chunks** — với 1 câu hỏi cụ thể, đây là top-5 chunks được chọn kèm score chính xác. Màu xanh ≥0.7 (highly relevant), vàng ≥0.5 (relevant), đỏ <0.5 (low confidence). Giúp debug xem hệ thống đang retrieve đúng không.

**(d) PCA projection** — chiếu 768 chiều xuống 2 chiều bằng PCA (giữ lại hướng có variance lớn nhất). Màu gradient theo chunk index (thứ tự trong tài liệu). Các chunks gần nhau trong không gian 2D có embedding tương tự nhau. PC1 và PC2 cho biết % thông tin được giữ lại sau khi giảm chiều.

**(e) t-SNE projection** — phương pháp giảm chiều khác, tốt hơn PCA trong việc giữ cấu trúc cục bộ. Các cụm (clusters) trong biểu đồ là các nhóm chunks có nội dung tương tự nhau về mặt ngữ nghĩa — dù chúng có thể ở các trang khác nhau trong tài liệu.

**(f) Pairwise similarity distribution** — histogram toàn bộ cặp chunks. Đường đứt P90 cho biết 90% cặp chunks có similarity thấp hơn giá trị đó. Dùng để chọn ngưỡng retrieval phù hợp: nếu đặt threshold quá cao sẽ miss nhiều chunks liên quan, quá thấp sẽ đưa vào quá nhiều noise.

---

## Tech Stack

| Layer | Technology | Vai trò |
|-------|-----------|---------|
| Backend | FastAPI + Uvicorn | REST API, xử lý upload và query |
| Graph DB | Neo4j 5.11+ | Lưu graph + vector index trong 1 database |
| LLM | Ollama / Groq / OpenAI / Anthropic | Extract entities, sinh summary, trả lời |
| Embedding | nomic-embed-text (dim=768) | Chuyển text thành vector |
| Orchestration | LangChain | Abstraction layer cho LLM và embeddings |
| Frontend | Vanilla HTML/JS | Chat UI đơn giản |

Toàn bộ stack có thể chạy **hoàn toàn local và miễn phí** với Ollama.

---

## Cài đặt

**1. Neo4j (Docker):**
```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.20
```
Neo4j Browser: http://localhost:7474

**2. Python:**
```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Linux/Mac
pip install -r requirements.txt
```

**3. `.env`:**
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

Muốn nhanh hơn với file lớn, đổi sang Groq (miễn phí, cần API key tại console.groq.com):
```env
LLM_PROVIDER=groq
GROQ_MODEL=llama-3.1-8b-instant
GROQ_API_KEY=your_key_here
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
| `POST` | `/api/upload` | Upload + index tài liệu vào Neo4j |
| `POST` | `/api/query` | Đặt câu hỏi, nhận câu trả lời + sources |
| `GET` | `/api/files` | Danh sách tài liệu đã index |
| `DELETE` | `/api/files/{filename}` | Xóa tài liệu khỏi Neo4j + disk |
| `GET` | `/health` | Trạng thái server + Neo4j connection |

---

## Cấu trúc dự án

```
app/
  main.py                  # FastAPI app, khởi tạo Neo4j indexes
  api/routes.py            # API endpoints
  core/
    config.py              # Đọc settings từ .env
    document_loader.py     # PDF/DOCX -> danh sách chunks
    graph_builder.py       # Build Hierarchical Lexical Graph vào Neo4j
    neo4j_store.py         # Neo4j CRUD, vector search, graph traversal
    rag_chain.py           # Query pipeline (embed -> search -> graph -> LLM)
    providers.py           # Factory cho LLM và Embeddings
research/
  01_pipeline_diagram.py   # Vẽ sơ đồ pipeline
  02_timing_benchmark.py   # Đo thời gian thực từng bước
  03_graph_visualization.py # Visualize knowledge graph từ Neo4j
  04_embedding_similarity.py # Phân tích embedding space
  05_architecture_comparison.py # So sánh Naive RAG vs Graph RAG
  run_all.py               # Chạy tất cả scripts
figures/                   # Hình ảnh được sinh ra bởi research/
requirements.txt
ui.html                    # Chat UI
```

---

## Giới hạn

- Upload file lớn với Ollama local rất chậm (~40s/section) — dùng Groq/OpenAI để nhanh hơn
- Entity resolution dùng normalize cơ bản (lowercase + collapse whitespace), chưa có fuzzy matching
- Chưa có community detection (Leiden/Louvain) như Microsoft GraphRAG gốc
- Neo4j vector index yêu cầu Neo4j 5.11+

---

## Tham khảo

- [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130) — Microsoft Research, 2024
- [awslabs/graphrag-toolkit](https://github.com/awslabs/graphrag-toolkit) — Hierarchical Lexical Graph (Apache-2.0)
- [microsoft/graphrag](https://github.com/microsoft/graphrag) — Official Microsoft GraphRAG (MIT)

---

MIT License
