# Graph RAG — Từ Lý Thuyết Đến Thực Hành
### Buổi thuyết trình & demo hệ thống hỏi đáp tài liệu thông minh

---

## Mục lục

1. [Vấn đề với LLM thông thường](#1-vấn-đề-với-llm-thông-thường)
2. [RAG là gì?](#2-rag-là-gì)
3. [Giới hạn của RAG thông thường](#3-giới-hạn-của-rag-thông-thường)
4. [Graph RAG ra đời](#4-graph-rag-ra-đời)
5. [Knowledge Graph là gì?](#5-knowledge-graph-là-gì)
6. [Kiến trúc Hierarchical Lexical Graph](#6-kiến-trúc-hierarchical-lexical-graph)
7. [Pipeline xử lý tài liệu](#7-pipeline-xử-lý-tài-liệu)
8. [Pipeline truy vấn](#8-pipeline-truy-vấn)
9. [Tech stack của dự án](#9-tech-stack-của-dự-án)
10. [Demo trực tiếp](#10-demo-trực-tiếp)
11. [Kết quả & đánh giá](#11-kết-quả--đánh-giá)
12. [Hướng phát triển](#12-hướng-phát-triển)
13. [Tài liệu tham khảo](#13-tài-liệu-tham-khảo)

---

## 1. Vấn đề với LLM thông thường

### Nói với khán giả:
> "Trước khi nói về Graph RAG, hãy hiểu tại sao chúng ta cần nó."

**LLM (Large Language Model) như ChatGPT, Llama, Claude có 3 vấn đề lớn:**

### Hallucination (Ảo giác)
LLM có thể tự bịa ra thông tin nghe có vẻ đúng nhưng hoàn toàn sai.

```
Hỏi: "Doanh thu Q3/2024 của công ty chúng ta là bao nhiêu?"
LLM: "Doanh thu Q3/2024 là 2.3 tỷ đồng, tăng 15% so với cùng kỳ."
→ Thông tin này LLM TỰ BỊA vì nó không có dữ liệu nội bộ.
```

### Knowledge Cutoff (Giới hạn thời gian)
LLM chỉ biết thông tin đến thời điểm training. Không biết gì về dữ liệu mới.

### Không có dữ liệu riêng tư
LLM không biết về tài liệu nội bộ, hợp đồng, báo cáo của tổ chức bạn.

---

## 2. RAG là gì?

### Nói với khán giả:
> "RAG là giải pháp cho 3 vấn đề trên."

**RAG = Retrieval-Augmented Generation**

Ý tưởng đơn giản: **Trước khi LLM trả lời, hãy tìm kiếm tài liệu liên quan và đưa vào context.**

```
Không có RAG:
  Câu hỏi → LLM → Trả lời (từ memory, có thể sai)

Có RAG:
  Câu hỏi → Tìm tài liệu liên quan → LLM + Tài liệu → Trả lời (có căn cứ)
```

### Quy trình RAG cơ bản:

```
[Indexing]
Tài liệu → Chia nhỏ (chunks) → Embedding → Lưu vào Vector Store

[Querying]
Câu hỏi → Embedding → Tìm chunks tương tự → LLM đọc chunks → Trả lời
```

### Embedding là gì?
Chuyển text thành vector số (ví dụ 768 chiều). Các đoạn text có nghĩa gần nhau → vector gần nhau trong không gian.

```
"con mèo ngồi trên ghế"  → [0.2, 0.8, 0.1, ...]
"chú mèo đang ngồi"     → [0.21, 0.79, 0.12, ...]  ← gần nhau
"thị trường chứng khoán" → [0.9, 0.1, 0.7, ...]    ← xa nhau
```

---

## 3. Giới hạn của RAG thông thường

### Nói với khán giả:
> "RAG tốt, nhưng chưa đủ. Hãy xem nó thiếu gì."

### Vấn đề 1: Flat retrieval — không hiểu quan hệ

```
Tài liệu: "Nguyễn Văn A là CEO của Công ty X. 
           Công ty X hợp tác với Công ty Y.
           Công ty Y do Trần Thị B điều hành."

Câu hỏi: "Nguyễn Văn A có liên quan gì đến Trần Thị B?"

RAG thường: Tìm chunks có "Nguyễn Văn A" → không thấy liên kết với Trần Thị B
→ Trả lời: "Không có thông tin"  ← SAI
```

### Vấn đề 2: Mất ngữ cảnh tổng quan

Khi chia tài liệu thành chunks nhỏ, thông tin tổng quan bị mất. Câu hỏi "Tài liệu này nói về gì?" sẽ cho kết quả kém.

### Vấn đề 3: Multi-hop reasoning

```
Fact 1: "A làm việc tại B"
Fact 2: "B là công ty con của C"  
Fact 3: "C có trụ sở tại Hà Nội"

Câu hỏi: "A làm việc ở đâu?"
→ Cần kết hợp 3 facts → RAG thường không làm được tốt
```

---

## 4. Graph RAG ra đời

### Nói với khán giả:
> "Microsoft Research công bố Graph RAG năm 2024 để giải quyết những vấn đề này."

**Paper gốc:** "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"  
Microsoft Research, 2024 — [arxiv.org/abs/2404.16130](https://arxiv.org/abs/2404.16130)

### Ý tưởng cốt lõi:
Thay vì chỉ lưu chunks, hãy **xây dựng Knowledge Graph** từ tài liệu — lưu entities và quan hệ giữa chúng — rồi dùng graph traversal khi truy vấn.

### So sánh:

| | RAG thông thường | Graph RAG |
|---|---|---|
| Lưu trữ | Flat chunks | Chunks + Knowledge Graph |
| Retrieval | Vector similarity | Vector + Graph traversal |
| Hiểu quan hệ | Yếu | Tốt |
| Multi-hop | Không | Có |
| Câu hỏi tổng quan | Kém | Tốt (community summaries) |
| Độ phức tạp | Thấp | Cao hơn |
| Chi phí indexing | Thấp | Cao hơn (nhiều LLM calls) |

---

## 5. Knowledge Graph là gì?

### Nói với khán giả:
> "Trước khi đi vào kiến trúc, cần hiểu Knowledge Graph."

**Knowledge Graph = Mạng lưới các thực thể và quan hệ giữa chúng.**

### Cấu trúc cơ bản:
```
Node (Entity): Người, tổ chức, khái niệm, địa điểm, kỹ năng...
Edge (Relation): Quan hệ có hướng giữa 2 entities

Ví dụ:
  Nguyễn Văn A  --[làm_việc_tại]-->  Công ty X
  Công ty X     --[hợp_tác_với]-->   Công ty Y
  Công ty Y     --[có_trụ_sở_tại]--> Hà Nội
```

### Ví dụ thực tế — từ CV/Resume:
```
  python        --[is_skill_of]-->    Nguyễn Văn A
  Nguyễn Văn A  --[studied_at]-->     Đại học Bách Khoa
  Nguyễn Văn A  --[worked_at]-->      Google
  Google        --[located_in]-->     Mountain View
  machine_learning --[requires]--> python
```

### Tại sao dùng Neo4j?
- Database graph native, tối ưu cho graph traversal
- Cypher query language trực quan
- Neo4j 5.11+ có native vector index → vừa lưu graph vừa làm vector search
- Community Edition miễn phí hoàn toàn

---

## 6. Kiến trúc Hierarchical Lexical Graph

### Nói với khán giả:
> "Đây là điểm khác biệt lớn nhất của dự án này so với Graph RAG thông thường."

**Hierarchical Lexical Graph** = Graph có phân cấp, lấy cảm hứng từ awslabs/graphrag-toolkit.

### Cấu trúc 4 tầng:

```
Tầng 1: Document
  (:Document {id, filename, summary})
         │
         │ HAS_SECTION
         ▼
Tầng 2: Section  ──NEXT_SECTION──► Section
  (:Section {title, summary, text})
         │
         │ HAS_CHUNK
         ▼
Tầng 3: Chunk  ──NEXT_CHUNK──► Chunk
  (:Chunk {text, embedding})
         │
         │ MENTIONS
         ▼
Tầng 4: Entity  ──RELATED_TO {relation}──► Entity
  (:Entity {name, type})
```

### Tại sao cần phân cấp?

```
Câu hỏi tổng quan: "Tài liệu này nói về gì?"
→ Dùng Document summary + Section summaries (tầng 1, 2)

Câu hỏi chi tiết: "Kỹ năng Python được đề cập ở đâu?"
→ Dùng Vector search trên Chunks (tầng 3)

Câu hỏi quan hệ: "Python liên quan đến những gì?"
→ Dùng Entity graph traversal (tầng 4)
```

---

## 7. Pipeline xử lý tài liệu

### Nói với khán giả:
> "Khi bạn upload 1 file PDF, hệ thống làm gì?"

```
Bước 1: LOAD
  PDF/DOCX → PyPDFLoader/Docx2txtLoader → raw text

Bước 2: CHUNK
  RecursiveCharacterTextSplitter
  chunk_size = 800 ký tự
  chunk_overlap = 100 ký tự
  → Danh sách chunks với metadata (source_file, chunk_index)

Bước 3: EMBED
  Mỗi chunk → Embedding model (nomic-embed-text, 768 chiều)
  → Vector lưu vào Neo4j Chunk node

Bước 4: ENTITY EXTRACTION (LLM call per chunk)
  Prompt → LLM → JSON {entities, relations}
  Normalize tên → lowercase, collapse whitespace
  → Entity nodes + RELATED_TO edges trong Neo4j

Bước 5: SECTION SUMMARY (LLM call per 6 chunks)
  Nhóm 6 chunks → LLM → {title, summary}
  → Section nodes với NEXT_SECTION links

Bước 6: DOCUMENT SUMMARY (1 LLM call)
  Tất cả section summaries → LLM → Document summary
  → Document node
```

### Chi phí LLM calls cho 1 file 30 chunks:
```
Entity extraction:  30 calls  (1 per chunk)
Section summary:     5 calls  (1 per 6 chunks)
Document summary:    1 call
─────────────────────────────
Total:              36 LLM calls
```

---

## 8. Pipeline truy vấn

### Nói với khán giả:
> "Khi bạn đặt câu hỏi, hệ thống làm gì?"

```
Bước 1: EMBED QUESTION
  Câu hỏi → Embedding model → Query vector

Bước 2: VECTOR SEARCH
  Query vector → Neo4j vector index
  → Top-8 chunks tương tự nhất (cosine similarity)

Bước 3: ENTITY EXTRACTION FROM QUESTION (LLM call)
  Câu hỏi → LLM → ["entity1", "entity2", ...]
  → Seed entities cho graph traversal

Bước 4: GRAPH TRAVERSAL (Cypher)
  MATCH (e:Entity)-[r1:RELATED_TO]->(e2)
  WHERE e.name CONTAINS seed_entity
  OPTIONAL MATCH (e2)-[r2:RELATED_TO]->(e3)
  → Relation facts (depth=2)

Bước 5: HIERARCHICAL CONTEXT
  Từ chunks tìm được → lấy Section summaries
  Từ sections → lấy Document summary

Bước 6: BUILD PROMPT (4-layer context)
  === Document Overview ===
  {doc_summary}

  === Section Summaries ===
  {section_summaries}

  === Knowledge Graph ===
  {entity_relation_facts}

  === Retrieved Passages ===
  {chunk_texts}

  Question: {question}

Bước 7: LLM GENERATE ANSWER
  → Answer + Sources
```

---

## 9. Tech stack của dự án

### Nói với khán giả:
> "Toàn bộ dự án dùng công nghệ mã nguồn mở, miễn phí."

| Layer | Technology | Vai trò |
|-------|-----------|---------|
| Backend | FastAPI (Python) | REST API server |
| Graph DB | Neo4j 5.20 (Docker) | Lưu graph + vector index |
| LLM | Ollama + Llama3.2/Qwen2.5 | Entity extraction + Answer generation |
| Embedding | nomic-embed-text (Ollama) | Text → vector 768 chiều |
| Frontend | Vanilla HTML/JS | Chat UI |
| Orchestration | LangChain | LLM abstraction layer |

### Tại sao chọn stack này?
- **Hoàn toàn miễn phí** — chạy local, không cần API key
- **Dễ thay thế** — đổi LLM provider chỉ cần sửa `.env`
- **Production-ready** — Neo4j, FastAPI đều dùng trong production thực tế

### Hỗ trợ nhiều LLM provider:
```
Ollama (local)  → miễn phí, chậm hơn
Groq            → miễn phí có limit, rất nhanh
OpenAI          → trả phí, chất lượng cao
Anthropic Claude → trả phí, chất lượng cao
```

---

## 10. Demo trực tiếp

### Checklist trước khi demo:
- [ ] Neo4j Docker container đang chạy (`docker ps`)
- [ ] Ollama đang chạy (`ollama list`)
- [ ] Server đang chạy (`uvicorn app.main:app`)
- [ ] Mở `ui.html` trong browser
- [ ] Có sẵn 1-2 file PDF để upload

### Kịch bản demo:

**Bước 1 — Upload tài liệu**
```
→ Kéo thả file PDF vào upload zone
→ Chỉ cho khán giả thấy progress bar
→ Giải thích: "Hệ thống đang gọi LLM để extract entities..."
→ Chờ hoàn tất → tóm tắt tự động hiện ra
```

**Bước 2 — Câu hỏi đơn giản**
```
Hỏi: "Tài liệu này nói về chủ đề gì?"
→ Chỉ ra: câu trả lời dùng Document summary (tầng 1)
```

**Bước 3 — Câu hỏi về quan hệ**
```
Hỏi: "[Entity trong tài liệu] liên quan đến gì?"
→ Chỉ ra: câu trả lời dùng Knowledge Graph facts
```

**Bước 4 — Mở Neo4j Browser**
```
→ Truy cập http://localhost:7474
→ Chạy query: MATCH (n) RETURN n LIMIT 50
→ Cho khán giả thấy graph trực quan
```

**Bước 5 — Chạy research scripts**
```
cd research
python 01_pipeline_diagram.py
python 03_graph_visualization.py
→ Mở figures/ để xem hình ảnh sinh ra
```

### Cypher queries hay để demo trên Neo4j Browser:

```cypher
// Xem toàn bộ graph
MATCH (n) RETURN n LIMIT 100

// Xem document hierarchy
MATCH (d:Document)-[:HAS_SECTION]->(s)-[:HAS_CHUNK]->(c)
RETURN d, s, c LIMIT 30

// Xem entity relationships
MATCH (e1:Entity)-[r:RELATED_TO]->(e2:Entity)
RETURN e1, r, e2 LIMIT 50

// Top entities theo số quan hệ
MATCH (e:Entity)-[r:RELATED_TO]->()
RETURN e.name, count(r) AS degree
ORDER BY degree DESC LIMIT 10

// Tìm path giữa 2 entities
MATCH path = shortestPath(
  (a:Entity {name: "entity_1"})-[*]-(b:Entity {name: "entity_2"})
)
RETURN path
```

---

## 11. Kết quả & đánh giá

### So sánh RAG vs Graph RAG (định tính):

| Loại câu hỏi | RAG thường | Graph RAG |
|---|---|---|
| "Tóm tắt tài liệu" | Trung bình | Tốt (doc summary) |
| "X là gì?" | Tốt | Tốt |
| "X liên quan đến Y như thế nào?" | Kém | Tốt (graph traversal) |
| "Những ai liên quan đến X?" | Kém | Tốt |
| "Kể các bước của quy trình Z" | Trung bình | Tốt (section context) |

### Điểm mạnh của kiến trúc này:
- Hierarchical context → trả lời tốt cả câu hỏi tổng quan lẫn chi tiết
- Graph traversal → hiểu quan hệ multi-hop
- Neo4j native vector index → không cần 2 database riêng biệt
- Xóa tài liệu sạch hoàn toàn (không như FAISS)

### Giới hạn hiện tại:
- Upload chậm vì nhiều LLM calls (giải pháp: dùng Groq/OpenAI)
- Entity resolution cơ bản (chưa có fuzzy matching)
- Chưa có community detection như Microsoft GraphRAG gốc

---

## 12. Hướng phát triển

### Ngắn hạn:
- **Batch embedding** — embed nhiều chunks cùng lúc thay vì từng cái
- **Async entity extraction** — xử lý song song các chunks
- **Entity deduplication** — dùng embedding similarity để merge entities giống nhau

### Trung hạn:
- **Community detection** — Leiden algorithm để nhóm entities thành communities
- **Global query mode** — trả lời câu hỏi tổng quan dùng community summaries
- **Multi-document reasoning** — kết nối entities across documents

### Dài hạn:
- **Incremental update** — cập nhật graph khi tài liệu thay đổi
- **Graph neural networks** — dùng GNN để improve retrieval
- **Evaluation framework** — đo lường chất lượng câu trả lời tự động (RAGAS)

---

## 13. Tài liệu tham khảo

### Papers

1. **Graph RAG gốc (Microsoft)**
   > Edge, D., Trinh, H., Cheng, N., et al. (2024).
   > *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.*
   > Microsoft Research. https://arxiv.org/abs/2404.16130

2. **RAG Survey**
   > Gao, Y., et al. (2023).
   > *Retrieval-Augmented Generation for Large Language Models: A Survey.*
   > https://arxiv.org/abs/2312.10997

3. **Knowledge Graphs + LLMs**
   > Pan, S., et al. (2024).
   > *Unifying Large Language Models and Knowledge Graphs: A Roadmap.*
   > IEEE TKDE. https://arxiv.org/abs/2306.08302

4. **HippoRAG — Graph-based RAG**
   > Guti­érrez, B.J., et al. (2024).
   > *HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models.*
   > https://arxiv.org/abs/2405.14831

### Repositories & Tools

5. **Microsoft GraphRAG**
   > https://github.com/microsoft/graphrag
   > Official implementation của paper trên.

6. **awslabs/graphrag-toolkit**
   > https://github.com/awslabs/graphrag-toolkit
   > AWS implementation với Hierarchical Lexical Graph (cảm hứng của dự án này).

7. **LangChain**
   > https://python.langchain.com/docs/
   > Framework orchestration LLM được dùng trong dự án.

8. **Neo4j Graph Data Science**
   > https://neo4j.com/docs/graph-data-science/current/
   > Tài liệu về graph algorithms, vector index.

9. **langchain-graph-retriever**
   > https://github.com/datastax/langchain-graph-retriever
   > Thư viện graph-aware retrieval cho LangChain.

### Blogs & Tutorials

10. **Introducing the GraphRAG Toolkit (AWS)**
    > https://aws.amazon.com/blogs/database/introducing-the-graphrag-toolkit/

11. **Neo4j + LLM: Building Knowledge Graphs**
    > https://neo4j.com/developer-blog/knowledge-graph-llm/

12. **RAGAS — RAG Evaluation Framework**
    > https://docs.ragas.io/
    > Framework đánh giá chất lượng RAG system.

---

## Câu hỏi thường gặp

**Q: Graph RAG có thay thế hoàn toàn RAG thông thường không?**
> Không nhất thiết. Với tài liệu đơn giản, RAG thường đủ dùng và nhanh hơn. Graph RAG phát huy khi tài liệu phức tạp, nhiều entities và quan hệ.

**Q: Chi phí so với RAG thường?**
> Indexing tốn nhiều LLM calls hơn (~30-40 calls/file thay vì 0). Query tốn thêm 1 LLM call để extract entities. Với Ollama local thì miễn phí nhưng chậm; với Groq/OpenAI thì nhanh nhưng tốn tiền.

**Q: Có thể dùng cho tiếng Việt không?**
> Có. Qwen2.5 và Claude đặc biệt tốt cho tiếng Việt. Llama3.2 cũng được nhưng kém hơn.

**Q: Khác gì so với Microsoft GraphRAG?**
> Microsoft GraphRAG có thêm community detection (Leiden algorithm) và global query mode. Dự án này implement Hierarchical Lexical Graph — phù hợp hơn cho Q&A trên tài liệu cụ thể, nhẹ hơn và không cần cloud.

**Q: Neo4j có thể thay bằng gì?**
> Có thể dùng Amazon Neptune (AWS), ArangoDB, hoặc giữ NetworkX + FAISS như v1. Neo4j được chọn vì có native vector index và Community Edition miễn phí.

---

*Dự án: Graph RAG v2 — Hierarchical Lexical Graph*
*Stack: FastAPI + Neo4j + LangChain + Ollama*
*License: MIT*
