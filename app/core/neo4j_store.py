"""
Neo4j graph store — Mô hình đồ thị phân cấp từ vựng (Hierarchical Lexical Graph).

Cấu trúc node:
  (:Document  {id, filename, summary, created_at})  -- Tài liệu gốc
  (:Section   {id, doc_id, index, title, summary, text})  -- Phần/chương trong tài liệu
  (:Chunk     {id, doc_id, section_id, index, text, embedding})  -- Đoạn văn bản nhỏ
  (:Entity    {id, name, type})  -- Thực thể được trích xuất (khái niệm, tên, ...)

Quan hệ giữa các node:
  (Document)-[:HAS_SECTION]->(Section)      -- Tài liệu chứa các section
  (Section)-[:HAS_CHUNK]->(Chunk)           -- Section chứa các chunk
  (Section)-[:NEXT_SECTION]->(Section)      -- Liên kết tuần tự giữa các section
  (Chunk)-[:NEXT_CHUNK]->(Chunk)            -- Liên kết tuần tự giữa các chunk
  (Chunk)-[:MENTIONS]->(Entity)             -- Chunk đề cập đến thực thể nào
  (Entity)-[:RELATED_TO {relation}]->(Entity)  -- Quan hệ giữa các thực thể
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from neo4j import GraphDatabase

from app.core.config import settings

# Số chiều embedding tương ứng với từng model
_EMBED_DIMS = {
    "nomic-embed-text": 768,
    "text-embedding-3-small": 1536,
    "all-MiniLM-L6-v2": 384,
}


@lru_cache(maxsize=1)
def get_driver():
    """Tạo và cache kết nối Neo4j driver (chỉ khởi tạo 1 lần duy nhất)."""
    return GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )


def run(query: str, params: dict | None = None) -> list[dict[str, Any]]:
    """Thực thi một câu Cypher query và trả về danh sách kết quả dạng dict."""
    with get_driver().session() as session:
        result = session.run(query, params or {})
        return [dict(r) for r in result]


def setup_indexes():
    """Tạo các constraint và vector index cho Neo4j. Gọi một lần khi khởi động app."""

    # Tạo unique constraint cho từng loại node để tránh trùng lặp
    constraints = [
        "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    ]
    for stmt in constraints:
        try:
            run(stmt)
        except Exception:
            pass  # Bỏ qua nếu constraint đã tồn tại

    # Xác định số chiều embedding dựa theo provider được cấu hình
    dims = _EMBED_DIMS.get(settings.OLLAMA_EMBED_MODEL, 768)
    if settings.EMBED_PROVIDER == "openai":
        dims = _EMBED_DIMS.get(settings.OPENAI_EMBED_MODEL, 1536)
    elif settings.EMBED_PROVIDER == "huggingface":
        dims = _EMBED_DIMS.get(settings.HF_EMBED_MODEL, 384)

    # Tạo vector index trên trường embedding của Chunk để hỗ trợ tìm kiếm ANN
    try:
        run(
            f"""CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {{indexConfig: {{
                  `vector.dimensions`: {dims},
                  `vector.similarity_function`: 'cosine'
                }}}}"""
        )
    except Exception:
        pass  # Bỏ qua nếu index đã tồn tại


# ── Các hàm ghi dữ liệu (write helpers) ──────────────────────────────────────

def upsert_document(doc_id: str, filename: str, summary: str):
    """Tạo mới hoặc cập nhật node Document trong đồ thị."""
    run(
        """MERGE (d:Document {id: $id})
           SET d.filename = $filename, d.summary = $summary,
               d.created_at = timestamp()""",
        {"id": doc_id, "filename": filename, "summary": summary},
    )


def upsert_section(section_id: str, doc_id: str, index: int,
                   title: str, summary: str, text: str):
    """Tạo mới hoặc cập nhật node Section, đồng thời liên kết với Document cha."""
    run(
        """MERGE (s:Section {id: $id})
           SET s.doc_id = $doc_id, s.index = $index,
               s.title = $title, s.summary = $summary, s.text = $text
           WITH s
           MATCH (d:Document {id: $doc_id})
           MERGE (d)-[:HAS_SECTION]->(s)""",
        {"id": section_id, "doc_id": doc_id, "index": index,
         "title": title, "summary": summary, "text": text},
    )


def link_sections(prev_id: str, next_id: str):
    """Tạo quan hệ NEXT_SECTION giữa 2 section liền kề nhau."""
    run(
        """MATCH (a:Section {id: $prev}), (b:Section {id: $next})
           MERGE (a)-[:NEXT_SECTION]->(b)""",
        {"prev": prev_id, "next": next_id},
    )


def upsert_chunk(chunk_id: str, doc_id: str, section_id: str,
                 index: int, text: str, embedding: list[float]):
    """Tạo mới hoặc cập nhật node Chunk (kèm embedding), đồng thời liên kết với Section cha."""
    run(
        """MERGE (c:Chunk {id: $id})
           SET c.doc_id = $doc_id, c.section_id = $section_id,
               c.index = $index, c.text = $text, c.embedding = $embedding
           WITH c
           MATCH (s:Section {id: $section_id})
           MERGE (s)-[:HAS_CHUNK]->(c)""",
        {"id": chunk_id, "doc_id": doc_id, "section_id": section_id,
         "index": index, "text": text, "embedding": embedding},
    )


def link_chunks(prev_id: str, next_id: str):
    """Tạo quan hệ NEXT_CHUNK giữa 2 chunk liền kề nhau."""
    run(
        """MATCH (a:Chunk {id: $prev}), (b:Chunk {id: $next})
           MERGE (a)-[:NEXT_CHUNK]->(b)""",
        {"prev": prev_id, "next": next_id},
    )


def upsert_entity(entity_id: str, name: str, etype: str = "CONCEPT"):
    """Tạo mới hoặc cập nhật node Entity (thực thể như khái niệm, tên riêng, ...)."""
    run(
        """MERGE (e:Entity {id: $id})
           SET e.name = $name, e.type = $etype""",
        {"id": entity_id, "name": name, "etype": etype},
    )


def link_chunk_entity(chunk_id: str, entity_id: str):
    """Tạo quan hệ MENTIONS: chunk này đề cập đến entity nào."""
    run(
        """MATCH (c:Chunk {id: $cid}), (e:Entity {id: $eid})
           MERGE (c)-[:MENTIONS]->(e)""",
        {"cid": chunk_id, "eid": entity_id},
    )


def link_entities(src_id: str, tgt_id: str, relation: str):
    """Tạo quan hệ RELATED_TO giữa 2 entity với nhãn quan hệ cụ thể."""
    run(
        """MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt})
           MERGE (a)-[r:RELATED_TO {relation: $rel}]->(b)""",
        {"src": src_id, "tgt": tgt_id, "rel": relation},
    )


# ── Các hàm đọc dữ liệu (read helpers) ───────────────────────────────────────

def vector_search_chunks(embedding: list[float], k: int = 8,
                         doc_ids: list[str] | None = None) -> list[dict]:
    """Tìm kiếm ANN (Approximate Nearest Neighbor) trên vector embedding của Chunk.

    Sử dụng vector index của Neo4j để tìm k chunk gần nhất với embedding đầu vào.
    doc_ids: nếu truyền vào thì chỉ tìm trong các document được chỉ định.
    """
    if doc_ids:
        # Lấy nhiều hơn k rồi filter theo doc_ids,
        # vì vector index không hỗ trợ WHERE trực tiếp
        rows = run(
            """CALL db.index.vector.queryNodes('chunk_embedding', $k, $embedding)
               YIELD node AS c, score
               WHERE c.doc_id IN $doc_ids
               RETURN c.id AS chunk_id, c.text AS text,
                      c.doc_id AS doc_id, c.section_id AS section_id,
                      c.index AS chunk_index, score""",
            {"k": k * 5, "embedding": embedding, "doc_ids": doc_ids},
        )
        return rows[:k]  # Cắt lại đúng k kết quả sau khi filter

    # Tìm kiếm toàn bộ không giới hạn document
    return run(
        """CALL db.index.vector.queryNodes('chunk_embedding', $k, $embedding)
           YIELD node AS c, score
           RETURN c.id AS chunk_id, c.text AS text,
                  c.doc_id AS doc_id, c.section_id AS section_id,
                  c.index AS chunk_index, score""",
        {"k": k, "embedding": embedding},
    )


def get_entity_subgraph_no_apoc(entity_names: list[str], depth: int = 2) -> list[str]:
    """Duyệt đồ thị entity theo BFS thuần Cypher (không dùng APOC), tối đa 2 bước.

    Trả về danh sách các "fact" dạng chuỗi: 'A --[relation]--> B'.
    """
    if not entity_names:
        return []

    results = run(
        """UNWIND $names AS name
           MATCH (e:Entity)-[r1:RELATED_TO]->(e2:Entity)
           WHERE toLower(e.name) CONTAINS toLower(name)
              OR toLower(e2.name) CONTAINS toLower(name)
           WITH e, r1, e2
           OPTIONAL MATCH (e2)-[r2:RELATED_TO]->(e3:Entity)
           RETURN
             e.name + ' --[' + r1.relation + ']--> ' + e2.name AS fact1,
             CASE WHEN r2 IS NOT NULL
               THEN e2.name + ' --[' + r2.relation + ']--> ' + e3.name
               ELSE null END AS fact2""",
        {"names": entity_names},
    )

    # Dùng set để loại bỏ các fact trùng lặp
    facts: set[str] = set()
    for r in results:
        if r.get("fact1"):
            facts.add(r["fact1"])
        if r.get("fact2"):
            facts.add(r["fact2"])
    return list(facts)


def get_section_summary_context(section_ids: list[str]) -> list[str]:
    """Lấy tiêu đề và tóm tắt của các section theo danh sách id, sắp xếp theo thứ tự."""
    if not section_ids:
        return []
    results = run(
        """UNWIND $ids AS sid
           MATCH (s:Section {id: sid})
           RETURN s.title AS title, s.summary AS summary
           ORDER BY s.index""",
        {"ids": section_ids},
    )
    # Định dạng kết quả: "[Tiêu đề] Tóm tắt"
    return [f"[{r['title']}] {r['summary']}" for r in results if r.get("summary")]


def get_document_summary(doc_id: str) -> str:
    """Lấy tóm tắt của một document theo id. Trả về chuỗi rỗng nếu không tìm thấy."""
    results = run(
        "MATCH (d:Document {id: $id}) RETURN d.summary AS summary",
        {"id": doc_id},
    )
    return results[0]["summary"] if results else ""


def list_documents() -> list[dict]:
    """Lấy danh sách tất cả document, sắp xếp theo thời gian tạo mới nhất trước."""
    return run(
        """MATCH (d:Document)
           RETURN d.id AS id, d.filename AS filename,
                  d.summary AS summary, d.created_at AS created_at
           ORDER BY d.created_at DESC"""
    )


def delete_document(doc_id: str):
    """Xóa document và toàn bộ node con (section, chunk). Entity liên quan được giữ lại."""
    run(
        """MATCH (d:Document {id: $id})
           OPTIONAL MATCH (d)-[:HAS_SECTION]->(s)
           OPTIONAL MATCH (s)-[:HAS_CHUNK]->(c)
           DETACH DELETE c, s, d""",
        {"id": doc_id},
    )
