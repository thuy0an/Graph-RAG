"""
Neo4j graph store — Hierarchical Lexical Graph model.

Node labels:
  (:Document  {id, filename, summary, created_at})
  (:Section   {id, doc_id, index, title, summary, text})
  (:Chunk     {id, doc_id, section_id, index, text, embedding})
  (:Entity    {id, name, type})

Relationships:
  (Document)-[:HAS_SECTION]->(Section)
  (Section)-[:HAS_CHUNK]->(Chunk)
  (Section)-[:NEXT_SECTION]->(Section)
  (Chunk)-[:NEXT_CHUNK]->(Chunk)
  (Chunk)-[:MENTIONS]->(Entity)
  (Entity)-[:RELATED_TO {relation}]->(Entity)
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from neo4j import GraphDatabase

from app.core.config import settings

# embedding dimensions per provider
_EMBED_DIMS = {
    "nomic-embed-text": 768,
    "text-embedding-3-small": 1536,
    "all-MiniLM-L6-v2": 384,
}


@lru_cache(maxsize=1)
def get_driver():
    return GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )


def run(query: str, params: dict | None = None) -> list[dict[str, Any]]:
    with get_driver().session() as session:
        result = session.run(query, params or {})
        return [dict(r) for r in result]


def setup_indexes():
    """Create constraints + vector index. Called once on startup."""
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
            pass

    # Determine embedding dimension from config
    dims = _EMBED_DIMS.get(settings.OLLAMA_EMBED_MODEL, 768)
    if settings.EMBED_PROVIDER == "openai":
        dims = _EMBED_DIMS.get(settings.OPENAI_EMBED_MODEL, 1536)
    elif settings.EMBED_PROVIDER == "huggingface":
        dims = _EMBED_DIMS.get(settings.HF_EMBED_MODEL, 384)

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
        pass


# ── write helpers ─────────────────────────────────────────────────────────────

def upsert_document(doc_id: str, filename: str, summary: str):
    run(
        """MERGE (d:Document {id: $id})
           SET d.filename = $filename, d.summary = $summary,
               d.created_at = timestamp()""",
        {"id": doc_id, "filename": filename, "summary": summary},
    )


def upsert_section(section_id: str, doc_id: str, index: int,
                   title: str, summary: str, text: str):
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
    run(
        """MATCH (a:Section {id: $prev}), (b:Section {id: $next})
           MERGE (a)-[:NEXT_SECTION]->(b)""",
        {"prev": prev_id, "next": next_id},
    )


def upsert_chunk(chunk_id: str, doc_id: str, section_id: str,
                 index: int, text: str, embedding: list[float]):
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
    run(
        """MATCH (a:Chunk {id: $prev}), (b:Chunk {id: $next})
           MERGE (a)-[:NEXT_CHUNK]->(b)""",
        {"prev": prev_id, "next": next_id},
    )


def upsert_entity(entity_id: str, name: str, etype: str = "CONCEPT"):
    run(
        """MERGE (e:Entity {id: $id})
           SET e.name = $name, e.type = $etype""",
        {"id": entity_id, "name": name, "etype": etype},
    )


def link_chunk_entity(chunk_id: str, entity_id: str):
    run(
        """MATCH (c:Chunk {id: $cid}), (e:Entity {id: $eid})
           MERGE (c)-[:MENTIONS]->(e)""",
        {"cid": chunk_id, "eid": entity_id},
    )


def link_entities(src_id: str, tgt_id: str, relation: str):
    run(
        """MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt})
           MERGE (a)-[r:RELATED_TO {relation: $rel}]->(b)""",
        {"src": src_id, "tgt": tgt_id, "rel": relation},
    )


# ── read helpers ──────────────────────────────────────────────────────────────

def vector_search_chunks(embedding: list[float], k: int = 8,
                         doc_ids: list[str] | None = None) -> list[dict]:
    """ANN search on Chunk.embedding using Neo4j vector index.
    doc_ids: nếu có thì chỉ search trong các document được chỉ định.
    """
    if doc_ids:
        # Lấy nhiều hơn rồi filter, vì vector index không hỗ trợ WHERE trực tiếp
        rows = run(
            """CALL db.index.vector.queryNodes('chunk_embedding', $k, $embedding)
               YIELD node AS c, score
               WHERE c.doc_id IN $doc_ids
               RETURN c.id AS chunk_id, c.text AS text,
                      c.doc_id AS doc_id, c.section_id AS section_id,
                      c.index AS chunk_index, score""",
            {"k": k * 5, "embedding": embedding, "doc_ids": doc_ids},
        )
        return rows[:k]
    return run(
        """CALL db.index.vector.queryNodes('chunk_embedding', $k, $embedding)
           YIELD node AS c, score
           RETURN c.id AS chunk_id, c.text AS text,
                  c.doc_id AS doc_id, c.section_id AS section_id,
                  c.index AS chunk_index, score""",
        {"k": k, "embedding": embedding},
    )


def get_entity_subgraph_no_apoc(entity_names: list[str], depth: int = 2) -> list[str]:
    """BFS traversal without APOC — pure Cypher up to depth 2."""
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
    facts: set[str] = set()
    for r in results:
        if r.get("fact1"):
            facts.add(r["fact1"])
        if r.get("fact2"):
            facts.add(r["fact2"])
    return list(facts)


def get_section_summary_context(section_ids: list[str]) -> list[str]:
    if not section_ids:
        return []
    results = run(
        """UNWIND $ids AS sid
           MATCH (s:Section {id: sid})
           RETURN s.title AS title, s.summary AS summary
           ORDER BY s.index""",
        {"ids": section_ids},
    )
    return [f"[{r['title']}] {r['summary']}" for r in results if r.get("summary")]


def get_document_summary(doc_id: str) -> str:
    results = run(
        "MATCH (d:Document {id: $id}) RETURN d.summary AS summary",
        {"id": doc_id},
    )
    return results[0]["summary"] if results else ""


def list_documents() -> list[dict]:
    return run(
        """MATCH (d:Document)
           RETURN d.id AS id, d.filename AS filename,
                  d.summary AS summary, d.created_at AS created_at
           ORDER BY d.created_at DESC"""
    )


def delete_document(doc_id: str):
    """Delete document + all child nodes (sections, chunks). Orphan entities kept."""
    run(
        """MATCH (d:Document {id: $id})
           OPTIONAL MATCH (d)-[:HAS_SECTION]->(s)
           OPTIONAL MATCH (s)-[:HAS_CHUNK]->(c)
           DETACH DELETE c, s, d""",
        {"id": doc_id},
    )
