"""
Graph RAG query pipeline — Neo4j Hierarchical Lexical Graph.

Query flow:
  1. Extract entities from question (LLM)
  2. Vector search → seed Chunks (Neo4j vector index)
  3. Entity subgraph traversal → relation facts
  4. Section summaries → hierarchical context
  5. Document summary → global context
  6. LLM answer generation
"""
from __future__ import annotations

import json
import re

from app.core.providers import get_llm, get_embeddings
from app.core.neo4j_store import (
    vector_search_chunks,
    get_entity_subgraph_no_apoc,
    get_section_summary_context,
    get_document_summary,
)

# ── prompts ───────────────────────────────────────────────────────────────────

_ENTITY_EXTRACT_PROMPT = """List the key entities (people, organizations, concepts, skills, places) in this question.
Return ONLY a JSON array, e.g. ["entity1", "entity2"]. If none, return [].

Question: {question}
Entities:"""

_ANSWER_PROMPT = """You are a helpful assistant. Answer the question using the context below.
If the answer is not in the context, say "I don't have enough information."

=== Document Overview ===
{doc_summary}

=== Section Summaries (hierarchical context) ===
{section_context}

=== Knowledge Graph (entity relationships) ===
{graph_context}

=== Retrieved Passages ===
{doc_context}

Question: {question}

Answer:"""


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_query_entities(question: str) -> list[str]:
    try:
        response = get_llm().invoke(_ENTITY_EXTRACT_PROMPT.format(question=question))
        match = re.search(r"\[.*?\]", response.content, re.DOTALL)
        if match:
            entities = json.loads(match.group())
            return [e for e in entities if isinstance(e, str) and e.strip()]
    except Exception:
        pass
    return []


# ── main query ────────────────────────────────────────────────────────────────

def query(question: str) -> tuple[str, list[str]]:
    embed_model = get_embeddings()

    # 1. Embed question
    q_embedding = embed_model.embed_query(question)

    # 2. Vector search in Neo4j
    hits = vector_search_chunks(q_embedding, k=8)
    if not hits:
        raise ValueError("No documents indexed yet. Please upload a document first.")

    print(f"  [query] Vector search returned {len(hits)} hits")
    for h in hits:
        print(f"    score={h.get('score', '?'):.3f}  doc={h.get('doc_id','?')[:8]}  sec={str(h.get('section_id','?'))[:8]}  text={h.get('text','')[:60]!r}")

    # 3. Dedupe and collect context
    seen_chunks: set[str] = set()
    doc_passages: list[str] = []
    section_ids: set[str] = set()
    doc_ids: set[str] = set()
    sources: set[str] = set()

    for hit in hits:
        cid = hit.get("chunk_id") or hit.get("id", "")
        if cid in seen_chunks:
            continue
        seen_chunks.add(cid)
        text = hit.get("text") or hit.get("content", "")
        if text:
            doc_passages.append(text)
        sid = hit.get("section_id")
        did = hit.get("doc_id")
        if sid:
            section_ids.add(sid)
        if did:
            doc_ids.add(did)

    print(f"  [query] section_ids={len(section_ids)}  doc_ids={len(doc_ids)}  passages={len(doc_passages)}")

    # 4. Extract query entities → graph traversal
    query_entities = _extract_query_entities(question)
    graph_facts = get_entity_subgraph_no_apoc(query_entities, depth=2)

    # 5. Section summaries (hierarchical context)
    section_summaries = get_section_summary_context(list(section_ids))

    # 6. Document-level summary (global context)
    doc_summary_parts: list[str] = []
    for did in doc_ids:
        s = get_document_summary(did)
        if s:
            doc_summary_parts.append(s)
            sources.add(did)

    # Resolve source filenames
    from app.core.neo4j_store import list_documents
    id_to_filename = {d["id"]: d["filename"] for d in list_documents()}
    source_names = [id_to_filename.get(did, did) for did in doc_ids]

    # 7. Build prompt
    prompt = _ANSWER_PROMPT.format(
        doc_summary="\n\n".join(doc_summary_parts) or "N/A",
        section_context="\n".join(section_summaries) or "N/A",
        graph_context="\n".join(graph_facts) or "No graph context found.",
        doc_context="\n\n---\n\n".join(doc_passages),
        question=question,
    )

    response = get_llm().invoke(prompt)
    return response.content.strip(), source_names
