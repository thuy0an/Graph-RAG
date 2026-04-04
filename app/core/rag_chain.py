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
import time

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

def query(question: str, filenames: list[str] | None = None) -> tuple[str, list[str]]:
    t_total_start = time.perf_counter()

    print(f"\n{'='*60}")
    print(f"  QUERY PIPELINE")
    print(f"{'='*60}")
    print(f"  Question : {question[:100]}")
    print(f"  Filter   : {filenames if filenames else 'all files'}")

    embed_model = get_embeddings()

    # 1. Embed question
    t0 = time.perf_counter()
    q_embedding = embed_model.embed_query(question)
    t_embed = time.perf_counter() - t0
    print(f"\n  [1/5] Embed question       ({t_embed*1000:.1f}ms)")

    # Resolve filenames -> doc_ids filter
    from app.core.neo4j_store import list_documents
    all_docs = list_documents()
    id_to_filename = {d["id"]: d["filename"] for d in all_docs}
    filename_to_id = {d["filename"]: d["id"] for d in all_docs}

    filter_doc_ids: list[str] | None = None
    if filenames:
        filter_doc_ids = [filename_to_id[f] for f in filenames if f in filename_to_id]
        print(f"  Filter doc_ids : {filter_doc_ids}")

    # 2. Vector search
    t0 = time.perf_counter()
    hits = vector_search_chunks(q_embedding, k=8, doc_ids=filter_doc_ids)
    t_search = time.perf_counter() - t0

    if not hits:
        raise ValueError("No documents indexed yet. Please upload a document first.")

    print(f"  [2/5] Vector search        ({t_search*1000:.1f}ms)  → {len(hits)} hits")
    for i, h in enumerate(hits):
        score = h.get('score', 0)
        text_preview = h.get('text', '')[:55]
        print(f"        #{i+1}  score={score:.4f}  {text_preview!r}")

    # 3. Dedupe and collect context
    seen_chunks: set[str] = set()
    doc_passages: list[str] = []
    section_ids: set[str] = set()
    doc_ids: set[str] = set()

    for hit in hits:
        cid = hit.get("chunk_id") or hit.get("id", "")
        if cid in seen_chunks:
            continue
        seen_chunks.add(cid)
        text = hit.get("text") or hit.get("content", "")
        if text:
            doc_passages.append(text)
        if hit.get("section_id"):
            section_ids.add(hit["section_id"])
        if hit.get("doc_id"):
            doc_ids.add(hit["doc_id"])

    print(f"\n  [3/5] Context collection")
    print(f"        Unique chunks  : {len(doc_passages)}")
    print(f"        Sections found : {len(section_ids)}")
    print(f"        Documents found: {len(doc_ids)}")

    # 4. Entity extraction + graph traversal
    t0 = time.perf_counter()
    query_entities = _extract_query_entities(question)
    t_ent = time.perf_counter() - t0

    t0 = time.perf_counter()
    graph_facts = get_entity_subgraph_no_apoc(query_entities, depth=2)
    t_graph = time.perf_counter() - t0

    print(f"\n  [4/5] Graph traversal      (entity extract={t_ent*1000:.1f}ms, traversal={t_graph*1000:.1f}ms)")
    print(f"        Query entities : {query_entities}")
    print(f"        Graph facts    : {len(graph_facts)}")
    for fact in graph_facts[:5]:
        print(f"        → {fact[:80]}")
    if len(graph_facts) > 5:
        print(f"        ... (+{len(graph_facts)-5} more)")

    # 5. Hierarchical context
    t0 = time.perf_counter()
    section_summaries = get_section_summary_context(list(section_ids))
    doc_summary_parts: list[str] = []
    for did in doc_ids:
        s = get_document_summary(did)
        if s:
            doc_summary_parts.append(s)
    t_ctx = time.perf_counter() - t0

    print(f"\n  [5/5] Hierarchical context ({t_ctx*1000:.1f}ms)")
    print(f"        Section summaries: {len(section_summaries)}")
    print(f"        Doc summaries    : {len(doc_summary_parts)}")

    # Resolve source filenames (dùng lại all_docs đã fetch)
    source_names = [id_to_filename.get(did, did) for did in doc_ids]

    # Build prompt & generate answer
    prompt = _ANSWER_PROMPT.format(
        doc_summary="\n\n".join(doc_summary_parts) or "N/A",
        section_context="\n".join(section_summaries) or "N/A",
        graph_context="\n".join(graph_facts) or "No graph context found.",
        doc_context="\n\n---\n\n".join(doc_passages),
        question=question,
    )

    prompt_tokens_est = len(prompt) // 4
    print(f"\n  Prompt size: ~{prompt_tokens_est} tokens (est.)")

    t0 = time.perf_counter()
    response = get_llm().invoke(prompt)
    t_llm = time.perf_counter() - t0

    t_total = time.perf_counter() - t_total_start
    answer = response.content.strip()

    print(f"\n{'─'*60}")
    print(f"  QUERY COMPLETE")
    print(f"{'─'*60}")
    print(f"  Embed      : {t_embed*1000:.1f}ms")
    print(f"  Search     : {t_search*1000:.1f}ms")
    print(f"  Graph      : {(t_ent+t_graph)*1000:.1f}ms")
    print(f"  LLM answer : {t_llm:.2f}s")
    print(f"  Total      : {t_total:.2f}s")
    print(f"  Sources    : {source_names}")
    print(f"{'='*60}\n")

    return answer, source_names
