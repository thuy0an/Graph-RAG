"""
Hierarchical Lexical Graph builder — optimized.

Optimizations vs naive version:
  - Batch embedding: embed ALL chunks in one call instead of one-by-one
  - Batch entity extraction: one LLM call per SECTION (not per chunk)
  - Section summary reuses the same LLM call as entity extraction
"""
from __future__ import annotations

import hashlib
import json
import re

from langchain_core.documents import Document

from app.core.providers import get_llm

# ── prompts ───────────────────────────────────────────────────────────────────

# Combined prompt: extract entities + summary in ONE LLM call per section
_SECTION_EXTRACT_PROMPT = """Analyze the following text and return a JSON object with:
1. "title": short section title (max 8 words)
2. "summary": concise summary (2-3 sentences)
3. "entities": list of entities with name and type
4. "relations": list of [subject, relation, object] triples

Entity types: PERSON, ORG, CONCEPT, SKILL, PLACE, OTHER
Normalize entity names to lowercase.

Return ONLY valid JSON:
{{
  "title": "...",
  "summary": "...",
  "entities": [{{"name": "...", "type": "..."}}],
  "relations": [["subject", "relation", "object"]]
}}

Text:
{text}

JSON:"""

_DOC_SUMMARY_PROMPT = """Summarize the following document in 3-5 sentences.
Focus on the main topics, key entities, and overall purpose.

Document sections:
{sections}

Summary:"""

# ── helpers ───────────────────────────────────────────────────────────────────

def _uid(*parts: str) -> str:
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def _normalize(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _call_json(prompt: str) -> dict:
    try:
        response = get_llm().invoke(prompt)
        match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {}


def _extract_section(text: str) -> dict:
    """Single LLM call: title + summary + entities + relations for a section."""
    data = _call_json(_SECTION_EXTRACT_PROMPT.format(text=text[:3000]))
    return {
        "title":    data.get("title", "Section"),
        "summary":  data.get("summary", ""),
        "entities": [
            {"name": _normalize(e["name"]), "type": e.get("type", "CONCEPT")}
            for e in data.get("entities", [])
            if isinstance(e, dict) and e.get("name")
        ],
        "relations": [
            (_normalize(str(r[0])), str(r[1]).strip(), _normalize(str(r[2])))
            for r in data.get("relations", [])
            if isinstance(r, list) and len(r) == 3 and r[0] and r[2]
        ],
    }


def _doc_summary(section_summaries: list[str]) -> str:
    joined = "\n\n".join(section_summaries[:10])
    try:
        response = get_llm().invoke(_DOC_SUMMARY_PROMPT.format(sections=joined))
        return response.content.strip()
    except Exception:
        return joined[:500]


# ── grouping ──────────────────────────────────────────────────────────────────

_CHUNKS_PER_SECTION = 6


def _group_into_sections(chunks: list[Document]) -> list[list[Document]]:
    sections = []
    for i in range(0, len(chunks), _CHUNKS_PER_SECTION):
        sections.append(chunks[i: i + _CHUNKS_PER_SECTION])
    return sections


# ── main entry point ──────────────────────────────────────────────────────────

def build_lexical_graph(chunks: list[Document], filename: str) -> dict:
    """
    Build Hierarchical Lexical Graph in Neo4j.

    Key optimizations:
      1. Batch embed ALL chunks in one call
      2. One LLM call per section (title + summary + entities + relations)
      3. One LLM call for document summary
    """
    from app.core.neo4j_store import (
        upsert_document, upsert_section, link_sections,
        upsert_chunk, link_chunks, link_chunk_entity,
        upsert_entity, link_entities,
    )
    from app.core.providers import get_embeddings

    embeddings_model = get_embeddings()
    doc_id = _uid(filename)
    section_groups = _group_into_sections(chunks)

    # ── OPTIMIZATION 1: Batch embed all chunks at once ────────────────────────
    print(f"  [embed] Embedding {len(chunks)} chunks in batch...")
    all_texts = [c.page_content for c in chunks]
    all_embeddings = embeddings_model.embed_documents(all_texts)
    # Map chunk_index → embedding
    chunk_embeddings: dict[int, list[float]] = {
        c.metadata.get("chunk_index", i): all_embeddings[i]
        for i, c in enumerate(chunks)
    }
    print(f"  [embed] Done.")

    section_summaries: list[str] = []
    all_section_ids: list[str] = []
    total_entities = 0

    for sec_idx, sec_chunks in enumerate(section_groups):
        sec_text = "\n\n".join(c.page_content for c in sec_chunks)

        # ── OPTIMIZATION 2: One LLM call per section ─────────────────────────
        print(f"  [llm] Section {sec_idx + 1}/{len(section_groups)}: extracting...")
        sec_data = _extract_section(sec_text)

        title   = sec_data["title"]
        summary = sec_data["summary"]
        section_summaries.append(f"{title}: {summary}")

        section_id = _uid(doc_id, str(sec_idx))
        upsert_section(
            section_id=section_id,
            doc_id=doc_id,
            index=sec_idx,
            title=title,
            summary=summary,
            text=sec_text[:1000],
        )
        all_section_ids.append(section_id)

        if sec_idx > 0:
            link_sections(all_section_ids[sec_idx - 1], section_id)

        # Persist chunks (embeddings already computed)
        prev_chunk_id: str | None = None
        for chunk in sec_chunks:
            chunk_idx = chunk.metadata.get("chunk_index", 0)
            chunk_id  = _uid(doc_id, str(chunk_idx))
            embedding = chunk_embeddings.get(chunk_idx, all_embeddings[0])

            upsert_chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                section_id=section_id,
                index=chunk_idx,
                text=chunk.page_content,
                embedding=embedding,
            )

            if prev_chunk_id:
                link_chunks(prev_chunk_id, chunk_id)
            prev_chunk_id = chunk_id

        # Persist entities + relations from this section
        total_entities += len(sec_data["entities"])
        for ent in sec_data["entities"]:
            eid = _uid(ent["name"])
            upsert_entity(eid, ent["name"], ent["type"])
            # Link all chunks in section to entities
            for chunk in sec_chunks:
                chunk_id = _uid(doc_id, str(chunk.metadata.get("chunk_index", 0)))
                link_chunk_entity(chunk_id, eid)

        for subj, label, obj in sec_data["relations"]:
            src_id = _uid(subj)
            tgt_id = _uid(obj)
            upsert_entity(src_id, subj)
            upsert_entity(tgt_id, obj)
            link_entities(src_id, tgt_id, label)

    # ── OPTIMIZATION 3: One doc summary call ─────────────────────────────────
    print(f"  [llm] Generating document summary...")
    doc_summary = _doc_summary(section_summaries)
    upsert_document(doc_id=doc_id, filename=filename, summary=doc_summary)

    return {
        "chunks":   len(chunks),
        "entities": total_entities,
        "sections": len(section_groups),
    }
