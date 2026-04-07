"""
Hierarchical Lexical Graph builder — phiên bản tối ưu.

Các tối ưu so với phiên bản naive:
  - Batch embedding: embed TẤT CẢ chunk trong 1 lần gọi thay vì từng chunk
  - Batch entity extraction: 1 lần gọi LLM cho mỗi SECTION (không phải mỗi chunk)
  - Section summary được tái sử dụng từ cùng lần gọi LLM với entity extraction
"""
from __future__ import annotations

import hashlib
import json
import re
import time

from langchain_core.documents import Document

from app.core.providers import get_llm

# ── Prompt templates ──────────────────────────────────────────────────────────

# Prompt yêu cầu LLM phân tích một section và trả về JSON gồm:
# tiêu đề, tóm tắt, danh sách entity và quan hệ
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

# Prompt tổng hợp tóm tắt toàn bộ tài liệu từ các section summary
_DOC_SUMMARY_PROMPT = """Summarize the following document in 3-5 sentences.
Focus on the main topics, key entities, and overall purpose.

Document sections:
{sections}

Summary:"""


# ── Các hàm tiện ích (helpers) ────────────────────────────────────────────────

def _uid(*parts: str) -> str:
    """Tạo ID duy nhất bằng MD5 hash từ các chuỗi đầu vào."""
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def _normalize(name: str) -> str:
    """Chuẩn hóa tên entity: trim, lowercase, gộp khoảng trắng thừa."""
    return re.sub(r"\s+", " ", name.strip().lower())


def _call_json(prompt: str) -> dict:
    """Gọi LLM và parse kết quả JSON từ response. Trả về dict rỗng nếu thất bại."""
    try:
        response = get_llm().invoke(prompt)
        # Tìm JSON object đầu tiên trong response (LLM đôi khi thêm text thừa)
        match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {}


def _extract_section(text: str) -> dict:
    """Gọi LLM một lần để trích xuất tiêu đề, tóm tắt, entity và quan hệ của một section.

    Chỉ dùng 3000 ký tự đầu để tránh vượt context window của LLM.
    Trả về dict chuẩn hóa với các key: title, summary, entities, relations.
    """
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
    """Tạo tóm tắt tổng thể cho tài liệu từ danh sách tóm tắt các section.

    Chỉ dùng tối đa 10 section đầu để tránh prompt quá dài.
    Fallback về 500 ký tự đầu nếu LLM gặp lỗi.
    """
    joined = "\n\n".join(section_summaries[:10])
    try:
        response = get_llm().invoke(_DOC_SUMMARY_PROMPT.format(sections=joined))
        return response.content.strip()
    except Exception:
        return joined[:500]


# ── Nhóm chunk thành section ──────────────────────────────────────────────────

# Số chunk tối đa trong mỗi section
_CHUNKS_PER_SECTION = 6


def _group_into_sections(chunks: list[Document]) -> list[list[Document]]:
    """Chia danh sách chunk thành các nhóm (section), mỗi nhóm _CHUNKS_PER_SECTION chunk."""
    sections = []
    for i in range(0, len(chunks), _CHUNKS_PER_SECTION):
        sections.append(chunks[i: i + _CHUNKS_PER_SECTION])
    return sections


# ── Hàm chính ─────────────────────────────────────────────────────────────────

def build_lexical_graph(chunks: list[Document], filename: str) -> dict:
    """Xây dựng Hierarchical Lexical Graph trong Neo4j từ danh sách chunk.

    3 bước chính:
      1. Batch embed toàn bộ chunk trong 1 lần gọi (tối ưu tốc độ)
      2. Với mỗi section: gọi LLM 1 lần để lấy title + summary + entities + relations
      3. Gọi LLM 1 lần để tạo tóm tắt tổng thể cho document

    Trả về dict thống kê: chunks, sections, entities, relations, thời gian từng bước.
    """
    from app.core.neo4j_store import (
        upsert_document, upsert_section, link_sections,
        upsert_chunk, link_chunks, link_chunk_entity,
        upsert_entity, link_entities,
    )
    from app.core.providers import get_embeddings

    t_total_start = time.perf_counter()

    embeddings_model = get_embeddings()
    doc_id = _uid(filename)  # ID document = MD5 của tên file
    section_groups = _group_into_sections(chunks)

    print(f"\n{'='*60}")
    print(f"  GRAPH BUILDER")
    print(f"{'='*60}")
    print(f"  File     : {filename}")
    print(f"  Chunks   : {len(chunks)}")
    print(f"  Sections : {len(section_groups)}  ({_CHUNKS_PER_SECTION} chunks/section)")
    print(f"  LLM calls: ~{len(section_groups) + 1} total  ({len(section_groups)} sections + 1 doc summary)")

    # ── BƯỚC 1: Batch Embedding ───────────────────────────────────────────────
    # Embed tất cả chunk cùng lúc thay vì từng cái → nhanh hơn nhiều
    print(f"\n  [1/3] Batch Embedding")
    t0 = time.perf_counter()
    all_texts = [c.page_content for c in chunks]
    all_embeddings = embeddings_model.embed_documents(all_texts)
    t_embed = time.perf_counter() - t0

    # Map chunk_index → embedding vector để tra cứu nhanh
    chunk_embeddings: dict[int, list[float]] = {
        c.metadata.get("chunk_index", i): all_embeddings[i]
        for i, c in enumerate(chunks)
    }
    embed_dim = len(all_embeddings[0]) if all_embeddings else 0
    print(f"        Embedded : {len(all_embeddings)} chunks")
    print(f"        Dimension: {embed_dim}")
    print(f"        Time     : {t_embed:.3f}s  ({t_embed/len(chunks)*1000:.1f}ms/chunk avg)")

    # ── BƯỚC 2: Trích xuất Section (LLM) ─────────────────────────────────────
    # Mỗi section gọi LLM 1 lần để lấy title, summary, entities, relations
    print(f"\n  [2/3] Section Extraction (LLM)")
    section_summaries: list[str] = []
    all_section_ids: list[str] = []
    total_entities = 0
    total_relations = 0
    t_llm_sections = 0.0

    for sec_idx, sec_chunks in enumerate(section_groups):
        # Ghép nội dung các chunk trong section thành một đoạn văn bản
        sec_text = "\n\n".join(c.page_content for c in sec_chunks)

        t0 = time.perf_counter()
        sec_data = _extract_section(sec_text)
        t_sec = time.perf_counter() - t0
        t_llm_sections += t_sec

        n_ent = len(sec_data["entities"])
        n_rel = len(sec_data["relations"])
        total_entities += n_ent
        total_relations += n_rel

        print(f"        Section {sec_idx+1:2d}/{len(section_groups)}"
              f"  title={sec_data['title'][:35]!r}"
              f"  entities={n_ent}  relations={n_rel}"
              f"  ({t_sec:.2f}s)")

        title   = sec_data["title"]
        summary = sec_data["summary"]
        section_summaries.append(f"{title}: {summary}")

        # Lưu Section node vào Neo4j
        section_id = _uid(doc_id, str(sec_idx))
        upsert_section(
            section_id=section_id,
            doc_id=doc_id,
            index=sec_idx,
            title=title,
            summary=summary,
            text=sec_text[:1000],  # Chỉ lưu 1000 ký tự đầu để tiết kiệm bộ nhớ
        )
        all_section_ids.append(section_id)

        # Liên kết section này với section trước (NEXT_SECTION)
        if sec_idx > 0:
            link_sections(all_section_ids[sec_idx - 1], section_id)

        # Lưu từng Chunk trong section và liên kết tuần tự (NEXT_CHUNK)
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

        # Lưu Entity và liên kết với tất cả chunk trong section (MENTIONS)
        for ent in sec_data["entities"]:
            eid = _uid(ent["name"])
            upsert_entity(eid, ent["name"], ent["type"])
            for chunk in sec_chunks:
                chunk_id = _uid(doc_id, str(chunk.metadata.get("chunk_index", 0)))
                link_chunk_entity(chunk_id, eid)

        # Lưu quan hệ giữa các entity (RELATED_TO)
        for subj, label, obj in sec_data["relations"]:
            src_id = _uid(subj)
            tgt_id = _uid(obj)
            upsert_entity(src_id, subj)
            upsert_entity(tgt_id, obj)
            link_entities(src_id, tgt_id, label)

    print(f"\n        Subtotal entities : {total_entities}")
    print(f"        Subtotal relations: {total_relations}")
    print(f"        LLM time (sections): {t_llm_sections:.2f}s  ({t_llm_sections/len(section_groups):.2f}s/section avg)")

    # ── BƯỚC 3: Tóm tắt Document ─────────────────────────────────────────────
    # Gọi LLM 1 lần để tổng hợp tóm tắt toàn bộ tài liệu
    print(f"\n  [3/3] Document Summary (LLM)")
    t0 = time.perf_counter()
    doc_summary = _doc_summary(section_summaries)
    t_doc = time.perf_counter() - t0

    upsert_document(doc_id=doc_id, filename=filename, summary=doc_summary)
    print(f"        Summary  : {doc_summary[:120]}...")
    print(f"        Time     : {t_doc:.2f}s")

    # ── Thống kê tổng kết ─────────────────────────────────────────────────────
    t_total = time.perf_counter() - t_total_start
    print(f"\n{'─'*60}")
    print(f"  GRAPH BUILD COMPLETE")
    print(f"{'─'*60}")
    print(f"  Chunks    : {len(chunks)}")
    print(f"  Sections  : {len(section_groups)}")
    print(f"  Entities  : {total_entities}")
    print(f"  Relations : {total_relations}")
    print(f"  Embed time: {t_embed:.2f}s")
    print(f"  LLM time  : {t_llm_sections + t_doc:.2f}s  ({len(section_groups)+1} calls)")
    print(f"  Total time: {t_total:.2f}s")
    print(f"{'='*60}\n")

    return {
        "chunks":    len(chunks),
        "sections":  len(section_groups),
        "entities":  total_entities,
        "relations": total_relations,
        "time_embed_s": round(t_embed, 2),
        "time_llm_s":   round(t_llm_sections + t_doc, 2),
        "time_total_s": round(t_total, 2),
    }
