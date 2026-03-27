"""
Figure 2 — Processing Time Benchmark
Đo thời gian thực từng bước khi xử lý tài liệu thật.
Chạy sau khi đã upload ít nhất 1 file.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

os.makedirs("figures", exist_ok=True)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app.core.document_loader import load_document
from app.core.providers import get_embeddings, get_llm
from app.core.graph_builder import _extract_section, _group_into_sections, _DOC_SUMMARY_PROMPT

# ── find a test file ──────────────────────────────────────────────────────────
upload_dir = os.path.join(os.path.dirname(__file__), "..", "uploads")
files = [f for f in os.listdir(upload_dir) if f.endswith((".pdf", ".docx", ".doc"))]
if not files:
    print("❌ No files in uploads/. Please upload a document first.")
    sys.exit(1)

test_file = os.path.join(upload_dir, files[0])
file_size_kb = os.path.getsize(test_file) / 1024
print(f"📄 Benchmarking: {files[0]}  ({file_size_kb:.1f} KB)")

results = {}
llm_call_times = []
entity_counts = []
relation_counts = []

# ── Step 1: Chunking ──────────────────────────────────────────────────────────
t0 = time.perf_counter()
chunks = load_document(test_file)
results["Chunking"] = time.perf_counter() - t0
chunk_lengths = [len(c.page_content) for c in chunks]
print(f"  Chunking:   {results['Chunking']:.3f}s  ({len(chunks)} chunks, avg {np.mean(chunk_lengths):.0f} chars)")

# ── Step 2: Embedding (batch, all chunks) ─────────────────────────────────────
embed_model = get_embeddings()
t0 = time.perf_counter()
all_embeddings = embed_model.embed_documents([c.page_content for c in chunks])
results["Embedding\n(batch)"] = time.perf_counter() - t0
embed_dim = len(all_embeddings[0]) if all_embeddings else 0
ms_per_chunk = results["Embedding\n(batch)"] / len(chunks) * 1000
print(f"  Embedding:  {results['Embedding (batch)' if 'Embedding (batch)' in results else list(results.keys())[-1]]:.3f}s  "
      f"({len(chunks)} chunks, dim={embed_dim}, {ms_per_chunk:.1f}ms/chunk)")

# ── Step 3: LLM section extraction (sample up to 5 sections) ─────────────────
sections = _group_into_sections(chunks)
sample_secs = sections[:min(5, len(sections))]
print(f"  LLM extract: sampling {len(sample_secs)}/{len(sections)} sections...")

for i, sec in enumerate(sample_secs):
    sec_text = "\n\n".join(c.page_content for c in sec)
    t0 = time.perf_counter()
    sec_data = _extract_section(sec_text)
    elapsed = time.perf_counter() - t0
    llm_call_times.append(elapsed)
    entity_counts.append(len(sec_data.get("entities", [])))
    relation_counts.append(len(sec_data.get("relations", [])))
    print(f"    Section {i+1}: {elapsed:.2f}s  entities={entity_counts[-1]}  relations={relation_counts[-1]}")

avg_llm_time = np.mean(llm_call_times)
results["LLM Extract\n(per section)"] = avg_llm_time
results["LLM Extract\n(all sections, est.)"] = avg_llm_time * len(sections)
print(f"  LLM avg:    {avg_llm_time:.2f}s/section  → est. {avg_llm_time * len(sections):.1f}s total")

# ── Step 4: Doc summary (1 LLM call) ─────────────────────────────────────────
dummy_summaries = [f"Section {i}: summary text." for i in range(len(sections))]
joined = "\n\n".join(dummy_summaries[:10])
t0 = time.perf_counter()
get_llm().invoke(_DOC_SUMMARY_PROMPT.format(sections=joined))
results["Doc Summary\n(1 LLM call)"] = time.perf_counter() - t0
print(f"  Doc summary:{results['Doc Summary\n(1 LLM call)']:.2f}s")

# ── Step 5: Neo4j write estimate ──────────────────────────────────────────────
results["Neo4j Write\n(est.)"] = len(chunks) * 0.012
print(f"  Neo4j write:{results['Neo4j Write\n(est.)']:.2f}s  (estimated ~12ms/chunk)")

# ── Derived stats ─────────────────────────────────────────────────────────────
embed_key   = "Embedding\n(batch)"
llm_sec_key = "LLM Extract\n(all sections, est.)"
doc_key     = "Doc Summary\n(1 LLM call)"
neo_key     = "Neo4j Write\n(est.)"
chunk_key   = "Chunking"

plot_keys   = [chunk_key, embed_key, llm_sec_key, doc_key, neo_key]
plot_values = [results[k] for k in plot_keys]
total       = sum(plot_values)

total_llm_calls = len(sections) + 1  # sections + doc summary
tokens_per_section_est = 3000 // 4   # ~3000 chars input
total_tokens_est = tokens_per_section_est * total_llm_calls

print(f"\n  === SUMMARY ===")
print(f"  File size   : {file_size_kb:.1f} KB")
print(f"  Chunks      : {len(chunks)}")
print(f"  Sections    : {len(sections)}")
print(f"  LLM calls   : {total_llm_calls}")
print(f"  Tokens est. : ~{total_tokens_est:,}")
print(f"  Total time  : {total:.1f}s")
print(f"  Avg entities/section: {np.mean(entity_counts):.1f}")
print(f"  Avg relations/section: {np.mean(relation_counts):.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════
BG, FG, GRID = "#0f172a", "#e2e8f0", "#1e293b"
COLORS = ["#0ea5e9", "#8b5cf6", "#ec4899", "#10b981", "#f59e0b"]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.patch.set_facecolor(BG)
fig.suptitle(
    f"Processing Benchmark — {files[0]}\n"
    f"{len(chunks)} chunks · {len(sections)} sections · {total_llm_calls} LLM calls · ~{total_tokens_est:,} tokens est.",
    color=FG, fontsize=13, fontweight="bold"
)

# ── (a) Step timing bar chart ─────────────────────────────────────────────────
ax = axes[0, 0]
ax.set_facecolor(BG)
short_labels = ["Chunking", "Embedding\n(batch)", "LLM Extract\n(sections)", "Doc Summary\n(LLM)", "Neo4j Write\n(est.)"]
bars = ax.barh(short_labels, plot_values, color=COLORS, edgecolor="#334155", height=0.55)
ax.set_xlabel("Time (seconds)", color=FG, fontsize=10)
ax.set_title("(a) Processing Time per Step", color=FG, fontsize=11, fontweight="bold")
ax.tick_params(colors=FG)
ax.spines[:].set_color("#334155")
for bar, val in zip(bars, plot_values):
    ax.text(val + total * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}s", va="center", color=FG, fontsize=9)
ax.text(0.98, 0.02, f"Total: {total:.1f}s", transform=ax.transAxes,
        ha="right", va="bottom", color="#38bdf8", fontsize=11, fontweight="bold")

# ── (b) Time distribution pie ─────────────────────────────────────────────────
ax = axes[0, 1]
ax.set_facecolor(BG)
wedges, texts, autotexts = ax.pie(
    plot_values, labels=short_labels, colors=COLORS,
    autopct="%1.1f%%", startangle=140,
    textprops={"color": FG, "fontsize": 8},
    wedgeprops={"edgecolor": BG, "linewidth": 2},
)
for at in autotexts:
    at.set_color(BG)
    at.set_fontweight("bold")
ax.set_title("(b) Time Distribution", color=FG, fontsize=11, fontweight="bold")

# ── (c) LLM call time per section ────────────────────────────────────────────
ax = axes[0, 2]
ax.set_facecolor(BG)
x = np.arange(len(llm_call_times))
bars2 = ax.bar(x, llm_call_times, color="#ec4899", edgecolor="#334155", width=0.6)
ax.axhline(avg_llm_time, color="#38bdf8", linestyle="--", lw=1.5,
           label=f"Avg: {avg_llm_time:.2f}s")
ax.set_xlabel("Section index (sample)", color=FG, fontsize=10)
ax.set_ylabel("Time (seconds)", color=FG, fontsize=10)
ax.set_title(
    f"(c) LLM Time per Section\n"
    f"avg={avg_llm_time:.2f}s · est. total={avg_llm_time*len(sections):.1f}s for {len(sections)} sections",
    color=FG, fontsize=10, fontweight="bold"
)
ax.tick_params(colors=FG)
ax.spines[:].set_color("#334155")
ax.legend(facecolor=GRID, edgecolor="#334155", labelcolor=FG, fontsize=9)
for bar, val in zip(bars2, llm_call_times):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.2f}s",
            ha="center", va="bottom", color=FG, fontsize=8)

# ── (d) Chunk length distribution ────────────────────────────────────────────
ax = axes[1, 0]
ax.set_facecolor(BG)
ax.hist(chunk_lengths, bins=20, color="#8b5cf6", edgecolor="#334155", alpha=0.85)
ax.axvline(np.mean(chunk_lengths), color="#38bdf8", linestyle="--", lw=1.5,
           label=f"Mean: {np.mean(chunk_lengths):.0f}")
ax.axvline(np.median(chunk_lengths), color="#f59e0b", linestyle="--", lw=1.5,
           label=f"Median: {np.median(chunk_lengths):.0f}")
ax.axvline(np.percentile(chunk_lengths, 10), color="#ef4444", linestyle=":", lw=1.2,
           label=f"P10: {np.percentile(chunk_lengths, 10):.0f}")
ax.axvline(np.percentile(chunk_lengths, 90), color="#10b981", linestyle=":", lw=1.2,
           label=f"P90: {np.percentile(chunk_lengths, 90):.0f}")
ax.set_xlabel("Chunk length (characters)", color=FG, fontsize=10)
ax.set_ylabel("Count", color=FG, fontsize=10)
ax.set_title(
    f"(d) Chunk Length Distribution\n"
    f"min={min(chunk_lengths)}  max={max(chunk_lengths)}  std={np.std(chunk_lengths):.0f}",
    color=FG, fontsize=10, fontweight="bold"
)
ax.tick_params(colors=FG)
ax.spines[:].set_color("#334155")
ax.legend(facecolor=GRID, edgecolor="#334155", labelcolor=FG, fontsize=8)

# ── (e) Entities & relations per section ─────────────────────────────────────
ax = axes[1, 1]
ax.set_facecolor(BG)
x = np.arange(len(entity_counts))
w = 0.35
b1 = ax.bar(x - w/2, entity_counts, w, label="Entities", color="#0ea5e9", edgecolor="#334155")
b2 = ax.bar(x + w/2, relation_counts, w, label="Relations", color="#ec4899", edgecolor="#334155")
ax.set_xlabel("Section index (sample)", color=FG, fontsize=10)
ax.set_ylabel("Count", color=FG, fontsize=10)
ax.set_title(
    f"(e) Entities & Relations per Section\n"
    f"avg entities={np.mean(entity_counts):.1f}  avg relations={np.mean(relation_counts):.1f}  "
    f"→ est. total entities={np.mean(entity_counts)*len(sections):.0f}",
    color=FG, fontsize=9, fontweight="bold"
)
ax.tick_params(colors=FG)
ax.spines[:].set_color("#334155")
ax.legend(facecolor=GRID, edgecolor="#334155", labelcolor=FG, fontsize=9)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, str(int(h)),
                ha="center", va="bottom", color=FG, fontsize=8)

# ── (f) Summary stats table ───────────────────────────────────────────────────
ax = axes[1, 2]
ax.set_facecolor(BG)
ax.axis("off")
ax.set_title("(f) Pipeline Summary", color=FG, fontsize=11, fontweight="bold")

rows = [
    ("File",              files[0][:30]),
    ("File size",         f"{file_size_kb:.1f} KB"),
    ("Pages/Docs",        f"{len(chunks)} chunks"),
    ("Sections",          f"{len(sections)}  ({6} chunks/section)"),
    ("Embed dimension",   f"{embed_dim}"),
    ("LLM calls",         f"{total_llm_calls}  ({len(sections)} sec + 1 doc)"),
    ("Tokens est.",       f"~{total_tokens_est:,}"),
    ("Avg entities/sec",  f"{np.mean(entity_counts):.1f}"),
    ("Avg relations/sec", f"{np.mean(relation_counts):.1f}"),
    ("Est. total entities", f"~{np.mean(entity_counts)*len(sections):.0f}"),
    ("─────────────",     "──────────────"),
    ("Chunking time",     f"{results['Chunking']:.3f}s"),
    ("Embedding time",    f"{results['Embedding\n(batch)']:.2f}s  ({ms_per_chunk:.1f}ms/chunk)"),
    ("LLM time (est.)",   f"{avg_llm_time*len(sections)+results['Doc Summary\n(1 LLM call)']:.1f}s"),
    ("Neo4j write (est.)",f"{results['Neo4j Write\n(est.)']:.2f}s"),
    ("TOTAL (est.)",      f"{total:.1f}s"),
]

y_start = 0.97
row_h = 0.062
for i, (label, value) in enumerate(rows):
    y = y_start - i * row_h
    is_sep = label.startswith("─")
    is_total = label == "TOTAL (est.)"
    color_val = "#38bdf8" if is_total else ("#475569" if is_sep else FG)
    color_lbl = "#94a3b8" if not is_total else "#38bdf8"
    ax.text(0.02, y, label, transform=ax.transAxes, fontsize=8.5,
            color=color_lbl, va="top", fontfamily="monospace")
    ax.text(0.52, y, value, transform=ax.transAxes, fontsize=8.5,
            color=color_val, va="top", fontfamily="monospace", fontweight="bold" if is_total else "normal")

plt.tight_layout()
plt.savefig("figures/02_timing.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("✅ Saved: figures/02_timing.png")

# ── Save JSON ─────────────────────────────────────────────────────────────────
import json
stats = {
    "file": files[0],
    "file_size_kb": round(file_size_kb, 1),
    "chunks": len(chunks),
    "sections": len(sections),
    "llm_calls": total_llm_calls,
    "tokens_estimated": total_tokens_est,
    "embed_dimension": embed_dim,
    "timing_seconds": {k.replace("\n", " "): round(v, 3) for k, v in results.items()},
    "total_seconds": round(total, 3),
    "avg_chunk_length": round(float(np.mean(chunk_lengths)), 1),
    "avg_entities_per_section": round(float(np.mean(entity_counts)), 2),
    "avg_relations_per_section": round(float(np.mean(relation_counts)), 2),
    "llm_avg_time_per_section_s": round(float(avg_llm_time), 3),
}
with open("figures/02_timing_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print("✅ Saved: figures/02_timing_stats.json")
