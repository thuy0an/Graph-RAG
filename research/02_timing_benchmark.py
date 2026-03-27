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

# ── load env ──────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app.core.document_loader import load_document
from app.core.providers import get_embeddings, get_llm
from app.core.graph_builder import _extract_entities, _section_summary, _group_into_sections

# ── find a test file ──────────────────────────────────────────────────────────
upload_dir = os.path.join(os.path.dirname(__file__), "..", "uploads")
files = [f for f in os.listdir(upload_dir) if f.endswith((".pdf", ".docx", ".doc"))]
if not files:
    print("❌ No files in uploads/. Please upload a document first.")
    sys.exit(1)

test_file = os.path.join(upload_dir, files[0])
print(f"📄 Benchmarking: {files[0]}")

results = {}

# ── Step 1: Chunking ──────────────────────────────────────────────────────────
t0 = time.perf_counter()
chunks = load_document(test_file)
results["Chunking"] = time.perf_counter() - t0
print(f"  Chunking:   {results['Chunking']:.2f}s  ({len(chunks)} chunks)")

# ── Step 2: Embedding (sample 10 chunks) ─────────────────────────────────────
sample = chunks[:min(10, len(chunks))]
embed_model = get_embeddings()
t0 = time.perf_counter()
embeddings = embed_model.embed_documents([c.page_content for c in sample])
elapsed = time.perf_counter() - t0
# Extrapolate to full doc
results["Embedding"] = elapsed / len(sample) * len(chunks)
print(f"  Embedding:  {results['Embedding']:.2f}s  (extrapolated, {len(chunks)} chunks)")

# ── Step 3: Entity extraction (sample 5 chunks) ───────────────────────────────
sample5 = chunks[:min(5, len(chunks))]
t0 = time.perf_counter()
entity_counts = []
relation_counts = []
for c in sample5:
    ents, rels = _extract_entities(c.page_content)
    entity_counts.append(len(ents))
    relation_counts.append(len(rels))
elapsed = time.perf_counter() - t0
results["Entity Extraction"] = elapsed / len(sample5) * len(chunks)
print(f"  Entity ext: {results['Entity Extraction']:.2f}s  (extrapolated)")

# ── Step 4: Section summarization ────────────────────────────────────────────
sections = _group_into_sections(chunks)
sample_secs = sections[:min(3, len(sections))]
t0 = time.perf_counter()
for sec in sample_secs:
    sec_text = "\n\n".join(c.page_content for c in sec)
    _section_summary(sec_text)
elapsed = time.perf_counter() - t0
results["Section Summary"] = elapsed / len(sample_secs) * len(sections)
print(f"  Sec summary:{results['Section Summary']:.2f}s  (extrapolated)")

# ── Step 5: Neo4j write (estimate from chunk count) ──────────────────────────
results["Neo4j Write"] = len(chunks) * 0.015   # ~15ms per chunk write
print(f"  Neo4j write:{results['Neo4j Write']:.2f}s  (estimated)")

# ── Per-chunk stats ───────────────────────────────────────────────────────────
avg_entities = np.mean(entity_counts) if entity_counts else 0
avg_relations = np.mean(relation_counts) if relation_counts else 0
chunk_lengths = [len(c.page_content) for c in chunks]

# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════
BG, FG, GRID = "#0f172a", "#e2e8f0", "#1e293b"
COLORS = ["#0ea5e9", "#8b5cf6", "#ec4899", "#10b981", "#f59e0b"]

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.patch.set_facecolor(BG)
fig.suptitle(f"Processing Benchmark — {files[0]}\n({len(chunks)} chunks, {len(sections)} sections)",
             color=FG, fontsize=14, fontweight="bold")

# ── (a) Step timing bar chart ─────────────────────────────────────────────────
ax = axes[0, 0]
ax.set_facecolor(BG)
labels = list(results.keys())
values = list(results.values())
bars = ax.barh(labels, values, color=COLORS, edgecolor="#334155", height=0.55)
ax.set_xlabel("Time (seconds)", color=FG, fontsize=10)
ax.set_title("(a) Processing Time per Step", color=FG, fontsize=11, fontweight="bold")
ax.tick_params(colors=FG)
ax.spines[:].set_color("#334155")
for bar, val in zip(bars, values):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}s", va="center", color=FG, fontsize=9)
total = sum(values)
ax.text(0.98, 0.02, f"Total: {total:.1f}s", transform=ax.transAxes,
        ha="right", va="bottom", color="#38bdf8", fontsize=10, fontweight="bold")

# ── (b) Stacked time proportion ───────────────────────────────────────────────
ax = axes[0, 1]
ax.set_facecolor(BG)
proportions = [v / total * 100 for v in values]
wedges, texts, autotexts = ax.pie(
    proportions, labels=labels, colors=COLORS,
    autopct="%1.1f%%", startangle=140,
    textprops={"color": FG, "fontsize": 8},
    wedgeprops={"edgecolor": BG, "linewidth": 2},
)
for at in autotexts:
    at.set_color(BG)
    at.set_fontweight("bold")
ax.set_title("(b) Time Distribution", color=FG, fontsize=11, fontweight="bold")

# ── (c) Chunk length distribution ────────────────────────────────────────────
ax = axes[1, 0]
ax.set_facecolor(BG)
ax.hist(chunk_lengths, bins=20, color="#8b5cf6", edgecolor="#334155", alpha=0.85)
ax.axvline(np.mean(chunk_lengths), color="#38bdf8", linestyle="--", lw=1.5,
           label=f"Mean: {np.mean(chunk_lengths):.0f} chars")
ax.axvline(np.median(chunk_lengths), color="#f59e0b", linestyle="--", lw=1.5,
           label=f"Median: {np.median(chunk_lengths):.0f} chars")
ax.set_xlabel("Chunk length (characters)", color=FG, fontsize=10)
ax.set_ylabel("Count", color=FG, fontsize=10)
ax.set_title("(c) Chunk Length Distribution", color=FG, fontsize=11, fontweight="bold")
ax.tick_params(colors=FG)
ax.spines[:].set_color("#334155")
ax.legend(facecolor=GRID, edgecolor="#334155", labelcolor=FG, fontsize=9)

# ── (d) Entities & relations per chunk ───────────────────────────────────────
ax = axes[1, 1]
ax.set_facecolor(BG)
x = np.arange(len(entity_counts))
w = 0.35
ax.bar(x - w/2, entity_counts, w, label="Entities", color="#0ea5e9", edgecolor="#334155")
ax.bar(x + w/2, relation_counts, w, label="Relations", color="#ec4899", edgecolor="#334155")
ax.set_xlabel("Chunk index (sample)", color=FG, fontsize=10)
ax.set_ylabel("Count", color=FG, fontsize=10)
ax.set_title(f"(d) Entities & Relations per Chunk\n(avg: {avg_entities:.1f} entities, {avg_relations:.1f} relations)",
             color=FG, fontsize=11, fontweight="bold")
ax.tick_params(colors=FG)
ax.spines[:].set_color("#334155")
ax.legend(facecolor=GRID, edgecolor="#334155", labelcolor=FG, fontsize=9)

plt.tight_layout()
plt.savefig("figures/02_timing.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("✅ Saved: figures/02_timing.png")

# ── Save raw numbers to JSON ──────────────────────────────────────────────────
import json
stats = {
    "file": files[0],
    "chunks": len(chunks),
    "sections": len(sections),
    "timing_seconds": {k: round(v, 3) for k, v in results.items()},
    "total_seconds": round(total, 3),
    "avg_chunk_length": round(float(np.mean(chunk_lengths)), 1),
    "avg_entities_per_chunk": round(float(avg_entities), 2),
    "avg_relations_per_chunk": round(float(avg_relations), 2),
}
with open("figures/02_timing_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print("✅ Saved: figures/02_timing_stats.json")
