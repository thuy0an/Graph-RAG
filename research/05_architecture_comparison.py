"""
Figure 5 — Architecture Comparison: Naive RAG vs Graph RAG
Sinh hình so sánh 2 kiến trúc cho paper/báo cáo.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

os.makedirs("figures", exist_ok=True)

BG, FG = "#0f172a", "#e2e8f0"
C = {
    "rag":    "#64748b",
    "graph":  "#0ea5e9",
    "shared": "#8b5cf6",
    "plus":   "#10b981",
    "minus":  "#ef4444",
    "neo4j":  "#f59e0b",
}

fig, axes = plt.subplots(1, 3, figsize=(22, 12), gridspec_kw={"width_ratios": [1, 1, 0.9]})
fig.patch.set_facecolor(BG)

def box(ax, x, y, w, h, text, color, alpha=0.25, fontsize=9.5, textcolor=FG):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.025",
                          linewidth=1.5, edgecolor=color,
                          facecolor=color, alpha=alpha)
    ax.add_patch(rect)
    rect2 = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle="round,pad=0.025",
                           linewidth=1.5, edgecolor=color,
                           facecolor="none")
    ax.add_patch(rect2)
    for i, line in enumerate(text.split("\n")):
        offset = (len(text.split("\n")) - 1) * 0.018
        ax.text(x, y + offset - i * 0.036, line,
                ha="center", va="center", fontsize=fontsize,
                color=textcolor, fontweight="bold" if i == 0 else "normal")

def arr(ax, x1, y1, x2, y2, color="#94a3b8", style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=1.4))

# ══════════════════════════════════════════════════════════════════════════════
# LEFT: Naive RAG
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[0]
ax.set_facecolor(BG); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
ax.set_title("Naive RAG", color=C["rag"], fontsize=14, fontweight="bold", pad=10)

steps_rag = [
    (0.5, 0.90, "Document", C["shared"]),
    (0.5, 0.75, "Chunking\n(fixed size)", C["shared"]),
    (0.5, 0.60, "Embedding\n(vector store)", C["shared"]),
    (0.5, 0.45, "Vector Search\n(cosine similarity)", C["rag"]),
    (0.5, 0.30, "Top-K Chunks\n(flat retrieval)", C["rag"]),
    (0.5, 0.15, "LLM Answer\n(chunks only)", C["rag"]),
]
for x, y, label, color in steps_rag:
    box(ax, x, y, 0.55, 0.10, label, color)
for i in range(len(steps_rag)-1):
    arr(ax, 0.5, steps_rag[i][1]-0.05, 0.5, steps_rag[i+1][1]+0.05)

# Limitations
ax.text(0.5, 0.04, "✗ No entity relationships\n✗ No document hierarchy\n✗ Flat context only",
        ha="center", va="center", fontsize=8, color=C["minus"],
        bbox=dict(boxstyle="round,pad=0.3", fc="#1e293b", ec=C["minus"], alpha=0.8))

# ══════════════════════════════════════════════════════════════════════════════
# MIDDLE: Graph RAG (this project)
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[1]
ax.set_facecolor(BG); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
ax.set_title("Graph RAG (this project)", color=C["graph"], fontsize=14, fontweight="bold", pad=10)

# Shared steps
box(ax, 0.5, 0.93, 0.55, 0.09, "Document", C["shared"])
box(ax, 0.5, 0.82, 0.55, 0.09, "Chunking + Embedding", C["shared"])
arr(ax, 0.5, 0.885, 0.5, 0.865)
arr(ax, 0.5, 0.775, 0.5, 0.735)

# Graph-specific
box(ax, 0.5, 0.70, 0.55, 0.09, "LLM Entity Extraction\n(entities + relations)", C["graph"])
arr(ax, 0.5, 0.655, 0.5, 0.615)
box(ax, 0.5, 0.58, 0.55, 0.09, "Section Summarization\n(title + summary per section)", C["graph"])
arr(ax, 0.5, 0.535, 0.5, 0.495)
box(ax, 0.5, 0.46, 0.55, 0.09, "Neo4j Hierarchical Graph\nDocument→Section→Chunk→Entity", C["neo4j"])

# Query side
arr(ax, 0.5, 0.415, 0.28, 0.345)
arr(ax, 0.5, 0.415, 0.72, 0.345)

box(ax, 0.28, 0.31, 0.46, 0.09, "Vector Search\n(Neo4j index)", C["graph"])
box(ax, 0.72, 0.31, 0.46, 0.09, "Graph Traversal\n(Cypher BFS)", C["graph"])

arr(ax, 0.28, 0.265, 0.5, 0.205)
arr(ax, 0.72, 0.265, 0.5, 0.205)

box(ax, 0.5, 0.17, 0.55, 0.09, "4-Layer Context\ndoc + sections + graph + passages", C["graph"])
arr(ax, 0.5, 0.125, 0.5, 0.085)
box(ax, 0.5, 0.05, 0.55, 0.09, "LLM Answer", C["graph"])

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT: Feature comparison table
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[2]
ax.set_facecolor(BG); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
ax.set_title("Feature Comparison", color=FG, fontsize=14, fontweight="bold", pad=10)

features = [
    ("Feature",                  "Naive RAG", "Graph RAG"),
    ("Vector search",            "✓",         "✓"),
    ("Entity extraction",        "✗",         "✓"),
    ("Knowledge graph",          "✗",         "✓"),
    ("Graph traversal",          "✗",         "✓"),
    ("Section summaries",        "✗",         "✓"),
    ("Document summary",         "✗",         "✓"),
    ("Hierarchical context",     "✗",         "✓"),
    ("Multi-hop reasoning",      "✗",         "✓"),
    ("Relationship-aware",       "✗",         "✓"),
    ("Persistent graph store",   "✗",         "✓ Neo4j"),
    ("Context layers",           "1",         "4"),
    ("Retrieval strategy",       "Flat",      "Hybrid"),
]

col_x = [0.08, 0.52, 0.82]
row_h = 0.065
start_y = 0.93

for i, (feat, naive, graph) in enumerate(features):
    y = start_y - i * row_h
    is_header = i == 0
    bg_color = "#1e293b" if i % 2 == 0 else "#0f172a"

    rect = FancyBboxPatch((0.02, y - row_h*0.45), 0.96, row_h*0.9,
                          boxstyle="round,pad=0.005",
                          facecolor="#334155" if is_header else bg_color,
                          edgecolor="none")
    ax.add_patch(rect)

    fw = "bold" if is_header else "normal"
    ax.text(col_x[0], y, feat,  color=FG,         fontsize=8.5, va="center", fontweight=fw)
    ax.text(col_x[1], y, naive, color=C["minus"] if naive == "✗" else FG,
            fontsize=8.5, va="center", ha="center", fontweight=fw)
    ax.text(col_x[2], y, graph, color=C["plus"]  if graph.startswith("✓") else FG,
            fontsize=8.5, va="center", ha="center", fontweight=fw)

# Column headers underline
ax.axhline(start_y - row_h * 0.55, color="#334155", lw=1.5, xmin=0.02, xmax=0.98)

# Summary boxes
y_sum = start_y - len(features) * row_h - 0.04
ax.text(0.5, y_sum, "Graph RAG advantages:", ha="center", color=FG,
        fontsize=9, fontweight="bold")
advantages = [
    "• Understands entity relationships",
    "• Multi-hop reasoning across docs",
    "• Hierarchical context (3 levels)",
    "• Richer answers for complex queries",
]
for j, adv in enumerate(advantages):
    ax.text(0.08, y_sum - 0.06 - j*0.055, adv, color=C["plus"], fontsize=8.5)

plt.suptitle("Architecture Comparison: Naive RAG vs Graph RAG",
             color=FG, fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("figures/05_comparison.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("✅ Saved: figures/05_comparison.png")
