"""
Figure 1 — Hierarchical Lexical Graph Pipeline Diagram
Sinh sơ đồ pipeline đầy đủ cho paper/báo cáo.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

os.makedirs("figures", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(20, 12))
fig.patch.set_facecolor("#0f172a")

# ── color palette ─────────────────────────────────────────────────────────────
C = {
    "doc":     "#0ea5e9",
    "chunk":   "#8b5cf6",
    "section": "#10b981",
    "entity":  "#f59e0b",
    "neo4j":   "#ef4444",
    "llm":     "#ec4899",
    "arrow":   "#94a3b8",
    "text":    "#e2e8f0",
    "bg":      "#1e293b",
    "dim":     "#475569",
}

def box(ax, x, y, w, h, label, sublabel="", color="#1e293b", textcolor="#e2e8f0", fontsize=10):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.02", linewidth=1.5,
                          edgecolor=color, facecolor=color + "33")
    ax.add_patch(rect)
    ax.text(x, y + (0.015 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=textcolor)
    if sublabel:
        ax.text(x, y - 0.045, sublabel, ha="center", va="center",
                fontsize=7.5, color=C["dim"])

def arrow(ax, x1, y1, x2, y2, label="", color="#94a3b8"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.02, my, label, fontsize=7, color=color, va="center")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT: INDEXING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[0]
ax.set_facecolor("#0f172a")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("(a) Indexing Pipeline", color=C["text"], fontsize=14, fontweight="bold", pad=12)

steps = [
    (0.5, 0.92, "📄 Document", "PDF / DOCX",          C["doc"],     0.38, 0.07),
    (0.5, 0.78, "✂️  Chunking", "size=800, overlap=100", C["chunk"],  0.38, 0.07),
    (0.5, 0.64, "🔢 Embedding", "nomic-embed-text / OpenAI", C["chunk"], 0.38, 0.07),
    (0.5, 0.50, "🤖 LLM Extract", "entities + relations per chunk", C["llm"], 0.38, 0.07),
    (0.5, 0.36, "📑 Section Summary", "title + summary (4 chunks/section)", C["section"], 0.38, 0.07),
    (0.5, 0.22, "📋 Doc Summary", "global document overview", C["section"], 0.38, 0.07),
    (0.5, 0.08, "🗄️  Neo4j", "Hierarchical Lexical Graph", C["neo4j"], 0.38, 0.07),
]

for (x, y, label, sub, color, w, h) in steps:
    box(ax, x, y, w, h, label, sub, color)

ys = [s[1] for s in steps]
for i in range(len(ys) - 1):
    arrow(ax, 0.5, ys[i] - 0.035, 0.5, ys[i+1] + 0.035, color=C["arrow"])

# Side annotations
annotations = [
    (0.5, 0.78, 0.83, 0.78, "RecursiveCharacterTextSplitter"),
    (0.5, 0.64, 0.83, 0.64, "LangChain Embeddings"),
    (0.5, 0.50, 0.83, 0.50, "JSON: {entities, relations}"),
    (0.5, 0.36, 0.83, 0.36, "LLM → title + summary"),
]
for x1, y1, x2, y2, label in annotations:
    ax.annotate("", xy=(x2, y2), xytext=(x1 + 0.19, y1),
                arrowprops=dict(arrowstyle="-", color=C["dim"], lw=0.8, linestyle="dashed"))
    ax.text(x2 + 0.01, y2, label, fontsize=7, color=C["dim"], va="center")

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT: QUERY PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[1]
ax.set_facecolor("#0f172a")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("(b) Query Pipeline", color=C["text"], fontsize=14, fontweight="bold", pad=12)

box(ax, 0.5, 0.93, 0.40, 0.07, "❓ Question", "", C["doc"])
arrow(ax, 0.5, 0.895, 0.5, 0.845, color=C["arrow"])
box(ax, 0.5, 0.81, 0.40, 0.07, "🔢 Embed Question", "get_embeddings().embed_query()", C["chunk"])

# Split into 2 branches
arrow(ax, 0.35, 0.775, 0.22, 0.695, color=C["arrow"])
arrow(ax, 0.65, 0.775, 0.78, 0.695, color=C["arrow"])

box(ax, 0.22, 0.66, 0.36, 0.07, "🔍 Vector Search", "Neo4j vector index\ntop-K chunks", C["neo4j"])
box(ax, 0.78, 0.66, 0.36, 0.07, "🤖 Entity Extract", "LLM → query entities", C["llm"])

arrow(ax, 0.22, 0.625, 0.22, 0.555, color=C["arrow"])
arrow(ax, 0.78, 0.625, 0.78, 0.555, color=C["arrow"])

box(ax, 0.22, 0.52, 0.36, 0.07, "📑 Section Summaries", "hierarchical context", C["section"])
box(ax, 0.78, 0.52, 0.36, 0.07, "🕸️  Graph Traversal", "Cypher BFS depth=2", C["entity"])

# Merge
arrow(ax, 0.22, 0.485, 0.40, 0.405, color=C["arrow"])
arrow(ax, 0.78, 0.485, 0.60, 0.405, color=C["arrow"])

box(ax, 0.5, 0.37, 0.55, 0.07, "🔗 Merge Context", "doc_summary + sections + graph_facts + passages", C["doc"])
arrow(ax, 0.5, 0.335, 0.5, 0.265, color=C["arrow"])
box(ax, 0.5, 0.23, 0.40, 0.07, "🤖 LLM Generate", "4-layer context prompt", C["llm"])
arrow(ax, 0.5, 0.195, 0.5, 0.125, color=C["arrow"])
box(ax, 0.5, 0.09, 0.40, 0.07, "💬 Answer + Sources", "", C["doc"])

# Context layer legend
legend_items = [
    mpatches.Patch(color=C["doc"]+"88",     label="Document layer"),
    mpatches.Patch(color=C["chunk"]+"88",   label="Embedding layer"),
    mpatches.Patch(color=C["section"]+"88", label="Section/Summary layer"),
    mpatches.Patch(color=C["entity"]+"88",  label="Entity/Graph layer"),
    mpatches.Patch(color=C["llm"]+"88",     label="LLM call"),
    mpatches.Patch(color=C["neo4j"]+"88",   label="Neo4j"),
]
axes[1].legend(handles=legend_items, loc="lower right", fontsize=8,
               facecolor="#1e293b", edgecolor="#334155", labelcolor=C["text"])

plt.suptitle("Graph RAG — Hierarchical Lexical Graph Architecture",
             color=C["text"], fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("figures/01_pipeline.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("✅ Saved: figures/01_pipeline.png")
