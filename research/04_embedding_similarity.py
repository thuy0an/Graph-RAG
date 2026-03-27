"""
Figure 4 — Embedding & Similarity Analysis
- Heatmap cosine similarity giữa các chunks
- t-SNE / PCA projection của embedding space
- Score distribution từ vector search
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

os.makedirs("figures", exist_ok=True)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app.core.neo4j_store import run
from app.core.providers import get_embeddings

# ── Fetch chunks from Neo4j ───────────────────────────────────────────────────
print("📡 Fetching chunks from Neo4j...")
rows = run("""
    MATCH (c:Chunk)
    RETURN c.id AS id, c.text AS text, c.embedding AS embedding,
           c.index AS idx, c.doc_id AS doc_id
    ORDER BY c.index
    LIMIT 30
""")

if not rows:
    print("❌ No chunks found. Please upload a document first.")
    sys.exit(1)

valid = [r for r in rows if r.get("embedding")]
if len(valid) < 3:
    print("❌ Not enough chunks with embeddings.")
    sys.exit(1)

print(f"  Loaded {len(valid)} chunks with embeddings")

texts     = [r["text"][:60] + "..." for r in valid]
embeddings = np.array([r["embedding"] for r in valid], dtype=np.float32)
chunk_ids  = [r["idx"] for r in valid]

# ── Cosine similarity matrix ──────────────────────────────────────────────────
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
normed = embeddings / (norms + 1e-9)
sim_matrix = normed @ normed.T

# ── Dimensionality reduction ──────────────────────────────────────────────────
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    pca = PCA(n_components=2, random_state=42)
    coords_pca = pca.fit_transform(embeddings)
    use_tsne = len(valid) >= 10
    if use_tsne:
        perp = min(5, len(valid) - 1)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=500)
        coords_tsne = tsne.fit_transform(embeddings)
    has_sklearn = True
except ImportError:
    has_sklearn = False
    print("⚠️  scikit-learn not installed. Skipping PCA/t-SNE plots.")

# ── Query similarity scores ───────────────────────────────────────────────────
embed_model = get_embeddings()
test_queries = [
    "What is the main topic?",
    "Who are the key people mentioned?",
    "What are the main concepts?",
]
query_scores = {}
for q in test_queries:
    q_emb = np.array(embed_model.embed_query(q), dtype=np.float32)
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    scores = normed @ q_norm
    query_scores[q] = scores

# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════
BG, FG = "#0f172a", "#e2e8f0"
n_plots = 4 if has_sklearn else 2
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.patch.set_facecolor(BG)

# ── (a) Cosine similarity heatmap ─────────────────────────────────────────────
ax = axes[0, 0]
ax.set_facecolor(BG)
cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom", ["#0f172a", "#1e3a5f", "#0ea5e9", "#38bdf8", "#e2e8f0"])
im = ax.imshow(sim_matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors=FG)

n = len(valid)
step = max(1, n // 10)
ax.set_xticks(range(0, n, step))
ax.set_yticks(range(0, n, step))
ax.set_xticklabels([str(chunk_ids[i]) for i in range(0, n, step)],
                   color=FG, fontsize=7, rotation=45)
ax.set_yticklabels([str(chunk_ids[i]) for i in range(0, n, step)],
                   color=FG, fontsize=7)
ax.set_xlabel("Chunk index", color=FG, fontsize=10)
ax.set_ylabel("Chunk index", color=FG, fontsize=10)
ax.set_title(f"(a) Cosine Similarity Matrix\n({n} chunks)", color=FG, fontsize=11, fontweight="bold")

# Annotate diagonal stats
diag_off = sim_matrix[np.triu_indices(n, k=1)]
ax.text(0.98, 0.02,
        f"Mean off-diag: {diag_off.mean():.3f}\nMax off-diag: {diag_off.max():.3f}",
        transform=ax.transAxes, ha="right", va="bottom",
        color="#38bdf8", fontsize=8, fontfamily="monospace")

# ── (b) Query similarity scores ───────────────────────────────────────────────
ax = axes[0, 1]
ax.set_facecolor(BG)
colors_q = ["#0ea5e9", "#10b981", "#f59e0b"]
for (q, scores), color in zip(query_scores.items(), colors_q):
    ax.plot(range(len(scores)), scores, marker="o", markersize=3,
            linewidth=1.2, color=color, alpha=0.85,
            label=f'"{q[:35]}..."')
ax.set_xlabel("Chunk index", color=FG, fontsize=10)
ax.set_ylabel("Cosine similarity", color=FG, fontsize=10)
ax.set_title("(b) Query–Chunk Similarity Scores", color=FG, fontsize=11, fontweight="bold")
ax.tick_params(colors=FG)
ax.spines[:].set_color("#334155")
ax.set_ylim(0, 1)
ax.axhline(0.5, color="#334155", linestyle="--", lw=0.8, alpha=0.5)
ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor=FG, fontsize=7.5)

# ── (c) PCA projection ────────────────────────────────────────────────────────
ax = axes[1, 0]
ax.set_facecolor(BG)
if has_sklearn:
    scatter = ax.scatter(coords_pca[:, 0], coords_pca[:, 1],
                         c=range(len(valid)), cmap="plasma",
                         s=60, alpha=0.85, edgecolors="#334155", linewidths=0.5)
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04,
                 label="Chunk index").ax.tick_params(colors=FG)
    for i, (x, y) in enumerate(coords_pca):
        if i % max(1, len(valid)//8) == 0:
            ax.annotate(str(chunk_ids[i]), (x, y), fontsize=6.5,
                        color="#94a3b8", xytext=(3, 3), textcoords="offset points")
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)", color=FG, fontsize=10)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)", color=FG, fontsize=10)
    ax.set_title("(c) PCA Projection of Chunk Embeddings", color=FG, fontsize=11, fontweight="bold")
else:
    ax.text(0.5, 0.5, "scikit-learn required\npip install scikit-learn",
            ha="center", va="center", color=FG, fontsize=12)
    ax.set_title("(c) PCA Projection (unavailable)", color=FG, fontsize=11)
ax.tick_params(colors=FG)
ax.spines[:].set_color("#334155")

# ── (d) t-SNE projection ──────────────────────────────────────────────────────
ax = axes[1, 1]
ax.set_facecolor(BG)
if has_sklearn and use_tsne:
    scatter2 = ax.scatter(coords_tsne[:, 0], coords_tsne[:, 1],
                          c=range(len(valid)), cmap="plasma",
                          s=60, alpha=0.85, edgecolors="#334155", linewidths=0.5)
    plt.colorbar(scatter2, ax=ax, fraction=0.046, pad=0.04,
                 label="Chunk index").ax.tick_params(colors=FG)
    ax.set_xlabel("t-SNE dim 1", color=FG, fontsize=10)
    ax.set_ylabel("t-SNE dim 2", color=FG, fontsize=10)
    ax.set_title("(d) t-SNE Projection of Chunk Embeddings", color=FG, fontsize=11, fontweight="bold")
elif has_sklearn:
    ax.text(0.5, 0.5, "Need ≥10 chunks for t-SNE",
            ha="center", va="center", color=FG, fontsize=12)
    ax.set_title("(d) t-SNE (insufficient data)", color=FG, fontsize=11)
else:
    ax.text(0.5, 0.5, "scikit-learn required\npip install scikit-learn",
            ha="center", va="center", color=FG, fontsize=12)
    ax.set_title("(d) t-SNE (unavailable)", color=FG, fontsize=11)
ax.tick_params(colors=FG)
ax.spines[:].set_color("#334155")

plt.suptitle("Embedding Space Analysis — Graph RAG",
             color=FG, fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/04_similarity.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("✅ Saved: figures/04_similarity.png")
