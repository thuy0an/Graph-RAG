"""
Figure 4 — Embedding & Similarity Analysis
- Cosine similarity heatmap giua cac chunks
- PCA / t-SNE projection
- Top-K retrieval score analysis
- Score threshold visualization
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from collections import Counter

os.makedirs("figures", exist_ok=True)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app.core.neo4j_store import run
from app.core.providers import get_embeddings

# Fetch chunks from Neo4j
print("Fetching chunks from Neo4j...")
rows = run("""
    MATCH (c:Chunk)
    RETURN c.id AS id, c.text AS text, c.embedding AS embedding,
           c.index AS idx, c.doc_id AS doc_id
    ORDER BY c.index
    LIMIT 30
""")

if not rows:
    print("No chunks found. Please upload a document first.")
    sys.exit(1)

valid = [r for r in rows if r.get("embedding")]
if len(valid) < 3:
    print("Not enough chunks with embeddings.")
    sys.exit(1)

print(f"  Loaded {len(valid)} chunks with embeddings")

texts      = [r["text"][:60] + "..." for r in valid]
embeddings = np.array([r["embedding"] for r in valid], dtype=np.float32)
chunk_ids  = [r["idx"] for r in valid]
embed_dim  = embeddings.shape[1]

# Cosine similarity matrix
norms  = np.linalg.norm(embeddings, axis=1, keepdims=True)
normed = embeddings / (norms + 1e-9)
sim_matrix = normed @ normed.T

# Off-diagonal stats
n = len(valid)
off_diag = sim_matrix[np.triu_indices(n, k=1)]
print(f"  Embedding dim : {embed_dim}")
print(f"  Sim mean      : {off_diag.mean():.4f}")
print(f"  Sim max       : {off_diag.max():.4f}")
print(f"  Sim min       : {off_diag.min():.4f}")
print(f"  Sim std       : {off_diag.std():.4f}")

# Dimensionality reduction
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    pca = PCA(n_components=2, random_state=42)
    coords_pca = pca.fit_transform(embeddings)
    use_tsne = len(valid) >= 10
    if use_tsne:
        perp = min(5, len(valid) - 1)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=500)
        coords_tsne = tsne.fit_transform(embeddings)
    has_sklearn = True
except ImportError:
    has_sklearn = False
    print("scikit-learn not installed. Skipping PCA/t-SNE.")

# Query similarity analysis
embed_model = get_embeddings()
test_queries = [
    "What is the main topic?",
    "Who are the key people mentioned?",
    "What are the main concepts?",
]
query_scores = {}
query_top_k  = {}
K = 5

for q in test_queries:
    q_emb  = np.array(embed_model.embed_query(q), dtype=np.float32)
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    scores = normed @ q_norm
    query_scores[q] = scores
    top_k_idx = np.argsort(scores)[::-1][:K]
    query_top_k[q] = [(chunk_ids[i], float(scores[i]), texts[i]) for i in top_k_idx]
    print(f"\n  Query: {q}")
    for rank, (cid, score, text) in enumerate(query_top_k[q]):
        print(f"    #{rank+1}  chunk={cid}  score={score:.4f}  {text[:50]}")

# PLOT — 2x3
BG, FG, GRID = "#0f172a", "#e2e8f0", "#1e293b"
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.patch.set_facecolor(BG)
fig.suptitle(
    f"Embedding Space Analysis  dim={embed_dim}  chunks={len(valid)}\n"
    f"Sim: mean={off_diag.mean():.3f}  max={off_diag.max():.3f}  std={off_diag.std():.3f}",
    color=FG, fontsize=13, fontweight="bold"
)

# (a) Cosine similarity heatmap
ax = axes[0, 0]
ax.set_facecolor(BG)
cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom", ["#0f172a", "#1e3a5f", "#0ea5e9", "#38bdf8", "#e2e8f0"])
im = ax.imshow(sim_matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(colors=FG)
cbar.set_label("Cosine similarity", color=FG, fontsize=8)

step = max(1, n // 10)
ax.set_xticks(range(0, n, step))
ax.set_yticks(range(0, n, step))
ax.set_xticklabels([str(chunk_ids[i]) for i in range(0, n, step)],
                   color=FG, fontsize=7, rotation=45)
ax.set_yticklabels([str(chunk_ids[i]) for i in range(0, n, step)],
                   color=FG, fontsize=7)
ax.set_xlabel("Chunk index", color=FG, fontsize=9)
ax.set_ylabel("Chunk index", color=FG, fontsize=9)
ax.set_title(
    f"(a) Cosine Similarity Matrix  ({n} chunks)\n"
    f"mean={off_diag.mean():.3f}  max={off_diag.max():.3f}  std={off_diag.std():.3f}",
    color=FG, fontsize=10, fontweight="bold"
)

# (b) Query similarity line chart
ax = axes[0, 1]
ax.set_facecolor(BG)
colors_q = ["#0ea5e9", "#10b981", "#f59e0b"]
for (q, scores), color in zip(query_scores.items(), colors_q):
    ax.plot(range(len(scores)), scores, marker="o", markersize=3,
            linewidth=1.2, color=color, alpha=0.85,
            label=f'"{q[:30]}"')
ax.axhline(0.7, color="#ef4444", linestyle="--", lw=1.0, alpha=0.6, label="threshold=0.7")
ax.axhline(0.5, color="#475569", linestyle="--", lw=0.8, alpha=0.5, label="threshold=0.5")
ax.fill_between(range(len(scores)), 0.7, 1.0, alpha=0.05, color="#10b981")
ax.set_xlabel("Chunk index", color=FG, fontsize=9)
ax.set_ylabel("Cosine similarity", color=FG, fontsize=9)
ax.set_title("(b) Query-Chunk Similarity Scores\n(dashed = retrieval thresholds)", color=FG, fontsize=10, fontweight="bold")
ax.tick_params(colors=FG)
ax.spines[:].set_color("#334155")
ax.set_ylim(0, 1)
ax.legend(facecolor=GRID, edgecolor="#334155", labelcolor=FG, fontsize=7.5)

# (c) Top-K retrieval bar chart (first query)
ax = axes[0, 2]
ax.set_facecolor(BG)
first_q = test_queries[0]
top_k_data = query_top_k[first_q]
tk_labels  = [f"chunk {cid}" for cid, _, _ in top_k_data]
tk_scores  = [score for _, score, _ in top_k_data]
tk_colors  = ["#10b981" if s >= 0.7 else "#f59e0b" if s >= 0.5 else "#ef4444" for s in tk_scores]
bars = ax.barh(tk_labels[::-1], tk_scores[::-1], color=tk_colors[::-1],
               edgecolor="#334155", height=0.55)
ax.axvline(0.7, color="#ef4444", linestyle="--", lw=1.2, label="threshold=0.7")
ax.set_xlabel("Cosine similarity", color=FG, fontsize=9)
ax.set_title(
    f'(c) Top-{K} Retrieved Chunks\nQuery: "{first_q[:35]}"',
    color=FG, fontsize=10, fontweight="bold"
)
ax.tick_params(colors=FG)
ax.spines[:].set_color("#334155")
ax.set_xlim(0, 1)
for bar, val in zip(bars, tk_scores[::-1]):
    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", color=FG, fontsize=8.5)
ax.legend(facecolor=GRID, edgecolor="#334155", labelcolor=FG, fontsize=8)

# (d) PCA projection
ax = axes[1, 0]
ax.set_facecolor(BG)
if has_sklearn:
    scatter = ax.scatter(coords_pca[:, 0], coords_pca[:, 1],
                         c=range(len(valid)), cmap="plasma",
                         s=60, alpha=0.85, edgecolors="#334155", linewidths=0.5)
    cbar2 = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(colors=FG)
    cbar2.set_label("Chunk index", color=FG, fontsize=8)
    for i, (x, y) in enumerate(coords_pca):
        if i % max(1, len(valid) // 8) == 0:
            ax.annotate(str(chunk_ids[i]), (x, y), fontsize=6.5,
                        color="#94a3b8", xytext=(3, 3), textcoords="offset points")
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)", color=FG, fontsize=9)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)", color=FG, fontsize=9)
    ax.set_title(
        f"(d) PCA Projection  (dim {embed_dim} -> 2)\n"
        f"PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%  cumulative={sum(var)*100:.1f}%",
        color=FG, fontsize=10, fontweight="bold"
    )
else:
    ax.text(0.5, 0.5, "pip install scikit-learn", ha="center", va="center", color=FG)
    ax.set_title("(d) PCA (unavailable)", color=FG, fontsize=10)
ax.tick_params(colors=FG)
ax.spines[:].set_color("#334155")

# (e) t-SNE projection
ax = axes[1, 1]
ax.set_facecolor(BG)
if has_sklearn and use_tsne:
    scatter2 = ax.scatter(coords_tsne[:, 0], coords_tsne[:, 1],
                          c=range(len(valid)), cmap="plasma",
                          s=60, alpha=0.85, edgecolors="#334155", linewidths=0.5)
    cbar3 = plt.colorbar(scatter2, ax=ax, fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(colors=FG)
    cbar3.set_label("Chunk index", color=FG, fontsize=8)
    ax.set_xlabel("t-SNE dim 1", color=FG, fontsize=9)
    ax.set_ylabel("t-SNE dim 2", color=FG, fontsize=9)
    ax.set_title(
        f"(e) t-SNE Projection  (perplexity={min(5, len(valid)-1)})\n"
        f"Clusters indicate semantically similar chunks",
        color=FG, fontsize=10, fontweight="bold"
    )
elif has_sklearn:
    ax.text(0.5, 0.5, "Need >= 10 chunks for t-SNE", ha="center", va="center", color=FG)
    ax.set_title("(e) t-SNE (insufficient data)", color=FG, fontsize=10)
else:
    ax.text(0.5, 0.5, "pip install scikit-learn", ha="center", va="center", color=FG)
    ax.set_title("(e) t-SNE (unavailable)", color=FG, fontsize=10)
ax.tick_params(colors=FG)
ax.spines[:].set_color("#334155")

# (f) Similarity score distribution
ax = axes[1, 2]
ax.set_facecolor(BG)
ax.hist(off_diag, bins=30, color="#8b5cf6", edgecolor="#334155", alpha=0.85,
        label=f"n={len(off_diag):,} pairs")
ax.axvline(off_diag.mean(), color="#38bdf8", linestyle="--", lw=1.5,
           label=f"Mean: {off_diag.mean():.3f}")
ax.axvline(np.percentile(off_diag, 90), color="#10b981", linestyle="--", lw=1.2,
           label=f"P90: {np.percentile(off_diag, 90):.3f}")
ax.axvline(0.7, color="#ef4444", linestyle=":", lw=1.2, label="threshold=0.7")
ax.set_xlabel("Cosine similarity", color=FG, fontsize=9)
ax.set_ylabel("Pair count", color=FG, fontsize=9)
ax.set_title(
    f"(f) Pairwise Similarity Distribution\n"
    f"mean={off_diag.mean():.3f}  std={off_diag.std():.3f}  "
    f"pairs>0.7: {(off_diag > 0.7).sum()} ({(off_diag > 0.7).mean()*100:.1f}%)",
    color=FG, fontsize=10, fontweight="bold"
)
ax.tick_params(colors=FG)
ax.spines[:].set_color("#334155")
ax.legend(facecolor=GRID, edgecolor="#334155", labelcolor=FG, fontsize=8)

plt.tight_layout()
plt.savefig("figures/04_similarity.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved: figures/04_similarity.png")
