"""
Figure 3 — Knowledge Graph Visualization
Lấy entity graph từ Neo4j và visualize bằng networkx + matplotlib.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

os.makedirs("figures", exist_ok=True)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app.core.neo4j_store import run

# ── Fetch graph data from Neo4j ───────────────────────────────────────────────
print("📡 Fetching graph from Neo4j...")

entity_rows = run("MATCH (e:Entity) RETURN e.id AS id, e.name AS name, e.type AS type LIMIT 80")
relation_rows = run("""
    MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
    RETURN a.name AS src, b.name AS tgt, r.relation AS rel
    LIMIT 120
""")

if not entity_rows:
    print("❌ No entities found. Please upload and process a document first.")
    sys.exit(1)

print(f"  Entities: {len(entity_rows)}, Relations: {len(relation_rows)}")

# ── Build NetworkX graph ──────────────────────────────────────────────────────
G = nx.DiGraph()

TYPE_COLORS = {
    "PERSON":  "#0ea5e9",
    "ORG":     "#10b981",
    "CONCEPT": "#8b5cf6",
    "SKILL":   "#f59e0b",
    "PLACE":   "#ef4444",
    "OTHER":   "#64748b",
}

for row in entity_rows:
    G.add_node(row["name"], etype=row.get("type", "OTHER"))

for row in relation_rows:
    if row["src"] in G and row["tgt"] in G:
        G.add_edge(row["src"], row["tgt"], relation=row.get("rel", "related"))

# Remove isolated nodes for cleaner viz
isolated = [n for n in G.nodes() if G.degree(n) == 0]
G.remove_nodes_from(isolated)

if len(G.nodes()) == 0:
    print("⚠️  No connected entities to visualize.")
    sys.exit(0)

# ── Layout ────────────────────────────────────────────────────────────────────
# Use spring layout with high-degree nodes as anchors
try:
    pos = nx.spring_layout(G, k=2.5 / np.sqrt(len(G.nodes())), iterations=60, seed=42)
except Exception:
    pos = nx.random_layout(G, seed=42)

# ── Node attributes ───────────────────────────────────────────────────────────
node_colors = [TYPE_COLORS.get(G.nodes[n].get("etype", "OTHER"), "#64748b") for n in G.nodes()]
degree = dict(G.degree())
node_sizes = [200 + degree[n] * 120 for n in G.nodes()]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(22, 11))
fig.patch.set_facecolor("#0f172a")

# ── Left: Full graph ──────────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor("#0f172a")
ax.set_title(f"(a) Full Entity Graph  ({len(G.nodes())} nodes, {len(G.edges())} edges)",
             color="#e2e8f0", fontsize=12, fontweight="bold", pad=10)

nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color="#475569",
                       arrows=True, arrowsize=10,
                       connectionstyle="arc3,rad=0.1", width=0.8)
nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                       node_size=node_sizes, alpha=0.9)

# Label only high-degree nodes
top_nodes = sorted(degree, key=degree.get, reverse=True)[:20]
labels = {n: n[:18] for n in top_nodes}
nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                        font_size=6.5, font_color="#e2e8f0")

ax.axis("off")

# Legend
legend_handles = [
    mpatches.Patch(color=c, label=t)
    for t, c in TYPE_COLORS.items()
    if any(G.nodes[n].get("etype") == t for n in G.nodes())
]
ax.legend(handles=legend_handles, loc="lower left", fontsize=8,
          facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

# ── Right: Ego graph of top node ─────────────────────────────────────────────
ax = axes[1]
ax.set_facecolor("#0f172a")

top_node = max(degree, key=degree.get)
ego = nx.ego_graph(G, top_node, radius=2)
ego_pos = nx.spring_layout(ego, k=3.0 / max(np.sqrt(len(ego.nodes())), 1), seed=42)

ego_colors = [TYPE_COLORS.get(ego.nodes[n].get("etype", "OTHER"), "#64748b") for n in ego.nodes()]
ego_sizes  = [400 + ego.degree(n) * 150 for n in ego.nodes()]

nx.draw_networkx_edges(ego, ego_pos, ax=ax, alpha=0.4, edge_color="#475569",
                       arrows=True, arrowsize=12,
                       connectionstyle="arc3,rad=0.1", width=1.0)
nx.draw_networkx_nodes(ego, ego_pos, ax=ax, node_color=ego_colors,
                       node_size=ego_sizes, alpha=0.95)
nx.draw_networkx_labels(ego, ego_pos, ax=ax, font_size=7.5, font_color="#e2e8f0")

# Edge labels (relation names)
edge_labels = {(u, v): d["relation"][:12] for u, v, d in ego.edges(data=True)}
nx.draw_networkx_edge_labels(ego, ego_pos, edge_labels=edge_labels, ax=ax,
                             font_size=6, font_color="#94a3b8",
                             bbox=dict(boxstyle="round,pad=0.1", fc="#1e293b", alpha=0.7))

ax.set_title(f'(b) Ego Graph — "{top_node}"\n(radius=2, {len(ego.nodes())} nodes)',
             color="#e2e8f0", fontsize=12, fontweight="bold", pad=10)
ax.axis("off")

# ── Stats annotation ──────────────────────────────────────────────────────────
type_counts = {}
for n in G.nodes():
    t = G.nodes[n].get("etype", "OTHER")
    type_counts[t] = type_counts.get(t, 0) + 1

stats_text = "Entity types:\n" + "\n".join(f"  {t}: {c}" for t, c in sorted(type_counts.items()))
fig.text(0.01, 0.02, stats_text, color="#94a3b8", fontsize=8,
         va="bottom", fontfamily="monospace")

plt.suptitle("Knowledge Graph — Hierarchical Lexical Graph (Neo4j)",
             color="#e2e8f0", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/03_graph.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("✅ Saved: figures/03_graph.png")
