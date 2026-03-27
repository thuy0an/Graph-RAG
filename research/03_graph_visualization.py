"""
Figure 3 — Knowledge Graph Visualization
Lay entity graph tu Neo4j va visualize bang networkx + matplotlib.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
from collections import Counter

os.makedirs("figures", exist_ok=True)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app.core.neo4j_store import run

# Fetch graph data
print("Fetching graph from Neo4j...")

entity_rows = run("MATCH (e:Entity) RETURN e.id AS id, e.name AS name, e.type AS type LIMIT 100")
relation_rows = run("""
    MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
    RETURN a.name AS src, b.name AS tgt, r.relation AS rel
    LIMIT 150
""")
doc_rows = run("MATCH (d:Document) RETURN d.filename AS filename, d.id AS id")
chunk_count = run("MATCH (c:Chunk) RETURN count(c) AS n")[0]["n"]
section_count = run("MATCH (s:Section) RETURN count(s) AS n")[0]["n"]

if not entity_rows:
    print("No entities found. Please upload and process a document first.")
    sys.exit(1)

print(f"  Documents : {len(doc_rows)}")
print(f"  Sections  : {section_count}")
print(f"  Chunks    : {chunk_count}")
print(f"  Entities  : {len(entity_rows)}")
print(f"  Relations : {len(relation_rows)}")

# Build NetworkX graph
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

isolated = [n for n in G.nodes() if G.degree(n) == 0]
G.remove_nodes_from(isolated)

if len(G.nodes()) == 0:
    print("No connected entities to visualize.")
    sys.exit(0)

# Derived stats
degree      = dict(G.degree())
in_degree   = dict(G.in_degree())
out_degree  = dict(G.out_degree())
type_counts = Counter(G.nodes[n].get("etype", "OTHER") for n in G.nodes())
rel_labels  = [d["relation"] for _, _, d in G.edges(data=True)]
rel_counts  = Counter(rel_labels)
top_nodes   = sorted(degree, key=degree.get, reverse=True)[:15]
top_node    = top_nodes[0]

print(f"\n  Top entities by degree:")
for n in top_nodes[:8]:
    print(f"    {n[:30]:30s}  degree={degree[n]}  in={in_degree[n]}  out={out_degree[n]}")

# Layout
try:
    pos = nx.spring_layout(G, k=2.5 / np.sqrt(len(G.nodes())), iterations=80, seed=42)
except Exception:
    pos = nx.random_layout(G, seed=42)

node_colors = [TYPE_COLORS.get(G.nodes[n].get("etype", "OTHER"), "#64748b") for n in G.nodes()]
node_sizes  = [200 + degree[n] * 130 for n in G.nodes()]

# PLOT
BG, FG = "#0f172a", "#e2e8f0"
fig = plt.figure(figsize=(24, 14))
fig.patch.set_facecolor(BG)

gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[2, 1.5, 1])
ax_main  = fig.add_subplot(gs[:, 0])
ax_ego   = fig.add_subplot(gs[0, 1])
ax_types = fig.add_subplot(gs[1, 1])
ax_deg   = fig.add_subplot(gs[0, 2])
ax_stats = fig.add_subplot(gs[1, 2])

# (a) Main graph
ax_main.set_facecolor(BG)
ax_main.set_title(
    f"(a) Full Entity Graph\n"
    f"{len(G.nodes())} nodes  {len(G.edges())} edges  {len(doc_rows)} document(s)",
    color=FG, fontsize=12, fontweight="bold", pad=10
)
nx.draw_networkx_edges(G, pos, ax=ax_main, alpha=0.25, edge_color="#475569",
                       arrows=True, arrowsize=8,
                       connectionstyle="arc3,rad=0.1", width=0.7)
nx.draw_networkx_nodes(G, pos, ax=ax_main, node_color=node_colors,
                       node_size=node_sizes, alpha=0.9)
labels = {n: n[:20] for n in top_nodes[:20]}
nx.draw_networkx_labels(G, pos, labels=labels, ax=ax_main,
                        font_size=6.5, font_color=FG)
ax_main.axis("off")

legend_handles = [
    mpatches.Patch(color=c, label=f"{t}  ({type_counts.get(t, 0)})")
    for t, c in TYPE_COLORS.items()
    if type_counts.get(t, 0) > 0
]
ax_main.legend(handles=legend_handles, loc="lower left", fontsize=8.5,
               facecolor="#1e293b", edgecolor="#334155", labelcolor=FG,
               title="Entity Types", title_fontsize=9)

# (b) Ego graph
ax_ego.set_facecolor(BG)
ego = nx.ego_graph(G, top_node, radius=2)
ego_pos = nx.spring_layout(ego, k=3.0 / max(np.sqrt(len(ego.nodes())), 1), seed=42)
ego_colors = [TYPE_COLORS.get(ego.nodes[n].get("etype", "OTHER"), "#64748b") for n in ego.nodes()]
ego_sizes  = [350 + ego.degree(n) * 140 for n in ego.nodes()]

nx.draw_networkx_edges(ego, ego_pos, ax=ax_ego, alpha=0.4, edge_color="#475569",
                       arrows=True, arrowsize=10,
                       connectionstyle="arc3,rad=0.1", width=1.0)
nx.draw_networkx_nodes(ego, ego_pos, ax=ax_ego, node_color=ego_colors,
                       node_size=ego_sizes, alpha=0.95)
nx.draw_networkx_labels(ego, ego_pos, ax=ax_ego, font_size=7, font_color=FG)
edge_labels = {(u, v): d["relation"][:14] for u, v, d in ego.edges(data=True)}
nx.draw_networkx_edge_labels(ego, ego_pos, edge_labels=edge_labels, ax=ax_ego,
                             font_size=5.5, font_color="#94a3b8",
                             bbox=dict(boxstyle="round,pad=0.1", fc="#1e293b", alpha=0.7))
ax_ego.set_title(
    f'(b) Ego Graph  "{top_node[:25]}"\n'
    f'radius=2  {len(ego.nodes())} nodes  {len(ego.edges())} edges',
    color=FG, fontsize=10, fontweight="bold", pad=8
)
ax_ego.axis("off")

# (c) Entity type distribution
ax_types.set_facecolor(BG)
type_labels = [t for t in TYPE_COLORS if type_counts.get(t, 0) > 0]
type_vals   = [type_counts[t] for t in type_labels]
type_clrs   = [TYPE_COLORS[t] for t in type_labels]
bars = ax_types.barh(type_labels, type_vals, color=type_clrs, edgecolor="#334155", height=0.6)
ax_types.set_xlabel("Count", color=FG, fontsize=9)
ax_types.set_title("(c) Entity Type Distribution", color=FG, fontsize=10, fontweight="bold")
ax_types.tick_params(colors=FG)
ax_types.spines[:].set_color("#334155")
for bar, val in zip(bars, type_vals):
    ax_types.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                  str(val), va="center", color=FG, fontsize=9)

# (d) Degree distribution
ax_deg.set_facecolor(BG)
deg_vals = list(degree.values())
ax_deg.hist(deg_vals, bins=min(20, max(deg_vals) + 1), color="#8b5cf6",
            edgecolor="#334155", alpha=0.85)
ax_deg.axvline(np.mean(deg_vals), color="#38bdf8", linestyle="--", lw=1.5,
               label=f"Mean: {np.mean(deg_vals):.1f}")
ax_deg.axvline(np.median(deg_vals), color="#f59e0b", linestyle="--", lw=1.5,
               label=f"Median: {np.median(deg_vals):.1f}")
ax_deg.set_xlabel("Degree", color=FG, fontsize=9)
ax_deg.set_ylabel("Count", color=FG, fontsize=9)
ax_deg.set_title(
    f"(d) Degree Distribution\nmax={max(deg_vals)}  nodes deg>=3: {sum(1 for d in deg_vals if d >= 3)}",
    color=FG, fontsize=10, fontweight="bold"
)
ax_deg.tick_params(colors=FG)
ax_deg.spines[:].set_color("#334155")
ax_deg.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor=FG, fontsize=8)

# (e) Stats table
ax_stats.set_facecolor(BG)
ax_stats.axis("off")
ax_stats.set_title("(e) Graph Statistics", color=FG, fontsize=10, fontweight="bold")

top5_rel = rel_counts.most_common(5)
top5_ent = [(n, degree[n]) for n in top_nodes[:5]]

rows = [
    ("-- Graph --",      ""),
    ("Documents",        str(len(doc_rows))),
    ("Sections",         str(section_count)),
    ("Chunks",           str(chunk_count)),
    ("Entities",         str(len(G.nodes()))),
    ("Relations",        str(len(G.edges()))),
    ("Isolated (rm)",    str(len(isolated))),
    ("-- Degree --",     ""),
    ("Avg degree",       f"{np.mean(deg_vals):.2f}"),
    ("Max degree",       f"{max(deg_vals)}"),
    ("Density",          f"{nx.density(G):.4f}"),
    ("-- Top Relations --", ""),
] + [(f"  {r[:18]}", str(c)) for r, c in top5_rel] + [
    ("-- Top Entities --", ""),
] + [(f"  {n[:18]}", f"deg={d}") for n, d in top5_ent]

y_start = 0.98
row_h   = 0.057
for i, (label, value) in enumerate(rows):
    y = y_start - i * row_h
    is_sep = label.startswith("--")
    color_lbl = "#475569" if is_sep else "#94a3b8"
    color_val = "#38bdf8"
    ax_stats.text(0.02, y, label, transform=ax_stats.transAxes, fontsize=7.8,
                  color=color_lbl, va="top", fontfamily="monospace")
    if value:
        ax_stats.text(0.62, y, value, transform=ax_stats.transAxes, fontsize=7.8,
                      color=color_val, va="top", fontfamily="monospace")

plt.suptitle(
    f"Knowledge Graph  Hierarchical Lexical Graph (Neo4j)\n"
    f"{len(doc_rows)} doc(s)  {len(G.nodes())} entities  {len(G.edges())} relations  {len(type_counts)} entity types",
    color=FG, fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig("figures/03_graph.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved: figures/03_graph.png")
