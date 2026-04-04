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


# =============================================================================
# DETAILED GRAPH ANALYSIS — sinh figures/03_graph_detailed.png
# =============================================================================

def plot_detailed_graph(G, degree, in_degree, out_degree, type_counts,
                        rel_counts, top_nodes, doc_rows, BG, FG):
    """
    6 bieu do chi tiet hon:
      (a) Top-20 entities by degree — horizontal bar
      (b) In-degree vs Out-degree scatter — ai la "nguon" vs "dich"
      (c) Top-15 relation types — tan suat cac loai quan he
      (d) Entity co-occurrence heatmap — cap entity nao hay xuat hien cung
      (e) Subgraph cua top-3 hub nodes — 3 ego graphs nho
      (f) Graph connectivity stats — so lien ket thanh phan
    """
    print("\nGenerating detailed graph analysis...")

    fig, axes = plt.subplots(2, 3, figsize=(24, 15))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"Knowledge Graph — Detailed Analysis\n"
        f"{len(G.nodes())} entities  {len(G.edges())} relations  {len(doc_rows)} document(s)",
        color=FG, fontsize=14, fontweight="bold"
    )

    COLORS = ["#0ea5e9", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444",
              "#ec4899", "#38bdf8", "#a78bfa", "#34d399", "#fbbf24"]

    # ── (a) Top-20 entities by degree ────────────────────────────────────────
    ax = axes[0, 0]
    ax.set_facecolor(BG)
    top20 = sorted(degree, key=degree.get, reverse=True)[:20]
    top20_vals = [degree[n] for n in top20]
    top20_in   = [in_degree[n] for n in top20]
    top20_out  = [out_degree[n] for n in top20]
    top20_labels = [n[:28] for n in top20]

    y = np.arange(len(top20))
    w = 0.35
    ax.barh(y + w/2, top20_in,  w, label="In-degree",  color="#0ea5e9", edgecolor="#334155")
    ax.barh(y - w/2, top20_out, w, label="Out-degree", color="#ec4899", edgecolor="#334155")
    ax.set_yticks(y)
    ax.set_yticklabels(top20_labels[::-1] if False else top20_labels,
                       color=FG, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Degree", color=FG, fontsize=9)
    ax.set_title(
        f"(a) Top-20 Entities by Degree\n"
        f"In-degree = so lan duoc tham chieu  |  Out-degree = so quan he di ra",
        color=FG, fontsize=10, fontweight="bold"
    )
    ax.tick_params(colors=FG)
    ax.spines[:].set_color("#334155")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor=FG, fontsize=8)
    # annotate total degree
    for i, (vi, vo) in enumerate(zip(top20_in, top20_out)):
        ax.text(max(vi, vo) + 0.1, i, str(vi + vo),
                va="center", color="#94a3b8", fontsize=7)

    # ── (b) In-degree vs Out-degree scatter ───────────────────────────────────
    ax = axes[0, 1]
    ax.set_facecolor(BG)
    all_in  = list(in_degree.values())
    all_out = list(out_degree.values())
    type_list = [G.nodes[n].get("etype", "OTHER") for n in G.nodes()]
    clr_map = {"PERSON": "#0ea5e9", "ORG": "#10b981", "CONCEPT": "#8b5cf6",
                "SKILL": "#f59e0b", "PLACE": "#ef4444", "OTHER": "#64748b"}
    scatter_colors = [clr_map.get(t, "#64748b") for t in type_list]

    ax.scatter(all_in, all_out, c=scatter_colors, alpha=0.6, s=40, edgecolors="#334155", lw=0.4)
    # diagonal line: in == out
    max_val = max(max(all_in), max(all_out)) + 1
    ax.plot([0, max_val], [0, max_val], color="#475569", linestyle="--", lw=1, alpha=0.5)
    # label top nodes
    for n in top_nodes[:8]:
        ax.annotate(n[:15], (in_degree[n], out_degree[n]),
                    fontsize=6, color="#94a3b8",
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("In-degree (so lan duoc tham chieu)", color=FG, fontsize=9)
    ax.set_ylabel("Out-degree (so quan he di ra)", color=FG, fontsize=9)
    ax.set_title(
        "(b) In-degree vs Out-degree\n"
        "Tren duong cheo = can bang  |  Duoi = hub nhan nhieu  |  Tren = hub phat nhieu",
        color=FG, fontsize=10, fontweight="bold"
    )
    ax.tick_params(colors=FG)
    ax.spines[:].set_color("#334155")
    # legend
    handles = [mpatches.Patch(color=c, label=t)
               for t, c in clr_map.items() if type_counts.get(t, 0) > 0]
    ax.legend(handles=handles, fontsize=7, facecolor="#1e293b",
              edgecolor="#334155", labelcolor=FG)

    # ── (c) Top-15 relation types ─────────────────────────────────────────────
    ax = axes[0, 2]
    ax.set_facecolor(BG)
    top15_rel = rel_counts.most_common(15)
    if top15_rel:
        r_labels = [r[:25] for r, _ in top15_rel]
        r_vals   = [c for _, c in top15_rel]
        r_colors = [COLORS[i % len(COLORS)] for i in range(len(r_labels))]
        bars = ax.barh(r_labels[::-1], r_vals[::-1], color=r_colors[::-1],
                       edgecolor="#334155", height=0.65)
        ax.set_xlabel("Count", color=FG, fontsize=9)
        ax.set_title(
            f"(c) Top-{len(top15_rel)} Relation Types\n"
            f"Tong {len(rel_counts)} loai quan he khac nhau",
            color=FG, fontsize=10, fontweight="bold"
        )
        ax.tick_params(colors=FG)
        ax.spines[:].set_color("#334155")
        for bar, val in zip(bars, r_vals[::-1]):
            ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", color=FG, fontsize=8)

    # ── (d) Co-occurrence heatmap (top-15 entities) ───────────────────────────
    ax = axes[1, 0]
    ax.set_facecolor(BG)
    top15_nodes = top_nodes[:15]
    n15 = len(top15_nodes)
    cooc = np.zeros((n15, n15))
    node_idx = {n: i for i, n in enumerate(top15_nodes)}

    for u, v in G.edges():
        if u in node_idx and v in node_idx:
            i, j = node_idx[u], node_idx[v]
            cooc[i][j] += 1
            cooc[j][i] += 1  # symmetric

    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cooc", ["#0f172a", "#1e3a5f", "#0ea5e9", "#38bdf8"])
    im = ax.imshow(cooc, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors=FG)
    short_labels = [n[:14] for n in top15_nodes]
    ax.set_xticks(range(n15))
    ax.set_yticks(range(n15))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", color=FG, fontsize=6.5)
    ax.set_yticklabels(short_labels, color=FG, fontsize=6.5)
    ax.set_title(
        "(d) Entity Co-occurrence (Top-15)\n"
        "Mau sang = hai entity co nhieu quan he truc tiep voi nhau",
        color=FG, fontsize=10, fontweight="bold"
    )

    # ── (e) Top-3 hub ego subgraphs ───────────────────────────────────────────
    ax = axes[1, 1]
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_title(
        "(e) Top-3 Hub Nodes — Ego Subgraphs (radius=1)\n"
        "Moi hub hien thi cac entity ket noi truc tiep",
        color=FG, fontsize=10, fontweight="bold"
    )

    hub_colors_list = ["#0ea5e9", "#10b981", "#f59e0b"]
    sub_positions = [(0.18, 0.5), (0.5, 0.5), (0.82, 0.5)]
    sub_radius    = 0.28

    for hub_idx, (hub_node, hub_color, (cx, cy)) in enumerate(
            zip(top_nodes[:3], hub_colors_list, sub_positions)):
        ego1 = nx.ego_graph(G, hub_node, radius=1)
        neighbors = [n for n in ego1.nodes() if n != hub_node][:8]
        all_nodes  = [hub_node] + neighbors
        n_neigh    = len(neighbors)

        # draw hub
        ax.add_patch(plt.Circle((cx, cy), 0.045, color=hub_color, zorder=3, alpha=0.9))
        ax.text(cx, cy, hub_node[:12], ha="center", va="center",
                fontsize=5.5, color="white", fontweight="bold", zorder=4)

        # draw neighbors in circle
        for j, nb in enumerate(neighbors):
            angle = 2 * np.pi * j / max(n_neigh, 1)
            nx_ = cx + sub_radius * 0.6 * np.cos(angle)
            ny_ = cy + sub_radius * 0.6 * np.sin(angle)
            nb_color = {"PERSON": "#0ea5e9", "ORG": "#10b981", "CONCEPT": "#8b5cf6",
                        "SKILL": "#f59e0b", "PLACE": "#ef4444"}.get(
                G.nodes[nb].get("etype", "OTHER"), "#64748b")
            ax.plot([cx, nx_], [cy, ny_], color="#475569", lw=0.8, alpha=0.5, zorder=1)
            ax.add_patch(plt.Circle((nx_, ny_), 0.028, color=nb_color, zorder=2, alpha=0.8))
            ax.text(nx_, ny_, nb[:10], ha="center", va="center",
                    fontsize=4.5, color="white", zorder=3)
            # relation label
            rel = G.edges.get((hub_node, nb), {}).get("relation", "")
            if rel:
                mx, my = (cx + nx_) / 2, (cy + ny_) / 2
                ax.text(mx, my, rel[:10], ha="center", va="center",
                        fontsize=4, color="#94a3b8", zorder=4)

        ax.text(cx, cy - sub_radius * 0.75,
                f"deg={degree[hub_node]}  neighbors={n_neigh}",
                ha="center", va="center", fontsize=6.5, color=hub_color)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 0.95)

    # ── (f) Connectivity & graph metrics ─────────────────────────────────────
    ax = axes[1, 2]
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_title("(f) Graph Metrics & Connectivity", color=FG, fontsize=10, fontweight="bold")

    # compute metrics
    G_undirected = G.to_undirected()
    components   = list(nx.connected_components(G_undirected))
    largest_cc   = max(components, key=len)
    G_lcc        = G_undirected.subgraph(largest_cc)

    try:
        avg_clustering = nx.average_clustering(G_undirected)
    except Exception:
        avg_clustering = 0.0

    try:
        avg_shortest = nx.average_shortest_path_length(G_lcc) if len(G_lcc) > 1 else 0
    except Exception:
        avg_shortest = 0.0

    try:
        diameter = nx.diameter(G_lcc) if len(G_lcc) > 1 else 0
    except Exception:
        diameter = 0

    deg_vals_all = list(degree.values())
    metrics = [
        ("── Connectivity ──────", ""),
        ("Components",        str(len(components))),
        ("Largest CC size",   f"{len(largest_cc)} nodes ({len(largest_cc)/len(G.nodes())*100:.1f}%)"),
        ("Density",           f"{nx.density(G):.5f}"),
        ("── Path Metrics ──────", ""),
        ("Avg shortest path", f"{avg_shortest:.3f}" if avg_shortest else "N/A"),
        ("Diameter (LCC)",    str(diameter) if diameter else "N/A"),
        ("── Clustering ────────", ""),
        ("Avg clustering",    f"{avg_clustering:.4f}"),
        ("── Degree Stats ──────", ""),
        ("Min degree",        str(min(deg_vals_all))),
        ("Max degree",        str(max(deg_vals_all))),
        ("Mean degree",       f"{np.mean(deg_vals_all):.2f}"),
        ("Std degree",        f"{np.std(deg_vals_all):.2f}"),
        ("Nodes deg >= 5",    str(sum(1 for d in deg_vals_all if d >= 5))),
        ("Nodes deg == 1",    str(sum(1 for d in deg_vals_all if d == 1))),
        ("── Entity Types ──────", ""),
    ] + [(f"  {t}", str(c)) for t, c in sorted(type_counts.items(), key=lambda x: -x[1])]

    y_start = 0.97
    row_h   = 0.052
    for i, (label, value) in enumerate(metrics):
        y = y_start - i * row_h
        if y < 0.02:
            break
        is_sep = label.startswith("──")
        ax.text(0.02, y, label, transform=ax.transAxes, fontsize=7.5,
                color="#475569" if is_sep else "#94a3b8",
                va="top", fontfamily="monospace")
        if value:
            ax.text(0.62, y, value, transform=ax.transAxes, fontsize=7.5,
                    color="#38bdf8", va="top", fontfamily="monospace")

    plt.tight_layout()
    plt.savefig("figures/03_graph_detailed.png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Saved: figures/03_graph_detailed.png")


# Run detailed analysis
plot_detailed_graph(G, degree, in_degree, out_degree, type_counts,
                    rel_counts, top_nodes, doc_rows, BG, FG)
