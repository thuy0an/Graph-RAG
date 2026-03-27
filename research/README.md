# Research Visualizations

Chạy từng script để sinh hình ảnh phục vụ nghiên cứu khoa học.

## Cài dependencies
```bash
pip install matplotlib networkx numpy scikit-learn pandas seaborn
```

## Các script

| Script | Mô tả | Output |
|--------|-------|--------|
| `01_pipeline_diagram.py` | Sơ đồ pipeline Graph RAG | `figures/01_pipeline.png` |
| `02_timing_benchmark.py` | Thời gian từng bước xử lý | `figures/02_timing.png` |
| `03_graph_visualization.py` | Visualize Knowledge Graph từ Neo4j | `figures/03_graph.png` |
| `04_embedding_similarity.py` | Heatmap similarity scores | `figures/04_similarity.png` |
| `05_architecture_comparison.py` | So sánh RAG vs Graph RAG | `figures/05_comparison.png` |

```bash
cd research
python 01_pipeline_diagram.py
python 02_timing_benchmark.py
python 03_graph_visualization.py
python 04_embedding_similarity.py
python 05_architecture_comparison.py
```
