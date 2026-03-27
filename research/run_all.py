"""
Chạy tất cả research scripts và sinh toàn bộ figures.
Usage: python research/run_all.py
"""
import subprocess, sys, os

scripts = [
    "01_pipeline_diagram.py",
    "02_timing_benchmark.py",
    "03_graph_visualization.py",
    "04_embedding_similarity.py",
    "05_architecture_comparison.py",
]

research_dir = os.path.dirname(os.path.abspath(__file__))

print("=" * 55)
print("  Graph RAG — Research Figure Generator")
print("=" * 55)

for script in scripts:
    path = os.path.join(research_dir, script)
    print(f"\n▶ Running {script}...")
    result = subprocess.run([sys.executable, path], capture_output=False)
    if result.returncode != 0:
        print(f"  ⚠️  {script} exited with error (see above)")

print("\n" + "=" * 55)
print("  Done. Check research/figures/ for all outputs.")
print("=" * 55)
