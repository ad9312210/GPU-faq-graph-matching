"""Generate CPU-only storage and candidate-support figures for the paper.

The script runs uniform Top-K candidate generation with the native executable.
It never allocates a dense correspondence matrix.  The output CSV files are
the data used for both figures and include the storage formulas and support
quality metrics.
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


DEFAULT_KS = (16, 32, 64, 128, 256, 512)
SEEDS = (11, 22, 33)
INT32_BYTES = 4
FLOAT32_BYTES = 4


def graph_vertices(path):
    with path.open(encoding="utf-8") as stream:
        return int(stream.readline().split()[0])


def ground_truth(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return {int(row["source_vertex"]): int(row["target_vertex"])
                for row in csv.DictReader(stream)}


def support_metrics(path, n, truth):
    rows = [[] for _ in range(n)]
    recall_hits = 0
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            source = int(row["source"])
            target = int(row["target"])
            rows[source].append(target)
    recall_hits = sum(truth[source] in targets
                      for source, targets in enumerate(rows))
    indptr = np.zeros(n + 1, dtype=np.int64)
    for source, targets in enumerate(rows):
        indptr[source + 1] = indptr[source] + len(targets)
    indices = np.fromiter((target for targets in rows for target in targets),
                          dtype=np.int32, count=int(indptr[-1]))
    support = csr_matrix((np.ones(len(indices), dtype=np.int8), indices, indptr),
                         shape=(n, n))
    matching = maximum_bipartite_matching(support, perm_type="column")
    return recall_hits / n, float(np.count_nonzero(matching >= 0)) / n


def run_cpu(executable, source, target, truth_path, k, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    candidates = run_dir / "candidates.csv"
    timings = run_dir / "native_timings.json"
    assignment = run_dir / "assignment.csv"
    command = [str(executable), "--source", str(source), "--target", str(target),
               "--k", str(k), "--cpu", "--candidates", str(candidates),
               "--output", str(assignment), "--timings", str(timings)]
    completed = subprocess.run(command, capture_output=True, text=True)
    if not candidates.is_file():
        raise RuntimeError(
            f"CPU run failed for {source.parent.name}, K={k}:\n"
            f"{completed.stdout}\n{completed.stderr}")
    n = graph_vertices(source)
    recall, coverage = support_metrics(candidates, n, ground_truth(truth_path))
    metadata = json.loads(timings.read_text(encoding="utf-8")) if timings.is_file() else {}
    if metadata.get("mode") != "cpu":
        raise RuntimeError(f"Refusing non-CPU artifact for {source.parent.name}, K={k}")
    return recall, coverage, n


def write_storage_data(path, ns, ks):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["n", "k", "dense_fp32_bytes", "csr_int32_float32_bytes"])
        for n in ns:
            for k in ks:
                dense = n * n * FLOAT32_BYTES
                csr = ((n + 1) * INT32_BYTES +
                       n * k * (INT32_BYTES + FLOAT32_BYTES))
                writer.writerow([n, k, dense, csr])


def write_quality_data(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", "k", "n",
                                                     "recall_percent",
                                                     "matching_coverage_percent"])
        writer.writeheader()
        writer.writerows(rows)


def plot_storage(path, output):
    data = list(csv.DictReader(path.open(encoding="utf-8")))
    ks = sorted({int(row["k"]) for row in data})
    ns = sorted({int(row["n"]) for row in data})
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    dense = [float(next(row["dense_fp32_bytes"] for row in data
                        if int(row["n"]) == n and int(row["k"]) == ks[0])) / 1e9
             for n in ns]
    ax.plot(ns, dense, "o-", label="Dense FP32 cost matrix")
    for k in ks:
        values = [float(next(row["csr_int32_float32_bytes"] for row in data
                             if int(row["n"]) == n and int(row["k"]) == k)) / 1e9
                  for n in ns]
        ax.plot(ns, values, "o-", label=f"CSR, K={k}")
    ax.set(xscale="log", yscale="log", xlabel="Number of vertices, n",
           ylabel="Analytical storage (GB)")
    ax.set_title("Dense versus sparse correspondence storage")
    ax.grid(True, which="both", alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=240)
    plt.close(fig)


def plot_quality(path, output):
    data = list(csv.DictReader(path.open(encoding="utf-8")))
    ks = sorted({int(row["k"]) for row in data})
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for field, label, color in (("recall_percent", "Recall@K", "#2764A5"),
                                ("matching_coverage_percent", "Maximum matching coverage", "#C56524")):
        means, lows, highs = [], [], []
        for k in ks:
            values = [float(row[field]) for row in data if int(row["k"]) == k]
            means.append(np.mean(values)); lows.append(np.min(values)); highs.append(np.max(values))
        ax.plot(ks, means, "o-", color=color, label=label)
        ax.fill_between(ks, lows, highs, color=color, alpha=.14, linewidth=0,
                        label=f"{label} range across seeds")
    ax.axhline(100, color="#697586", linestyle="--", linewidth=1, label="100%")
    ax.set(xlabel="Uniform candidate budget, K", ylabel="Percentage",
           ylim=(0, 105), xticks=ks)
    ax.set_title("Candidate quality and assignment feasibility")
    ax.grid(axis="y", alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("build-l40s/scgm_cuda"))
    parser.add_argument("--data-root", type=Path, default=Path("data/G01-ca-GrQc"))
    parser.add_argument("--output", type=Path, default=Path("results/paper_figures_cpu"))
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    args = parser.parse_args()
    if not args.executable.is_file():
        parser.error(f"executable does not exist: {args.executable}")
    args.output.mkdir(parents=True, exist_ok=True)
    quality_rows = []
    for seed in SEEDS:
        pair = args.data_root / f"seed_{seed}"
        for k in args.ks:
            recall, coverage, n = run_cpu(
                args.executable, pair / "source.graph", pair / "target.graph",
                pair / "ground_truth.csv", k,
                args.output / "runs" / f"seed_{seed}" / f"K{k:03d}")
            quality_rows.append({"seed": seed, "k": k, "n": n,
                                "recall_percent": 100 * recall,
                                "matching_coverage_percent": 100 * coverage})
    ns = [10**i for i in range(3, 8)]
    write_storage_data(args.output / "storage_data.csv", ns, args.ks)
    write_quality_data(args.output / "quality_data.csv", quality_rows)
    plot_storage(args.output / "storage_data.csv", args.output / "dense_vs_sparse_storage")
    plot_quality(args.output / "quality_data.csv", args.output / "candidate_quality_vs_feasibility")
    print(f"CPU-only figures and CSV data written to {args.output}")


if __name__ == "__main__":
    main()