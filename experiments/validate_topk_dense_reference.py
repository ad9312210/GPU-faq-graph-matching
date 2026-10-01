"""Compare selected native Top-K rows with an independent dense feature reference."""

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def load_graph(path):
    with path.open(encoding="utf-8") as stream:
        n, declared_edges = map(int, stream.readline().split())
        edges = [tuple(map(int, line.split()[:2])) for line in stream if line.strip()]
    if len(edges) != declared_edges:
        raise ValueError(f"{path}: header says {declared_edges} edges, read {len(edges)}")
    adjacency = [set() for _ in range(n)]
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n) or u == v:
            raise ValueError(f"{path}: invalid edge {(u, v)}")
        if v in adjacency[u]:
            raise ValueError(f"{path}: duplicate edge {(u, v)}")
        adjacency[u].add(v)
        adjacency[v].add(u)
    return adjacency


def reference_features(adjacency):
    """Independent transparent reference; returns raw six-column features."""
    n = len(adjacency)
    degree = np.asarray([len(neighbors) for neighbors in adjacency], dtype=np.float64)
    triangles = np.zeros(n, dtype=np.float64)
    for u in range(n):
        for w in adjacency[u]:
            if w <= u:
                continue
            for z in adjacency[w]:
                if z > w and z in adjacency[u]:
                    triangles[u] += 1
                    triangles[w] += 1
                    triangles[z] += 1
    features = np.zeros((n, 6), dtype=np.float64)
    features[:, 0] = degree
    features[:, 4] = triangles
    features[:, 5] = degree * degree
    for v, neighbors in enumerate(adjacency):
        d = len(neighbors)
        if d >= 2:
            features[v, 1] = triangles[v] / (d * (d - 1) / 2)
        if d:
            neighbor_degrees = degree[list(neighbors)]
            features[v, 2] = neighbor_degrees.mean()
            if d > 1:
                features[v, 3] = neighbor_degrees.std(ddof=1)
    return features


def joint_normalize(source, target):
    # Match the native method's shared graph-pair population z-score.
    combined = np.concatenate((source, target), axis=0)
    mean = combined.mean(axis=0)
    std = combined.std(axis=0)
    std[std < 1e-12] = 1.0
    return (source - mean) / std, (target - mean) / std


def read_native_candidates(path, selected_rows):
    candidates = {row: [] for row in selected_rows}
    with path.open(newline="", encoding="utf-8") as stream:
        for item in csv.DictReader(stream):
            source = int(item["source"])
            if source in candidates:
                candidates[source].append((int(item["target"]), float(item["similarity"])))
    for source, row in candidates.items():
        if not row:
            raise ValueError(f"native candidate export has no row for source {source}")
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--executable", type=Path, default=Path("build-l40s/scgm_cuda"))
    parser.add_argument("--mode", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--rows", type=int, nargs="+", default=(0, 1, 17, 512, 2048, 4096, 5240))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tie-tolerance", type=float, default=2e-5)
    args = parser.parse_args()

    source_adj = load_graph(args.source)
    target_adj = load_graph(args.target)
    if len(source_adj) != len(target_adj):
        raise ValueError("source and target must have the same number of vertices")
    if any(row < 0 or row >= len(source_adj) for row in args.rows):
        raise ValueError("selected source row is out of range")
    if not args.executable.is_file():
        raise FileNotFoundError(args.executable)

    x, y = joint_normalize(reference_features(source_adj), reference_features(target_adj))
    selected = sorted(set(args.rows))
    with tempfile.TemporaryDirectory(prefix="scgm_dense_ref_") as temp:
        candidate_path = Path(temp) / "candidates.csv"
        assignment_path = Path(temp) / "assignment.csv"
        timing_path = Path(temp) / "timings.json"
        command = [
            str(args.executable), "--source", str(args.source), "--target", str(args.target),
            "--k", str(args.k), f"--{args.mode}", "--candidates", str(candidate_path),
            "--output", str(assignment_path), "--timings", str(timing_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if not candidate_path.exists():
            raise RuntimeError(
                f"native candidate export failed ({completed.returncode}): {completed.stderr}"
            )
        native = read_native_candidates(candidate_path, selected)

    row_reports = []
    failures = []
    for source in selected:
        dots = y @ x[source]
        denominators = np.linalg.norm(y, axis=1) * np.linalg.norm(x[source])
        dense_scores = np.divide(dots, denominators, out=np.zeros_like(dots),
                                 where=denominators >= 1e-12)
        order = np.argsort(-dense_scores, kind="stable")
        cutoff = float(dense_scores[order[args.k - 1]])
        dense_top = set(int(target) for target in order[:args.k])
        saved = native[source]
        saved_targets = {target for target, _ in saved}
        saved_score_error = max(
            abs(score - float(dense_scores[target])) for target, score in saved
        )
        outside_cutoff = [
            target for target, _ in saved
            if dense_scores[target] < cutoff - args.tie_tolerance
        ]
        tied_at_cutoff = int(np.count_nonzero(np.abs(dense_scores - cutoff) <= args.tie_tolerance))
        if outside_cutoff or len(saved) != args.k:
            failures.append(source)
        row_reports.append({
            "source": source,
            "k": args.k,
            "dense_cutoff_similarity": cutoff,
            "native_minimum_similarity": min(score for _, score in saved),
            "dense_cutoff_tie_count": tied_at_cutoff,
            "native_dense_topk_overlap": len(saved_targets & dense_top),
            "native_saved_candidate_count": len(saved),
            "saved_candidate_count_matches_k": len(saved) == args.k,
            "max_native_vs_dense_similarity_error": saved_score_error,
            "native_candidates_below_dense_cutoff": outside_cutoff,
            "pass": not outside_cutoff,
        })

    report = {
        "source_graph": str(args.source),
        "target_graph": str(args.target),
        "mode": args.mode,
        "k": args.k,
        "selected_rows": selected,
        "native_exit_code": completed.returncode,
        "native_exit_note": "Nonzero exit is expected if global candidate support is infeasible at this K; candidate export is still checked.",
        "reference": "Independent dense cosine computation over six explicitly recomputed features with joint graph-pair normalization; ties at the K boundary are accepted within tolerance.",
        "all_selected_rows_pass": not failures,
        "failed_rows": failures,
        "rows": row_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
