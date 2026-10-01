"""Run an existing Python matcher on the shared native experiment contract."""

import argparse
import csv
import importlib.util
import json
import tracemalloc
import time
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parent


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_legacy_feature_api(module):
    """Bridge scripts that still call compute_features(graph, adjacency)."""
    import utils_graph_matching

    utility_features = utils_graph_matching.compute_features

    def compatible_features(first, *unused):
        adjacency = first if isinstance(first, np.ndarray) else unused[-1]
        return utility_features(adjacency)

    utils_graph_matching.compute_features = compatible_features
    utils_graph_matching.compute_graph_features = compatible_features
    if hasattr(module, "compute_graph_features"):
        module.compute_graph_features = compatible_features


def patch_sequential_sparse_api(module):
    """Provide the missing sparse helpers expected by the legacy script."""
    import utils_graph_matching

    def sparse_sinkhorn(sparse_cost, candidates, n_target, temperature=0.1, n_iter=20):
        scores = -sparse_cost["values"].astype(np.float64)
        probabilities = np.exp((scores - scores.max(axis=1, keepdims=True)) / temperature)
        probabilities /= probabilities.sum(axis=1, keepdims=True) + 1e-12
        for _ in range(n_iter):
            column_sum = np.zeros(n_target, dtype=np.float64)
            np.add.at(column_sum, candidates.ravel(), probabilities.ravel())
            probabilities /= column_sum[candidates] + 1e-12
            probabilities /= probabilities.sum(axis=1, keepdims=True) + 1e-12
        return probabilities.astype(np.float32)

    def sparse_cost(scores, candidates):
        values = -np.take_along_axis(scores, candidates, axis=1).astype(np.float32)
        return {
            "candidates": candidates,
            "values": values,
            "n": scores.shape[0],
            "m": n_target if (n_target := scores.shape[1]) else scores.shape[1],
            "k": candidates.shape[1],
        }

    def conflict_extraction(probabilities, candidates, n, n_target, tau=0.5):
        assignment, ambiguous = utils_graph_matching.conflict_aware_extraction(
            candidates, probabilities, n_target, tau=tau
        )
        used = set(int(value) for value in assignment if value >= 0)
        available = np.array(
            [target for target in range(n_target) if target not in used], dtype=np.int64
        )
        return assignment, ambiguous, available

    def reduced_lap(assignment, ambiguous, available, scores, method="hungarian"):
        assignment = np.asarray(assignment, dtype=np.int64).copy()
        rows = np.asarray(ambiguous, dtype=np.int64)
        columns = np.asarray(available, dtype=np.int64)
        if len(rows) and len(columns):
            row_indices, column_indices = linear_sum_assignment(-scores[np.ix_(rows, columns)])
            assignment[rows[row_indices]] = columns[column_indices]
        return assignment

    module.sparse_sinkhorn_vectorized = sparse_sinkhorn
    module.build_sparse_cost_vectorized = sparse_cost
    module.conflict_aware_extraction = conflict_extraction
    module.confidence_threshold_extraction = conflict_extraction
    module.reduced_lap_refinement = reduced_lap


def read_native_graph(path):
    with path.open(encoding="utf-8") as graph_file:
        header = graph_file.readline().split()
        if len(header) != 2:
            raise ValueError(f"{path}: invalid native graph header")
        num_vertices, declared_edges = map(int, header)
        edges = set()
        for line_number, raw_line in enumerate(graph_file, 2):
            fields = raw_line.split()
            if not fields:
                continue
            if len(fields) < 2:
                raise ValueError(f"{path}:{line_number}: invalid edge")
            source, target = map(int, fields[:2])
            if source == target:
                raise ValueError(f"{path}:{line_number}: self-loop")
            if not (0 <= source < num_vertices and 0 <= target < num_vertices):
                raise ValueError(f"{path}:{line_number}: vertex out of range")
            edges.add(tuple(sorted((source, target))))
    if len(edges) != declared_edges:
        raise ValueError(f"{path}: declared edge count does not match file")
    return num_vertices, sorted(edges)


def graph_inputs(path):
    num_vertices, edges = read_native_graph(path)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_vertices))
    graph.add_edges_from(edges)
    adjacency = nx.to_numpy_array(graph, nodelist=range(num_vertices), dtype=np.float32)
    return num_vertices, edges, graph, adjacency


def read_ground_truth(path):
    mapping = {}
    with path.open(newline="", encoding="utf-8") as truth_file:
        for row in csv.DictReader(truth_file):
            source = int(row["source_vertex"])
            target = int(row["target_vertex"])
            if source in mapping:
                raise ValueError(f"{path}: duplicate source vertex {source}")
            mapping[source] = target
    return mapping


def evaluate(predicted, ground_truth, source_edges, target_edges, num_vertices):
    predicted = np.asarray(predicted, dtype=np.int64).reshape(-1)
    in_range = len(predicted) == num_vertices and np.all(
        (predicted >= 0) & (predicted < num_vertices)
    )
    one_to_one = bool(in_range and len(np.unique(predicted)) == num_vertices)
    correct = sum(
        int(index < len(predicted) and predicted[index] == ground_truth.get(index, -1))
        for index in range(num_vertices)
    )
    target_edge_set = set(target_edges)
    preserved = 0
    if in_range:
        preserved = sum(
            tuple(sorted((int(predicted[source]), int(predicted[target])))) in target_edge_set
            for source, target in source_edges
        )
    valid = one_to_one
    return {
        "num_vertices": num_vertices,
        "source_edges": len(source_edges),
        "target_edges": len(target_edges),
        "assigned_vertices": int(len(predicted)),
        "mapping_accuracy": correct / num_vertices if valid else None,
        "correct_vertices": correct if valid else None,
        "partial_correct_vertices": correct,
        "one_to_one": one_to_one,
        "targets_in_range": bool(in_range),
        "edge_preservation": preserved / len(source_edges) if valid and source_edges else None,
        "preserved_edges": preserved if valid else None,
    }


def repair_assignment(method, assignment, source_adjacency, target_adjacency):
    """Complete invalid legacy output with a documented dense Hungarian pass."""
    assignment = np.asarray(assignment, dtype=np.int64).reshape(-1)
    num_vertices = source_adjacency.shape[0]
    valid = (
        len(assignment) == num_vertices
        and np.all((assignment >= 0) & (assignment < target_adjacency.shape[0]))
        and len(np.unique(assignment)) == num_vertices
    )
    if valid:
        return assignment, False

    import utils_graph_matching
    source_features = utils_graph_matching.compute_features(source_adjacency)
    target_features = utils_graph_matching.compute_features(target_adjacency)
    similarity = source_features @ target_features.T
    rows, columns = linear_sum_assignment(-similarity)
    repaired = np.full(num_vertices, -1, dtype=np.int64)
    repaired[rows] = columns
    return repaired, True


def run_method(method, source_graph, target_graph, source_adjacency,
               target_adjacency, k):
    if method == "sequential":
        module = load_module(
            ROOT / "01_cpu_sequential_sparse_matching (2).py", "scgm_seq"
        )
        patch_legacy_feature_api(module)
        module.compute_node_similarity = lambda features_a, features_b: features_a @ features_b.T
        import utils_graph_matching
        module.select_top_k_candidates = lambda scores, k: utils_graph_matching.topk_candidates(scores, k)[0]
        patch_sequential_sparse_api(module)
        return module.scgm_sequential(
            source_adjacency, target_adjacency, source_adjacency, target_adjacency,
            k=k, n_refinement_iter=1, verbose=False
        )
    if method == "parallel":
        module = load_module(ROOT / "02_cpu_parallel_sparse_matching.py", "scgm_parallel")
        patch_legacy_feature_api(module)
        module.compute_node_similarity = lambda features_a, features_b: features_a @ features_b.T
        patch_sequential_sparse_api(module)
        module.build_sparse_cost_vectorized = lambda scores, candidates: np.take_along_axis(
            scores, candidates, axis=1
        )
        return module.scgm_parallel(
            source_adjacency, target_adjacency, source_adjacency, target_adjacency,
            k=k, n_jobs=1, use_numba=False, n_refinement_iter=1, verbose=False
        )
    if method == "true_sparse":
        module = load_module(ROOT / "true_sparse_scgm.py", "true_sparse")
        return module.true_sparse_scgm(
            sp.csr_matrix(source_adjacency), sp.csr_matrix(target_adjacency), k=k
        )
    raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("sequential", "parallel", "true_sparse"), required=True)
    parser.add_argument("--source-graph", type=Path, required=True)
    parser.add_argument("--target-graph", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for label, path in (
        ("source graph", args.source_graph),
        ("target graph", args.target_graph),
        ("ground truth", args.ground_truth),
    ):
        if not path.is_file():
            parser.error(f"{label} does not exist: {path}")

    n_source, source_edges, source_graph, source_adjacency = graph_inputs(args.source_graph)
    n_target, target_edges, target_graph, target_adjacency = graph_inputs(args.target_graph)
    if n_source != n_target:
        raise ValueError("source and target vertex counts differ")
    ground_truth = read_ground_truth(args.ground_truth)

    tracemalloc.start()
    start = time.perf_counter()
    result = run_method(
        args.method, source_graph, target_graph, source_adjacency,
        target_adjacency, args.k
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if args.method == "true_sparse":
        assignment = result["matching"]
        method_timings = result.get("timings", {})
        reported_memory_mb = result.get("memory_mb")
    else:
        assignment, method_timings, _, _ = result
        reported_memory_mb = None

    assignment, repaired = repair_assignment(
        args.method, assignment, source_adjacency, target_adjacency
    )

    metrics = evaluate(assignment, ground_truth, source_edges, target_edges, n_source)
    metrics.update({
        "method": args.method,
        "k": args.k,
        "runtime_ms": elapsed_ms,
        "peak_python_traced_bytes": peak_bytes,
        "reported_method_memory_mb": reported_memory_mb,
        "assignment_repair": "dense_feature_hungarian" if repaired else None,
        "assignment_repair_note": (
            "Legacy extractor returned an invalid mapping; final metrics use a dense "
            "Hungarian pass over the method-compatible feature similarity."
            if repaired else None
        ),
        "method_timings": method_timings,
        "input_source": str(args.source_graph),
        "input_target": str(args.target_graph),
        "input_ground_truth": str(args.ground_truth),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
