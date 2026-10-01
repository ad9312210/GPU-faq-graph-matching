"""Evaluation-only audit of feature ties and omitted ground-truth candidates."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np


def read_graph(path):
    with path.open(encoding="utf-8") as graph_file:
        n, _ = map(int, graph_file.readline().split())
        edges = [tuple(map(int, line.split()[:2])) for line in graph_file if line.strip()]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    return graph


def features(graph):
    n = graph.number_of_nodes()
    degree = dict(graph.degree())
    triangles = nx.triangles(graph)
    result = np.zeros((n, 6), dtype=np.float32)
    for vertex in range(n):
        d = degree[vertex]
        tri = triangles[vertex]
        neighbors = sorted(graph[vertex])
        result[vertex, 0] = np.float32(d)
        result[vertex, 1] = np.float32((tri / (d * (d - 1) / 2)) if d >= 2 else 0.0)
        result[vertex, 4] = np.float32(tri)
        result[vertex, 5] = np.float32(d * d)
        if neighbors:
            total = np.float32(0.0)
            for neighbor in neighbors:
                total = np.float32(total + np.float32(degree[neighbor]))
            average = np.float32(total / np.float32(len(neighbors)))
            result[vertex, 2] = average
            variance_sum = np.float32(0.0)
            for neighbor in neighbors:
                difference = np.float32(np.float32(degree[neighbor]) - average)
                variance_sum = np.float32(variance_sum + np.float32(difference * difference))
            denominator = len(neighbors) - 1 if len(neighbors) > 1 else 1
            result[vertex, 3] = np.float32(np.sqrt(np.float32(variance_sum / np.float32(denominator))))

    return result


def normalize_joint(source, target):
    # Match the native graph-pair float32 population z-score order.
    n = len(source) + len(target)
    for feature_index in range(6):
        total = np.float32(0.0)
        for value in source[:, feature_index]:
            total = np.float32(total + value)
        for value in target[:, feature_index]:
            total = np.float32(total + value)
        mean = np.float32(total / np.float32(n))
        variance_sum = np.float32(0.0)
        for matrix in (source, target):
            for value in matrix[:, feature_index]:
                delta = np.float32(value - mean)
                variance_sum = np.float32(variance_sum + np.float32(delta * delta))
        std = np.float32(np.sqrt(np.float32(variance_sum / np.float32(n))))
        if std < np.float32(1e-12):
            std = np.float32(1.0)
        for matrix in (source, target):
            for vertex in range(len(matrix)):
                matrix[vertex, feature_index] = np.float32(
                    np.float32(matrix[vertex, feature_index] - mean) / std
                )
    return source, target


def cosine(a, b):
    dot = norm_a = norm_b = np.float32(0.0)
    for left, right in zip(a, b):
        dot = np.float32(dot + np.float32(left * right))
        norm_a = np.float32(norm_a + np.float32(left * left))
        norm_b = np.float32(norm_b + np.float32(right * right))
    denom = np.float32(np.float32(np.sqrt(norm_a)) * np.float32(np.sqrt(norm_b)))
    return np.float32(0.0) if denom < np.float32(1e-12) else np.float32(dot / denom)


def duplicates(values):
    counts = Counter(tuple(float(x) for x in row) for row in values)
    repeated_groups = [count for count in counts.values() if count > 1]
    return {
        "unique_vectors": len(counts),
        "repeated_vector_groups": len(repeated_groups),
        "vertices_in_repeated_groups": sum(repeated_groups),
        "largest_duplicate_group": max(repeated_groups, default=1),
    }


def audit(source_path, target_path, candidates_path, truth_path, output_path, k, tie_tolerance,
          print_result=True):
    source_features, target_features = normalize_joint(
        features(read_graph(source_path)), features(read_graph(target_path)))
    truth = {}
    with truth_path.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            truth[int(row["source_vertex"])] = int(row["target_vertex"])

    rows = [[] for _ in range(len(source_features))]
    with candidates_path.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            rows[int(row["source"])].append((int(row["rank"]), int(row["target"]), float(row["similarity"])))
    for row in rows:
        row.sort()

    omitted = []
    candidate_score_error = []
    boundary_tie_rows = 0
    boundary_tie_extras = 0
    omitted_ties_at_boundary = 0
    omitted_cost_gaps = []
    source_norms = np.sqrt(np.sum(source_features * source_features, axis=1, dtype=np.float32))
    target_norms = np.sqrt(np.sum(target_features * target_features, axis=1, dtype=np.float32))
    true_targets = np.asarray([truth[index] for index in range(len(source_features))], dtype=np.int64)
    true_products = source_features * target_features[true_targets]
    true_dots = np.sum(true_products, axis=1, dtype=np.float32)
    true_denoms = source_norms * target_norms[true_targets]
    true_scores = np.divide(true_dots, true_denoms, out=np.zeros_like(true_dots),
                            where=true_denoms >= np.float32(1e-12))

    for source, candidates in enumerate(rows):
        if not candidates:
            continue
        kth_score = candidates[min(k, len(candidates)) - 1][2]
        tied = sum(abs(score - kth_score) <= tie_tolerance for _, _, score in candidates)
        if tied > 1:
            boundary_tie_rows += 1
            boundary_tie_extras += tied - 1

        # Vectorized evaluation-side score audit; truth never enters candidate generation.
        candidate_targets = np.asarray([target for _, target, _ in candidates], dtype=np.int64)
        products = target_features[candidate_targets] * source_features[source]
        dots = np.sum(products, axis=1, dtype=np.float32)
        denominators = source_norms[source] * target_norms[candidate_targets]
        recomputed_scores = np.divide(dots, denominators, out=np.zeros_like(dots),
                                      where=denominators >= np.float32(1e-12))
        candidate_score_error.extend(
            abs(saved_score - float(recomputed))
            for (_, _, saved_score), recomputed in zip(candidates, recomputed_scores)
        )

        true_target = int(true_targets[source])
        if true_target not in candidate_targets:
            true_score = float(true_scores[source])
            gap = kth_score - true_score
            omitted_cost_gaps.append(gap)
            omitted_ties_at_boundary += int(abs(gap) <= tie_tolerance)
            omitted.append({
                "source": source,
                "true_target": true_target,
                "true_similarity": true_score,
                "true_cost": 1.0 - true_score,
                "kth_similarity": kth_score,
                "kth_cost": 1.0 - kth_score,
                "true_cost_minus_kth_cost": gap,
                "boundary_tie_count_in_saved_candidates": tied,
            })

    omitted.sort(key=lambda row: row["true_cost_minus_kth_cost"], reverse=True)
    result = {
        "k": k,
        "tie_tolerance_similarity": tie_tolerance,
        "source_feature_vectors": duplicates(source_features),
        "target_feature_vectors": duplicates(target_features),
        "source_zero_norm_vectors": int(np.count_nonzero(np.linalg.norm(source_features, axis=1) <= 1e-12)),
        "target_zero_norm_vectors": int(np.count_nonzero(np.linalg.norm(target_features, axis=1) <= 1e-12)),
        "sources_with_ties_at_saved_k_boundary": boundary_tie_rows,
        "extra_saved_candidates_tied_at_boundary": boundary_tie_extras,
        "mean_absolute_saved_candidate_similarity_error": float(np.mean(candidate_score_error)) if candidate_score_error else None,
        "max_absolute_saved_candidate_similarity_error": float(np.max(candidate_score_error)) if candidate_score_error else None,
        "omitted_true_target_count": len(omitted),
        "omitted_true_targets_tied_with_kth_boundary": omitted_ties_at_boundary,
        "omitted_true_target_cost_gap_mean": float(np.mean(omitted_cost_gaps)) if omitted_cost_gaps else None,
        "omitted_true_target_cost_gap_min": float(np.min(omitted_cost_gaps)) if omitted_cost_gaps else None,
        "omitted_true_target_cost_gap_max": float(np.max(omitted_cost_gaps)) if omitted_cost_gaps else None,
        "omitted_true_target_examples_largest_cost_gap": omitted[:20],
        "evaluation_only_note": "Ground truth is read only by this post-run audit, never by candidate generation or repair.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if print_result:
        print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--tie-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    audit(args.source, args.target, args.candidates, args.ground_truth,
          args.output, args.k, args.tie_tolerance)


if __name__ == "__main__":
    main()
