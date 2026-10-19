"""Evaluation-only audit of feature ties and omitted ground-truth candidates."""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
try:
    from .quality_metrics import support_metrics, validate_truth, fingerprint
    from .evaluate_assignment import read_graph as read_checked_graph
except ImportError:
    from quality_metrics import support_metrics, validate_truth, fingerprint
    from evaluate_assignment import read_graph as read_checked_graph


def read_graph(path):
    n, edges = read_checked_graph(path)
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
    if n == 0:
        return source, target
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
        "largest_duplicate_group": max(counts.values(), default=0),
    }


def audit(source_path, target_path, candidates_path, truth_path, output_path, k, tie_tolerance,
          print_result=True, support_kind="ranked-prefix", normalization="joint",
          feature_bin_width=1e-4, variant="baseline", timings_path=None):
    if k <= 0 or not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("K must be positive and tie tolerance finite/nonnegative")
    if not np.isfinite(feature_bin_width) or feature_bin_width <= 0:
        raise ValueError("feature bin width must be finite/positive")
    if support_kind not in ("ranked-prefix", "retained-support") or normalization not in ("joint", "separate"):
        raise ValueError("invalid support or normalization mode")
    source_graph, target_graph = read_graph(source_path), read_graph(target_path)
    started = time.perf_counter()
    source_features, target_features = features(source_graph), features(target_graph)
    extraction_ms = (time.perf_counter() - started) * 1000
    raw_duplicates = {"source": duplicates(source_features), "target": duplicates(target_features)}
    descriptor_bytes = source_features.nbytes + target_features.nbytes
    started = time.perf_counter()
    if normalization == "joint":
        normalize_joint(source_features, target_features)
    else:
        normalize_joint(source_features, np.empty((0, 6), dtype=np.float32))
        normalize_joint(target_features, np.empty((0, 6), dtype=np.float32))
    normalization_ms = (time.perf_counter() - started) * 1000
    if not np.isfinite(source_features).all() or not np.isfinite(target_features).all():
        raise ValueError("nonfinite normalized reference features")
    truth = {}
    with truth_path.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            s = int(row["source_vertex"])
            if s in truth:
                raise ValueError("duplicate ground-truth source")
            truth[s] = int(row["target_vertex"])
    validate_truth(truth, len(source_features), len(target_features))

    rows = [[] for _ in range(len(source_features))]
    with candidates_path.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            s, rank, t = int(row["source"]), int(row["rank"]), int(row["target"])
            score = float(row["similarity"])
            if not (0 <= s < len(rows) and 0 <= t < len(target_features)) or rank < 0 or not np.isfinite(score):
                raise ValueError("invalid candidate endpoint, rank, or score")
            rows[s].append((rank, t, score))
    for row in rows:
        row.sort()
        if [item[0] for item in row] != list(range(len(row))) or len({item[1] for item in row}) != len(row):
            raise ValueError("candidate ranks must be contiguous and targets unique")
        if support_kind == "ranked-prefix":
            del row[k:]

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
    np.clip(true_scores, -1.0, 1.0, out=true_scores)

    for source, candidates in enumerate(rows):
        if not candidates:
            omitted.append({"source": source, "true_target": int(true_targets[source]),
                            "true_similarity": float(true_scores[source]),
                            "true_cost_minus_kth_cost": None, "empty_row": True})
            continue
        kth_score = candidates[min(k, len(candidates)) - 1][2]
        tied = sum(abs(score - kth_score) <= tie_tolerance for _, _, score in candidates)
        if tied > 1 and support_kind == "ranked-prefix":
            boundary_tie_rows += 1
            boundary_tie_extras += tied - 1

        # Vectorized evaluation-side score audit; truth never enters candidate generation.
        candidate_targets = np.asarray([target for _, target, _ in candidates], dtype=np.int64)
        products = target_features[candidate_targets] * source_features[source]
        dots = np.sum(products, axis=1, dtype=np.float32)
        denominators = source_norms[source] * target_norms[candidate_targets]
        recomputed_scores = np.divide(dots, denominators, out=np.zeros_like(dots),
                                      where=denominators >= np.float32(1e-12))
        np.clip(recomputed_scores, -1.0, 1.0, out=recomputed_scores)
        candidate_score_error.extend(
            abs(saved_score - float(recomputed))
            for (_, _, saved_score), recomputed in zip(candidates, recomputed_scores)
        )

        true_target = int(true_targets[source])
        if true_target not in candidate_targets:
            true_score = float(true_scores[source])
            gap = kth_score - true_score
            if support_kind == "ranked-prefix":
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

    omitted.sort(key=lambda row: row["true_cost_minus_kth_cost"] if row["true_cost_minus_kth_cost"] is not None else float("inf"), reverse=True)
    support = [dict((target, score) for _, target, score in row) for row in rows]
    matching_started = time.perf_counter()
    feasibility = support_metrics(support, len(target_features))
    matching_ms = (time.perf_counter() - matching_started) * 1000
    result = {
        "variant": variant,
        "descriptor_variant": "native_baseline_six_structural_features_python_reference",
        "input_fingerprints": {"source": fingerprint(source_path), "target": fingerprint(target_path),
                               "candidates": fingerprint(candidates_path), "ground_truth": fingerprint(truth_path)},
        "support_kind": support_kind,
        "normalization_mode": normalization,
        "candidate_recall": (len(rows) - len(omitted)) / len(rows) if rows else 1.0,
        "support": feasibility,
        "audit_matching_ms": matching_ms,
        "python_reference_feature_extraction_ms": extraction_ms,
        "python_reference_normalization_ms": normalization_ms,
        "descriptor_array_payload_bytes": descriptor_bytes,
        "descriptor_cost_note": "Python reference timings exclude graph loading and are not native/GPU performance timings.",
        "raw_feature_vectors": raw_duplicates,
        "quantized_feature_ambiguity": {
            "normalized_bin_width": feature_bin_width,
            "definition": "Equal round-to-nearest quantization cells, not an exhaustive pairwise distance test",
            "source": duplicates(np.rint(source_features.astype(np.float64) / feature_bin_width)),
            "target": duplicates(np.rint(target_features.astype(np.float64) / feature_bin_width)),
        },
        "native_timings": json.loads(timings_path.read_text()) if timings_path else None,
        "inputs": {"source": str(source_path), "target": str(target_path),
                   "candidates": str(candidates_path), "ground_truth": str(truth_path)},
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
    if support_kind == "retained-support":
        for key in ("sources_with_ties_at_saved_k_boundary", "extra_saved_candidates_tied_at_boundary",
                    "omitted_true_targets_tied_with_kth_boundary"):
            result[key] = None
        result["omitted_true_target_examples_largest_cost_gap"] = [
            {"source": row["source"], "true_target": row["true_target"],
             "true_similarity": row["true_similarity"]} for row in omitted[:20]]
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
    parser.add_argument("--feature-bin-width", type=float, default=1e-4)
    parser.add_argument("--support-kind", choices=("ranked-prefix", "retained-support"), default="ranked-prefix")
    parser.add_argument("--normalization", choices=("joint", "separate"), default="joint")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--timings", type=Path)
    args = parser.parse_args()
    audit(args.source, args.target, args.candidates, args.ground_truth,
          args.output, args.k, args.tie_tolerance, support_kind=args.support_kind,
          normalization=args.normalization, feature_bin_width=args.feature_bin_width,
          variant=args.variant, timings_path=args.timings)


if __name__ == "__main__":
    main()
