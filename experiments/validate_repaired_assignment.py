"""Compare a repaired sparse assignment with an exact LAP reference."""
import argparse
import csv
import json
from array import array
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


def read_assignment(path: Path, n: int):
    assignment = np.full(n, -1, dtype=np.int32)
    stated_cost = 0.0
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            source, target = int(row["source"]), int(row["target"])
            assignment[source] = target
            stated_cost += float(row["cost"])
    return assignment, stated_cost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True,
                        help="Read only for post-solve evaluation metrics")
    parser.add_argument("--initial-candidates", type=Path, required=True,
                        help="Original K64 candidate file; used only for Recall@64")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.ground_truth.open(newline="", encoding="utf-8") as stream:
        truth_rows = list(csv.DictReader(stream))
    n = len(truth_rows)
    assignment, stated_cost = read_assignment(args.assignment, n)
    valid = bool(np.all(assignment >= 0) and np.all(assignment < n) and
                 len(np.unique(assignment)) == n)

    source_values, target_values, similarity_values = array("i"), array("i"), array("f")
    with args.candidates.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            source_values.append(int(row["source"]))
            target_values.append(int(row["target"]))
            similarity_values.append(float(row["similarity"]))
    source = np.frombuffer(source_values, dtype=np.int32)
    target = np.frombuffer(target_values, dtype=np.int32)
    similarity = np.frombuffer(similarity_values, dtype=np.float32)
    costs = (np.float32(1.0) - similarity).astype(np.float64)

    # Verify selected edges directly against the exact saved candidate file.
    if valid:
        selected = target == assignment[source]
        selected_per_source = np.bincount(source[selected], minlength=n)
        support_valid = bool(np.all(selected_per_source == 1))
        missing = int(n - np.count_nonzero(selected_per_source))
        candidate_cost = float(costs[selected].sum()) if support_valid else None
    else:
        support_valid = False
        missing = None
        candidate_cost = None

    # If every candidate cost is nonnegative and a complete assignment uses
    # only zero-cost edges, zero is a rigorous global lower-bound certificate.
    # This avoids a slow second solver while still proving exact LAP optimality.
    zero_cost_certificate = bool(
        valid and support_valid and costs.size > 0 and
        float(costs.min()) >= 0.0 and candidate_cost == 0.0)
    if zero_cost_certificate:
        exact_cost = 0.0
        reference_method = "nonnegative_cost_zero_optimum_certificate"
        identical_to_reference = None
    else:
        offset = max(1.0, 1.0 - float(costs.min()))
        weighted = coo_matrix((costs + offset, (source, target)), shape=(n, n)).tocsr()
        ref_rows, ref_targets = min_weight_full_bipartite_matching(weighted)
        reference = np.full(n, -1, dtype=np.int32)
        reference[ref_rows] = ref_targets
        exact_cost = float(weighted[np.arange(n), reference].sum() - n * offset)
        identical_to_reference = bool(valid and np.array_equal(assignment, reference))
        reference_method = "scipy_sparse_min_weight_full_bipartite_matching"

    assignment_cost_gap = candidate_cost - exact_cost if valid and support_valid else None
    recall_hits = 0
    truth = np.full(n, -1, dtype=np.int32)
    for row in truth_rows:
        truth[int(row["source_vertex"])] = int(row["target_vertex"])
    with args.initial_candidates.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            recall_hits += int(int(row["target"]) == int(truth[int(row["source"])]))
    recall = recall_hits / n

    result = {"vertices": n, "candidate_edges": int(len(source)),
              "assignment_complete_one_to_one": valid,
              "assignment_edges_missing_from_support": missing,
              "exact_lap_cost": exact_cost,
              "exact_lap_validation_method": reference_method,
              "assignment_cost_recomputed_from_float32_native_costs": candidate_cost,
              "native_assignment_cost_sum": stated_cost,
              "assignment_cost_gap_to_exact_lap": assignment_cost_gap,
              "assignment_mapping_identical_to_exact_reference": identical_to_reference,
              "zero_cost_optimality_certificate": zero_cost_certificate,
              "minimum_candidate_cost": float(costs.min()) if costs.size else None,
              "final_accuracy": float(np.count_nonzero(assignment == truth) / n) if valid else None,
              "recall_at_k64": recall,
              "label_use": "ground truth read only after support repair and assignment for evaluation"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not valid or missing != 0 or result["assignment_cost_gap_to_exact_lap"] > 1e-8:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
