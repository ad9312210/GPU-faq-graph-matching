"""Candidate-support-constrained 2-swap hill climb on undirected edge agreement.

The initial assignment and candidate support are matcher inputs. Ground truth is
read only after refinement to report accuracy.
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path
try:
    from .evaluate_assignment import read_graph as read_checked_graph, read_mapping
    from .quality_metrics import load_support, mapping_metrics, fingerprint
except ImportError:
    from evaluate_assignment import read_graph as read_checked_graph, read_mapping
    from quality_metrics import load_support, mapping_metrics, fingerprint


def read_graph(path):
    n, edges = read_checked_graph(path)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def read_assignment(path, n):
    p = [-1] * n
    seen = set()
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            s, t = int(row["source"]), int(row["target"])
            if not (0 <= s < n and 0 <= t < n) or s in seen:
                raise ValueError("duplicate/out-of-range assignment endpoint")
            seen.add(s)
            p[s] = t
    if any(x < 0 for x in p) or len(set(p)) != n:
        raise ValueError("input assignment is not complete and one-to-one")
    return p


def read_support(path, n):
    similarity = load_support(path, n, n)
    return [set(row) for row in similarity], similarity


def preserved_edges(source_adj, target_adj, p):
    return sum(1 for u in range(len(p)) for v in source_adj[u]
               if u < v and p[v] in target_adj[p[u]])


def refine(source_adj, target_adj, support, initial, attempts, seed, time_limit_seconds):
    """Ground-truth-free search. Returns a new mapping and bounded-search evidence."""
    import math
    if attempts < 0 or not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
        raise ValueError("attempts must be nonnegative and time limit finite/positive")
    p = list(initial)
    n = len(p)
    if len(source_adj) != n or len(target_adj) != n or len(support) != n or sorted(p) != list(range(n)):
        raise ValueError("refinement requires a square permutation and matching dimensions")
    if any(p[s] not in support[s] for s in range(n)):
        raise ValueError("initial assignment lies outside candidate support")
    rng = random.Random(seed)
    started = time.perf_counter()
    accepted = checked = attempted = 0
    reason = "attempt_limit" if n >= 2 else "fewer_than_two_vertices"
    for _ in range(attempts if n >= 2 else 0):
        if time.perf_counter() - started >= time_limit_seconds:
            reason = "time_limit"
            break
        attempted += 1
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        a, b = p[i], p[j]
        if b not in support[i] or a not in support[j]:
            continue
        checked += 1
        affected = source_adj[i] | source_adj[j]
        before = after = 0
        for v in affected:
            if v == i or v == j:
                continue
            pv = p[v]
            if v in source_adj[i]:
                before += int(pv in target_adj[a])
                after += int(pv in target_adj[b])
            if v in source_adj[j]:
                before += int(pv in target_adj[b])
                after += int(pv in target_adj[a])
        # The i--j edge is unchanged by swapping its endpoints in undirected graphs.
        if after > before:
            p[i], p[j] = b, a
            accepted += 1
    return p, {"attempted_swaps": attempted, "support_eligible_swaps_checked": checked,
               "accepted_strict_improvement_swaps": accepted, "stop_reason": reason,
               "refinement_seconds": time.perf_counter() - started}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--support", type=Path, required=True)
    ap.add_argument("--assignment", type=Path, required=True)
    ap.add_argument("--ground-truth", type=Path, help="Optional post-refinement evaluation only")
    ap.add_argument("--output-assignment", type=Path, required=True)
    ap.add_argument("--output-report", type=Path, required=True)
    ap.add_argument("--attempts", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--time-limit-seconds", type=float, default=60.0)
    ap.add_argument("--variant", default="six_features_edge_refinement")
    args = ap.parse_args()
    start = time.perf_counter()
    ga, ha = read_graph(args.source), read_graph(args.target)
    n = len(ga)
    if len(ha) != n:
        raise ValueError("requires equal graph sizes")
    support, similarity = read_support(args.support, n)
    p = read_assignment(args.assignment, n)
    if any(p[u] not in similarity[u] for u in range(n)):
        raise ValueError("initial assignment lies outside candidate support")
    initial = preserved_edges(ga, ha, p)
    initial_descriptor_cost = sum(1.0 - similarity[u][p[u]] for u in range(n))
    original = list(p)
    p, search = refine(ga, ha, support, p, args.attempts, args.seed, args.time_limit_seconds)
    final = preserved_edges(ga, ha, p)
    if final < initial:
        raise AssertionError("edge agreement unexpectedly decreased")
    # Ground truth is opened only after the search is finished.
    truth = read_mapping(args.ground_truth, "source_vertex", "target_vertex") if args.ground_truth else None
    edges_a = {(u, v) for u in range(n) for v in ga[u] if u < v}
    edges_b = {(u, v) for u in range(n) for v in ha[u] if u < v}
    before_metrics = mapping_metrics(edges_a, edges_b, dict(enumerate(original)), truth,
                                    n, n, similarity) if truth is not None else None
    after_metrics = mapping_metrics(edges_a, edges_b, dict(enumerate(p)), truth,
                                   n, n, similarity) if truth is not None else None
    accuracy = after_metrics["mapping_accuracy"] if after_metrics else None
    final_descriptor_cost = sum(1.0 - similarity[u][p[u]] for u in range(n))
    args.output_assignment.parent.mkdir(parents=True, exist_ok=True)
    with args.output_assignment.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source", "target"])
        writer.writerows(enumerate(p))
    report = {
        "variant": args.variant,
        "input_fingerprints": {"source": fingerprint(args.source), "target": fingerprint(args.target),
                               "support": fingerprint(args.support), "assignment": fingerprint(args.assignment)},
        "inputs": {"source": str(args.source), "target": str(args.target),
                   "support": str(args.support), "assignment": str(args.assignment)},
        "evaluation_before": before_metrics,
        "evaluation_after": after_metrics,
        "changed_matches": sum(a != b for a, b in zip(original, p)),
        "time_limit_seconds": args.time_limit_seconds,
        "time_limit_note": "Checked between proposals; one proposal can overrun the deadline. Time-limited runs need not replay identically.",
        "method": "candidate-support-constrained strict-improvement 2-swap hill climb",
        "objective": "maximize preserved undirected source edges in target under the permutation",
        "candidate_support_edges": sum(map(len, support)),
        "descriptor_assignment_cost_initial": initial_descriptor_cost,
        "descriptor_assignment_cost_after_refinement": final_descriptor_cost,
        "descriptor_cost_increase": final_descriptor_cost - initial_descriptor_cost,
        "initial_preserved_edges": initial,
        "final_preserved_edges": final,
        "source_edges": sum(map(len, ga)) // 2,
        "edge_agreement_initial": initial / len(edges_a) if edges_a else 1.0,
        "edge_agreement_final": final / len(edges_a) if edges_a else 1.0,
        "reverse_edge_agreement_initial": initial / len(edges_b) if edges_b else 1.0,
        "reverse_edge_agreement_final": final / len(edges_b) if edges_b else 1.0,
        "attempt_limit": args.attempts,
        **search,
        "seed": args.seed,
        "final_accuracy_evaluation_only": accuracy,
        "ground_truth_used_for_refinement": False,
        "runtime_seconds_including_io_and_evaluation": time.perf_counter() - start,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
