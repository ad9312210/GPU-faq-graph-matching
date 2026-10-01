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


def read_graph(path):
    with path.open(encoding="utf-8") as stream:
        n, _ = map(int, stream.readline().split())
        adj = [set() for _ in range(n)]
        for line in stream:
            if not line.strip():
                continue
            u, v = map(int, line.split()[:2])
            if u != v:
                adj[u].add(v)
                adj[v].add(u)
    return adj


def read_assignment(path, n):
    p = [-1] * n
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            p[int(row["source"])] = int(row["target"])
    if any(x < 0 for x in p) or len(set(p)) != n:
        raise ValueError("input assignment is not complete and one-to-one")
    return p


def read_support(path, n):
    support = [set() for _ in range(n)]
    similarity = [dict() for _ in range(n)]
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            source, target = int(row["source"]), int(row["target"])
            support[source].add(target)
            if "similarity" in row and row["similarity"]:
                similarity[source][target] = float(row["similarity"])
    return support, similarity


def preserved_edges(source_adj, target_adj, p):
    return sum(1 for u in range(len(p)) for v in source_adj[u]
               if u < v and p[v] in target_adj[p[u]])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--support", type=Path, required=True)
    ap.add_argument("--assignment", type=Path, required=True)
    ap.add_argument("--ground-truth", type=Path, required=True)
    ap.add_argument("--output-assignment", type=Path, required=True)
    ap.add_argument("--output-report", type=Path, required=True)
    ap.add_argument("--attempts", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=20260929)
    args = ap.parse_args()
    start = time.perf_counter()
    ga, ha = read_graph(args.source), read_graph(args.target)
    n = len(ga)
    if len(ha) != n:
        raise ValueError("requires equal graph sizes")
    support, similarity = read_support(args.support, n)
    p = read_assignment(args.assignment, n)
    initial = preserved_edges(ga, ha, p)
    initial_descriptor_cost = sum(1.0 - similarity[u][p[u]] for u in range(n))
    rng = random.Random(args.seed)
    accepted = 0
    checked = 0
    for _ in range(args.attempts):
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        a, b = p[i], p[j]
        if b not in support[i] or a not in support[j]:
            continue
        checked += 1
        affected = set(ga[i]) | set(ga[j]) | {i, j}
        before = after = 0
        for v in affected:
            if v == i or v == j:
                continue
            pv = p[v]
            if v in ga[i]:
                before += int(pv in ha[a])
                after += int(pv in ha[b])
            if v in ga[j]:
                before += int(pv in ha[b])
                after += int(pv in ha[a])
        if j in ga[i]:
            before += int(b in ha[a])
            after += int(a in ha[b])
        delta = after - before
        if delta > 0:
            p[i], p[j] = b, a
            accepted += 1
    final = preserved_edges(ga, ha, p)
    if final < initial:
        raise AssertionError("edge agreement unexpectedly decreased")
    truth = {}
    with args.ground_truth.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            truth[int(row["source_vertex"])] = int(row["target_vertex"])
    accuracy = sum(p[u] == truth[u] for u in range(n)) / n
    final_descriptor_cost = sum(1.0 - similarity[u][p[u]] for u in range(n))
    args.output_assignment.parent.mkdir(parents=True, exist_ok=True)
    with args.output_assignment.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source", "target"])
        writer.writerows(enumerate(p))
    report = {
        "method": "candidate-support-constrained strict-improvement 2-swap hill climb",
        "objective": "maximize preserved undirected source edges in target under the permutation",
        "candidate_support_edges": sum(map(len, support)),
        "descriptor_assignment_cost_initial": initial_descriptor_cost,
        "descriptor_assignment_cost_after_refinement": final_descriptor_cost,
        "descriptor_cost_increase": final_descriptor_cost - initial_descriptor_cost,
        "initial_preserved_edges": initial,
        "final_preserved_edges": final,
        "source_edges": sum(map(len, ga)) // 2,
        "edge_agreement_initial": initial / (sum(map(len, ga)) // 2),
        "edge_agreement_final": final / (sum(map(len, ga)) // 2),
        "attempted_swaps": args.attempts,
        "support_eligible_swaps_checked": checked,
        "accepted_strict_improvement_swaps": accepted,
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
