"""Shared post-run quality checks; never used for candidate generation."""
import csv
import math
import hashlib
from collections import deque


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def validate_truth(truth, rows, columns):
    if set(truth) != set(range(rows)):
        raise ValueError("ground truth must cover every source exactly once")
    if any(t < 0 or t >= columns for t in truth.values()):
        raise ValueError("ground-truth target out of range")
    if len(set(truth.values())) != rows:
        raise ValueError("ground truth must be injective")


def load_support(path, rows, columns):
    support = [dict() for _ in range(rows)]
    with path.open(newline="", encoding="utf-8") as stream:
        for record in csv.DictReader(stream):
            s, t = int(record["source"]), int(record["target"])
            score = float(record["similarity"])
            if not (0 <= s < rows and 0 <= t < columns) or not math.isfinite(score):
                raise ValueError("invalid candidate endpoint or nonfinite similarity")
            if t in support[s]:
                raise ValueError("duplicate candidate target within a source")
            support[s][t] = score
    return support


def support_metrics(support, columns):
    """Hopcroft-Karp with iterative DFS; no dense matrix or recursion limit."""
    rows = len(support)
    adjacency = [list(row) for row in support]
    incoming = [0] * columns
    for row in adjacency:
        for t in row:
            if not 0 <= t < columns:
                raise ValueError("candidate target out of range")
            incoming[t] += 1
    left, right = [-1] * rows, [-1] * columns
    matched = 0
    while True:
        distance = [-1] * rows
        queue = deque()
        for s in range(rows):
            if left[s] < 0:
                distance[s] = 0
                queue.append(s)
        shortest = rows + 1
        while queue:
            s = queue.popleft()
            if distance[s] >= shortest:
                continue
            for t in adjacency[s]:
                nxt = right[t]
                if nxt < 0:
                    shortest = distance[s] + 1
                elif distance[nxt] < 0:
                    distance[nxt] = distance[s] + 1
                    queue.append(nxt)
        if shortest == rows + 1:
            break
        cursor = [0] * rows
        for root in range(rows):
            if left[root] >= 0:
                continue
            stack, links = [root], []
            while stack:
                s = stack[-1]
                advanced = False
                while cursor[s] < len(adjacency[s]):
                    t = adjacency[s][cursor[s]]
                    cursor[s] += 1
                    nxt = right[t]
                    if nxt < 0 and distance[s] + 1 == shortest:
                        left[s], right[t] = t, s
                        for index in range(len(links) - 1, -1, -1):
                            parent, column = stack[index], links[index]
                            left[parent], right[column] = column, parent
                        matched += 1
                        stack.clear()
                        advanced = True
                        break
                    if nxt >= 0 and distance[nxt] == distance[s] + 1:
                        stack.append(nxt)
                        links.append(t)
                        advanced = True
                        break
                if not advanced:
                    distance[s] = -1
                    stack.pop()
                    if links:
                        links.pop()
    return {"candidate_edges": sum(map(len, adjacency)),
            "empty_source_rows": sum(not row for row in adjacency),
            "zero_incoming_targets": incoming.count(0),
            "maximum_matching_size": matched, "support_feasible": matched == rows,
            "row_size_min": min(map(len, adjacency), default=0),
            "row_size_max": max(map(len, adjacency), default=0)}


def mapping_metrics(source_edges, target_edges, assignment, truth, rows, columns,
                    support=None):
    validate_truth(truth, rows, columns)
    if any(s < 0 or s >= rows for s in assignment):
        raise ValueError("assignment source out of range")
    valid = {s: t for s, t in assignment.items() if 0 <= t < columns}
    complete = len(valid) == rows and len(set(valid.values())) == rows
    correct = sum(valid.get(s) == truth[s] for s in range(rows))
    preserved = sum(s in valid and t in valid and
                    tuple(sorted((valid[s], valid[t]))) in target_edges
                    for s, t in source_edges)
    on_support = None if support is None else complete and all(
        assignment[s] in support[s] for s in range(rows))
    cost = sum(1.0 - support[s][assignment[s]] for s in range(rows)) if on_support else None
    return {"vertices_source": rows, "vertices_target": columns,
            "num_vertices": rows, "source_edges": len(source_edges),
            "target_edges": len(target_edges), "assigned_vertices": len(valid),
            "targets_in_range": all(0 <= t < columns for t in assignment.values()),
            "one_to_one": complete, "mapping_complete": complete,
            "mapping_accuracy": (correct / rows if rows else 1.0) if complete else None,
            "correct_vertices": correct if complete else None, "partial_correct_vertices": correct,
            "edge_preservation": (preserved / len(source_edges) if source_edges else 1.0) if complete else None,
            "reverse_edge_preservation": (preserved / len(target_edges) if target_edges else 1.0) if complete else None,
            "preserved_edges": preserved if complete else None,
            "assignment_on_candidate_support": on_support,
            "restricted_assignment_feasible": None if support is None else bool(on_support),
            "descriptor_assignment_cost": cost,
            "descriptor_cost_scope": "sum(1 - exported_similarity), float64 audit arithmetic",
            "evaluation_only_ground_truth": True}
