"""Repair a Top-K support by expanding only its alternating conflict region."""
import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np


def read_ranked_prefix(path: Path, n: int, k: int):
    targets = np.full((n, k), -1, dtype=np.int32)
    similarities = np.full((n, k), np.nan, dtype=np.float32)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            source, rank = int(row["source"]), int(row["rank"])
            if source < 0 or source >= n or rank < 0 or rank >= k:
                raise ValueError(f"bad source/rank in {path}: {source}/{rank}")
            targets[source, rank] = int(row["target"])
            similarities[source, rank] = np.float32(row["similarity"])
    if np.any(targets < 0) or np.any(~np.isfinite(similarities)):
        raise ValueError(f"incomplete Top-{k} candidate file: {path}")
    return targets, similarities


def write_support(path: Path, row_targets, row_scores):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("source", "rank", "target", "similarity"))
        for source, (targets, scores) in enumerate(zip(row_targets, row_scores)):
            for rank, (target, score) in enumerate(zip(targets, scores)):
                writer.writerow((source, rank, int(target), format(float(score), ".9g")))


def read_matching(path: Path, n: int):
    matching = np.full(n, -1, dtype=np.int32)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            matching[int(row["source"])] = int(row["target"])
    return matching


def matching_conflict_region(row_targets, source_to_target):
    """Alternating reachability from unmatched source vertices."""
    n = len(row_targets)
    target_to_source = np.full(n, -1, dtype=np.int32)
    for source, target in enumerate(source_to_target):
        if target >= 0:
            target_to_source[target] = source
    unmatched = np.flatnonzero(source_to_target < 0)
    seen_sources = np.zeros(n, dtype=bool)
    seen_targets = np.zeros(n, dtype=bool)
    seen_sources[unmatched] = True
    queue = unmatched.tolist()
    head = 0
    while head < len(queue):
        source = queue[head]
        head += 1
        for target in row_targets[source]:
            if source_to_target[source] == target or seen_targets[target]:
                continue
            seen_targets[target] = True
            matched_source = int(target_to_source[target])
            if matched_source >= 0 and not seen_sources[matched_source]:
                seen_sources[matched_source] = True
                queue.append(matched_source)
    return np.flatnonzero(seen_sources), unmatched


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("build-l40s/scgm_cuda"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--ranked-dir", type=Path, required=True,
                        help="Directory containing support_checks/K064,candidates.csv, etc.")
    parser.add_argument("--initial-k", type=int, default=64)
    parser.add_argument("--k-limit", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.initial_k not in (16, 32, 64, 128, 256) or args.k_limit not in (128, 256, 512, 1024):
        parser.error("supported initial K values are 16..256 and limits are 128..1024")
    if not (args.executable.is_file() and args.source.is_file() and args.target.is_file()):
        parser.error("executable and graph files must exist")
    n = int(args.source.open(encoding="utf-8").readline().split()[0])
    target_n = int(args.target.open(encoding="utf-8").readline().split()[0])
    if n != target_n:
        parser.error("current assignment workflow requires equal graph sizes")
    ks = [args.initial_k]
    while ks[-1] < args.k_limit:
        ks.append(min(ks[-1] * 2, args.k_limit))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    checks_dir = output / "support_checks"
    checks_dir.mkdir(exist_ok=True)
    # Store each row as a ranked prefix. Expansion only lengthens selected
    # prefixes, so support remains monotone without an n-by-n matrix.
    current_k = args.initial_k
    ranked_targets, ranked_scores = read_ranked_prefix(
        args.ranked_dir / f"K{current_k:03d}" / "candidates.csv", n, current_k)
    if any(np.unique(row).size != current_k for row in ranked_targets):
        raise ValueError("ranked candidate rows must not contain duplicate targets")
    # Own the initial row arrays so the dense n-by-K loader buffer can be freed.
    row_targets = [ranked_targets[source].copy() for source in range(n)]
    row_scores = [ranked_scores[source].copy() for source in range(n)]
    del ranked_targets, ranked_scores
    history = []
    repair_start = time.perf_counter()
    total_added = 0
    final_paths = None

    for index, k in enumerate(ks):
        check_dir = checks_dir / f"K{k:03d}"
        check_dir.mkdir(exist_ok=True)
        candidates_path = check_dir / "candidates.csv"
        assignment_path = check_dir / "assignment.csv"
        matching_path = check_dir / "maximum_matching.csv"
        timings_path = check_dir / "native_timings.json"
        write_support(candidates_path, row_targets, row_scores)
        command = [str(args.executable), "--source", str(args.source), "--target", str(args.target),
                   "--cpu", "--k", str(k), "--candidate-input", str(candidates_path),
                   "--matching-output", str(matching_path), "--output", str(assignment_path),
                   "--timings", str(timings_path)]
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        matching_ms = (time.perf_counter() - started) * 1000.0
        if not matching_path.is_file() or not timings_path.is_file() or timings_path.stat().st_size == 0:
            raise RuntimeError(f"native matching failed at K={k}: {completed.stderr}\n{completed.stdout}")
        native = json.loads(timings_path.read_text(encoding="utf-8"))
        source_to_target = read_matching(matching_path, n)
        matched = int(np.count_nonzero(source_to_target >= 0))
        if matched != int(native["maximum_matching_size"]):
            raise RuntimeError(f"native matching output disagrees with its diagnostic at K={k}")
        edges = sum(len(row) for row in row_targets)
        retained_targets = np.concatenate(row_targets)
        incoming = np.bincount(retained_targets, minlength=target_n)
        checkpoint = {"k": k, "candidate_edges": edges,
                      "zero_incoming_targets": int(np.count_nonzero(incoming == 0)),
                      "maximum_matching_size": matched,
                      "feasible": matched == n,
                      "native_matching_and_assignment_ms": matching_ms,
                      "repair_edges_added_this_round": 0,
                      "conflict_sources_expanded": 0,
                      "candidate_file": str(candidates_path),
                      "matching_file": str(matching_path),
                      "assignment_file": str(assignment_path)}
        history.append(checkpoint)
        final_paths = (candidates_path, assignment_path, timings_path)
        if matched == n or index == len(ks) - 1:
            break

        conflict_sources, unmatched_sources = matching_conflict_region(
            row_targets, source_to_target)
        checkpoint["conflict_sources_expanded"] = int(conflict_sources.size)
        next_k = ks[index + 1]
        prefix_targets, prefix_scores = read_ranked_prefix(
            args.ranked_dir / f"K{next_k:03d}" / "candidates.csv", n, next_k)
        if any(np.unique(row).size != next_k for row in prefix_targets):
            raise ValueError("ranked candidate rows must not contain duplicate targets")
        added = 0
        for source in conflict_sources:
            source = int(source)
            old_targets = row_targets[source]
            new_mask = ~np.isin(prefix_targets[source], old_targets, assume_unique=True)
            if np.any(new_mask):
                row_targets[source] = np.concatenate((old_targets, prefix_targets[source, new_mask]))
                row_scores[source] = np.concatenate((row_scores[source], prefix_scores[source, new_mask]))
                added += int(np.count_nonzero(new_mask))
        del prefix_targets, prefix_scores
        checkpoint["unmatched_sources_before_expansion"] = int(unmatched_sources.size)
        checkpoint["conflict_region_sources"] = int(conflict_sources.size)
        checkpoint["repair_edges_added_this_round"] = added
        checkpoint["next_k_prefix"] = next_k
        total_added += added

    repair_ms = (time.perf_counter() - repair_start) * 1000.0
    final = history[-1]
    report = {"source_graph": str(args.source), "target_graph": str(args.target),
              "source_sha256": sha256(args.source), "target_sha256": sha256(args.target),
              "vertices": n, "initial_k": args.initial_k, "k_limit": args.k_limit,
              "repair_policy": "Expand feature-ranked candidates only for sources reachable from currently unmatched sources in the alternating conflict graph; retain every previously accepted edge.",
              "ground_truth_used_for_candidate_generation_or_repair": False,
              "repair_rounds": max(0, len(history) - 1),
              "initial_candidate_edges": history[0]["candidate_edges"],
              "final_candidate_edges": final["candidate_edges"],
              "edges_added_by_repair": final["candidate_edges"] - history[0]["candidate_edges"],
              "repair_time_ms_including_matching_checks_and_writes": repair_ms,
              "zero_incoming_targets_initial": history[0]["zero_incoming_targets"],
              "zero_incoming_targets_final": final["zero_incoming_targets"],
              "maximum_matching_initial": history[0]["maximum_matching_size"],
              "maximum_matching_final": final["maximum_matching_size"],
              "feasible_after_repair": final["feasible"],
              "assignment_status": "success" if final["feasible"] and completed.returncode == 0 else "infeasible_or_solver_failure",
              "history": history,
              "final_candidates": str(final_paths[0]),
              "final_assignment": str(final_paths[1]),
              "final_native_timings": str(final_paths[2])}
    report_path = output / "repair_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not final["feasible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
