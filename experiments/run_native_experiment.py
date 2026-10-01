"""Run one native matcher, expand infeasible candidate support, and save results."""

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from analyze_feature_candidates import audit as audit_features
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def run_name(dataset, pair_seed, noise_percent, method, k):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", dataset):
        raise ValueError("dataset label may contain only letters, digits, '_' and '-'")
    if not re.fullmatch(r"[a-z0-9_]+", method):
        raise ValueError("method label must use lowercase letters, digits, and '_'")
    if not 0 <= noise_percent <= 100 or k < 0 or pair_seed < 0:
        raise ValueError("pair seed, noise percentage, or K is out of range")
    return f"{dataset}__seed{pair_seed}__noise{noise_percent:02d}__{method}__K{k:03d}"


def graph_size(path):
    with path.open(encoding="utf-8") as graph_file:
        fields = graph_file.readline().split()
    if len(fields) != 2:
        raise ValueError(f"invalid graph header in {path}")
    return {"vertices": int(fields[0]), "edges": int(fields[1])}


def read_mapping(path, source_key, target_key):
    with path.open(newline="", encoding="utf-8") as input_file:
        return {int(row[source_key]): int(row[target_key]) for row in csv.DictReader(input_file)}


def read_candidate_rows(path, num_sources, num_targets):
    rows = [[] for _ in range(num_sources)]
    with path.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            source, target = int(row["source"]), int(row["target"])
            if not (0 <= source < num_sources and 0 <= target < num_targets):
                raise ValueError(f"candidate endpoint out of range in {path}: {source}, {target}")
            rows[source].append(target)
    return rows


def candidate_diagnostics(path, num_sources, num_targets, truth, coverage_path,
                          native_matching_size=None, verify_with_scipy=True):
    rows = read_candidate_rows(path, num_sources, num_targets)
    incoming = [0] * num_targets
    found_truth = 0
    for source, targets in enumerate(rows):
        for target in targets:
            incoming[target] += 1
        found_truth += int(truth.get(source) in targets)

    indptr = np.zeros(num_sources + 1, dtype=np.int64)
    for source, targets in enumerate(rows):
        indptr[source + 1] = indptr[source] + len(targets)
    indices = np.fromiter((target for targets in rows for target in targets),
                          dtype=np.int32, count=int(indptr[-1]))
    if verify_with_scipy or native_matching_size is None:
        graph = csr_matrix((np.ones(len(indices), dtype=np.int8), indices, indptr),
                           shape=(num_sources, num_targets))
        matching = maximum_bipartite_matching(graph, perm_type="column")
        maximum_size = int(np.count_nonzero(matching >= 0))
        matching_check = "independent_scipy"
    else:
        maximum_size = int(native_matching_size)
        matching_check = "native_hopcroft_karp"

    histogram = {}
    for count in incoming:
        histogram[str(count)] = histogram.get(str(count), 0) + 1
    with coverage_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("target", "incoming_candidate_count"))
        writer.writerows(enumerate(incoming))

    return {
        "candidate_recall": found_truth / len(truth) if truth else None,
        "zero_incoming_targets": sum(count == 0 for count in incoming),
        "maximum_matching_size": maximum_size,
        "matching_check": matching_check,
        "feasible_candidate_support": maximum_size == num_sources,
        "incoming_count_histogram": histogram,
        "candidate_edge_count": int(indptr[-1]),
        "candidate_rows": rows,
    }


def read_assignment(path, n):
    assignment = {}
    costs = []
    with path.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            source, target = int(row["source"]), int(row["target"])
            assignment[source] = target
            if target >= 0:
                costs.append(float(row["cost"]))
    valid = (
        len(assignment) == n
        and set(assignment) == set(range(n))
        and all(0 <= target < n for target in assignment.values())
        and len(set(assignment.values())) == n
    )
    return assignment, valid, sum(costs) if valid else None


def environment_metadata():
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    worktree = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    metadata = {
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_worktree_dirty": bool(worktree.stdout.strip()) if worktree.returncode == 0 else None,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "gpu": None,
    }
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    )
    if gpu.returncode == 0:
        metadata["gpu"] = gpu.stdout.strip().splitlines()
    return metadata


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_delta(initial_rows, final_rows):
    added = removed = 0
    for initial, final in zip(initial_rows, final_rows):
        initial_set, final_set = set(initial), set(final)
        added += len(final_set - initial_set)
        removed += len(initial_set - final_set)
    return added, removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("build-l40s/scgm_cuda"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pair-seed", type=int, required=True)
    parser.add_argument("--noise-percent", type=int, required=True)
    parser.add_argument("--method", required=True, help="Run label, for example native_gpu_hybrid")
    parser.add_argument("--mode", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--repair-max-k", type=int, default=512,
                        help="Largest feature-ranked Top-K support to try")
    parser.add_argument("--no-support-repair", action="store_true",
                        help="Diagnose only at --k without expanding candidate support")
    parser.add_argument("--skip-feature-audit", action="store_true",
                        help="Skip evaluation-only feature/tie recomputation")
    parser.add_argument("--skip-independent-matching-check", action="store_true",
                        help="Use the native Hopcroft-Karp result instead of recomputing it with SciPy")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.executable, args.source, args.target, args.ground_truth):
        if not path.is_file():
            parser.error(f"file does not exist: {path}")
    pair_parents = {path.resolve().parent for path in (args.source, args.target, args.ground_truth)}
    if len(pair_parents) != 1:
        parser.error("source, target, and ground-truth files must come from the same saved pair")
    if args.repair_max_k < args.k:
        parser.error("--repair-max-k must be at least --k")

    name = run_name(args.dataset, args.pair_seed, args.noise_percent, args.method, args.k)
    pair_manifest_path = args.target.parent / "manifest.json"
    pair_manifest = None
    if pair_manifest_path.is_file():
        pair_manifest = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
        manifest_seed = pair_manifest.get("pair_seed", pair_manifest.get("seed"))
        manifest_noise = pair_manifest.get("noise_percent")
        if manifest_noise is None and "edge_noise" in pair_manifest:
            manifest_noise = round(100 * float(pair_manifest["edge_noise"]))
        if manifest_seed is not None and int(manifest_seed) != args.pair_seed:
            parser.error("--pair-seed does not match the pair manifest")
        if manifest_noise is not None and int(manifest_noise) != args.noise_percent:
            parser.error("--noise-percent does not match the pair manifest")
        if pair_manifest.get("dataset") not in (None, args.dataset):
            parser.error("--dataset does not match the pair manifest")

    n_source = graph_size(args.source)["vertices"]
    n_target = graph_size(args.target)["vertices"]
    truth = read_mapping(args.ground_truth, "source_vertex", "target_vertex")
    if len(truth) != n_source or n_source != n_target:
        parser.error("ground truth and graph sizes do not agree")

    run_dir = args.output_dir / name
    checks_dir = run_dir / "support_checks"
    checks_dir.mkdir(parents=True, exist_ok=True)
    history = []
    attempt_wall_ms = []
    support_diagnostic_ms = []
    feature_audit_ms = 0.0
    final_rows = None
    final_native = {}
    final_attempt = None
    initial_rows = None
    initial_native = {}
    current_k = args.k
    maximum_k = min(args.repair_max_k, n_target)
    attempt_index = 0

    while True:
        attempt_dir = checks_dir / f"K{current_k:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        assignment_path = attempt_dir / "assignment.csv"
        candidates_path = attempt_dir / "candidates.csv"
        timings_path = attempt_dir / "native_timings.json"
        command = [
            str(args.executable), "--source", str(args.source), "--target", str(args.target),
            "--k", str(current_k), f"--{args.mode}", "--candidates", str(candidates_path),
            "--output", str(assignment_path), "--timings", str(timings_path),
        ]
        start = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        wall_ms = (time.perf_counter() - start) * 1000.0
        attempt_wall_ms.append(wall_ms)
        native = json.loads(timings_path.read_text(encoding="utf-8")) if timings_path.exists() else {}
        if not candidates_path.exists():
            raise RuntimeError(f"native run did not write candidates at K={current_k}: {completed.stderr}")
        coverage_path = attempt_dir / "target_candidate_coverage.csv"
        support_start = time.perf_counter()
        support = candidate_diagnostics(
            candidates_path, n_source, n_target, truth, coverage_path,
            native_matching_size=native.get("maximum_matching_size"),
            verify_with_scipy=not args.skip_independent_matching_check,
        )
        support_elapsed_ms = (time.perf_counter() - support_start) * 1000.0
        support_diagnostic_ms.append(support_elapsed_ms)
        feature_diagnostics_path = attempt_dir / "feature_diagnostics.json"
        if attempt_index == 0 and not args.skip_feature_audit:
            audit_start = time.perf_counter()
            audit_features(args.source, args.target, candidates_path, args.ground_truth,
                           feature_diagnostics_path, current_k, 1e-6, print_result=False)
            feature_audit_ms = (time.perf_counter() - audit_start) * 1000.0
        native_matching = native.get("maximum_matching_size")
        if native_matching is not None and int(native_matching) != support["maximum_matching_size"]:
            raise RuntimeError(f"native/Python maximum-matching mismatch at K={current_k}")

        checkpoint = {
            "k": current_k,
            "candidate_recall": support["candidate_recall"],
            "zero_incoming_targets": support["zero_incoming_targets"],
            "maximum_matching_size": support["maximum_matching_size"],
            "feasible_candidate_support": support["feasible_candidate_support"],
            "matching_check": support["matching_check"],
            "candidate_edge_count": support["candidate_edge_count"],
            "incoming_count_histogram": support["incoming_count_histogram"],
            "native_wall_time_ms": wall_ms,
            "support_diagnostic_time_ms": support_elapsed_ms,
            "feature_audit_time_ms": feature_audit_ms if attempt_index == 0 else 0.0,
            "stage_timings_ms": {
                key: native.get(key) for key in (
                    "feature_extraction_ms", "similarity_ms", "topk_selection_ms",
                    "candidate_generation_ms", "host_device_transfer_ms",
                    "csr_construction_ms", "lapjv_ms", "sinkhorn_ms",
                )
            },
            "native_assignment_status": native.get("assignment_status"),
            "native_exit_code": completed.returncode,
            "coverage_file": str(coverage_path),
            "candidate_file": str(candidates_path),
            "feature_diagnostics_file": str(feature_diagnostics_path) if attempt_index == 0 and not args.skip_feature_audit else None,
        }
        history.append(checkpoint)
        if attempt_index == 0:
            initial_rows = support["candidate_rows"]
            initial_native = native
        final_rows = support["candidate_rows"]
        final_native = native
        final_attempt = (attempt_dir, assignment_path, candidates_path, timings_path, completed)

        support_is_perfect = support["maximum_matching_size"] == n_source
        if support_is_perfect:
            # If the support is perfect but the solver failed, classify that distinctly.
            break
        if args.no_support_repair or current_k >= maximum_k or completed.returncode not in (0, 2):
            break
        next_k = min(max(current_k + 1, current_k * 2), maximum_k)
        if next_k <= current_k:
            break
        current_k = next_k
        attempt_index += 1

    assert final_attempt is not None
    attempt_dir, assignment_path, candidates_path, timings_path, completed = final_attempt
    final_assignment = {}
    assignment_valid = False
    assignment_cost = None
    if assignment_path.exists():
        final_assignment, assignment_valid, assignment_cost = read_assignment(assignment_path, n_source)
    final_accuracy = None
    partial_accuracy = None
    if final_assignment:
        assigned = [source for source, target in final_assignment.items() if target >= 0]
        if assigned:
            partial_accuracy = sum(final_assignment[source] == truth.get(source) for source in assigned) / len(assigned)
        if assignment_valid:
            final_accuracy = sum(final_assignment.get(i) == truth.get(i) for i in range(n_source)) / n_source

    support_before_repair = history[0]["maximum_matching_size"] == n_source
    support_after_repair = history[-1]["maximum_matching_size"] == n_source
    added_edges, removed_edges = _candidate_delta(initial_rows or [], final_rows or [])
    native_assignment_status = final_native.get("assignment_status")
    if not support_after_repair:
        assignment_status = "infeasible_candidate_support"
    elif final_native.get("assignment_feasible") and assignment_valid and completed.returncode == 0:
        assignment_status = "success"
    else:
        assignment_status = "solver_failure"
    success = assignment_status == "success"

    # Preserve convenient top-level artifact paths while retaining every K checkpoint.
    final_files = {
        "assignment.csv": assignment_path,
        "candidates.csv": candidates_path,
        "native_timings.json": timings_path,
    }
    for filename, source_path in final_files.items():
        if source_path.exists():
            shutil.copy2(source_path, run_dir / filename)
    final_native.update({
        "support_diagnostics_by_k": history,
        "feasible_before_repair": support_before_repair,
        "repair_rounds": max(0, len(history) - 1),
        "edges_added_by_repair": added_edges,
        "edges_removed_by_topk_churn": removed_edges,
        "feasible_after_repair": support_after_repair,
        "effective_k": history[-1]["k"],
        "repair_policy": "rerun feature-ranked Top-K with doubled K; no ground truth passed to matcher",
        "assignment_status": assignment_status,
    })
    (run_dir / "native_timings.json").write_text(json.dumps(final_native, indent=2) + "\n", encoding="utf-8")

    native_wall_time_ms = sum(attempt_wall_ms)
    support_diagnostic_time_ms = sum(support_diagnostic_ms)
    total_time_ms = native_wall_time_ms + support_diagnostic_time_ms
    aggregate_stage_timings = {}
    for stage in (
        "feature_extraction_ms", "similarity_ms", "topk_selection_ms",
        "candidate_generation_ms", "host_device_transfer_ms",
        "csr_construction_ms", "lapjv_ms", "sinkhorn_ms",
    ):
        aggregate_stage_timings[stage] = sum(
            float(check["stage_timings_ms"].get(stage) or 0.0) for check in history
        )
    peak_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak_rss_bytes = int(peak_rss * (1024 if platform.system() == "Linux" else 1))
    failure_reason = None
    if not success:
        failure_reason = completed.stderr.strip() or assignment_status

    result = {
        "run_name": name,
        "dataset": args.dataset,
        "pair_seed": args.pair_seed,
        "noise_percent": args.noise_percent,
        "method": args.method,
        "mode": args.mode,
        "pipeline_description": final_native.get("pipeline_description"),
        "k": args.k,
        "normalization_mode": final_native.get("normalization_mode", "joint_graph_pair"),
        "effective_k": history[-1]["k"],
        "source_graph": str(args.source),
        "target_graph": str(args.target),
        "ground_truth_file": str(args.ground_truth),
        "source_vertices": n_source,
        "target_vertices": n_target,
        "source_edges": graph_size(args.source)["edges"],
        "target_edges": graph_size(args.target)["edges"],
        "stage_timings_ms": aggregate_stage_timings,
        "final_attempt_stage_timings_ms": {
            key: final_native.get(key) for key in (
                "feature_extraction_ms", "similarity_ms", "topk_selection_ms",
                "candidate_generation_ms", "host_device_transfer_ms",
                "csr_construction_ms", "lapjv_ms", "sinkhorn_ms",
            )
        },
        "native_wall_time_ms": native_wall_time_ms,
        "support_diagnostic_time_ms": support_diagnostic_time_ms,
        "evaluation_feature_audit_ms": feature_audit_ms,
        "total_time_ms": total_time_ms,
        "support_repair_time_ms": sum(attempt_wall_ms[1:]) + sum(support_diagnostic_ms[1:]),
        "peak_host_child_rss_bytes": peak_rss_bytes,
        "peak_gpu_memory_bytes": final_native.get("peak_device_process_memory_bytes"),
        "peak_gpu_memory_sampling_interval_ms": final_native.get("device_memory_sampling_interval_ms"),
        "peak_gpu_application_allocated_bytes": final_native.get("peak_device_application_allocated_bytes"),
        "candidate_recall_at_k": history[0]["candidate_recall"],
        "candidate_recall_after_repair": history[-1]["candidate_recall"],
        "partial_mapping_accuracy": partial_accuracy,
        "final_accuracy": final_accuracy,
        "assignment_cost": assignment_cost,
        "zero_incoming_targets": history[0]["zero_incoming_targets"],
        "zero_incoming_targets_after_repair": history[-1]["zero_incoming_targets"],
        "maximum_matching_size": history[0]["maximum_matching_size"],
        "maximum_matching_size_after_repair": history[-1]["maximum_matching_size"],
        "feasible_before_repair": support_before_repair,
        "repair_rounds": max(0, len(history) - 1),
        "edges_added_by_repair": added_edges,
        "feasible_after_repair": support_after_repair,
        "support_diagnostics_by_k": history,
        "native_assignment_status": native_assignment_status,
        "assignment_status": assignment_status,
        "assignment_feasible": bool(final_native.get("assignment_feasible")),
        "assignment_one_to_one": assignment_valid,
        "status": "success" if success else "failure",
        "exit_code": completed.returncode,
        "failure_reason": failure_reason,
        "code_and_hardware": environment_metadata(),
        "code_artifacts_sha256": {
            "native_executable": sha256_file(args.executable),
            "run_wrapper": sha256_file(Path(__file__).resolve()),
            "pair_generator": sha256_file(Path(__file__).with_name("generate_paired_graphs.py")),
            "dataset_variant_generator": sha256_file(Path(__file__).with_name("generate_dataset_variants.py")),
        },
        "pair_manifest": pair_manifest,
        "feature_diagnostics_file": (
            str(checks_dir / f"K{args.k:03d}" / "feature_diagnostics.json")
            if (checks_dir / f"K{args.k:03d}" / "feature_diagnostics.json").is_file() else None
        ),
    }
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Result row: {result_path}")


if __name__ == "__main__":
    main()
