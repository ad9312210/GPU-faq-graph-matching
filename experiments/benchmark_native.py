"""Repeat native runs and optionally record truth-based quality for each trial."""

import argparse
import csv
import json
import statistics
import subprocess
import time
from pathlib import Path


def load_mapping(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return {int(row["source_vertex"]): int(row["target_vertex"])
                for row in csv.DictReader(stream)}


def evaluate(candidates_path, assignment_path, truth, n):
    candidate_hits = 0
    zero_incoming = [True] * n
    with candidates_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            src, dst = int(row["source"]), int(row["target"])
            candidate_hits += int(truth.get(src) == dst)
            zero_incoming[dst] = False
    assignment = {}
    costs = []
    with assignment_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            src, dst = int(row["source"]), int(row["target"])
            assignment[src] = dst
            if dst >= 0:
                costs.append(float(row["cost"]))
    valid = (len(assignment) == n and set(assignment) == set(range(n))
             and all(0 <= d < n for d in assignment.values())
             and len(set(assignment.values())) == n)
    return {
        "candidate_recall_at_k": candidate_hits / n,
        "zero_incoming_targets": sum(zero_incoming),
        "assignment_valid_one_to_one": valid,
        "final_accuracy": (sum(assignment[i] == truth.get(i) for i in range(n)) / n
                           if valid else None),
        "assignment_cost": sum(costs) if valid else None,
    }


def run_once(executable, source, target, k, mode, directory, index, truth=None,
             write_candidates=True):
    timing_path = directory / f"timings_{mode}_{index}.json"
    assignment_path = directory / f"assignment_{mode}_{index}.csv"
    candidates_path = directory / f"candidates_{mode}_{index}.csv"
    command = [str(executable), "--source", str(source), "--target", str(target),
               "--k", str(k), f"--{mode}"]
    if write_candidates:
        command.extend(("--candidates", str(candidates_path)))
    command.extend(("--output", str(assignment_path), "--timings", str(timing_path)))
    start = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    wall_ms = (time.perf_counter() - start) * 1000.0
    timings = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else {}
    run = {**timings, "wall_clock_ms": wall_ms,
           "candidate_export_included": write_candidates,
           "exit_code": completed.returncode, "stderr": completed.stderr.strip()}
    if truth is not None and candidates_path.is_file() and assignment_path.is_file():
        n = len(truth)
        run.update(evaluate(candidates_path, assignment_path, truth, n))
    # Candidate support is deterministic for a fixed pair and K. Keep one measured
    # artifact set per config; repeated measurements remain fully represented in JSON.
    keep_artifacts = str(index) == "0" and not getattr(run_once, "discard_artifacts", False)
    if not keep_artifacts:
        candidates_path.unlink(missing_ok=True)
        assignment_path.unlink(missing_ok=True)
    timing_path.unlink(missing_ok=True)
    return run


def summarize(runs):
    fields = ("wall_clock_ms", "feature_extraction_ms", "similarity_ms", "topk_selection_ms",
              "candidate_generation_ms", "host_device_transfer_ms", "csr_construction_ms",
              "lapjv_ms", "sinkhorn_ms", "csr_memory_bytes",
              "peak_device_application_allocated_bytes", "peak_device_process_memory_bytes",
              "candidate_recall_at_k", "final_accuracy", "assignment_cost",
              "zero_incoming_targets", "maximum_matching_size")
    summary = {"runs": len(runs)}
    for field in fields:
        values = [float(run[field]) for run in runs if run.get(field) is not None
                  and not (field == "wall_clock_ms" and run.get("candidate_export_included"))]
        summary[field] = ({"median": statistics.median(values), "min": min(values),
                           "max": max(values)} if values else None)
    summary["all_assignments_valid"] = all(run.get("assignment_one_to_one",
                                                    run.get("assignment_valid_one_to_one", False))
                                             for run in runs)
    summary["all_candidate_support_feasible"] = all(
        run.get("maximum_matching_size") is not None
        and run.get("assignment_feasible") for run in runs)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("build-l40s/scgm_cuda"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--mode", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--discard-artifacts", action="store_true",
                        help="Delete candidate and assignment CSVs after evaluating each run")
    args = parser.parse_args()
    for path in (args.executable, args.source, args.target):
        if not path.is_file():
            parser.error(f"file does not exist: {path}")
    if args.ground_truth and not args.ground_truth.is_file():
        parser.error(f"ground truth does not exist: {args.ground_truth}")
    if args.warmup < 0 or args.trials <= 0:
        parser.error("warmup must be nonnegative and trials must be positive")
    truth = load_mapping(args.ground_truth) if args.ground_truth else None
    run_once.discard_artifacts = args.discard_artifacts
    args.output.mkdir(parents=True, exist_ok=True)
    for index in range(args.warmup):
        run_once(args.executable, args.source, args.target, args.k, args.mode,
                 args.output, f"warmup_{index}", write_candidates=False)
    runs = []
    invariant_quality = None
    quality_fields = ("candidate_recall_at_k", "zero_incoming_targets",
                      "assignment_valid_one_to_one", "final_accuracy", "assignment_cost")
    for index in range(args.trials):
        run = run_once(args.executable, args.source, args.target, args.k, args.mode,
                       args.output, index, truth if index == 0 else None,
                       write_candidates=(index == 0))
        if index == 0:
            invariant_quality = {field: run[field] for field in quality_fields if field in run}
        elif invariant_quality:
            # A fixed pair/K yields the same deterministic candidate support each trial.
            run.update(invariant_quality)
            run["quality_reused_from_trial"] = 0
        runs.append(run)
    result = {"mode": args.mode,
              "pipeline_description": ("GPU feature/candidate processing with CPU assignment and Sinkhorn"
                                       if args.mode == "gpu" else "CPU feature/candidate processing, assignment, and Sinkhorn"),
              "source": str(args.source), "target": str(args.target),
              "ground_truth": str(args.ground_truth) if args.ground_truth else None,
              "k": args.k, "warmup": args.warmup, "trials": args.trials,
              "summary": summarize(runs), "runs": runs}
    output_path = args.output / "benchmark.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Benchmark written to: {output_path}")


if __name__ == "__main__":
    main()
