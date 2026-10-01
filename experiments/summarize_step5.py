"""Build Step 5 summary CSV and exact planned-configuration manifest."""

import csv
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "step5"
DATASET_DIRS = {"G01": "G01-ca-GrQc", "G02": "G02-email-Enron",
                "G03": "G03-com-DBLP", "S16": "S16-Graph500"}
FIELDS = [
    "run_name", "experiment", "dataset", "pair_seed", "noise_percent", "k",
    "method", "mode", "vertices", "trials", "warmup", "median_wall_ms",
    "median_feature_ms", "median_similarity_ms", "median_topk_ms",
    "median_transfer_ms", "median_csr_ms", "median_assignment_ms",
    "median_sinkhorn_ms", "median_csr_memory_bytes",
    "median_gpu_application_peak_bytes", "median_gpu_process_peak_bytes",
    "median_candidate_recall", "median_final_accuracy", "median_assignment_cost",
    "median_zero_incoming_targets", "median_maximum_matching_size",
    "support_feasible", "assignment_valid", "pipeline_description",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def median(summary, field):
    item = summary.get(field)
    return item.get("median") if isinstance(item, dict) else None


def attach_metadata():
    env_path = RESULTS / "environment.json"
    if not env_path.is_file():
        return
    for path in RESULTS.glob("**/benchmark.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["environment"] = os.path.relpath(env_path, path.parent)
        legacy_small = path.parts[-5] in ("G01", "G02")
        for index, run in enumerate(data.get("runs", [])):
            if "candidate_export_included" not in run:
                run["candidate_export_included"] = True if legacy_small else index == 0
        data["input_sha256"] = {
            key: sha256(ROOT / data[key])
            for key in ("source", "target", "ground_truth")
            if data.get(key) and (ROOT / data[key]).is_file()
        }
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def planned_configs():
    specs = []
    for dataset in ("G01", "G02", "G03", "S16"):
        for seed in (11, 22, 33):
            for k in (16, 64, 128):
                specs.append(("E1", dataset, seed, 0, k, "native_gpu_hybrid"))
    for dataset in ("G01", "G03"):
        for seed in (11, 22, 33):
            for noise in (5, 10, 20, 30):
                specs.append(("E2", dataset, seed, noise, 64, "native_gpu_hybrid"))
    for dataset in ("G01", "G03"):
        for seed in (11, 22, 33):
            for method in ("native_cpu_sparse_lap", "standalone_gpu_lap"):
                specs.append(("E3", dataset, seed, 0, 64, method))
    return specs


def main():
    attach_metadata()
    rows = []
    run_paths = {}
    for path in sorted(RESULTS.glob("**/benchmark.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        parts = path.relative_to(RESULTS).parts
        if len(parts) < 6:
            continue
        experiment, dataset, seed_part, noise_part, config = parts[:5]
        seed = int(seed_part.replace("seed_", ""))
        noise = int(noise_part.replace("noise_", ""))
        k = int(re.search(r"K(\d+)", config).group(1))
        method = "native_gpu_hybrid"
        manifest_path = ROOT / "data" / DATASET_DIRS[dataset] / "dataset_manifest.json"
        dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = data["summary"]
        n = int(dataset_manifest["num_vertices"])
        matching = median(summary, "maximum_matching_size")
        run_name = f"{dataset}__seed{seed}__noise{noise:02d}__{method}__K{k:03d}"
        row = {
            "run_name": run_name, "experiment": experiment, "dataset": dataset,
            "pair_seed": seed, "noise_percent": noise, "k": k, "method": method,
            "mode": data["mode"], "vertices": n, "trials": data["trials"],
            "warmup": data["warmup"], "median_wall_ms": median(summary, "wall_clock_ms"),
            "median_feature_ms": median(summary, "feature_extraction_ms"),
            "median_similarity_ms": median(summary, "similarity_ms"),
            "median_topk_ms": median(summary, "topk_selection_ms"),
            "median_transfer_ms": median(summary, "host_device_transfer_ms"),
            "median_csr_ms": median(summary, "csr_construction_ms"),
            "median_assignment_ms": median(summary, "lapjv_ms"),
            "median_sinkhorn_ms": median(summary, "sinkhorn_ms"),
            "median_csr_memory_bytes": median(summary, "csr_memory_bytes"),
            "median_gpu_application_peak_bytes": median(summary, "peak_device_application_allocated_bytes"),
            "median_gpu_process_peak_bytes": median(summary, "peak_device_process_memory_bytes"),
            "median_candidate_recall": median(summary, "candidate_recall_at_k"),
            "median_final_accuracy": median(summary, "final_accuracy"),
            "median_assignment_cost": median(summary, "assignment_cost"),
            "median_zero_incoming_targets": median(summary, "zero_incoming_targets"),
            "median_maximum_matching_size": matching,
            "support_feasible": bool(matching == n) if matching is not None else None,
            "assignment_valid": summary.get("all_assignments_valid"),
            "pipeline_description": data["pipeline_description"],
        }
        rows.append(row)
        run_paths[(experiment, dataset, seed, noise, k, method)] = path

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    rows_by_key = {(r["experiment"], r["dataset"], r["pair_seed"],
                    r["noise_percent"], r["k"], r["method"]): r for r in rows}
    configs = []
    for experiment, dataset, seed, noise, k, method in planned_configs():
        key = (experiment, dataset, seed, noise, k, method)
        row = rows_by_key.get(key)
        run_name = f"{dataset}__seed{seed}__noise{noise:02d}__{method}__K{k:03d}"
        item = {"run_name": run_name, "experiment": experiment, "dataset": dataset,
                "seed": seed, "noise_percent": noise, "k": k, "method": method}
        if row:
            full = row["trials"] == 5 and row["warmup"] >= 1
            if not row["support_feasible"]:
                status = "completed_infeasible_support" if full else "pilot_infeasible_support"
            elif row["assignment_valid"]:
                status = "completed_success" if full else "pilot_success"
            else:
                status = "completed_solver_failure" if full else "pilot_solver_failure"
            item.update({"status": status, "maximum_matching_size": row["median_maximum_matching_size"],
                         "candidate_recall": row["median_candidate_recall"],
                         "benchmark_json": os.path.relpath(run_paths[key], RESULTS)})
        elif experiment == "E1" and dataset == "S16":
            item["status"] = "blocked_dataset_pair_missing"
        elif experiment == "E3" and method == "standalone_gpu_lap":
            item["status"] = "blocked_no_standalone_gpu_lap"
        elif experiment == "E3":
            item["status"] = "blocked_until_feasible_identical_support"
        elif dataset == "G03":
            item["status"] = "pending_after_feasibility_gate"
        else:
            item["status"] = "pending"
        configs.append(item)

    counts = Counter(item["status"] for item in configs)
    env = json.loads((RESULTS / "environment.json").read_text()) if (RESULTS / "environment.json").is_file() else None
    manifest = {
        "planned_configurations": len(configs),
        "completed_full_protocol_configurations": sum(1 for c in configs if c["status"].startswith("completed_")),
        "pilot_only_configurations": sum(1 for c in configs if c["status"].startswith("pilot_")),
        "status_counts": dict(counts),
        "repetition_protocol": "one warm-up plus five timed trials for full-protocol configurations; pilot exceptions are marked per row",
        "mode": "GPU feature/candidate processing with CPU assignment and Sinkhorn",
        "noise_policy": "after permutation, delete floor(p*|E(H)|) undirected edges uniformly, then insert the same count of uniformly sampled nonedges",
        "results_csv": "summary.csv",
        "hardware_and_code_metadata": "environment.json; input graph/truth SHA-256 values are in each benchmark.json",
        "configurations": configs,
        "experiment_environment": env,
    }
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} observed configurations; {len(configs)} planned configurations total")
    print(f"Status counts: {dict(counts)}")
    print(f"CSV: {RESULTS / 'summary.csv'}")
    print(f"Manifest: {RESULTS / 'manifest.json'}")


if __name__ == "__main__":
    main()
