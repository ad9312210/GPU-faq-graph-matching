# SCGM Graph Matching

SCGM is a native C++17/CUDA implementation and Python experiment harness for
sparse candidate graph matching. It provides CPU references and CUDA feature,
candidate-selection, sparse-cost, assignment, and Sinkhorn implementations.
The GPU assignment algorithm is primal-dual augmenting-path matching; the CPU
reference retains the historical `solve_lapjv` interface name. Select backends
explicitly and inspect saved diagnostics to see which implementations executed.

This repository is a research artifact, not a complete reproduction of the
unpublished paper draft. Results and capabilities should be interpreted using
the implementation that is actually present here.

## Capabilities

- Six structural features: degree, clustering coefficient, average neighbor
  degree, neighbor-degree standard deviation, triangle participation, and
  degree squared.
- CPU and CUDA feature extraction and Top-K candidate generation.
- CPU and GPU sparse CSR cost construction, with a GPU reverse column index.
- CPU assignment and GPU primal-dual assignment with host optimality certificates.
- Optional CPU and GPU structural-index candidates, with recorded policy and exhaustive fallback.
- CPU and GPU sparse Sinkhorn diagnostics over square candidate support.
- An opt-in device-resident single-GPU feature/candidate/cost pipeline.
- Native CPU alternating-conflict support repair with explicit limits.
- Independent assignment components with explicit warp/block/cooperative dispatch.
- Multi-GPU source scoring and assignment of whole independent components.
- Native unit tests, CPU/GPU parity tests, paired-graph generation, and
  experiment and benchmark scripts.

Defaults preserve the CPU assignment/Sinkhorn compatibility path. GPU stages,
component scheduling, repair, and multi-GPU execution require explicit options.
GPU certification and support analysis still use host handoffs. Repair currently
supports CPU exhaustive standard-feature generation only. Scheduling uses
explicit choices; automatic experimental thresholds are not implemented.
Multi-GPU execution uses host staging and never splits an assignment component.
Implementation and a successful example do not establish scaling guarantees.

For direct working commands and development guidance, see
[Checking the complete pipeline](#checking-the-complete-pipeline) below.

## Supported structural-index result

The [CPU pilot findings](docs/structural_index_findings.md) document one noiseless
ca-GrQc seed-11 pair at K64: 65.27% fewer scored pairs, 2.10x median candidate-stage
ratio including index setup, and unchanged 89.73% candidate recall. Both supports
matched only 4,703/5,241 sources. Final accuracy and complete-pipeline speedup are
unavailable; separate host-RSS profiles showed no measured reduction.

GPU indexed scoring is now integrated, including the resident pipeline. The
historical CPU pilot above provides no GPU performance evidence. See the findings for the exact
three-repeat reproduction command and provenance limitations; broader K/seed
commands later in this README are proposed additional experiments, not completed
results. The [design](docs/structural_index_design.md) distinguishes completed and
pending milestones.

## Repository Layout

```text
include/       Public C++ headers
src/           C++ graph, feature, candidate, CSR, assignment, and evaluation code
cuda/          CUDA kernels and GPU matching implementation
apps/          Native command-line applications
tests/         Native unit and parity tests
experiments/   Dataset preparation, baselines, evaluation, and benchmarking
datasets/      Optional raw input datasets; normally downloaded locally
```

Generated build trees, experiment outputs, logs, reports, diagrams, figures,
local environments, and the unpublished PDF are ignored by `.gitignore`.
Source code, tests, scripts, configuration, and small metadata files remain
eligible for commits.

## Requirements

Native builds require:

- CMake 3.21 or newer
- A C++17 compiler
- CUDA Toolkit and `nvcc`
- An NVIDIA GPU for CUDA execution

Python experiments are optional. Install the dependencies required by the
selected script, typically NumPy, SciPy, NetworkX, Joblib, Numba, or CuPy.

## Build And Test

For an NVIDIA L40S with CUDA 12:

```bash
cmake --preset release-l40s
cmake --build --preset release-l40s
ctest --preset release-l40s
```

For another supported GPU, use the native architecture preset:

```bash
cmake --preset release-native
cmake --build --preset release-native
ctest --preset release-native
```

Use `debug-native` for a debug build. The binaries are written to the matching
`build-*` directory. Build directories are disposable and are never required
in source control.

Run the basic native applications with:

```bash
./build-l40s/scgm_cuda --toy --cpu
./build-l40s/toy_scgm
./build-l40s/test_cpu_gpu_parity
```

The parity test requires a working CUDA runtime for GPU validation; CPU-only
tests can still be run with `ctest` when CUDA is available at build time.

## Native Graph Format

The native loader expects an undirected graph in this format:

```text
<number_of_vertices> <number_of_unique_undirected_edges>
<source_vertex> <target_vertex>
...
```

Vertex IDs must be contiguous, zero-based, and free of self-loops and duplicate
undirected edges. Raw SNAP gzip files and Cora files must be converted first.

## Prepare Data And Run A Match

The preparation script converts a raw edge list into cleaned graph pairs with
fixed seeds, permutations, manifests, and ground truth:

```bash
python3 experiments/prepare_dataset_pairs.py \
  --input datasets/ca-GrQc.txt.gz \
  --output-dir data/G01-ca-GrQc \
  --label G01
```

Run the native matcher on one prepared pair:

```bash
./build-l40s/scgm_cuda \
  --source data/G01-ca-GrQc/seed_11/source.graph \
  --target data/G01-ca-GrQc/seed_11/target.graph \
  --k 64 --cpu \
  --candidates data/G01-ca-GrQc/seed_11/candidates.csv \
  --output data/G01-ca-GrQc/seed_11/assignment.csv \
  --timings data/G01-ca-GrQc/seed_11/timings.json
```

Replace `--cpu` with `--gpu` for CUDA feature and candidate generation.
Evaluate an assignment separately with:

```bash
python3 experiments/evaluate_assignment.py \
  --source-graph data/G01-ca-GrQc/seed_11/source.graph \
  --target-graph data/G01-ca-GrQc/seed_11/target.graph \
  --assignment data/G01-ca-GrQc/seed_11/assignment.csv \
  --ground-truth data/G01-ca-GrQc/seed_11/ground_truth.csv \
  --candidates data/G01-ca-GrQc/seed_11/candidates.csv \
  --output data/G01-ca-GrQc/seed_11/metrics.json
```

## Experiments

Useful entry points include:

- `experiments/generate_dataset_variants.py` for controlled edge-noise variants.
- `experiments/run_native_experiment.py` for named native benchmark runs.
- `experiments/benchmark_native.py` for repeated stage timings.
- `experiments/run_python_baseline.py` for the legacy Python baselines.
- `experiments/repair_candidate_support.py` and
  `experiments/validate_repaired_assignment.py` for support-repair analysis.

Experiment outputs are intentionally local. They are written under `data/`,
`results/`, `logs/`, `reports/`, and `experiments/diagrams/`, all of which are
ignored by default. Keep reproducible source scripts and metadata in Git.

## Data And Reproducibility

Raw datasets are external inputs and may be large or subject to their own
licenses. The repository does not require them to be committed; place them in
`datasets/` locally and document their source and checksum when publishing
results. Generated graph pairs, assignments, timings, and figures should be
recreated from the scripts rather than committed as source.

Do not claim exact paper reproduction, distributed solving of a connected
assignment component, directed-graph support, million-node/billion-edge capability,
or universal speedups. Report executed implementations, correctness evidence,
quality, hardware, memory scope, and repeated comparable timings separately.

## Development Notes

Before opening a change, run the relevant CMake preset and `ctest`, then inspect
the diff with `git status` and `git diff`. Never commit build directories,
Python caches, local credentials, generated results, logs, reports, diagrams,
figures, or unpublished research material. Use `git add -f` only when a normally
ignored artifact is intentionally being released.

## Optional structural-index candidates

The default candidate backend remains `exhaustive`. The indexed backend
uses raw log-degree/log-mean-neighbor-degree buckets, complete Chebyshev shells,
and exhaustive per-source fallback. It ranks retrieved targets with the existing
normalized six-feature cosine score. Retrieval is not an exhaustive Top-K guarantee.

```bash
./build-l40s/scgm_cuda \
  --source data/G01-ca-GrQc/seed_11/source.graph \
  --target data/G01-ca-GrQc/seed_11/target.graph \
  --cpu --k 64 --candidate-backend structural-index \
  --index-degree-width 0.5 --index-neighbor-width 0.5 \
  --index-min-radius 1 --index-max-radius 3 \
  --index-pool-multiplier 4 --index-fallback exhaustive \
  --timings /tmp/scgm-indexed-timings.json
```

Index options require the structural-index backend. Fixed candidate input and
scoring ablations are rejected with this backend. GPU indexed scoring uses the
same bounded selector as exhaustive GPU scoring; use `--gpu --gpu-selector parallel`
or `--gpu --gpu-selector streaming-serial`. The legacy host-input path schedules
from downloaded metadata; the resident path consumes device index/query data.
Benchmark the actual selected path before claiming gains.

`experiments/run_native_experiment.py` and `experiments/benchmark_native.py`
accept the same candidate/index options (use `--mode cpu|gpu` rather than `--cpu|--gpu`).
They verify native-reported backend/settings and append a configuration hash to
run directories; benchmark output is now under `OUTPUT/config_HASH/benchmark.json`.
The hash includes input/executable fingerprints and execution settings. Native
attempt metadata preserves effective K and index diagnostics separately from
runner-controlled support repair. These legacy runners do not expose all newer
resident/assignment/scheduler/multi-GPU CLI options. Their wrapper-controlled
reruns are distinct from native alternating-conflict repair. Use the direct
commands below to check the current complete pipelines; do not infer newer
backend coverage from the old runner's method label.

Timing JSON records requested/effective mode, configuration, index build time,
occupied buckets, pool sizes, fallback count, and actual scored pairs. Indexed
`candidate_generation_ms` includes retrieval, scoring, and Top-K but excludes
`index_build_ms`; setup-inclusive comparisons must count both. Separate indexed
similarity/Top-K/retrieval times and unmeasured scratch/host peak memory are null,
not zero. Index storage bytes describe stored arrays, not whole-process peak RSS.
These integration checks do not establish a real-data quality or speedup claim.

## Paired quality/performance comparison

Use `experiments/benchmark_structural_index.py` to compare exhaustive and indexed
CPU generation on the identical saved pair. For example:

```bash
python3 experiments/benchmark_structural_index.py \
  --source data/G01-ca-GrQc/seed_11/source.graph \
  --target data/G01-ca-GrQc/seed_11/target.graph \
  --ground-truth data/G01-ca-GrQc/seed_11/ground_truth.csv \
  --ks 16 64 128 --warmup 1 --trials 5 \
  --index-degree-width 0.5 --index-neighbor-width 0.5 \
  --index-min-radius 1 --index-max-radius 3 --index-pool-multiplier 4 \
  --output results/structural_index_comparison
```

Outputs are under `OUTPUT/comparison_HASH/comparison.json`, with per-K backend
candidate/assignment artifacts and trial metadata. This command performs the
listed runs, not just plotting. Repeat explicitly on separate pairs/seeds and
noise levels; choose index settings independently of reported test labels.

Quality/export runs and separate GNU `/usr/bin/time` RSS profiles are excluded
from timing trials. Timed backends alternate order by trial and do not export
candidate CSVs (both still write assignments/timing metadata). Host RSS is the
native process high-water mark from an isolated profile, not the Python process
or a cumulative child-process maximum. Use `--skip-memory-profile` to leave RSS
unavailable; CPU comparisons have no applicable GPU memory reduction metric.

Reports distinguish exact candidate-ID overlap from the fraction of indexed
scores strictly below the exhaustive cutoff (within the saved tolerance). Recall
loss/gain refers to retained Top-K/support, not pre-Top-K retrieved-pool recall.
Final accuracy and edge preservation are unavailable for invalid assignments.
No repair is performed. If either support is infeasible, complete-pipeline
speedup is null: its assignment/Sinkhorn work was skipped. Raw wall ratios remain
labeled observations and must not be presented as successful matching speedups.
Candidate-stage ratios include index construction. Whole-process RSS, CSR storage,
and index-array storage are different metrics and must not be conflated.

Helper-metric checks:

```bash
python3 -m unittest discover -s tests -p 'test_benchmark_structural_index.py'
```


## Parallel Top-K: validation and measured claims

The default `--gpu-selector serial` preserves the original exhaustive tiled heap.
`streaming-serial` and `parallel` share bounded GPU scoring batches and are the
fair pipeline comparison. Parallel selection supports K=1..256 with 256-candidate
chunks. Other positive K values use a recorded bounded serial fallback. The
64 MiB default cap covers allocated global scoring/selection scratch, not resident
features, index/query storage, final host output, or runtime overhead. Block shared
memory is reported separately. Legacy serial does not enforce this cap.

Native timing JSON includes a versioned `gpu_topk` object with requested policy,
effective implementation, fallback reason, chunk and batch sizes, scratch limit
and peak, and selection/candidate-stage timings. Both Python native runners accept
`--gpu-selector`, `--gpu-topk-scratch-mib`, and `--gpu-topk-batch-candidates`; requested
settings are hashed into output configurations and checked against native output.
GPU fallback to CPU is skipped evidence, never a successful GPU measurement.

```bash
cmake --preset release-l40s &&
cmake --build --preset release-l40s --target scgm_cuda test_gpu_topk test_cpu_gpu_parity test_gpu_topk_streaming benchmark_gpu_topk_streaming &&
ctest --test-dir build-l40s -R '^(test_gpu_topk|test_cpu_gpu_parity|test_gpu_topk_streaming)$' --output-on-failure

python3 -m unittest discover -s tests -p 'test_topk_experiments.py' -v
python3 tests/test_gpu_topk_cli.py --executable build-l40s/scgm_cuda
```

`test_gpu_topk` checks exact IDs, ordering and score bits for supplied scores.
`test_cpu_gpu_parity` separately checks complete candidate generation with declared
floating-point tolerances and exact parity between the shared GPU scoring paths.
GPU-unavailable native runs use skip return code 77. The dense validator checks
candidate counts, uniqueness, native ordering, score errors and strict cutoff
losses; its tolerance-aware end-to-end result is not an exact selection oracle.

Run the selection benchmark on the intended GPU:

```bash
python3 experiments/benchmark_gpu_topk.py \
  --benchmark-executable build-l40s/benchmark_gpu_topk_streaming \
  --output results/parallel_topk
```

For pipeline impact and matching quality, also supply `--source`, `--target`, and
optionally `--ground-truth`. Choose `--ks` no larger than the target count. Trials
alternate selector order; candidate exports for quality are outside timed trials.
Reports preserve raw timings, actual fallback, scratch peaks, GPU/build information,
input/executable hashes, and exact-oracle evidence. K512 fallback is measured as
serial execution and cannot establish a parallel speedup. Infeasible candidate
support cannot establish a completed matching-pipeline speedup. Accuracy is
unmeasured when ground truth is absent.

See [parallel Top-K findings](docs/parallel_topk_findings.md) for the current evidence
status. The benchmark generates a findings report from actual runs; use
`--findings-output docs/parallel_topk_findings.md` only when ready to replace the
pending-evidence document with measurements. No universal speedup is claimed.

## Checking the complete pipeline

Run these commands from the repository root on the actual GPU machine. There
is no new reproduction framework or configuration suite required. The historical
`reproduce_all.sh` is not the validation entry point for these native pipelines.
On the shared mount, source edits are already visible; build on the GPU machine.

### 1. Build and run all registered tests

```bash
cmake --preset release-l40s &&
cmake --build --preset release-l40s --parallel 4 &&
ctest --test-dir build-l40s --parallel 1 --output-on-failure
```

Build includes the existing CPU references, GPU implementations, and test targets.
Tests run serially to avoid competing GPU workloads. GPU tests return skip code
77 when hardware is unavailable; multi-GPU tests require two visible devices.
A skipped test is not a passed GPU correctness check. Python quality tests require
NumPy and NetworkX. If any test fails, resolve it before interpreting timings.

For a focused check of the newer stages:

```bash
ctest --test-dir build-l40s --parallel 1 \
  -R '^(test_candidate_repair|test_candidate_repair_cli|test_device_csr|test_gpu_device_costs|test_gpu_resident_pipeline|test_cpu_gpu_parity|test_gpu_assignment|test_assignment_components|test_assignment_scheduler|test_gpu_sinkhorn|test_multi_gpu_matching|test_correspondence_quality)$' \
  --output-on-failure
```

Check device memory access/cleanup and synchronization after tests pass:

```bash
(
set -e
for test in test_device_csr test_gpu_resident_pipeline test_gpu_assignment test_assignment_scheduler test_gpu_sinkhorn test_multi_gpu_matching; do
  compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 \
    "./build-l40s/$test"
done

for test in test_gpu_assignment test_assignment_scheduler test_gpu_sinkhorn test_multi_gpu_matching; do
  compute-sanitizer --tool synccheck --error-exitcode 99 \
    "./build-l40s/$test"
done
)
```

Stop at a sanitizer failure; do not report that loop as completed validation.
Sanitizer runs are correctness checks, not performance measurements.

### 2. Run a feasible single-GPU pipeline

The following real pair was previously observed feasible at K512. This is a
dataset-specific example, not a guarantee that K512 makes every support feasible.

```bash
./build-l40s/scgm_cuda \
  --source data/G01-ca-GrQc/seed_33/source.graph \
  --target data/G01-ca-GrQc/seed_33/target.graph \
  --k 512 --gpu --gpu-resident --gpu-selector parallel \
  --assignment gpu --sinkhorn gpu \
  --candidates candidates_checked.csv \
  --output matching_checked.csv \
  --timings timings_checked.json
```

This path uploads adjacency, extracts raw features, normalizes on GPU, scores and
selects candidates, and builds a separate target-sorted cost CSR. Device owners
retain allocations until consumers finish. GPU assignment and GPU Sinkhorn use
the cost support. Host support analysis and assignment certification remain
explicit downloads. Candidates remain score-descending/target-ID-ascending;
cost rows remain target-ID-ascending with `cost = 1 - similarity`.

Inspect effective execution rather than just requested flags:

```bash
python3 - <<'PY'
import json
with open("timings_checked.json") as f:
    d = json.load(f)
for key in ("maximum_matching_size", "assignment_status", "assignment_one_to_one",
            "gpu_assignment", "sinkhorn_status", "sinkhorn_converged",
            "gpu_topk", "gpu_residency"):
    print(key, d.get(key))
PY
```

Expected success evidence: complete support matching, a feasible one-to-one
assignment, a passing primal/dual certificate, and converged Sinkhorn residuals
within tolerance. K512 requests parallel selection but executes the recorded
serial fallback because optimized parallel selection supports K1..256.
K20/K256 on this pair were previously infeasible; all `-1` assignment outputs
in such a run are failure reporting, not a usable correspondence.

### 3. Check optional paths separately

- **CPU reference:** use `--cpu` without resident/GPU-stage options.
- **Native repair:** start with `--cpu --k 64 --repair --repair-k-limit 512
  --repair-max-rounds 3 --repair-max-edges 4000000 --repair-payload-mib 256`.
  Repair preserves accepted edges, expands conflict sources, and reports its
  stopping reason. The payload budget is not a total process-memory cap.
- **Indexed candidates:** add `--candidate-backend structural-index` to an
  appropriate CPU or GPU command. Save recall/feasibility as well as index setup,
  retrieval, and fallback costs; indexed support need not equal exhaustive Top-K.
- **Component scheduling:** add `--assignment-execution warp`, `block`, or
  `cooperative` to the single-GPU GPU-assignment command. For cooperative execution,
  set `--assignment-cooperative-blocks 2`. Check `assignment_scheduler` diagnostics.
  These are explicit variants, not an experimentally tuned adaptive policy.
- **Confidence export:** add `--sinkhorn-output confidence_checked.csv` when
  needed. Sinkhorn confidence does not automatically change the hard mapping.

Native repair and indexed retrieval are different support policies. Do not combine
unsupported options or treat a bounded repair failure as proof of solver failure.

### 4. Check actual multi-GPU distribution

```bash
nvidia-smi -L

./build-l40s/scgm_cuda \
  --source data/G01-ca-GrQc/seed_33/source.graph \
  --target data/G01-ca-GrQc/seed_33/target.graph \
  --k 512 --gpu --devices 0,1 --gpu-selector parallel \
  --assignment gpu --assignment-execution block --sinkhorn cpu \
  --output matching_multi_checked.csv --timings timings_multi_checked.json
```

This first multi-GPU path uses one global CPU feature/normalization computation,
then disjoint source scoring with replicated targets, CUDA streams, and host
threads/staging. It does not accept `--gpu-resident` or GPU Sinkhorn. Verify
`multi_gpu.workers`: both active workers must execute and their intervals must
cover all sources exactly once. Assignment distributes only whole connected
components. If the graph has one connected component, one assignment GPU is
correct; it is not evidence of assignment scaling. Original-CSR certification
checks global target uniqueness. Do not compare its cost/timing to GPU-normalized
single-GPU runs as though normalization and transfer paths were identical.

### 5. Check correspondence quality independently

```bash
python3 experiments/evaluate_assignment.py \
  --source-graph data/G01-ca-GrQc/seed_33/source.graph \
  --target-graph data/G01-ca-GrQc/seed_33/target.graph \
  --assignment matching_checked.csv --candidates candidates_checked.csv \
  --ground-truth data/G01-ca-GrQc/seed_33/ground_truth.csv \
  --timings timings_checked.json --output quality_checked.json

python3 experiments/analyze_feature_candidates.py \
  --source data/G01-ca-GrQc/seed_33/source.graph \
  --target data/G01-ca-GrQc/seed_33/target.graph \
  --candidates candidates_checked.csv --k 512 --normalization joint \
  --ground-truth data/G01-ca-GrQc/seed_33/ground_truth.csv \
  --output feature_audit_checked.json
```

For repaired variable-length rows, audit with `--support-kind retained-support`.
To evaluate the existing structural refinement as a separate variant:

```bash
python3 experiments/refine_assignment_by_edge_agreement.py \
  --source data/G01-ca-GrQc/seed_33/source.graph \
  --target data/G01-ca-GrQc/seed_33/target.graph \
  --support candidates_checked.csv --assignment matching_checked.csv \
  --ground-truth data/G01-ca-GrQc/seed_33/ground_truth.csv \
  --attempts 500000 --time-limit-seconds 60 --seed 20260929 \
  --output-assignment matching_refined_checked.csv \
  --output-report refinement_checked.json
```

Labels are used only after generation/refinement for evaluation. Report candidate
recall, support feasibility, descriptor cost, label accuracy, and edge preservation
separately. A previous seed-33 run had 100% candidate recall but only 46.54% label
accuracy; refinement reached 47.28% and increased edge preservation from 81.88%
to 83.08%, while increasing descriptor cost. This is one observed case, not a
general accuracy claim or evidence that the correspondence problem is resolved.

## Pipeline and development guide

```text
Load and validate source/target graphs
  -> raw structural features
  -> joint or separate normalization
  -> exhaustive or indexed scored pools
  -> ordered candidate selection
  -> optional native CPU conflict-region repair (supported CPU mode only)
  -> support maximum-matching analysis
  -> separate, target-sorted sparse cost CSR
  -> CPU assignment OR GPU assignment
       optionally discover/compact independent components and explicitly dispatch
       optionally place whole components on separate GPUs
  -> feasibility / one-to-one / GPU primal-dual certificate
  -> CPU or GPU sparse Sinkhorn on the same support (square matrices)
  -> explicit outputs and evaluation-only correspondence metrics
```

Device ownership is managed through `DeviceBuffer<T>`. Candidate ranking and cost
CSR order are distinct representations. Assignment uses exactly the retained
support; missing edges are forbidden. Sinkhorn leaves the hard assignment
unchanged. Edge refinement is a separate quality variant that can change the
mapping and give up descriptor optimality.

| Change | Main files | Checks to run |
|---|---|---|
| Feature definition/normalization | `include/features.hpp`, `src/features.cpp`, `cuda/gpu_matching.cu` | Feature tests, CPU/GPU parity, resident pipeline |
| Candidate scoring/selection | `cuda/gpu_candidate_scoring.cu`, `cuda/topk.cu`, GPU headers | Top-K chunk/merge/streaming tests, parity |
| Support repair | `include/candidate_repair.hpp`, `src/candidate_repair.cpp` | Repair unit and CLI tests; preserve accepted edges |
| Sparse costs/reverse index | `include/device_csr.hpp`, `cuda/csr_matrix.cu` | Device CSR and device-cost tests |
| GPU assignment | `cuda/gpu_assignment.cu`, `include/gpu_assignment.hpp` | CPU objective agreement, primal/dual certificates, sanitizers |
| Component scheduling | Component/scheduler headers and CUDA sources | Component independence, remapping, scheduler tests |
| GPU Sinkhorn | `cuda/sparse_sinkhorn.cu`, `include/gpu_sinkhorn.hpp` | CPU parity, residuals, non-convergence, sanitizers |
| Multi-GPU coordination | `include/multi_gpu_matching.hpp`, `cuda/multi_gpu_matching.cu`, `include/common.hpp` | Two-GPU tests, source coverage, global target uniqueness |
| CLI/diagnostics | `apps/main.cpp`, `CMakeLists.txt` | Relevant CLI tests and complete commands above |

Keep CPU references independent. When changing features, update CPU and GPU
together and use numerical tolerances for reductions. Preserve finite scores,
unique target IDs, deterministic tie ordering, and K512 fallback. Check allocation
arithmetic and owning device; complete stream work before destroying owners.
Register new tests in CMake and use skip 77 only for genuinely unavailable GPUs.

## Traditional timing and paper reporting

First pass tests, sanitizers, and a complete feasible run. Then repeat the same
command with fixed inputs/options: one warm-up followed by at least five measured
runs. Change only the option under study. For a direct single-GPU example:

```bash
(
set -e
mkdir -p results/manual_checks
for trial in warmup 1 2 3 4 5; do
  /usr/bin/time -f 'wall_seconds=%e peak_process_rss_kib=%M' \
    -o "results/manual_checks/process_${trial}.txt" \
    ./build-l40s/scgm_cuda \
      --source data/G01-ca-GrQc/seed_33/source.graph \
      --target data/G01-ca-GrQc/seed_33/target.graph \
      --k 512 --gpu --gpu-resident --gpu-selector parallel \
      --assignment gpu --sinkhorn gpu \
      --output "results/manual_checks/matching_${trial}.csv" \
      --timings "results/manual_checks/timings_${trial}.json"
done
)
```

Exclude the warm-up and report median plus spread. Process wall time includes
loading, allocations, transfers, solver work, cleanup, and requested output.
Native `candidate_repair.elapsed_ms` is already inside process wall time; do not
add it twice. If Python refinement is included in the claimed pipeline, time
both processes and intermediate staging. Use identical export policies in all
comparisons and retain nonzero exits/infeasible cases instead of dropping them.

GNU time reports the process CPU-RSS high-water mark. Native JSON distinguishes
CSR/owned allocation bytes, scratch budgets, and sampled GPU process memory.
They are different quantities. Null means unavailable, not zero. The current
multi-GPU CLI does not provide a complete aggregate GPU process-memory peak;
do not substitute GPU 0 memory or the sum of independent sampled peaks for it.

For a paper, include:

- Algorithm scope and real CPU/GPU/host-transfer boundaries, including repair and
  certification. State that connected components are never arbitrarily split.
- Dataset preparation, graph sizes/noise/seeds, ground-truth evaluation protocol,
  normalization, K, index policy, repair limits, selector fallback, and solver
  tolerances. Record input/executable hashes, revision, compiler/CUDA/driver,
  hardware, build flags, warm-ups, repeats, and concurrent machine activity.
- Correctness evidence separately from timing: complete support, one-to-one
  assignments, CPU comparisons, dual certificates, Sinkhorn residuals, and
  sanitizer results. Report failures and GPU-unavailable skips.
- Quality tables separating ambiguity, recall, feasibility, descriptor cost,
  label accuracy, and forward/reverse edge preservation. Refinement can improve
  edge agreement while worsening descriptor cost or label accuracy.
- Controlled exhaustive/indexed, CPU/GPU assignment, explicit scheduler,
  repair/no-repair, refinement/no-refinement, and one-/multi-GPU comparisons.
  Include index setup, repair, transfer, coordination, and output costs in the
  stated end-to-end scope. K512 fallback is not a parallel Top-K result.
- Per-stage and full-process timings with memory scopes and limitations. A
  connected workload using one assignment GPU cannot establish assignment scaling.
  Compare identical normalized scored support when claiming solver speedups.

No editable paper source was found in the repository during this documentation
update. These instructions provide the implementation/reporting checklist; they
do not assert that an external paper or its performance claims have been updated.
