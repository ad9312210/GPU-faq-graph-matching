# SCGM Graph Matching

SCGM is a native C++17/CUDA implementation and Python experiment harness for
sparse candidate graph matching. The native backend extracts structural vertex
features, generates Top-K candidates, builds sparse CSR costs, and solves the
assignment with CPU LAPJV. CUDA accelerates feature extraction and candidate
generation; assignment and sparse Sinkhorn diagnostics remain on the CPU.

This repository is a research artifact, not a complete reproduction of the
unpublished paper draft. Results and capabilities should be interpreted using
the implementation that is actually present here.

## Capabilities

- Six structural features: degree, clustering coefficient, average neighbor
  degree, neighbor-degree standard deviation, triangle participation, and
  degree squared.
- CPU and CUDA feature extraction and Top-K candidate generation.
- Sparse CSR cost construction and native CPU LAPJV assignment.
- CPU sparse Sinkhorn diagnostics.
- Native unit tests, CPU/GPU parity tests, paired-graph generation, and
  experiment and benchmark scripts.

The GPU path is hybrid: feature and candidate processing run on the GPU, while
the CSR assignment and Sinkhorn stages run on the host CPU. The implementation
does not provide GPU LAPJV, GPU-resident Sinkhorn, multi-GPU execution, or the
large-scale guarantees described in the paper draft.

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

Do not claim exact paper reproduction, GPU-resident assignment, multi-GPU
execution, directed-graph support, million-node/billion-edge capability, or the
paper's reported speedups unless those capabilities are implemented and
independently verified.

## Development Notes

Before opening a change, run the relevant CMake preset and `ctest`, then inspect
the diff with `git status` and `git diff`. Never commit build directories,
Python caches, local credentials, generated results, logs, reports, diagrams,
figures, or unpublished research material. Use `git add -f` only when a normally
ignored artifact is intentionally being released.