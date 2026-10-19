# Parallel Top-K findings

Status: GPU measurements pending. Code preparation and compilation do not establish
selection correctness on hardware, a performance gain, or unchanged matching quality.
No numerical speedup is claimed by this document.

## Supported implementation scope

Parallel selection sorts 256-candidate chunks and merges partial lists while
retaining K. K=1..256 is supported by the parallel implementation. Larger valid K,
insufficient parallel scratch/batch capacity, and parallel workspace allocation
failure use recorded bounded serial selection. Invalid scores and unexpected CUDA
errors are rejected, not converted to successful fallback.

The default global scratch cap is 64 MiB for scoring batches and selection only.
Resident normalized/raw graph features, structural index/query storage, index build
and feature extraction, final host output, and CUDA runtime overhead are outside
this phase-specific cap. Per-block shared memory is reported separately. The
legacy exhaustive serial reference has no new scratch cap and remains available.

## Evidence required before a claim

1. Run `test_gpu_topk`: compare exact ordered target IDs and score bits against an
   independent full sort on explicit and seeded random scores. This isolates
   selection from float32 feature extraction and cosine calculations.
2. Run `test_cpu_gpu_parity` and existing chunk, merge, streaming, and index tests.
   Feature/score parity has declared tolerances. Native ordering remains exact.
   The streaming-serial and parallel pipelines share a GPU scoring kernel and
   must produce identical output for the same computed scores. Indexed pools
   are checked against their own pool references, not assumed exhaustive.
3. Run the CPU experiment-helper tests, CLI smoke test, and GPU memory/race checks.
   GPU-unavailable runs must be skipped. They are not correctness passes.
4. Run the identical-input microbenchmark. Retain every trial's actual selector,
   fallback reason, memory peak, kernel times, transfer times, and allocation-
   inclusive wall time. Require exact oracle checks. Record the GPU, CUDA versions,
   build settings, input fingerprint/seed and executable SHA256.
5. Run complete pipeline comparisons within each candidate backend using
   streaming-serial versus parallel. Keep K, input pair, normalization, retrieval
   configuration, batch maximum and scratch budget fixed. Export quality outside
   timed trials and alternate selector execution order. Record support feasibility,
   candidate IDs/scores, candidate recall (if truth exists), assignment cost,
   one-to-one validity, edge preservation and final accuracy (if truth exists).

## Commands on the selected GPU device

```bash
cmake --preset release-l40s &&
cmake --build --preset release-l40s --target scgm_cuda test_gpu_topk test_cpu_gpu_parity test_gpu_topk_chunks test_gpu_topk_merge test_gpu_topk_streaming test_gpu_structural_index benchmark_gpu_topk_streaming &&
ctest --test-dir build-l40s -R '^(test_gpu_topk|test_cpu_gpu_parity|test_gpu_topk_chunks|test_gpu_topk_merge|test_gpu_topk_streaming|test_gpu_structural_index)$' --output-on-failure

python3 -m unittest discover -s tests -p 'test_topk_experiments.py' -v
python3 tests/test_gpu_topk_cli.py --executable build-l40s/scgm_cuda
compute-sanitizer --tool memcheck --error-exitcode 1 ./build-l40s/test_gpu_topk
compute-sanitizer --tool racecheck --error-exitcode 1 ./build-l40s/test_gpu_topk

python3 experiments/benchmark_gpu_topk.py --output results/parallel_topk
```

For a saved graph pair, append `--source SOURCE --target TARGET --ground-truth TRUTH`
and select `--ks` within the target count. `--include-legacy` records the old
exhaustive baseline separately; it does not isolate selection from changes in
scoring orchestration. An unmodified default run measures selection only and
makes no pipeline/quality claim. NumPy is required for the dense-reference tests.

## Interpretation

Ratios above one favor parallel. Raw timings and median/min/max variability are
saved. A requested parallel policy that executed serial fallback receives no
parallel speedup field. A pipeline with infeasible support or failed assignment
receives no completed pipeline speedup field. Changed candidate support, scores beyond tolerance, or
measured matching quality also blocks an unchanged-quality speedup claim. Without
ground truth, final accuracy is unmeasured. Repeated runs on one saved graph pair
are timing repetitions, not independent graph samples.

Small pools can be dominated by allocation, transfer, synchronization, and host
scheduling. Indexed scoring initially schedules compatible bucket spans using host
metadata and processes source pools sequentially. Kernel speedup does not imply
candidate-stage or full-pipeline speedup. Claims must name the measured GPU,
workload, K, distribution, batch size and budget; no general subquadratic bound or
universal speedup follows from this implementation.

The benchmark writes `OUTPUT/config_HASH/benchmark.json` and a generated findings
report. After reviewing evidence, `--findings-output docs/parallel_topk_findings.md`
can replace this pending-evidence document with the actual measured report.
