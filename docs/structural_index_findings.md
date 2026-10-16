# Structural-index CPU pilot findings

Date: 2026-10-16. Scope: one saved noiseless ca-GrQc pair, seed 11, K64.

## Supported claim

On this pair, CPU structural indexing reduced scored source–target pairs by
65.27% and achieved a 2.10x ratio of median exhaustive/indexed
candidate-generation time including index construction. Ground-truth candidate
recall was unchanged at 89.73%. Both candidate supports were infeasible,
so this is evidence of candidate-stage savings only, not successful complete
matching, final accuracy preservation, or an end-to-end matching speedup.

## Evidence and operating conditions

- [Saved comparison](../results/structural_index_comparison/comparison_0f1371c775570bc2/comparison.json) and [configuration](../results/structural_index_comparison/comparison_0f1371c775570bc2/configuration.json).
- Input pair: `data/G01-ca-GrQc/seed_11/`; its manifest specifies edge noise 0.0.
- Each graph: 5,241 vertices and 14,484 unique undirected edges.
- Execution: CPU for both backends; standard six-feature cosine scoring and joint graph-pair normalization.
- Index coordinates: raw degree and raw mean neighbor degree, transformed with natural `log1p`; both widths 0.5.
- Chebyshev complete shells, minimum radius 1, maximum radius 3, desired pool `min(M, 4*K)`; exhaustive fallback.
- No support repair; no labels passed to native matching. Ground truth is used afterward for evaluation.
- One warmup per backend; three sequential timing repeats per backend, alternating order by trial.
- Candidate CSV export/quality runs and separate RSS profiles are excluded from timing repeats.
- Timed runs still write assignment CSV and timing JSON for both methods. Wall times include process startup, graph loading, diagnostics and those writes.
- Candidate-stage time is `candidate_generation_ms + index_build_ms`; excludes feature extraction and normalization, includes index setup/retrieval/scoring/Top-K.

## Measurements

| Metric | Exhaustive | Indexed |
|---|---:|---:|
| Scored pairs | 27,468,081 | 9,538,717 |
| Median candidate stage including index setup (ms) | 443.203 | 210.812 |
| Candidate-stage min–max (ms) | 442.902–443.516 | 210.811–210.893 |
| Observed median wall time (ms), incomplete pipeline | 490.034 | 248.036 |
| Ground-truth candidate recall (%) | 89.735 | 89.735 |
| Maximum matching size | 4,703 / 5,241 | 4,703 / 5,241 |
| Complete assignment | No | No |
| Final correspondence accuracy / edge preservation | Unavailable | Unavailable |
| Separate host peak RSS profile (MiB) | 8.00 | 8.00 |
| CSR storage (bytes) | 2,704,360 | 2,704,360 |

Additional indexed diagnostics (from the quality/export run, not timing medians):

- Occupied buckets: 50; occupancy minimum/maximum: 1/529 targets.
- Final pool size minimum/mean/maximum: 262/1820.02/2741.
- Exhaustive fallback sources: 0; maximum observed radius: 3.
- Index array storage: 22,172 bytes; this is not peak index-construction scratch or whole-process memory.
- Mean candidate-ID overlap with exhaustive Top-K: 98.38%.
- Indexed candidates strictly below the exhaustive cutoff by more than 2e-6: 1.57%.
- Zero ground-truth targets lost/gained relative to exhaustive Top-K on this pair. This does not mean the candidate sets are identical.

The recorded 1.98x wall ratio is an observation for infeasible-support runs.
Assignment and Sinkhorn were skipped. Do not label it a complete-pipeline speedup.
The two isolated RSS observations show no measured memory reduction; no GPU memory
comparison is applicable. RSS covers native execution without candidate export,
not Python-side overlap/quality analysis or the separate candidate-export run.

## CPU/GPU boundary

| Component | Current boundary |
|---|---|
| Exhaustive reference | CPU and hybrid GPU candidate backends exist |
| Indexed candidate CLI | CPU only |
| GPU bucket construction | Implemented with device sort/scan and host orchestration |
| GPU retrieval planning | Implemented as a correctness-first occupied-bucket scan per source/shell |
| GPU indexed scoring and Top-K integration | Pending; indexed GPU CLI explicitly rejects execution |
| CSR assignment and sparse Sinkhorn | CPU |
| Adaptive support repair | Separate orchestration; disabled in this comparison |

GPU construction/retrieval test code exists; the saved CPU comparison does not
record an executed GPU parity result. Do not infer GPU parity or GPU performance
from these CPU measurements. The planning kernel scans occupied buckets and is
not evidence of an optimized bucket-lookup implementation.

## Reproduce the measured protocol

From the repository root, with dependencies, CUDA build tools and GNU `/usr/bin/time` available:

```bash
cmake --preset release-l40s
cmake --build --preset release-l40s --target scgm_cuda test_structural_index test_topk
ctest --test-dir build-l40s -R '^(test_structural_index|test_topk)$' --output-on-failure
python3 -m unittest discover -s tests -p 'test_benchmark_structural_index.py'
python3 experiments/benchmark_structural_index.py \
  --source data/G01-ca-GrQc/seed_11/source.graph \
  --target data/G01-ca-GrQc/seed_11/target.graph \
  --ground-truth data/G01-ca-GrQc/seed_11/ground_truth.csv \
  --ks 64 --warmup 1 --trials 3 \
  --index-degree-width 0.5 --index-neighbor-width 0.5 \
  --index-min-radius 1 --index-max-radius 3 \
  --index-pool-multiplier 4 --index-fallback exhaustive \
  --output results/structural_index_comparison
```

Prepare the input pair first if absent, following README dataset preparation.
Use another supported preset on different hardware. A changed executable or
configuration produces a different output hash; timings are not expected to be
identical. `--skip-memory-profile` omits RSS measurement and changes the protocol.

## Provenance and limitations

The artifact records input/truth/executable hashes, runner hash, Python/platform,
and Git revision `1cba01fb03cb86e31a01f160eddc4e339c08f57f` with a dirty working tree.
Python was 3.10.12. The saved artifact does not include a CPU
model, compiler/build flags, full uncommitted patch or a full source snapshot.
Thus the revision alone cannot reconstruct the exact measured executable; hashes
identify artifacts but do not supply missing code or machine metadata. Preserve
the executable/source changes and collect fuller environment metadata for future
publication-grade experiments. Do not retroactively assert unrecorded hardware.

This is one graph and one planted permutation; timing repeats are not independent
graph replications. No confidence intervals or cross-dataset robustness are
established. Candidate recall is not final accuracy. Exactness on retained support
is not unrestricted assignment optimality or a QAP guarantee. Bucket parameters
are experimental choices, not proven optimal settings or established novelty.
Separate indexed retrieval/scoring/Top-K timers and scratch peak memory remain
unmeasured/null. Large buckets or fallback can restore all-pairs work; there is no
worst-case subquadratic guarantee. Billion-edge, multi-GPU and headline paper
speedup/memory/accuracy claims are not supported by this pilot.

## Wording for the paper

Suggested results wording:

> On one noiseless ca-GrQc pair (5,241 vertices; seed 11; K=64), the CPU indexed
> backend evaluated 65.27% fewer source–target similarities and reduced median
> candidate-generation time, including index construction, from 443.20 ms to
> 210.81 ms (2.10x; three timing repeats per backend). Ground-truth candidate
> recall remained 89.73%. Both retained supports matched only 4,703 sources;
> consequently, final assignment quality and complete-pipeline speedup were
> not established. Separate host-RSS profiles showed no measured reduction.

The method section must specify coordinates, widths, shell stopping, fallback and
unchanged scoring. Reconcile the introduction's indexed retrieval with the
all-pairs Algorithms 2–3 by naming exhaustive reference and indexed variant.
Do not describe the measured backend as GPU-resident. Describe the current CPU
assignment as a primal-dual shortest-augmenting-path implementation with
row-minimum initialization, not the removed full LAPJV initialization sequence.

## Next evidence needed

1. Independent graph pairs/seeds, noisy observations and K/bin/radius sensitivity.
2. Feasible supports and successful validated assignment before end-to-end claims.
3. Separate repair-inclusive experiments with identical accounting boundaries.
4. GPU scoring integration, actual-consumer parity and measured GPU comparisons.
5. Full source/build/hardware capture, repeated RSS measurements and scratch profiling.
