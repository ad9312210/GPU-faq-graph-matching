# Structural indexing: implementation and paper checklist

Status (2026-10-16): CPU indexed backend and CLI are implemented; a single-pair
CPU pilot measures candidate-stage savings. GPU index construction and retrieval
planning exist, but GPU indexed scoring/CLI integration remains pending.
See [supported findings](structural_index_findings.md) for evidence and limitations.
The stepwise sections below retain prospective instructions for unfinished work.
Repository: GPU-faq-graph-matching. Date: 2026-10-16.

## 1. Understand the paper/code gap

**Do:** Keep exhaustive generation as the reference and introduce indexing as an optional backend.
**Why:** Sparse Top-K storage reduces retained memory; it does not itself reduce all-pairs similarity work.

| Aspect | Paper | Current implementation / proposed work |
|---|---|---|
| Structural buckets | PDF p. 2 introduction and p. 3 contribution 2 claim quantized signatures and compatible bucket retrieval | CPU indexed backend implemented; GPU construction/planning exist; integrated GPU indexed scoring pending |
| Similarity generation | PDF pp. 9–10, Algorithms 2–3 describe all-pairs evaluation followed by Top-K | Current CUDA scores all target tiles; revise the paper to distinguish exhaustive reference and indexed method |
| Quantization details | Coordinates, widths, compatibility metric and fallback are not fully specified | The rules below are new implementation decisions, not reproduced paper details |
| GPU residency | Paper claims GPU-resident indexing and pipeline stages | CPU reference and GPU construction/planning exist; CPU-only indexed CLI; assignment/Sinkhorn remain on CPU |
| Exactness | Restricted assignment can be exact over retained support | Bucket retrieval does not guarantee exhaustive Top-K or unrestricted assignment optimality |

Page references above use PDF/printed page numbers for the supplied draft. No claim of novelty follows merely from implementing buckets.

## 2. Fix inputs and scope

**Do:** Index raw degree and raw average-neighbor-degree. Keep separately normalized six-feature vectors for cosine scoring.
**Why:** This changes retrieval, not the existing scoring objective. Raw values are nonnegative; z-score values need not be.

- Retain raw indexing coordinates before in-place normalization.
- Source and target use identical transformation and bin widths.
- Initial scope: standard six-feature scoring. Extra neighbor-profile signals remain separate variants.
- Ground-truth mappings must not enter construction, retrieval, stopping, repair, or fallback. Use labels only in evaluation; any later supervised parameter selection needs a declared development/holdout protocol.

**Later files:** `include/structural_index.hpp`, `src/structural_index.cpp`; integration in candidate generation and GPU orchestration.

## 3. Specify quantization

**Do:** Use natural-log transforms and fixed-width bins:

```text
q0(v) = floor(log1p(raw_degree(v)) / degree_bin_width)
q1(v) = floor(log1p(raw_avg_neighbor_degree(v)) / neighbor_bin_width)
key(v) = (q0(v), q1(v))
```

**Why:** Log transformation compresses large values and handles zero. Fixed boundaries avoid independently fitted source/target scales.

- Bin coordinate k represents transformed interval `[k*w, (k+1)*w)`.
- Reject negative/nonfinite indexed values, nonpositive/nonfinite widths, and coordinate overflow.
- Use checked signed 64-bit coordinates and a pair key; do not silently truncate or pack keys into insufficient bits.
- Do not assume CPU/GPU `log1p` produces identical keys near boundaries. Require boundary/adjacent-value tests and resolve or report differences.
- CPU-computed keys uploaded to GPU are an interim reference, not GPU index construction.

**Later files:** CPU quantization and tests, followed by `cuda/structural_index.cu`.

## 4. Build a deterministic target index

**Do:** Store occupied bucket keys, bucket offsets, and target IDs grouped by key. Sort keys lexicographically and IDs increasingly within each bucket.
**Why:** This provides reproducible retrieval without allocating a dense grid of mostly empty buckets.

Every target appears once. CPU/GPU construction must agree on key membership and offsets for identical keys. Rebuild when indexed target features or configuration change; incremental index maintenance is outside this first implementation.

**Later files:** `include/structural_index.hpp`, `src/structural_index.cpp`, then `cuda/structural_index.cu`.

## 5. Define compatible buckets and retrieval

**Do:** Search shells using Chebyshev distance from the source key:

```text
shell r: max(abs(target_key0-source_key0),
             abs(target_key1-source_key1)) == r
```

**Why:** “Neighboring compatible buckets” becomes an implementable rule.

1. Start at radius zero.
2. Visit each shell once, with keys in lexicographic order.
3. Skip negative bucket coordinates; use overflow-safe arithmetic.
4. Process a complete shell before deciding to stop.
5. Search at least `minimum_radius`.
6. Stop when the unique pool reaches `min(M, pool_multiplier*K)` and the minimum radius has been reached.
7. Otherwise continue through `maximum_radius`, then apply Step 6 below.

Require `0 <= minimum_radius <= maximum_radius`, integer `pool_multiplier >= 1`, and `1 <= K <= M`; calculate the desired pool without multiplication overflow. Empty source sets return no rows; nonempty sources with zero targets cannot satisfy positive K.

**Important:** Reaching the desired pool size is not a recall or feasibility guarantee. Similar bucket coordinates do not certify high six-feature cosine similarity.

## 6. Specify fallback, scoring, and oversized buckets

**Do:** If the desired pool is not reached at maximum radius, retrieve all M targets for that source. Otherwise score the retrieved pool only.
**Why:** Guarantees enough candidates when K is valid, while preserving a measurable restricted-search mode.

- Fallback is triggered by pool size, not ground-truth recall or assignment correctness.
- Deduplicate before scoring; full shells must not create duplicate targets.
- Score unique targets once where practical; count any duplicate work explicitly.
- Rank by decreasing existing cosine similarity, then increasing target ID on exactly equal scores.
- Retain K candidates and construct costs using the existing `1 - similarity` rule.
- Oversized buckets are processed fully in bounded chunks. Do not truncate by target ID.
- Preserve current score normalization, zero-vector handling, feature-mask and weight semantics in like-for-like comparisons. Initial comparisons use the standard six-feature mode.
- Candidate-support repair remains a separate stage. Report pre-repair support and include repair costs in repaired pipeline totals.

**Later files:** `src/candidate_generation.cpp`, its header, `cuda/gpu_matching.cu`, `include/gpu_matching.hpp`.

## 7. Record a starting configuration

**Do:** Begin correctness development with the following explicit configuration, then evaluate sensitivity before adopting performance defaults.
**Why:** The implementation needs concrete values; these values have no demonstrated optimality.

| Setting | Initial value |
|---|---|
| Coordinates | Raw degree, raw mean neighbor degree |
| Transform | Natural `log1p` |
| Widths | 0.5 for each coordinate |
| Minimum radius | 1 |
| Maximum radius | 3 |
| Pool multiplier | 4 |
| Fallback | Exhaustive per source |
| K | Caller supplied; record in every run |

These are proposed starting values, not values specified or validated by the paper. Record actual configurations and do not select final settings on reported test labels. Broad parameter sensitivity is research work, not a prerequisite for initial unit tests.

## 8. Implement in small checkpoints

**Do / why / files:**

1. **CPU API:** configuration, build/query operations, pool results and counters. Add `include/structural_index.hpp`. Separates retrieval from scoring.
2. **CPU implementation:** quantization, occupied-bucket index, shells and fallback. Add `src/structural_index.cpp`. Provides the behavior reference.
3. **CPU tests:** add `tests/test_structural_index.cpp`; register sources/tests in `CMakeLists.txt`. Catch policy errors before CUDA work.
4. **CPU integration:** add optional indexed mode in candidate-generation interfaces and `src/candidate_generation.cpp`; keep exhaustive mode available. Compare rankings over identical pools.
5. **GPU index:** add `cuda/structural_index.cu` and device interfaces. Compare keys, memberships and pool sets with CPU.
6. **GPU scoring:** edit `cuda/gpu_matching.cu` and `include/gpu_matching.hpp` to score retrieved pools without a dense cross-graph matrix. Record device/host boundaries honestly.
7. **CLI/metadata:** edit `apps/main.cpp` and experiment runners to expose and save the configuration, backend and fallback policy.
8. **Evaluation:** add `experiments/benchmark_structural_index.py`; reuse existing evaluation/benchmark helpers. Keep identical graph pairs and scoring settings.

Do not modify assignment or Sinkhorn as part of the indexing implementation unless an independently diagnosed bug requires it.

## 9. Test before timing

**Do:** Test these behaviors with known expected pools and rankings.
**Why:** A fast implementation of a different retrieval policy is not CPU/GPU parity.

- Bin boundaries, immediately adjacent representable inputs, zero values, invalid widths/features and overflow.
- Empty buckets, isolated vertices and repeated signatures.
- All targets in one bucket; no target truncation.
- Full-shell expansion, minimum radius, maximum radius, fallback and deduplication.
- K and desired-pool limits, rectangular source/target sizes and empty source sets.
- Index invariant: every target appears exactly once, offsets valid.
- Exact Top-K within a retrieved pool against an independent reference, including equal-score tie handling.
- CPU/GPU quantization and retrieved-set agreement; score tolerances documented separately from key equality.
- Exhaustive fallback reproduces exhaustive candidate ranking within declared numerical/tie rules.

## 10. Measure usefulness and preserve evidence

**Do:** Compare indexed and exhaustive generation on saved identical pairs with repeated timings.
**Why:** Reduced retained storage alone does not prove reduced work or end-to-end speedup.

Record:

- Build, retrieval, similarity, Top-K, transfer, assignment and repair times separately.
- Setup-inclusive total runtime; distinguish one-shot use from index reuse.
- Bucket occupancy, visited buckets, pool sizes, radii and fallback rates.
- Actual scored pairs, duplicate scoring if any, and exhaustive baseline pair count `N*M`.
- Index, scratch and candidate memory plus whole-pipeline host/device peak memory.
- Exhaustive Top-K overlap with documented cutoff-tie handling.
- Evaluation-only ground-truth candidate recall, maximum matching size, assignment accuracy and edge preservation.
- Revision, local changes/source hashes, executable hash, hardware, dependencies, input hashes, seeds, configuration and repeated-run boundaries.

The similarity-scoring work is proportional to the actual retrieved/scored pairs times feature dimension, plus index construction, lookup and Top-K overhead. Worst case remains all-pairs; large buckets and fallback may eliminate any gain. No universal subquadratic runtime or speedup is promised.

## 11. Update the paper at the appropriate milestone

**Historical wording for the pre-implementation milestone (not current status):**

Suggested wording:

> We propose quantized structural indexing to restrict candidate evaluation. The current exhaustive implementation remains the reference; indexing quality and performance have not yet been validated.

**After CPU-only implementation:**

> Our CPU reference groups targets by quantized log-degree and log-mean-neighbor-degree. It retrieves complete neighboring bucket shells and uses exhaustive fallback when the configured pool target is not reached. Candidates are ranked using the unchanged six-feature cosine score.

Do not call this GPU-resident indexing.

**Future wording only after integrated GPU scoring and validation:**

> The indexed backend constructs target buckets on GPU and evaluates candidates retrieved from compatible bucket neighborhoods. The quantization, neighborhood, stopping and fallback rules are specified explicitly. Exhaustive generation is retained as a reference.

Add concrete configuration, measured quality/runtime/memory results and CPU/GPU validation evidence. Do not imply the entire pipeline is GPU-resident while assignment or other major stages remain on CPU.

**Required manuscript reconciliation:**

- Revise pp. 9–10, Algorithms 2–3: distinguish exhaustive reference from indexed retrieval followed by scoring and Top-K.
- Add equations/rules for keys, compatibility, full-shell stopping and fallback to the method section.
- Qualify p. 3 complexity: scoring depends on retrieved pool sizes; index/lookup/selection overhead and worst-case fallback remain.
- State exact assignment optimality only over retained feasible support; no guarantee of exhaustive Top-K, dense optimality or correct correspondence follows.
- Identify the coordinates/log transform/bin widths/radii as design choices; provide ablations rather than asserting novelty or optimality.
- Report repair independently; do not hide support infeasibility or exclude repair overhead from repaired totals.

## 12. Completion gates

**Specification done:** this document defines behavior sufficiently to implement it, with concrete starting parameters and explicit limitations.

**Implementation done:** unit tests, CPU/GPU parity and reproducibility checks pass; CLI and metadata expose the policy; measured comparisons document quality and work saved.

**Research claim supported:** experiments establish the claimed benefits in stated conditions. A correct implementation may provide little speedup or lose quality; report that result rather than changing the acceptance criteria afterward.

## Related CPU solver note for the paper

The revised `solve_lapjv` retains its interface name but uses a sparse primal-dual shortest-augmenting-path CPU procedure with row-minimum initialization, not the full LAPJV initialization sequence. Describe it accordingly; it does not establish the paper's GPU LAPJV or scheduling claims. The existing 208-case SciPy comparison is small-input validation, not a proof of general correctness or a performance result. Keep this distinction when updating Figure 5 and solver text.


## CLI integration checkpoint (2026-10-16)

CPU indexed candidate generation is selectable through `--candidate-backend
structural-index`. The `--index-degree-width`, `--index-neighbor-width`,
`--index-min-radius`, `--index-max-radius`, `--index-pool-multiplier`, and
`--index-fallback exhaustive` flags are validated and saved. Exhaustive generation
remains the default. GPU indexed scoring is rejected pending its implementation;
fixed-support input and feature/profile ablations cannot be combined with the
initial indexed backend. Merely passing GPU construction/retrieval tests is not
proof of an integrated GPU indexed pipeline.

Both native runners forward identical configuration, check effective native mode
and settings, and separate different configurations using hashed directory names.
The native experiment runner stores per-K index metadata during support repair;
its aggregate unmeasured stage timings remain null. Benchmark trials preserve
native diagnostics and do not perform support repair. Raw source coordinates are
retained before normalization; the target index is built before normalization.

Current measurements: index build time, combined retrieval/scoring/Top-K time,
occupied bucket counts and min/max occupancy, pre-fallback pool mean, final pool
min/mean/max, fallback source count, final maximum radius, scored pair count,
and array storage bytes. Separate retrieval/scoring/selection timing and scratch
or host peak memory remain unavailable (null). Index setup time must be included
in one-shot totals; it is not part of candidate_generation_ms.

Validation of CLI wiring uses small identical graph pairs and explicit invalid
option checks. Real-data quality/performance evaluation, GPU indexed scoring and
its CLI dispatch remain pending. Do not promote this checkpoint into a speedup,
full GPU-residency, or exhaustive-optimality claim in the paper.

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


## Supported-claim checkpoint

See [CPU pilot findings](structural_index_findings.md) for exact operating conditions,
reproduction commands, provenance gaps and paper-ready qualified wording. Current
evidence supports a 2.10x candidate-stage ratio on one pair, not a complete matching
speedup: both K64 supports are infeasible. Do not interpret roadmap acceptance
criteria or future-tense GPU examples as completed evidence.
