# Parallel Top-K Selection Specification

Status: Proposed design. This document does not describe an implemented kernel.

## Purpose

Replace serial GPU selection with cooperative selection while preserving
candidate IDs and ordering for identical input scores.

Selection correctness is evaluated using identical precomputed scores.
CPU and GPU cosine calculations may differ slightly due to floating-point
arithmetic; those differences are evaluated separately.

## Candidate ordering

A candidate consists of a target ID and a float32 score.

Candidate A precedes candidate B when:
1. A has a higher score; or
2. their scores are exactly equal and A has a lower target ID.

Do not use an epsilon when comparing scores.
Positive and negative zero compare as equal.

Output rows are sorted using this ordering.

## Valid inputs

- K must be a positive integer.
- Scores must be finite. Reject NaN and positive or negative infinity.
- Target IDs must be valid for the target graph.
- Target IDs must be unique within each source pool.
- The same target may appear in different source pools.

Pool producers are responsible for uniqueness. Tests must verify this
invariant; selection does not silently deduplicate candidates.

Reject invalid inputs instead of sending them through a fallback that
could accept them.

For integrated cosine scoring, validate nonfinite inputs and intermediate
results before clamping can conceal an invalid score.

## Generic selector output

For a source pool containing P candidates, return min(K, P) candidates.

An empty pool produces an empty output row.
An empty source batch produces no output rows.

The matching pipeline separately enforces its candidate-count requirements.
Returning K candidates does not guarantee that a full matching exists.

Do not relax existing matching-pipeline validation merely because the
generic selector supports small pools.

## Proposed input and output layouts

Represent source pools as contiguous arrays:

- pool_offsets: N + 1 offsets into the candidate arrays.
- target_ids: one target ID per candidate.
- scores: one float32 score per candidate.

Offsets start at zero, are nondecreasing, and end at the candidate count.
The target and score arrays have the same length.

Use offsets and allocation arithmetic wide enough for the total candidate
count. Check conversions to narrower kernel index types.

Represent output with corresponding row offsets, target IDs, and scores.
Output row i contains min(K, pool_size[i]) candidates.

The existing exhaustive generator may continue producing scores in tiles.
This layout does not require materializing an N-by-M score matrix.

## Proposed optimized range and chunk size

Start with optimized support for integer K in [1, 256].

Use a fixed chunk size of 256 candidates.
Handle a final partial chunk without emitting padding as valid candidates.

Other valid positive K values use the preserved serial selector.
In particular, K=512 with 513 targets remains supported.

The optimized bound and chunk size are implementation choices, not changes
to generic selection semantics.

## Cooperative selection and merging

Use a cooperative block sorting or selection primitive for each chunk.
Retain at most min(K, chunk_size) candidates from that chunk.

Combine partial lists using parallel merging or sort-and-truncate stages.
Use the same candidate ordering in every stage.

Partial-list truncation is exact: a candidate excluded from its chunk's
Top-K cannot enter the complete pool's Top-K.

Keep the existing serial selection implementation available as a reference.

## Memory limits

Do not introduce a dense N-by-M score matrix.

Before allocating, calculate and check the sizes of:
- output buffers;
- partial-list buffers;
- temporary sorting or merging storage.

Check integer overflow in element counts and byte calculations.

Bound temporary workspace using a documented workspace budget.
Choose the numerical budget when the concrete primitive and workspace
requirements are known; do not claim an unmeasured universal limit.

Process chunks or merge stages in bounded batches when necessary.

## Fallback and errors

Use serial selection when a valid request exceeds the optimized K range
or supported kernel resource limits.

Record the selector actually used and its fallback reason.

If optimized temporary allocation fails:
1. release partially allocated temporary resources;
2. attempt the serial path if its required resources are available;
3. record the fallback reason.

If required input, output, or fallback storage cannot be allocated,
report an explicit allocation error. Do not return incomplete results.

Do not treat invalid inputs or unexpected kernel failures as successful
fallback cases. Report those errors.

Use exception-safe ownership for allocations and CUDA events.

## Required correctness coverage

Compare against an independent full-sort reference over identical scores.

Cover:
- K=1, K=256, K=257, and K=512;
- empty pools and empty source batches;
- pool sizes below, equal to, and above K;
- chunk boundaries at 255, 256, 257, 512, and 513 candidates;
- all-equal scores, ties at the cutoff, and signed zero;
- negative scores and arbitrary valid target-ID order;
- nonfinite-score rejection;
- uniqueness of each source pool and output row;
- fallback selection, recorded reasons, and allocation-failure cleanup.

For identical input scores, require exact target IDs and ordering.

## Performance evaluation

This documentation step makes no performance claim.

After implementation, compare serial and parallel selection using identical
input pools, scores, K values, and CUDA streams.

Warm up both paths and measure repeated runs using CUDA events.
Report selection latency, temporary-memory usage, and fallback counts.
Measure allocation and transfer costs separately, and report candidate-stage
wall time when evaluating pipeline impact.

Include small pools, multiple chunks, oversized pools, and fallback cases.
Report the GPU, build configuration, workload sizes, and timing variability