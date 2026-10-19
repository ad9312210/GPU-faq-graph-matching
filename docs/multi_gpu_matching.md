# Multi-GPU baseline

This implementation uses CUDA runtime streams and C++ host threads
(`Threads::Threads`). Transfers pass through host memory; NCCL and peer access
are not required. This is explicitly a host-coordinated first implementation,
not a device-resident distributed matching algorithm.

## Candidate generation

Pass `--gpu --devices 0,1` for exhaustive scoring. The CLI computes features and
joint/separate normalization once on CPU for the full graph pair. Disjoint
source intervals are scored concurrently on the requested GPUs; each active
worker owns an identical target-feature replica. Scoring uses the existing
device-input streaming selector. Global target IDs and original source order are
preserved. K=512 remains supported through the recorded serial selector fallback.

Device IDs are visible CUDA ordinals (after `CUDA_VISIBLE_DEVICES` remapping).
Unavailable or duplicate explicit IDs are errors. A single explicit ID selects
that device for the existing single-GPU path.

## Assignment

With `--assignment gpu`, connected components are discovered on the coordinator
GPU using the complete original sparse support. The original CSR is staged on
that GPU for discovery, then released before worker solves. Components are
placed whole using stable largest-edge-first greedy placement. Their local
CSR is constructed on host from the original support, then uploaded to one GPU.
No arbitrary row partition is treated as an independent assignment problem.

Each successful component is certified. Matching and duals are remapped to
original IDs on host, staged once on the coordinator, then passed through the
existing original-CSR certificate implementation. This extra transfer is
intentional reuse of the independent certificate; its known bytes are recorded.
An infeasible component makes the aggregate assignment infeasible. Failure
on any worker drains/joins workers before propagation; there is no silent CPU
assignment fallback. A single large connected component uses only one GPU for
assignment and receives no assignment speedup merely from additional contexts.

`--assignment-execution warp|block|cooperative` chooses the existing per-component
execution variant. Unsupported cooperative launch can use the same explicitly
enabled block fallback; occupancy and allocation errors remain explicit.

## Limits and diagnostics

`--multi-gpu-resident-mib` defaults to 1024 MiB per worker and bounds admitted
application payload. `--multi-gpu-reserve-mib` defaults to 64 MiB, left free
beyond the payload and workspace estimate. Candidate scoring/selection keeps
its separate per-worker 64 MiB scratch budget. Assignment has its existing
workspace budget. Before allocating payload, each worker checks CUDA free
memory and these budgets. Allocation can still fail if available memory changes.
The original CSR must fit on the coordinator for component discovery. Target
replicas must fit on every active scoring worker; otherwise the call fails
explicitly. Target sharding and cross-device connected-component solving are
not implemented.

These are admission checks, not a hard process memory cap. Host staging,
allocator/runtime overhead, and validation/library internals are excluded.
Reports distinguish per-worker target/payload bytes and selector fallback,
candidate wall time, cumulative kernel work across GPUs, component/device
placement, assignment wall time, and known transfer bytes. Transfer tracking
is marked incomplete. No misleading aggregate single-device NVML peak is emitted.

Initial CLI restrictions: exhaustive standard six-feature scoring, CPU Sinkhorn,
no candidate-input, and no `--gpu-resident`. Existing single-GPU resident,
indexed, and GPU Sinkhorn paths remain available when one device is selected.
These restrictions avoid silently routing requested resident/GPU stages through
host fallbacks. Do not compare this CLI's normalization to a GPU-normalized
baseline without reporting the change; the candidate API can compare identical
pre-normalized inputs directly.

## Validation

`test_multi_gpu_matching` requires two visible GPUs; fewer returns skip code 77.
It checks disjoint source intervals, actual execution on both GPUs, single-GPU
equivalence, ties, uneven/empty shards, global normalization inputs, K=512
fallback, reversed device order, memory rejection/recovery, independent
components assigned to both GPUs, a connected component remaining on one GPU,
global target uniqueness, CPU objective agreement, original-CSR certificates,
infeasibility, and deterministic remapping. Tests must run on the GPU machine;
a skipped test is not evidence of correctness or scaling.

Before reporting scaling, measure candidate generation and assignment separately
on identical inputs and verify actual per-device work. Include transfer and
coordination time, target replication cost, and component-size distribution.
