# Explicit component assignment scheduling

Enable component discovery, compaction, and explicit dispatch with
`--assignment gpu --assignment-execution warp|block|cooperative` alongside the
existing GPU resident pipeline flags. Omitting `--assignment-execution` preserves
the existing whole-matrix block solver. No automatic thresholds are defined.

Components are connected components of the original bipartite support, including
isolated vertices. Each is independent: no candidate edge crosses components.
More source rows than columns proves a component infeasible. Otherwise the
assignment solver establishes feasibility; equal counts do not establish Hall's
condition. Column-only components require no solve and retain unmatched columns
with zero dual. Infeasible components make the aggregate result infeasible.

Execution is sequential across components on the supplied stream. Warp launches
32 threads, block launches 256 threads, and cooperative launches the explicitly
requested grid (default two blocks) using grid synchronization. Launch capability
and resident occupancy are checked. `--assignment-component-fallback block`
permits fallback only when cooperative launch capability is absent; default
`error` rejects that request. Invalid sizes, occupancy, allocation failures, and
budget violations do not silently fall back. Sizes/thresholds must be measured
before adding an automatic policy.

Matching and dual arrays are scattered to original IDs on GPU. The existing CPU
certificate validates the combined result against the original CSR at the public
download boundary. Component labels/count metadata are downloaded for scheduling;
this implementation is not wholly device-controlled orchestration.

`assignment_scheduler` JSON records requested mode, discovery and total wall time,
workspace peak, temporary component payload peak, metadata download bytes, and
per-component dimensions, edges, actual dispatch, block count, reason, status,
workspace, compaction time, assignment kernel time, and total time. Skipped
components have null effective execution. Kernel timings exclude discovery,
compaction, scattering, and CPU certification. The enclosing assignment wall time
includes the public certificate handoff.

Component workspace and assignment workspace have independent 64 MiB defaults.
Reported peak is their maximum because these workspaces have separate lifetimes.
Original CSR, global matching/duals, host component maps, and temporary compact
CSR/local results are excluded; temporary component payload is reported separately.
Validation helper allocations are excluded. This is not a process memory cap.

GPU tests compare original-matrix certificates and CPU objectives, deterministic
remapping, disconnected/rectangular/isolated support, Hall deficiency, and budgets.
GPU absence returns 77. Cooperative capability fallback is exercised where the
device lacks that capability. Sanitizer and runtime results must be collected on
the target GPU; adding source/tests alone does not validate a performance claim.
