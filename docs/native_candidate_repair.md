# Native CPU candidate repair

`repair_candidate_support()` accepts ranked initial rows and a callback returning
one source's ranked Top-K prefix. The callback permits feature scoring or saved
prefixes without allocating an entire source-by-target matrix.

For each checkpoint, compute maximum matching. If not all sources are matched,
find alternating reachability from unmatched sources. Double the requested prefix
size, capped at `k_limit`, and expand only reachable source rows. Append targets
not already accepted, preserving their prefix order. Never replace an accepted
target or its score. This is the policy in
`experiments/repair_candidate_support.py`.

Initial rows contain exactly `initial_k` unique, finite, in-range candidates in
descending score order, with ascending target ID for exact ties. Each callback
returns exactly the requested prefix length with the same validation. After
expansion, accepted row order is append order, as in Python; if independently
computed prefixes change scores or order, no global re-sort is imposed.

Stop when feasible or when the prefix, round, candidate-entry, or logical payload
limit is reached. Zero rounds permits only the initial matching check. The round
count measures committed expansions; history includes the initial checkpoint.
No-new-edge rounds still advance the prefix, since a larger later prefix may
repair support. A stopped infeasible result is not an assignment solution.

An expansion is staged and validated before row values change. Invalid callback
output or allocation failure throws; accepted values from earlier rounds remain.
Capacity reservations may change even if a later reservation throws.

The payload budget counts retained entries, staged new entries, and one requested
prefix in units of `sizeof(Candidate)`. It is **not** a hard allocation or RSS cap:
vector spare capacity, allocation overhead, matching/conflict workspace, validation
sets, and callback scoring workspace are excluded. Device scoring's 64 MiB scratch
budget remains independent. A hard total-memory guarantee requires additional
allocation accounting before production integration.

The result reports initial/final entries, committed rounds, edges added, maximum
matching, checkpoint history, and a stable stopping-reason string. Rectangular
support is accepted, although the Python orchestration currently requires square
graphs. Neither implementation uses ground truth to expand candidates.

CLI integration, build registration, Python/native execution parity, and GPU
operations are subsequent work. The CPU assignment reference remains unchanged.
