# Controlled correspondence-quality comparisons

Do not interpret optimal assignment for six-feature descriptor costs as correct
graph correspondence. Graph automorphisms and feature collisions can yield equal
descriptor costs and equal edge preservation with different ground-truth labels.

Keep graph pair, seed/noise, normalization, feature definition, initial K, support
policy, solver tolerances, and implementation fixed when isolating a change.
Reports record variant names and input hashes. Attach native timings when
evaluating a mapping. Python reference extraction/normalization and support-audit
times are separate from native/GPU stage timings.

Compare these explicit variants:

1. Six-feature baseline with fixed ranked support.
2. The same baseline plus native support repair, reporting variable row lengths,
   accepted edge count, repair limits/stopping reason, recall, and feasibility.
   Compare native CPU repair to a CPU baseline to avoid conflating scoring changes.
3. Edge-agreement refinement of an existing feasible assignment, with exactly the
   same exported support. Report initial/final descriptor costs, edge preservation,
   label accuracy, changed matches, proposal budget, seed, and runtime limit.

`analyze_feature_candidates.py` supports `--support-kind ranked-prefix` (audit
the first K entries) or `retained-support` (audit every exported edge). The latter
does not claim a K-boundary tie metric. Exact feature duplicates and normalized
quantization-cell collisions are distinct: a quantization collision is not an
exhaustive pairwise near-equality test. Ties counted in saved lists do not count
every tied omitted target in the full target pool.

`evaluate_assignment.py` reports candidate recall, maximum support matching,
feasibility, descriptor cost from exported scores, complete/injective mapping,
forward/reverse edge preservation, and evaluation-only label accuracy separately.
An invalid/incomplete mapping has null full-mapping quality metrics, with partial
correct-label count available. Exported-score audit costs use float64 subtraction;
native solver costs use stored float32 CSR values and may differ slightly.

`refine_assignment_by_edge_agreement.py` performs support-constrained strictly
improving swaps. Its search function has no ground-truth argument. Labels are
optional and opened only after search. A better edge score can worsen descriptor
cost or label accuracy; reports retain both outcomes. The default search stops at
500000 proposals or 60 seconds. Deadline checks occur between proposals, so one
proposal can overrun. Use the same attempt budget/seed and report stopping reason;
wall-time-limited runs need not replay identically across devices.

Run across graph seeds and noise levels and retain failures rather than averaging
only successful mappings. Compare baseline/refined reports for the same support
hash. Support repair changes support, so attribute recall/feasibility differences
to that intervention separately from refinement.

No stronger descriptor is adopted in this change. If later measurements justify
one, introduce a named variant and update CPU/GPU features and parity tests
together before comparing speed or scaling claims. Ground truth must remain
outside generation, repair, refinement objectives, and parameter selection on the
reported test pairs. Select parameters on separate development pairs.

Start with `python3 tests/test_correspondence_quality.py`; this checks matching
against a small exhaustive oracle, automorphism-related label ambiguity,
truth-free bounded refinement, and empty-row recall accounting. NumPy and
NetworkX are the same dependencies used by the existing feature audit.
