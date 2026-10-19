# GPU-resident pipeline boundary

## Execution flow

1. Upload source and target graph adjacency once per pipeline execution.
2. Compute raw features on the GPU.
3. Compute joint normalization on the GPU.
4. Score and select candidates on the GPU.
5. Construct sparse candidate costs on the GPU.
6. Download data when a CPU consumer or requested output requires it.

Separate-per-graph normalization remains available when requested.

## Persistent device data

Keep the following allocations alive across stages:

- Source and target CSR adjacency.
- Raw source and target features.
- Normalized source and target features.
- Selected candidate target IDs, scores, and row offsets.
- Sparse cost matrix offsets, target IDs, and values.
- Structural index and retrieval data when indexed selection is enabled.

Raw features remain unchanged because structural indexing uses them.
Scoring uses normalized features.

Candidates retain score-descending order, with lower target ID first
for exact ties. Cost CSR rows are ordered by ascending target ID.
Costs remain 1.0f - similarity.

## CPU handoff

Assignment, candidate-support analysis, and Sinkhorn remain CPU consumers.

Download the inputs those consumers require at explicit handoff points.
Download features and detailed diagnostics only when required by a
consumer or requested output.

This scope does not make assignment or Sinkhorn GPU-resident.

## Ownership and compatibility

Each device allocation has one owner and automatic cleanup.
Borrowed stage inputs do not free their owners' allocations.

The execution context and stream outlive operations using their buffers.
Complete pending work before releasing buffers or destroying the context.

Existing host APIs remain available as compatibility wrappers around
the device pipeline and explicit downloads.

## Memory scope

The existing 64 MiB limit covers scoring batches and selection workspace.

Persistent adjacency, features, index data, final candidates, and costs
are reported separately. Do not allocate a dense source-by-target matrix.

Check allocation arithmetic and report allocation failures explicitly.

## Acceptance criteria

- No feature download and re-upload between extraction and scoring.
- Joint normalization statistics and application execute on GPU.
- Selection does not download intermediate candidate rows.
- Sparse costs are constructed on GPU.
- Downloads occur at documented consumer or output boundaries.
- Existing host callers remain supported.
- Transfer bytes, persistent memory, and scratch memory are reported.