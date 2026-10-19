#ifndef SCGM_GPU_MATCHING_HPP
#define SCGM_GPU_MATCHING_HPP
#include "common.hpp"
#include "graph.hpp"
#include "features.hpp"
#include "candidate_generation.hpp"
#include "csr_matrix.hpp"
#include "gpu_topk.hpp"
#include "gpu_structural_index.hpp"
#include "device_buffers.hpp"
#include "device_csr.hpp"
#include <cstddef>


struct GPUTimings {
    double feature_extraction_ms = 0.0;
    double candidate_generation_ms = 0.0;
    double similarity_ms = 0.0;
    double topk_selection_ms = 0.0;
    double host_device_transfer_ms = 0.0;
    size_t peak_device_application_allocated_bytes = 0;
    double total_ms = 0.0;
};
struct DeviceGraph {
    int num_vertices = 0;
    std::size_t adjacency_entries = 0;

    // CSR adjacency:
    // row_ptr.size() == num_vertices + 1
    // col_idx.size() == adjacency_entries
    DeviceBuffer<int> row_ptr;
    DeviceBuffer<int> col_idx;
};

struct DeviceFeatureSet {
    int num_vertices = 0;

    // One NodeFeatures element per vertex in each allocated array.
    //
    // raw:
    //   Unmodified structural descriptors used by the structural index.
    //
    // normalized:
    //   Separate z-score descriptors used for similarity scoring.
    //
    // Both arrays use typed NodeFeatures access. Kernels must read/write
    // named fields rather than reinterpret these allocations as float*.
    DeviceBuffer<NodeFeatures> raw;
    DeviceBuffer<NodeFeatures> normalized;
};

struct DeviceCandidates {
    int num_sources = 0;
    int num_targets = 0;
    int requested_k = 0;
    std::size_t entry_count = 0;

    // Compact row layout:
    // row_offsets.size() == num_sources + 1
    // row_offsets[0] == 0
    // row_offsets[num_sources] == entry_count
    // target_ids.size() == similarities.size() == entry_count
    //
    // Each row is ordered by descending similarity, then ascending
    // target ID for exact ties. Target IDs are unique within a row.
    // Empty rows have equal consecutive offsets.
    DeviceBuffer<std::size_t> row_offsets;
    DeviceBuffer<int> target_ids;
    DeviceBuffer<float> similarities;
};

// struct DeviceCosts {
//     int num_rows = 0;
//     int num_cols = 0;
//     std::size_t entry_count = 0;

//     // CSR layout compatible with the existing host CSRMatrix:
//     // row_ptr.size() == num_rows + 1
//     // col_idx.size() == values.size() == entry_count
//     //
//     // Columns are strictly increasing within each row.
//     // Each stored value is 1.0f - the corresponding similarity.
//     // Missing entries remain forbidden assignments.
//     //
//     // Construction must reject entry_count > INT_MAX before converting
//     // offsets to int. Allocation sizes also require checked arithmetic.
//     DeviceBuffer<int> row_ptr;
//     DeviceBuffer<int> col_idx;
//     DeviceBuffer<float> values;
// };
DeviceCosts gpu_build_device_costs(
    const DeviceCandidates& candidates,
    const GPUContext& context);

    
// Builds a separate target-sorted CSR cost matrix on the GPU.
//
// Borrows candidates without modifying them.
// Cost = 1.0f - similarity.
// Rejects invalid offsets, IDs, duplicate IDs, and nonfinite scores.
// Rejects entry counts exceeding the existing host CSR's int capacity.
//
// Returns completed, device-owned costs.
// Downloads only a four-byte validation flag, never candidate/cost arrays.
DeviceCosts gpu_build_device_costs(
    const DeviceCandidates& candidates,
    const GPUContext& context);


//

struct GPUResidentDiagnostics {
    // Exact allocation sizes for DeviceBuffer-owned pipeline arrays.
    std::size_t adjacency_bytes = 0;
    std::size_t raw_feature_bytes = 0;
    std::size_t normalized_feature_bytes = 0;
    std::size_t candidate_bytes = 0;
    std::size_t cost_bytes = 0;

    // Logical payload sizes. Library temporary allocations are excluded.
    std::size_t index_payload_bytes = 0;
    std::size_t query_payload_bytes = 0;

    // Separate workspace categories; do not add these to the scratch budget.
    std::size_t scoring_selection_scratch_peak_bytes = 0;
    std::size_t normalization_workspace_peak_bytes = 0;
    std::size_t cost_workspace_peak_bytes = 0;

    // Count explicit application transfers, including validation flags.
    // Indexed library internals are not fully instrumented yet.
    std::size_t known_h2d_bytes = 0;
    std::size_t known_d2h_bytes = 0;
    bool transfer_tracking_complete = true;

    // Synchronized stage wall times, not kernel-only timings.
    double normalization_ms = 0.0;
    double cost_construction_ms = 0.0;
    double index_build_ms = 0.0;
    double retrieval_ms = 0.0;

    // Time spent downloading requested host inputs/diagnostics.
    double cpu_handoff_ms = 0.0;
};

    struct DeviceMatchingResult {
    // Retain adjacency throughout GPU stages.
    DeviceGraph source_graph;
    DeviceGraph target_graph;

    // Raw and normalized features have independent allocations.
    DeviceFeatureSet source_features;
    DeviceFeatureSet target_features;

    // Keep candidate rank order separate from target-sorted cost CSR.
    DeviceCandidates candidates;
    DeviceCosts costs;

    // Used only for the indexed candidate path.
    // Queries borrow the index logically; both owners remain alive here.
    bool indexed = false;
    GPUStructuralIndex structural_index;
    GPUStructuralQueries structural_queries;

    // Small host-side metadata; feature/candidate arrays stay on device.
    GPUTimings timings;
    GPUSelectionInfo selection;

    // Remains false if any allocation source is not accounted for.
    bool device_allocation_tracking_complete = false;

    bool device_resident = false;
    GPUResidentDiagnostics residency;
};

struct GPUDeviceCandidateSelectionResult {
    DeviceCandidates candidates;
    GPUSelectionInfo selection;
    double similarity_ms = 0.0;

    // Validation-status transfers only; candidate arrays stay on device.
    double transfer_ms = 0.0;

    // Borrowed normalized feature allocations, excluding raw features.
    std::size_t resident_feature_bytes = 0;
};

// Borrow completed normalized features on context.device_id.
// Inputs remain owned by the caller.
//
// Matching requires exactly K valid candidates per source.
// Returned candidates use compact rows of length K.
//
// Public calls complete context.stream before returning.
GPUDeviceCandidateSelectionResult gpu_score_exhaustive_device_pools(
    const DeviceFeatureSet& sources,
    const DeviceFeatureSet& targets,
    int k,
    const GPUContext& context,
    GPUSelector selector = GPUSelector::Parallel,
    const GPUTopKOptions& options = {});

GPUDeviceCandidateSelectionResult gpu_score_indexed_device_pools(
    const DeviceFeatureSet& sources,
    const DeviceFeatureSet& targets,
    const GPUStructuralIndex& index,
    const GPUStructuralQueries& queries,
    int k,
    const GPUContext& context,
    GPUSelector selector = GPUSelector::Parallel,
    const GPUTopKOptions& options = {});

struct GPUMatchingResult {
    std::vector<NodeFeatures> feats_source;
    std::vector<NodeFeatures> feats_target;
    std::vector<std::vector<Candidate>> candidates;
    GPUTimings timings;
    GPUSelectionInfo selection;
    // GPU index construction uses temporary library allocations not tracked by
    // the legacy allocation counter. Report an unavailable peak as null in CLI.
    bool device_allocation_tracking_complete = true;
    bool device_resident = false;
    GPUResidentDiagnostics residency;
};
struct GPUCandidateSelectionResult {
    std::vector<std::vector<Candidate>> candidates;
    GPUSelectionInfo selection;
    double similarity_ms = 0.0;
    double transfer_ms = 0.0;
    std::size_t resident_feature_bytes = 0;
};
struct GPUIndexedMatchingResult {
    GPUMatchingResult matching;
    StructuralIndex index_diagnostics;
    std::vector<StructuralRetrievalStats> retrieval_stats;
    double index_build_ms = 0.0;
    double retrieval_ms = 0.0;
};
std::vector<NodeFeatures> gpu_compute_features(const Graph&, const GPUContext&,
    double* transfer_ms = nullptr, bool normalize = true,
    size_t* allocated_bytes = nullptr);

// Preserved default serial reference. Explicit Parallel requests use the new
// bounded streaming path, including its bounded serial fallback.
std::vector<std::vector<Candidate>> gpu_compute_topk(
    const std::vector<NodeFeatures>&, const std::vector<NodeFeatures>&,
    int k, const GPUContext&, double* transfer_ms = nullptr,
    size_t* allocated_bytes = nullptr, double* similarity_kernel_ms = nullptr,
    double* topk_kernel_ms = nullptr, GPUSelector selector = GPUSelector::SerialHeap,
    const GPUTopKOptions& options = {}, GPUSelectionInfo* info = nullptr);

GPUMatchingResult gpu_compute_candidates(const Graph&, const Graph&, int k,
    const GPUContext&, bool joint_normalize = true,
    GPUSelector selector = GPUSelector::SerialHeap,
    const GPUTopKOptions& options = {});

// Scoring arrays are already normalized. Index and queries were built from raw
// features on the same device. Only selected bucket spans are scored.
GPUCandidateSelectionResult gpu_score_indexed_pools(
    const std::vector<NodeFeatures>& scoring_sources,
    const std::vector<NodeFeatures>& scoring_targets,
    const GPUStructuralIndex&, const GPUStructuralQueries&, int k,
    const GPUContext&, GPUSelector selector = GPUSelector::Parallel,
    const GPUTopKOptions& options = {});
GPUCandidateSelectionResult gpu_score_exhaustive_pools(
    const std::vector<NodeFeatures>& scoring_sources,
    const std::vector<NodeFeatures>& scoring_targets, int k, const GPUContext&,
    GPUSelector selector = GPUSelector::Parallel,
    const GPUTopKOptions& options = {});

// Complete indexed graph pipeline: raw features -> GPU index/query -> normalize
// -> selected GPU bucket scoring -> shared bounded selector.
GPUIndexedMatchingResult gpu_compute_indexed_candidates(
    const Graph&, const Graph&, int k, const StructuralIndexConfig&,
    const GPUContext&, bool joint_normalize = true,
    GPUSelector selector = GPUSelector::Parallel,
    const GPUTopKOptions& options = {});

size_t gpu_feature_memory_bytes(int num_vertices);
size_t gpu_candidate_memory_bytes(int num_source_vertices, int k);



// Resident graph pipeline. A nonnull index_config enables indexed scoring.
DeviceMatchingResult gpu_compute_device_candidates(
    const Graph& source,
    const Graph& target,
    int k,
    const GPUContext& context,
    bool joint_normalize = true,
    GPUSelector selector = GPUSelector::Parallel,
    const GPUTopKOptions& options = {},
    const StructuralIndexConfig* index_config = nullptr);

// Explicit normalization interface for independent correctness tests.
void gpu_normalize_device_features(
    DeviceFeatureSet& source,
    DeviceFeatureSet& target,
    bool joint_normalize,
    const GPUContext& context);

// Explicit host-consumer boundaries. These update residency diagnostics.
std::vector<std::vector<Candidate>> gpu_download_candidates(
    DeviceMatchingResult& result,
    const GPUContext& context);

CSRMatrix gpu_download_costs(
    DeviceMatchingResult& result,
    const GPUContext& context);

void gpu_download_normalized_features(
    DeviceMatchingResult& result,
    std::vector<NodeFeatures>& source,
    std::vector<NodeFeatures>& target,
    const GPUContext& context);

GPUMatchingResult gpu_download_matching_result(
    DeviceMatchingResult& result,
    const GPUContext& context);

void gpu_download_index_diagnostics(
    DeviceMatchingResult& result,
    StructuralIndex& index,
    std::vector<StructuralRetrievalStats>& retrieval,
    const GPUContext& context);

// Explicitly retained legacy graph pipeline.
GPUMatchingResult gpu_compute_candidates_legacy(
    const Graph& source,
    const Graph& target,
    int k,
    const GPUContext& context,
    bool joint_normalize,
    GPUSelector selector,
    const GPUTopKOptions& options);
#endif
