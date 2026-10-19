#ifndef SCGM_MULTI_GPU_MATCHING_HPP
#define SCGM_MULTI_GPU_MATCHING_HPP
#include "gpu_matching.hpp"
#include "assignment_scheduler.hpp"

struct MultiGPUOptions {
    // Application payload admitted per worker; scoring/solver workspace is
    // added separately during admission. This is not a runtime/process cap.
    std::size_t resident_budget_bytes = 1024u * 1024u * 1024u;
    std::size_t free_memory_reserve_bytes = 64u * 1024u * 1024u;
    GPUSelector selector = GPUSelector::Parallel;
    GPUTopKOptions topk;
    GPUAssignmentSchedulerOptions scheduler;
};
struct MultiGPUCandidateWorker {
    int device = -1;
    bool executed = false;
    std::size_t source_begin = 0, source_end = 0;
    std::size_t target_replica_bytes = 0, resident_payload_bytes = 0;
    std::size_t free_bytes_at_admission = 0;
    std::size_t candidate_download_bytes = 0;
    GPUSelectionInfo selection;
    double similarity_ms = 0, wall_ms = 0;
};
struct MultiGPUCandidateResult {
    std::vector<std::vector<Candidate>> candidates;
    std::vector<MultiGPUCandidateWorker> workers;
    double wall_ms = 0;
};
// Inputs must already have GLOBAL pair normalization; never normalize shards.
// One host thread/stream per device; source intervals are disjoint and target
// replicas identical. Outputs retain global source and target ID order.
MultiGPUCandidateResult multi_gpu_generate_candidates(
    const std::vector<NodeFeatures>& sources, const std::vector<NodeFeatures>& targets,
    int k, const std::vector<int>& devices, const MultiGPUOptions& options = {});

struct MultiGPUComponentDispatch {
    GPUComponentDispatch component;
    int device = -1;
};
struct MultiGPUAssignmentResult {
    GPUAssignmentHostResult assignment;
    std::vector<MultiGPUComponentDispatch> dispatch;
    std::size_t known_h2d_bytes = 0, known_d2h_bytes = 0;
    // Validation flags/library internals are excluded from known transfers.
    bool transfer_tracking_complete = false;
    double discovery_ms = 0, wall_ms = 0;
};
// Discover components on the coordinator GPU, download only component metadata,
// then build host compact workloads and upload each to ONE worker GPU.
// Stable greedy placement by edge count; no component is split across devices.
// Matching and duals are remapped and certified against the original CSR.
// Host staging/CUDA runtime only: no NCCL, peer-access, or distributed dual solve.
MultiGPUAssignmentResult multi_gpu_solve_assignment(
    const CSRMatrix& costs, const std::vector<int>& devices,
    const MultiGPUOptions& options = {});
#endif
