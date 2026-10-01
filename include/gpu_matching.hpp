#ifndef SCGM_GPU_MATCHING_HPP
#define SCGM_GPU_MATCHING_HPP

// Public interface for CUDA feature and candidate stages.
// Sparse CSR construction, LAPJV assignment, and Sinkhorn remain on the CPU.

#include "common.hpp"
#include "graph.hpp"
#include "features.hpp"
#include "candidate_generation.hpp"

struct GPUTimings {
    double feature_extraction_ms = 0.0;
    double candidate_generation_ms = 0.0;
    double similarity_ms = 0.0;
    double topk_selection_ms = 0.0;
    double host_device_transfer_ms = 0.0;
    size_t peak_device_application_allocated_bytes = 0;
    double total_ms = 0.0;
};

struct GPUMatchingResult {
    std::vector<NodeFeatures> feats_source;
    std::vector<NodeFeatures> feats_target;
    std::vector<std::vector<Candidate>> candidates;
    GPUTimings timings;
};

std::vector<NodeFeatures> gpu_compute_features(
    const Graph& graph,
    const GPUContext& context,
    double* transfer_ms = nullptr,
    bool normalize = true,
    size_t* allocated_bytes = nullptr);

std::vector<std::vector<Candidate>> gpu_compute_topk(
    const std::vector<NodeFeatures>& feats_source,
    const std::vector<NodeFeatures>& feats_target,
    int k,
    const GPUContext& context,
    double* transfer_ms = nullptr,
    size_t* allocated_bytes = nullptr,
    double* similarity_kernel_ms = nullptr,
    double* topk_kernel_ms = nullptr);

GPUMatchingResult gpu_compute_candidates(
    const Graph& source,
    const Graph& target,
    int k,
    const GPUContext& context,
    bool joint_normalize = true);

size_t gpu_feature_memory_bytes(int num_vertices);
size_t gpu_candidate_memory_bytes(int num_source_vertices, int k);

#endif // SCGM_GPU_MATCHING_HPP
