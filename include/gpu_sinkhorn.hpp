#ifndef SCGM_GPU_SINKHORN_HPP
#define SCGM_GPU_SINKHORN_HPP

#include "common.hpp"
#include "device_csr.hpp"
#include "sparse_sinkhorn.hpp"

#include <cstddef>
#include <limits>

enum class GPUSinkhornStatus {
    Converged = 0,
    IterationLimit = 1,
    InvalidSupport = 2,
    NumericalFailure = 3
};

struct GPUSinkhornOptions {
    double epsilon = 0.1;
    int max_iterations = 500;
    double tolerance = 1e-3;

    // Iteration workspace only. CSR, reverse-index construction, and owned
    // probability/scaling outputs are separate from this budget.
    std::size_t workspace_budget_bytes = 64u * 1024u * 1024u;
};

struct GPUSinkhornDiagnostics {
    GPUSinkhornStatus status = GPUSinkhornStatus::IterationLimit;
    bool probabilities_valid = false;
    int iterations = 0;
    double max_row_marginal_error = std::numeric_limits<double>::infinity();
    double max_column_marginal_error = std::numeric_limits<double>::infinity();
    double setup_ms = 0.0;
    double reverse_index_build_ms = 0.0;
    double iteration_gpu_ms = 0.0;
    double total_ms = 0.0;
    std::size_t workspace_bytes = 0;
    std::size_t persistent_result_bytes = 0;
    std::size_t reverse_index_bytes = 0;
};

struct GPUSinkhornResult {
    // Probabilities have exactly the CSR entry order. Scaling remains in log
    // form. No assignment or CSR cost entries are modified.
    DeviceBuffer<double> edge_probabilities;
    DeviceBuffer<double> row_log_scale;
    DeviceBuffer<double> column_log_scale;
    GPUSinkhornDiagnostics diagnostics;
};

// Square matrices, unit row/column marginals, matching the CPU reference.
// Borrows costs and may populate its optional reverse index.
// Malformed inputs/parameters throw; scaling failures have explicit statuses.
// Nonempty row/column support alone does not guarantee convergence.
// Completes the supplied stream before returning.
GPUSinkhornResult gpu_sparse_sinkhorn(
    DeviceCSR& costs,
    const GPUContext& context,
    const GPUSinkhornOptions& options = {});

// Explicit confidence download, including valid nonconverged iterates.
// Rejects results without a valid finite probability array.
SparseSinkhornResult gpu_download_sinkhorn(
    const GPUSinkhornResult& result,
    const GPUContext& context);

const char* gpu_sinkhorn_status_name(GPUSinkhornStatus status);

#endif
