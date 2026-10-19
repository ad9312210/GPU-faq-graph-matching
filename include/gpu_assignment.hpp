#ifndef SCGM_GPU_ASSIGNMENT_HPP
#define SCGM_GPU_ASSIGNMENT_HPP

#include "common.hpp"
#include "device_csr.hpp"
#include "lapjv.hpp"

#include <cstddef>
#include <vector>

enum class GPUAssignmentStatus {
    Success = 0,
    Infeasible = 1,
    NumericalFailure = 2
};

enum class GPUAssignmentExecution { Warp, Block, Cooperative };

struct GPUAssignmentOptions {
    // Assignment workspace only.
    // CSR and persistent matching/dual arrays are excluded.
    std::size_t workspace_budget_bytes = 64u * 1024u * 1024u;
    GPUAssignmentExecution execution = GPUAssignmentExecution::Block;
    // Explicit experimental launch size; no automatic dispatch threshold yet.
    int cooperative_blocks = 2;
};

struct GPUAssignmentResult {
    int num_rows = 0;
    int num_cols = 0;

    // One owner per allocation; these remain available after solving.
    DeviceBuffer<int> row_to_col;
    DeviceBuffer<int> col_to_row;
    DeviceBuffer<double> row_dual;
    DeviceBuffer<double> col_dual;

    GPUAssignmentStatus status = GPUAssignmentStatus::NumericalFailure;

    double kernel_ms = 0.0;
    unsigned long long search_steps = 0;
    std::size_t workspace_bytes = 0;
    GPUAssignmentExecution execution = GPUAssignmentExecution::Block;
    int execution_blocks = 1;
};

struct GPUAssignmentCertificate {
    bool primal_feasible = false;
    bool dual_feasible = false;
    bool complementary_slackness = false;
    bool optimal = false;

    double primal_objective = 0.0;
    double dual_objective = 0.0;
    double objective_gap = 0.0;

    double max_dual_violation = 0.0;
    double max_matched_slack = 0.0;
};

struct GPUAssignmentHostResult {
    AssignmentResult assignment;
    std::vector<double> row_dual;
    std::vector<double> col_dual;
    GPUAssignmentCertificate certificate;

    GPUAssignmentStatus status = GPUAssignmentStatus::NumericalFailure;
    double kernel_ms = 0.0;
    unsigned long long search_steps = 0;
    std::size_t workspace_bytes = 0;
    std::size_t download_bytes = 0;
};

// Covers every source row using distinct target columns.
// Requires num_rows <= num_cols.
//
// Missing CSR entries are forbidden assignments.
// Costs may be negative, but must be finite.
// Inputs remain owned by the caller.
//
// Returns completed device matching and dual state.
// Infeasible support is reported explicitly; there is no automatic fallback.
GPUAssignmentResult gpu_solve_sparse_assignment(
    const DeviceCSR& costs,
    const GPUContext& context,
    const GPUAssignmentOptions& options = {});

// Explicit host handoff and independent CPU certificate evaluation.
// host_costs must represent the same matrix passed to the solver.
//
// Dual conditions for row equality / column capacity constraints:
//   row_dual[i] + col_dual[j] <= cost[i,j]
//   col_dual[j] <= 0
// Matched edges are tight; unused columns have zero dual.
GPUAssignmentHostResult gpu_download_assignment(
    const GPUAssignmentResult& result,
    const CSRMatrix& host_costs,
    const GPUContext& context,
    double absolute_tolerance = 1e-9,
    double relative_tolerance = 1e-9);

#endif
