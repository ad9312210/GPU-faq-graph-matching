#ifndef SCGM_ASSIGNMENT_SCHEDULER_HPP
#define SCGM_ASSIGNMENT_SCHEDULER_HPP

#include "assignment_components.hpp"
#include "gpu_assignment.hpp"
#include <string>

struct GPUAssignmentSchedulerOptions {
    GPUAssignmentOptions assignment;
    std::size_t component_workspace_budget_bytes = 64u * 1024u * 1024u;
    // Only unsupported cooperative-launch capability permits this fallback.
    // Invalid launch sizes, occupancy, allocations, and budgets still throw.
    bool allow_unsupported_cooperative_fallback = false;
};

struct GPUComponentDispatch {
    int component = -1;
    int rows = 0;
    int columns = 0;
    std::size_t edges = 0;
    GPUAssignmentExecution requested = GPUAssignmentExecution::Block;
    GPUAssignmentExecution effective = GPUAssignmentExecution::Block;
    bool executed = false;
    std::string reason = "none";
    GPUAssignmentStatus status = GPUAssignmentStatus::NumericalFailure;
    int blocks = 0;
    std::size_t workspace_peak_bytes = 0;
    std::size_t compact_payload_bytes = 0;
    double compaction_ms = 0.0;
    double assignment_kernel_ms = 0.0;
    double total_ms = 0.0;
};

struct GPUAssignmentScheduleResult {
    // Original matrix dimensions/order; existing download/certificate API
    // validates this combined result against the original host CSR.
    GPUAssignmentResult assignment;
    std::vector<GPUComponentDispatch> dispatch;
    std::size_t discovery_workspace_bytes = 0;
    std::size_t workspace_peak_bytes = 0;
    // Compact CSR/maps + local solver outputs, excluding original CSR and
    // global solver outputs. Sequential components reuse their lifetimes.
    std::size_t component_payload_peak_bytes = 0;
    std::size_t metadata_download_bytes = 0;
    double discovery_ms = 0.0;
    double total_ms = 0.0;
};

const char* gpu_assignment_execution_name(GPUAssignmentExecution execution) noexcept;

// Experimental explicit dispatch. Components run sequentially on one stream;
// this does not yet batch components or select size thresholds automatically.
// Isolated columns need no solve; row-heavy components prove infeasibility.
// Other components are solved to determine feasibility, including Hall failures.
// Public completion is synchronized. No matching/dual arrays are downloaded.
GPUAssignmentScheduleResult gpu_schedule_sparse_assignment(
    const DeviceCSR& costs, const GPUContext& context,
    const GPUAssignmentSchedulerOptions& options = {});

#endif
