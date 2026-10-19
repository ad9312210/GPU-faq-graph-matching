#ifndef SCGM_ASSIGNMENT_COMPONENTS_HPP
#define SCGM_ASSIGNMENT_COMPONENTS_HPP

#include "common.hpp"
#include "device_csr.hpp"
#include <vector>

struct AssignmentComponent {
    // Local-to-original maps: rows[local_row] and columns[local_column].
    // IDs are sorted in original matrix order, never arbitrary partitions.
    std::vector<int> rows;
    std::vector<int> columns;
    std::size_t edges = 0;
    // Necessary condition only; true does not establish Hall feasibility.
    bool passes_cardinality_check = true;
};

struct AssignmentComponents {
    std::vector<AssignmentComponent> components;
    std::vector<int> row_component;
    std::vector<int> column_component;
    // Original-to-local maps, interpreted within the component above:
    // components[row_component[r]].rows[row_local_index[r]] == r
    // components[column_component[c]].columns[column_local_index[c]] == c
    // Every vertex occurs exactly once, including isolated vertices.
    std::vector<int> row_local_index;
    std::vector<int> column_local_index;
    int propagation_rounds = 0;
    std::size_t workspace_bytes = 0;
    std::size_t download_bytes = 0;
};

// GPU minimum-label propagation on the undirected bipartite support.
// Includes isolated vertices. Returns explicit host scheduling metadata;
// CSR costs/target arrays remain on device and are never modified.
// Budget covers labels + per-row edge counts + one convergence flag only.
// CSR validation workspace and returned host metadata are excluded.
AssignmentComponents gpu_find_assignment_components(
    const DeviceCSR& costs, const GPUContext& context,
    std::size_t workspace_budget_bytes = 64u * 1024u * 1024u);

struct CompactAssignmentComponent {
    DeviceCSR costs;
    DeviceBuffer<int> original_rows;
    DeviceBuffer<int> original_columns;
    bool passes_cardinality_check = true;
    std::size_t workspace_bytes = 0;
};

// Compact one component at a time without changing the original matrix.
// Local IDs follow the sorted original-ID maps. A failed cardinality check
// proves infeasibility; otherwise pass costs to matching/assignment to decide.
// Workspace budget covers a global-column inverse map and validation flag.
// Owned output CSR/maps, input CSR, and validation helpers are excluded.
// Requires a component produced for this exact, unchanged support.
CompactAssignmentComponent gpu_compact_assignment_component(
    const DeviceCSR& costs, const AssignmentComponent& component,
    const GPUContext& context,
    std::size_t workspace_budget_bytes = 64u * 1024u * 1024u);

#endif
