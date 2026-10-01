#include "lapjv.hpp"

/*
 * Native C++ LAPJV for SPARSE cost matrices in CSR format.
 *
 * This is a Shortest Augmenting Path (SAP) based solver for rectangular
 * or square linear assignment problems on sparse cost matrices.
 *
 * PERMUTATION CONVENTION:
 *   row_to_col[i] = j  means  source i -> target j
 *
 * SPARSE HANDLING:
 *   Only entries present in the CSR are considered as allowed assignments.
 *   Missing entries are FORBIDDEN (not zero cost).
 *
 * Implementation approach:
 *   Phase 1: Column reduction - for each column, find the minimum cost row
 *            among sparse entries.
 *   Phase 2: Reduction transfer - for rows assigned in phase 1, adjust duals.
 *   Phase 3: Augmenting row reduction.
 *   Phase 4: Augmentation via shortest augmenting path (Dijkstra-like).
 *
 * Because the matrix is sparse, we iterate over CSR entries rather than
 * scanning dense N*M arrays.
 *
 * NO call to scipy, Python, or external programs.
 */

AssignmentResult solve_lapjv(const CSRMatrix& cost) {
    int N = cost.rows;
    int M = cost.cols;

    AssignmentResult result;
    result.feasible = false;
    result.total_cost = 0.0;

    if (N == 0) {
        result.feasible = true;
        return result;
    }

    if (N > M) {
        // More rows than columns - cannot assign each row to a unique column
        result.feasible = false;
        return result;
    }

    // Check that each row has at least one candidate
    for (int i = 0; i < N; i++) {
        if (cost.row_ptr[i + 1] - cost.row_ptr[i] == 0) {
            std::cerr << "LAPJV: Row " << i << " has no candidates. Infeasible." << std::endl;
            result.row_to_col.assign(N, -1);
            result.col_to_row.assign(M, -1);
            result.feasible = false;
            return result;
        }
    }

    // Build column-to-row reverse index for sparse matrix
    // For each column j, store the list of (row, value_index) that have j as a candidate
    std::vector<std::vector<std::pair<int, int>>> col_to_rows(M);
    for (int i = 0; i < N; i++) {
        for (int idx = cost.row_ptr[i]; idx < cost.row_ptr[i + 1]; idx++) {
            int j = cost.col_idx[idx];
            col_to_rows[j].push_back({i, idx});
        }
    }

    // Dual variables
    std::vector<double> u(N, 0.0);  // row duals
    std::vector<double> v(M, 0.0);  // column duals

    // Assignment
    std::vector<int> row_to_col(N, -1);
    std::vector<int> col_to_row(M, -1);

    // Phase 1: Column reduction
    // For each column, find the row with minimum cost for that column
    // and tentatively assign it.
    for (int j = 0; j < M; j++) {
        if (col_to_rows[j].empty()) continue;

        double min_cost = std::numeric_limits<double>::max();
        int min_row = -1;
        for (auto& [row, vidx] : col_to_rows[j]) {
            double c = (double)cost.values[vidx];
            if (c < min_cost) {
                min_cost = c;
                min_row = row;
            }
        }
        v[j] = min_cost;

        if (min_row >= 0 && row_to_col[min_row] < 0) {
            row_to_col[min_row] = j;
            col_to_row[j] = min_row;
        }
    }

    // Phase 2: Reduction transfer
    // For already-assigned rows, update u[i]
    for (int i = 0; i < N; i++) {
        if (row_to_col[i] >= 0) {
            // u[i] = cost[i][row_to_col[i]] - v[row_to_col[i]]
            // This is 0 from phase 1, but let's compute properly
            int j = row_to_col[i];
            double c = (double)get_value(cost, i, j);
            u[i] = c - v[j];
        }
    }

    // Phase 3: Augmenting row reduction
    // Try to assign remaining unassigned rows
    for (int i = 0; i < N; i++) {
        if (row_to_col[i] >= 0) continue;

        // Find the two smallest reduced costs for row i
        double min1 = std::numeric_limits<double>::max();
        double min2 = std::numeric_limits<double>::max();
        int jmin = -1;

        for (int idx = cost.row_ptr[i]; idx < cost.row_ptr[i + 1]; idx++) {
            int j = cost.col_idx[idx];
            double reduced = (double)cost.values[idx] - v[j];
            if (reduced < min1) {
                min2 = min1;
                min1 = reduced;
                jmin = j;
            } else if (reduced < min2) {
                min2 = reduced;
            }
        }

        if (jmin < 0) continue;

        u[i] = min1;
        // If min1 < min2, unique minimum -> assign
        if (min1 < min2 - 1e-10) {
            if (col_to_row[jmin] < 0) {
                row_to_col[i] = jmin;
                col_to_row[jmin] = i;
            }
        }
    }

    // Phase 4: Augmentation via shortest augmenting path
    // For each unassigned row, find shortest augmenting path
    // using Dijkstra on the reduced cost graph.
    std::vector<double> dist(M);
    std::vector<int> pred(M);
    std::vector<bool> scanned(M);
    std::vector<int> cols_in_list;

    for (int i = 0; i < N; i++) {
        if (row_to_col[i] >= 0) continue;

        // Dijkstra-like shortest augmenting path from row i
        for (int j = 0; j < M; j++) {
            dist[j] = std::numeric_limits<double>::max();
            pred[j] = -1;
            scanned[j] = false;
        }
        cols_in_list.clear();

        // Initialize: scan row i
        for (int idx = cost.row_ptr[i]; idx < cost.row_ptr[i + 1]; idx++) {
            int j = cost.col_idx[idx];
            double reduced = (double)cost.values[idx] - u[i] - v[j];
            if (reduced < dist[j]) {
                dist[j] = reduced;
                pred[j] = i;
            }
        }

        int sink = -1;

        while (true) {
            // Find unscanned column with minimum dist
            double min_dist = std::numeric_limits<double>::max();
            int jmin = -1;
            for (int j = 0; j < M; j++) {
                if (!scanned[j] && dist[j] < min_dist) {
                    min_dist = dist[j];
                    jmin = j;
                }
            }

            if (jmin < 0 || min_dist >= std::numeric_limits<double>::max() * 0.5) {
                break;  // No augmenting path found
            }

            scanned[jmin] = true;
            cols_in_list.push_back(jmin);

            if (col_to_row[jmin] < 0) {
                // Found free column -> augmenting path found
                sink = jmin;
                break;
            }

            // Scan the row assigned to jmin
            int row_assigned = col_to_row[jmin];
            for (int idx = cost.row_ptr[row_assigned]; idx < cost.row_ptr[row_assigned + 1]; idx++) {
                int j2 = cost.col_idx[idx];
                if (scanned[j2]) continue;
                double new_dist = min_dist + (double)cost.values[idx]
                                  - u[row_assigned] - v[j2];
                if (new_dist < dist[j2]) {
                    dist[j2] = new_dist;
                    pred[j2] = row_assigned;
                }
            }
        }

        if (sink < 0) {
            // No augmenting path: infeasible
            result.row_to_col = row_to_col;
            result.col_to_row = col_to_row;
            result.feasible = false;
            std::cerr << "LAPJV: No augmenting path for row " << i
                      << ". Assignment infeasible." << std::endl;
            return result;
        }

        // Update dual variables
        for (int j : cols_in_list) {
            if (j == sink) {
                v[j] += dist[j] - dist[sink];
            } else {
                // Only update scanned columns
                double delta = dist[j] - dist[sink];
                v[j] += delta;
                if (col_to_row[j] >= 0) {
                    u[col_to_row[j]] -= delta;
                }
            }
        }
        u[i] += dist[sink];

        // Trace back augmenting path and flip assignments
        int j = sink;
        while (true) {
            int row_in_path = pred[j];
            // The row pred[j] was previously assigned to some column
            // (or is the starting row i)
            int prev_col = row_to_col[row_in_path];
            row_to_col[row_in_path] = j;
            col_to_row[j] = row_in_path;

            if (row_in_path == i) break;
            j = prev_col;
        }
    }

    // Verify complete assignment
    result.row_to_col = row_to_col;
    result.col_to_row = col_to_row;
    result.total_cost = 0.0;
    result.feasible = true;

    for (int i = 0; i < N; i++) {
        if (row_to_col[i] < 0) {
            result.feasible = false;
            std::cerr << "LAPJV: Row " << i << " unassigned after augmentation." << std::endl;
            return result;
        }
        float c = get_value(cost, i, row_to_col[i]);
        if (c >= SCGM_INF * 0.5f) {
            result.feasible = false;
            std::cerr << "LAPJV: Row " << i << " assigned to forbidden column "
                      << row_to_col[i] << "." << std::endl;
            return result;
        }
        result.total_cost += (double)c;
    }

    return result;
}