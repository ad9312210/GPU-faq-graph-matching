#include "lapjv.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

/*
 * Native CPU sparse primal-dual shortest-augmenting-path solver.
 *
 * Contract:
 *   - Every source row must receive one distinct target column.
 *   - Rectangular inputs are supported when rows <= columns.
 *   - Only stored CSR entries are allowed assignments.
 *   - Stored zero and negative costs are supported.
 *   - Malformed input throws std::invalid_argument.
 *   - Valid but infeasible input returns feasible = false.
 *
 * Initialization:
 *   - Empty matching.
 *   - Row potentials equal row minima.
 *   - Column potentials equal zero.
 *
 * This retains the solve_lapjv interface for compatibility.
 * It does not implement the full LAPJV initialization sequence.
 *
 * Sparse edges are used for relaxation. Selecting the next
 * minimum-distance column currently scans all columns.
 */

AssignmentResult solve_lapjv(const CSRMatrix& cost) {
    const int N = cost.rows;
    const int M = cost.cols;
    const double infinity = std::numeric_limits<double>::infinity();

    // Validate dimensions before allocating dimension-dependent arrays.
    if (N < 0 || M < 0) {
        throw std::invalid_argument(
            "solve_lapjv: negative matrix dimensions");
    }

    // Validate the CSR structure.
    if (cost.row_ptr.size() != static_cast<std::size_t>(N) + 1 ||
        cost.col_idx.size() != cost.values.size() ||
        cost.row_ptr.empty() ||
        cost.row_ptr.front() != 0) {
        throw std::invalid_argument(
            "solve_lapjv: malformed CSR arrays");
    }

    const std::size_t nnz = cost.values.size();

    for (int row = 0; row < N; ++row) {
        const int begin = cost.row_ptr[row];
        const int end = cost.row_ptr[row + 1];

        if (begin < 0 ||
            end < begin ||
            static_cast<std::size_t>(end) > nnz) {
            throw std::invalid_argument(
                "solve_lapjv: invalid CSR row offsets");
        }

        int previous_column = -1;

        for (int edge = begin; edge < end; ++edge) {
            const int column = cost.col_idx[edge];

            if (column < 0 || column >= M ||
                column <= previous_column) {
                throw std::invalid_argument(
                    "solve_lapjv: columns must be in range, "
                    "sorted, and unique within each row");
            }

            if (!std::isfinite(cost.values[edge])) {
                throw std::invalid_argument(
                    "solve_lapjv: stored costs must be finite");
            }

            previous_column = column;
        }
    }

    if (cost.row_ptr.back() < 0 ||
        static_cast<std::size_t>(cost.row_ptr.back()) != nnz) {
        throw std::invalid_argument(
            "solve_lapjv: final CSR offset does not equal nnz");
    }

    AssignmentResult result;
    result.feasible = false;
    result.total_cost = 0.0;
    result.row_to_col.assign(N, -1);
    result.col_to_row.assign(M, -1);

    // The empty assignment has cost zero.
    if (N == 0) {
        result.feasible = true;
        return result;
    }

    if (N > M) {
        return result;
    }

    for (int row = 0; row < N; ++row) {
        if (cost.row_ptr[row] == cost.row_ptr[row + 1]) {
            return result;
        }
    }

    // Dual potentials:
    // reduced_cost(row, column) = cost - u[row] - v[column].
    std::vector<double> u(N, 0.0);
    std::vector<double> v(M, 0.0);

    for (int row = 0; row < N; ++row) {
        double minimum = infinity;

        for (int edge = cost.row_ptr[row];
             edge < cost.row_ptr[row + 1];
             ++edge) {
            minimum = std::min(
                minimum,
                static_cast<double>(cost.values[edge]));
        }

        u[row] = minimum;
    }

    std::vector<double> distance(M);
    std::vector<int> predecessor(M);
    std::vector<bool> scanned(M);

    // Add one source to the matching at each iteration.
    for (int root = 0; root < N; ++root) {
        std::fill(distance.begin(), distance.end(), infinity);
        std::fill(predecessor.begin(), predecessor.end(), -1);
        std::fill(scanned.begin(), scanned.end(), false);

        // Initial distances from the unassigned source.
        for (int edge = cost.row_ptr[root];
             edge < cost.row_ptr[root + 1];
             ++edge) {
            const int column = cost.col_idx[edge];

            distance[column] =
                static_cast<double>(cost.values[edge])
                - u[root] - v[column];

            predecessor[column] = root;
        }

        int sink = -1;

        while (true) {
            int next_column = -1;
            double minimum_distance = infinity;

            for (int column = 0; column < M; ++column) {
                if (!scanned[column] &&
                    distance[column] < minimum_distance) {
                    minimum_distance = distance[column];
                    next_column = column;
                }
            }

            // No free target is reachable from this source.
            if (next_column == -1) {
                return result;
            }

            scanned[next_column] = true;

            const int matched_row =
                result.col_to_row[next_column];

            if (matched_row == -1) {
                sink = next_column;
                break;
            }

            // Traverse the matched edge back to its source,
            // then relax that source's allowed outgoing edges.
            for (int edge = cost.row_ptr[matched_row];
                 edge < cost.row_ptr[matched_row + 1];
                 ++edge) {
                const int column = cost.col_idx[edge];

                if (scanned[column]) {
                    continue;
                }

                const double candidate_distance =
                    minimum_distance
                    + static_cast<double>(cost.values[edge])
                    - u[matched_row] - v[column];

                if (candidate_distance < distance[column]) {
                    distance[column] = candidate_distance;
                    predecessor[column] = matched_row;
                }
            }
        }

        const double path_distance = distance[sink];

        // Update potentials while preserving tight matched edges.
        u[root] += path_distance;

        for (int column = 0; column < M; ++column) {
            if (!scanned[column]) {
                continue;
            }

            const double change =
                path_distance - distance[column];

            v[column] -= change;

            const int matched_row = result.col_to_row[column];

            if (matched_row != -1) {
                u[matched_row] += change;
            }
        }

        // Flip assignments along the augmenting path.
        int column = sink;

        while (true) {
            const int row = predecessor[column];
            const int previous_column = result.row_to_col[row];

            result.row_to_col[row] = column;
            result.col_to_row[column] = row;

            if (row == root) {
                break;
            }

            column = previous_column;
        }
    }

    // Verify the final matching and recompute its original cost.
    std::vector<bool> target_used(M, false);
    double total_cost = 0.0;

    for (int row = 0; row < N; ++row) {
        const int column = result.row_to_col[row];

        if (column < 0 || column >= M ||
            target_used[column] ||
            result.col_to_row[column] != row) {
            throw std::logic_error(
                "solve_lapjv: invalid matching produced");
        }

        target_used[column] = true;
        bool found = false;

        for (int edge = cost.row_ptr[row];
             edge < cost.row_ptr[row + 1];
             ++edge) {
            if (cost.col_idx[edge] == column) {
                total_cost +=
                    static_cast<double>(cost.values[edge]);
                found = true;
                break;
            }
        }

        if (!found) {
            throw std::logic_error(
                "solve_lapjv: forbidden edge selected");
        }
    }

    result.total_cost = total_cost;
    result.feasible = true;
    return result;
}