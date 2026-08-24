#ifndef SCGM_LAPJV_HPP
#define SCGM_LAPJV_HPP

#include "common.hpp"
#include "csr_matrix.hpp"

/*
 * Native C++ LAPJV-style linear assignment solver for SPARSE cost matrices
 * stored in CSR format.
 *
 * This is a NATIVE implementation. It does NOT call:
 *   - scipy.optimize.linear_sum_assignment
 *   - Any Python code
 *   - Any external program or library
 *
 * PERMUTATION CONVENTION:
 *   row_to_col[i] = j means source node i is assigned to target node j.
 *
 * SPARSE HANDLING:
 *   Missing CSR entries represent FORBIDDEN assignments.
 *   The solver distinguishes stored candidates from missing candidates.
 *   Missing edges are NOT treated as zero cost.
 *
 * If no feasible assignment exists, feasible is set to false.
 */

struct AssignmentResult {
    std::vector<int> row_to_col;   // row_to_col[i] = j: source i -> target j
    std::vector<int> col_to_row;   // col_to_row[j] = i: target j <- source i
    double total_cost;
    bool feasible;
};

AssignmentResult solve_lapjv(const CSRMatrix& cost);

#endif // SCGM_LAPJV_HPP