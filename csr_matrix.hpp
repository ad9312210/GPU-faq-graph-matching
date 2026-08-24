#ifndef SCGM_CSR_MATRIX_HPP
#define SCGM_CSR_MATRIX_HPP

#include "common.hpp"
#include "candidate_generation.hpp"

/*
 * CSR sparse cost matrix.
 *
 * row_ptr.size() = rows + 1
 * col_idx.size() = NNZ
 * values.size()  = NNZ
 *
 * NNZ <= rows * K
 *
 * Missing entries represent FORBIDDEN assignments, not zero cost.
 * No INF values are stored for non-candidates.
 */
struct CSRMatrix {
    int rows;
    int cols;

    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<float> values;

    int nnz() const { return static_cast<int>(col_idx.size()); }
};

CSRMatrix build_sparse_cost_matrix(
    const std::vector<std::vector<Candidate>>& candidates,
    int num_source,
    int num_target);

/*
 * Retrieve a stored value. Returns SCGM_INF if (row,col) is not stored
 * (i.e., the assignment is forbidden). This function is for TESTING ONLY;
 * the main algorithm never constructs a dense view.
 */
float get_value(const CSRMatrix& matrix, int row, int col);

bool validate_csr(const CSRMatrix& matrix, int K);

void print_csr_info(const CSRMatrix& matrix);

size_t csr_memory_bytes(const CSRMatrix& matrix);

#endif // SCGM_CSR_MATRIX_HPP