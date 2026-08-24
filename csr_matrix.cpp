#include "csr_matrix.hpp"

CSRMatrix build_sparse_cost_matrix(
    const std::vector<std::vector<Candidate>>& candidates,
    int num_source,
    int num_target)
{
    CSRMatrix csr;
    csr.rows = num_source;
    csr.cols = num_target;
    csr.row_ptr.resize(num_source + 1);

    // Count entries per row
    csr.row_ptr[0] = 0;
    for (int i = 0; i < num_source; i++) {
        csr.row_ptr[i + 1] = csr.row_ptr[i] + (int)candidates[i].size();
    }

    int nnz = csr.row_ptr[num_source];
    csr.col_idx.resize(nnz);
    csr.values.resize(nnz);

    for (int i = 0; i < num_source; i++) {
        int offset = csr.row_ptr[i];
        // Sort candidates by target index for CSR convention
        std::vector<Candidate> sorted_cands = candidates[i];
        std::sort(sorted_cands.begin(), sorted_cands.end(),
                  [](const Candidate& a, const Candidate& b) {
                      return a.target < b.target;
                  });
        for (int j = 0; j < (int)sorted_cands.size(); j++) {
            csr.col_idx[offset + j] = sorted_cands[j].target;
            // cost = 1 - similarity
            csr.values[offset + j] = 1.0f - sorted_cands[j].similarity;
        }
    }

    return csr;
}

float get_value(const CSRMatrix& matrix, int row, int col) {
    if (row < 0 || row >= matrix.rows || col < 0 || col >= matrix.cols) {
        return SCGM_INF;
    }
    int start = matrix.row_ptr[row];
    int end = matrix.row_ptr[row + 1];
    for (int j = start; j < end; j++) {
        if (matrix.col_idx[j] == col) {
            return matrix.values[j];
        }
    }
    return SCGM_INF;  // Not a candidate -> forbidden
}

bool validate_csr(const CSRMatrix& matrix, int K) {
    if ((int)matrix.row_ptr.size() != matrix.rows + 1) {
        std::cerr << "CSR validation failed: row_ptr size = " << matrix.row_ptr.size()
                  << ", expected " << matrix.rows + 1 << std::endl;
        return false;
    }
    if (matrix.row_ptr[0] != 0) {
        std::cerr << "CSR validation failed: row_ptr[0] = " << matrix.row_ptr[0] << std::endl;
        return false;
    }
    int nnz = matrix.row_ptr[matrix.rows];
    if ((int)matrix.col_idx.size() != nnz) {
        std::cerr << "CSR validation failed: col_idx.size() = " << matrix.col_idx.size()
                  << ", expected " << nnz << std::endl;
        return false;
    }
    if ((int)matrix.values.size() != nnz) {
        std::cerr << "CSR validation failed: values.size() = " << matrix.values.size()
                  << ", expected " << nnz << std::endl;
        return false;
    }
    for (int i = 0; i < matrix.rows; i++) {
        if (matrix.row_ptr[i + 1] < matrix.row_ptr[i]) {
            std::cerr << "CSR validation failed: row_ptr not monotonically increasing at row "
                      << i << std::endl;
            return false;
        }
        int row_nnz = matrix.row_ptr[i + 1] - matrix.row_ptr[i];
        if (row_nnz > K) {
            std::cerr << "CSR validation failed: row " << i << " has " << row_nnz
                      << " entries, exceeds K=" << K << std::endl;
            return false;
        }
    }
    if (nnz > matrix.rows * K) {
        std::cerr << "CSR validation failed: total NNZ=" << nnz
                  << " exceeds N*K=" << matrix.rows * K << std::endl;
        return false;
    }
    // Check col_idx range
    for (int j = 0; j < nnz; j++) {
        if (matrix.col_idx[j] < 0 || matrix.col_idx[j] >= matrix.cols) {
            std::cerr << "CSR validation failed: col_idx[" << j << "] = "
                      << matrix.col_idx[j] << " out of range [0, " << matrix.cols << ")"
                      << std::endl;
            return false;
        }
    }
    return true;
}

void print_csr_info(const CSRMatrix& matrix) {
    int nnz = matrix.nnz();
    int max_row_nnz = 0;
    for (int i = 0; i < matrix.rows; i++) {
        int row_nnz = matrix.row_ptr[i + 1] - matrix.row_ptr[i];
        max_row_nnz = std::max(max_row_nnz, row_nnz);
    }
    std::cout << "CSR:" << std::endl;
    std::cout << "    rows = " << matrix.rows << std::endl;
    std::cout << "    cols = " << matrix.cols << std::endl;
    std::cout << "    nnz = " << nnz << std::endl;
    std::cout << "    max row nnz = " << max_row_nnz << std::endl;
    std::cout << "    memory = " << csr_memory_bytes(matrix) << " bytes" << std::endl;
}

size_t csr_memory_bytes(const CSRMatrix& matrix) {
    return (matrix.row_ptr.size() * sizeof(int)) +
           (matrix.col_idx.size() * sizeof(int)) +
           (matrix.values.size() * sizeof(float));
}