#include "sparse_sinkhorn.hpp"

int main() {
    std::vector<std::vector<Candidate>> candidates = {
        {{0, 0.95f}, {1, 0.80f}},
        {{0, 0.80f}, {1, 0.95f}},
    };
    const CSRMatrix csr = build_sparse_cost_matrix(candidates, 2, 2);
    const auto result = sparse_sinkhorn(csr, 0.1, 100, 1e-8);
    if (!result.converged) {
        std::cerr << "test_sparse_sinkhorn: did not converge" << std::endl;
        return 1;
    }
    if (result.max_row_marginal_error > 1e-8 ||
        result.max_column_marginal_error > 1e-8) {
        std::cerr << "test_sparse_sinkhorn: marginal residual too large" << std::endl;
        return 1;
    }
    for (int row = 0; row < csr.rows; ++row) {
        double sum = 0.0;
        for (int edge = csr.row_ptr[row]; edge < csr.row_ptr[row + 1]; ++edge) {
            sum += result.edge_probabilities[edge];
        }
        if (std::abs(sum - 1.0) > 1e-8) {
            std::cerr << "test_sparse_sinkhorn: row sum mismatch" << std::endl;
            return 1;
        }
    }
    std::cout << "test_sparse_sinkhorn: ALL PASSED in " << result.iterations
              << " iterations" << std::endl;
    return 0;
}
