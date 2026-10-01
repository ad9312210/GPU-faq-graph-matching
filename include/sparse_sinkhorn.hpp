#ifndef SCGM_SPARSE_SINKHORN_HPP
#define SCGM_SPARSE_SINKHORN_HPP

#include "csr_matrix.hpp"

struct SparseSinkhornResult {
    std::vector<double> edge_probabilities;
    int iterations = 0;
    double max_row_marginal_error = 0.0;
    double max_column_marginal_error = 0.0;
    bool converged = false;
};

// Entropic row/column scaling over exactly the candidate edges in CSR.
// Requires a square matrix and positive support for every row and column.
SparseSinkhornResult sparse_sinkhorn(
    const CSRMatrix& costs,
    double epsilon = 0.1,
    int max_iterations = 500,
    double tolerance = 1e-3);

#endif
