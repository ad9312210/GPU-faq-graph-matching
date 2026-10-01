#include "sparse_sinkhorn.hpp"

SparseSinkhornResult sparse_sinkhorn(
    const CSRMatrix& costs,
    double epsilon,
    int max_iterations,
    double tolerance) {
    if (costs.rows != costs.cols) {
        throw std::invalid_argument("sparse_sinkhorn requires a square assignment matrix");
    }
    if (epsilon <= 0.0 || max_iterations <= 0 || tolerance <= 0.0) {
        throw std::invalid_argument("invalid sparse_sinkhorn parameters");
    }
    if (costs.row_ptr.size() != static_cast<size_t>(costs.rows + 1) ||
        costs.col_idx.size() != costs.values.size() ||
        costs.row_ptr.back() != static_cast<int>(costs.values.size())) {
        throw std::invalid_argument("malformed CSR matrix passed to sparse_sinkhorn");
    }

    SparseSinkhornResult result;
    const int n = costs.rows;
    const int nnz = costs.nnz();
    result.edge_probabilities.assign(nnz, 0.0);
    if (n == 0) {
        result.converged = true;
        return result;
    }
    if (nnz == 0) {
        throw std::invalid_argument("sparse_sinkhorn requires nonempty candidate support");
    }

    double min_cost = std::numeric_limits<double>::infinity();
    std::vector<double> row_sums(n, 0.0), column_sums(n, 0.0);
    for (int row = 0; row < n; ++row) {
        if (costs.row_ptr[row] == costs.row_ptr[row + 1]) {
            throw std::invalid_argument("sparse_sinkhorn row has no candidate edges");
        }
        for (int edge = costs.row_ptr[row]; edge < costs.row_ptr[row + 1]; ++edge) {
            const int column = costs.col_idx[edge];
            if (column < 0 || column >= n) {
                throw std::invalid_argument("sparse_sinkhorn target index out of range");
            }
            min_cost = std::min(min_cost, static_cast<double>(costs.values[edge]));
        }
    }
    std::vector<double> kernel(nnz);
    for (int edge = 0; edge < nnz; ++edge) {
        kernel[edge] = std::exp(-(static_cast<double>(costs.values[edge]) - min_cost) / epsilon);
    }
    std::vector<double> row_scale(n, 1.0), column_scale(n, 1.0);

    for (int iteration = 1; iteration <= max_iterations; ++iteration) {
        for (int row = 0; row < n; ++row) {
            double sum = 0.0;
            for (int edge = costs.row_ptr[row]; edge < costs.row_ptr[row + 1]; ++edge) {
                sum += kernel[edge] * column_scale[costs.col_idx[edge]];
            }
            if (!(sum > 0.0) || !std::isfinite(sum)) {
                throw std::runtime_error("sparse_sinkhorn encountered a zero/nonfinite row scale");
            }
            row_scale[row] = 1.0 / sum;
        }

        std::fill(column_sums.begin(), column_sums.end(), 0.0);
        for (int row = 0; row < n; ++row) {
            for (int edge = costs.row_ptr[row]; edge < costs.row_ptr[row + 1]; ++edge) {
                column_sums[costs.col_idx[edge]] += kernel[edge] * row_scale[row];
            }
        }
        for (int column = 0; column < n; ++column) {
            if (!(column_sums[column] > 0.0) || !std::isfinite(column_sums[column])) {
                throw std::invalid_argument("sparse_sinkhorn target has no positive incoming support");
            }
            column_scale[column] = 1.0 / column_sums[column];
        }

        std::fill(row_sums.begin(), row_sums.end(), 0.0);
        std::fill(column_sums.begin(), column_sums.end(), 0.0);
        for (int row = 0; row < n; ++row) {
            for (int edge = costs.row_ptr[row]; edge < costs.row_ptr[row + 1]; ++edge) {
                const double probability = row_scale[row] * kernel[edge] *
                                           column_scale[costs.col_idx[edge]];
                result.edge_probabilities[edge] = probability;
                row_sums[row] += probability;
                column_sums[costs.col_idx[edge]] += probability;
            }
        }
        result.iterations = iteration;
        result.max_row_marginal_error = 0.0;
        result.max_column_marginal_error = 0.0;
        for (int i = 0; i < n; ++i) {
            result.max_row_marginal_error = std::max(
                result.max_row_marginal_error, std::abs(row_sums[i] - 1.0));
            result.max_column_marginal_error = std::max(
                result.max_column_marginal_error, std::abs(column_sums[i] - 1.0));
        }
        if (result.max_row_marginal_error <= tolerance &&
            result.max_column_marginal_error <= tolerance) {
            result.converged = true;
            break;
        }
    }
    return result;
}
