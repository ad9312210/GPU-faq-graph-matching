#include "gpu_sinkhorn.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Exception, class Operation>
void reject(Operation operation) {
    try { operation(); }
    catch (const Exception&) { return; }
    throw std::runtime_error("Expected rejection did not occur");
}

using Rows = std::vector<std::vector<std::pair<int, float>>>;

CSRMatrix make_costs(const Rows& rows, int columns) {
    CSRMatrix matrix;
    matrix.rows = static_cast<int>(rows.size());
    matrix.cols = columns;
    matrix.row_ptr.push_back(0);
    for (auto row : rows) {
        std::sort(row.begin(), row.end());
        for (const auto& edge : row) {
            matrix.col_idx.push_back(edge.first);
            matrix.values.push_back(edge.second);
        }
        matrix.row_ptr.push_back(static_cast<int>(matrix.values.size()));
    }
    return matrix;
}

DeviceCSR upload(const CSRMatrix& host, const GPUContext& context) {
    DeviceCSR device;
    device.num_rows = host.rows;
    device.num_cols = host.cols;
    device.entry_count = host.values.size();
    device.row_ptr = DeviceBuffer<int>(host.row_ptr.size(), context.device_id);
    device.col_idx = DeviceBuffer<int>(host.col_idx.size(), context.device_id);
    device.values = DeviceBuffer<float>(host.values.size(), context.device_id);
    device.row_ptr.upload(host.row_ptr.data(), host.row_ptr.size(), context.stream);
    device.col_idx.upload(host.col_idx.data(), host.col_idx.size(), context.stream);
    device.values.upload(host.values.data(), host.values.size(), context.stream);
    return device;
}

template<class T>
std::vector<T> read(const DeviceBuffer<T>& buffer, const GPUContext& context) {
    std::vector<T> values(buffer.size());
    buffer.download(values.data(), values.size(), context.stream);
    return values;
}

void check_residuals(const CSRMatrix& costs, const SparseSinkhornResult& result) {
    require(result.edge_probabilities.size() == costs.values.size(),
            "Incorrect confidence count");
    std::vector<double> rows(costs.rows, 0.0), columns(costs.cols, 0.0);
    for (int row = 0; row < costs.rows; ++row) {
        for (int p = costs.row_ptr[row]; p < costs.row_ptr[row + 1]; ++p) {
            const double value = result.edge_probabilities[p];
            require(std::isfinite(value) && value >= 0.0,
                    "Invalid confidence value");
            rows[row] += value;
            columns[costs.col_idx[p]] += value;
        }
    }
    double row_error = 0.0, column_error = 0.0;
    for (double sum : rows) row_error = std::max(row_error, std::abs(sum - 1.0));
    for (double sum : columns)
        column_error = std::max(column_error, std::abs(sum - 1.0));
    require(std::abs(row_error - result.max_row_marginal_error) < 1e-11,
            "Reported row residual differs from downloaded probabilities");
    require(std::abs(column_error - result.max_column_marginal_error) < 1e-11,
            "Reported column residual differs from downloaded probabilities");
}

void check_parity(const CSRMatrix& costs, const GPUContext& context,
                  GPUSinkhornOptions options = {}) {
    auto device = upload(costs, context);
    const auto cpu = sparse_sinkhorn(costs, options.epsilon,
        options.max_iterations, options.tolerance);
    auto result = gpu_sparse_sinkhorn(device, context, options);
    const auto host = gpu_download_sinkhorn(result, context);
    require(host.converged == cpu.converged, "CPU/GPU convergence differs");
    require(host.iterations <= options.max_iterations, "Iteration limit exceeded");
    require(result.diagnostics.workspace_bytes <= options.workspace_budget_bytes,
            "Sinkhorn workspace budget exceeded");
    require(host.edge_probabilities.size() == cpu.edge_probabilities.size(),
            "CPU/GPU confidence sizes differ");
    for (std::size_t p = 0; p < cpu.edge_probabilities.size(); ++p) {
        require(std::abs(host.edge_probabilities[p] - cpu.edge_probabilities[p]) < 5e-8,
                "CPU/GPU confidence values differ");
    }
    check_residuals(costs, host);
    if (host.converged) {
        require(host.max_row_marginal_error <= options.tolerance &&
                host.max_column_marginal_error <= options.tolerance,
                "Convergence declared before both residuals met tolerance");
    } else {
        require(result.diagnostics.status == GPUSinkhornStatus::IterationLimit,
                "Non-convergence did not report iteration limit");
    }
    require(read(device.row_ptr, context) == costs.row_ptr &&
            read(device.col_idx, context) == costs.col_idx &&
            read(device.values, context) == costs.values,
            "Sinkhorn modified CSR support or costs");

    // Reuse an existing reverse index on a repeated call.
    auto repeated = gpu_sparse_sinkhorn(device, context, options);
    const auto again = gpu_download_sinkhorn(repeated, context);
    require(again.edge_probabilities == host.edge_probabilities,
            "Repeated GPU Sinkhorn results differ");
}

void run_tests(const GPUContext& context) {
    GPUSinkhornOptions accurate;
    accurate.epsilon = 0.5;
    accurate.tolerance = 1e-9;
    accurate.max_iterations = 2000;

    check_parity(make_costs({}, 0), context, accurate);
    check_parity(make_costs({{{0, -3.0f}}}, 1), context, accurate);
    check_parity(make_costs({{{0, 0.0f}, {1, 0.5f}},
                            {{0, 0.75f}, {1, -0.25f}}}, 2), context, accurate);
    check_parity(make_costs({{{0, 0.0f}, {1, 0.25f}},
                            {{1, 0.5f}, {2, -0.5f}},
                            {{0, 0.25f}, {2, 0.0f}}}, 3), context, accurate);
    check_parity(make_costs({{{0, 0.0f}}, {{1, 0.0f}}, {{2, 0.0f}}}, 3),
                 context, accurate);

    std::mt19937 random(20261019u);
    for (int trial = 0; trial < 12; ++trial) {
        const int n = 2 + static_cast<int>(random() % 7);
        Rows rows(n);
        for (int row = 0; row < n; ++row) {
            for (int column = 0; column < n; ++column) {
                rows[row].push_back({column,
                    (static_cast<int>(random() % 17) - 8) / 8.0f});
            }
        }
        check_parity(make_costs(rows, n), context, accurate);
    }

    // A valid perfect matching does not guarantee fast Sinkhorn convergence.
    auto one_iteration = accurate;
    one_iteration.max_iterations = 1;
    const auto triangular = make_costs(
        {{{0, 0.0f}, {1, 0.0f}}, {{1, 0.0f}}}, 2);
    check_parity(triangular, context, one_iteration);
    auto triangular_device = upload(triangular, context);
    auto limited = gpu_sparse_sinkhorn(triangular_device, context, one_iteration);
    require(limited.diagnostics.status == GPUSinkhornStatus::IterationLimit,
            "Deliberate non-convergence was not reported");

    // Positive rows and columns, but Hall-deficient marginal support.
    const auto deficient = make_costs(
        {{{0, 0.0f}}, {{0, 0.0f}}, {{1, 0.0f}, {2, 0.0f}}}, 3);
    auto deficient_options = accurate;
    deficient_options.max_iterations = 30;
    check_parity(deficient, context, deficient_options);

    for (const auto& costs : {
            make_costs({{}, {{0, 0.0f}, {1, 0.0f}}}, 2),
            make_costs({{{0, 0.0f}}, {{0, 0.0f}}}, 2),
            make_costs({{}, {}}, 2)}) {
        auto device = upload(costs, context);
        auto result = gpu_sparse_sinkhorn(device, context);
        require(result.diagnostics.status == GPUSinkhornStatus::InvalidSupport &&
                !result.diagnostics.probabilities_valid,
                "Empty marginal support was accepted");
        reject<std::invalid_argument>([&] {
            (void)gpu_download_sinkhorn(result, context);
        });
    }

    auto rectangular = upload(make_costs({{{0, 0.0f}}}, 2), context);
    reject<std::invalid_argument>([&] {
        auto unused = gpu_sparse_sinkhorn(rectangular, context);
    });

    // CPU exponential weights underflow for the whole second row.
    // Log-domain scaling still has the analytic solution p=1/2 on every edge.
    const auto extreme = make_costs(
        {{{0, 0.0f}, {1, 0.0f}}, {{0, 10000.0f}, {1, 10000.0f}}}, 2);
    reject<std::runtime_error>([&] {
        (void)sparse_sinkhorn(extreme, 0.1, 500, 1e-8);
    });
    auto extreme_device = upload(extreme, context);
    auto extreme_options = accurate;
    extreme_options.epsilon = 0.1;
    extreme_options.tolerance = 1e-8;
    auto stable = gpu_sparse_sinkhorn(extreme_device, context, extreme_options);
    const auto stable_host = gpu_download_sinkhorn(stable, context);
    require(stable_host.converged, "Log-domain extreme-cost case did not converge");
    for (double value : stable_host.edge_probabilities)
        require(std::abs(value - 0.5) < 1e-8, "Extreme-cost analytic solution differs");
    check_residuals(extreme, stable_host);

    auto finite = upload(make_costs(
        {{{0, 0.0f}, {1, 1.0f}}, {{0, 1.0f}, {1, 0.0f}}}, 2), context);
    auto impossible_scale = accurate;
    impossible_scale.epsilon = 1e-310;
    auto failed = gpu_sparse_sinkhorn(finite, context, impossible_scale);
    require(failed.diagnostics.status == GPUSinkhornStatus::NumericalFailure &&
            !failed.diagnostics.probabilities_valid,
            "Nonfinite log-kernel arithmetic was not reported");

    for (double epsilon : {0.0, -1.0,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::infinity()}) {
        auto options = accurate;
        options.epsilon = epsilon;
        reject<std::invalid_argument>([&] {
            auto unused = gpu_sparse_sinkhorn(finite, context, options);
        });
    }
    auto bad = accurate;
    bad.max_iterations = 0;
    reject<std::invalid_argument>([&] {
        auto unused = gpu_sparse_sinkhorn(finite, context, bad);
    });
    bad = accurate;
    bad.tolerance = std::numeric_limits<double>::quiet_NaN();
    reject<std::invalid_argument>([&] {
        auto unused = gpu_sparse_sinkhorn(finite, context, bad);
    });
    bad = accurate;
    bad.workspace_budget_bytes = 0;
    reject<std::invalid_argument>([&] {
        auto unused = gpu_sparse_sinkhorn(finite, context, bad);
    });

    const float invalid_cost = std::numeric_limits<float>::quiet_NaN();
    finite.values.upload(&invalid_cost, 1, context.stream);
    reject<std::invalid_argument>([&] {
        auto unused = gpu_sparse_sinkhorn(finite, context, accurate);
    });
}

} // namespace

int main() {
    cudaStream_t stream = nullptr;
    try {
        int count = 0;
        const auto available = cudaGetDeviceCount(&count);
        if (available == cudaErrorNoDevice || available == cudaErrorInsufficientDriver ||
            (available == cudaSuccess && count == 0)) {
            std::cout << "SKIP: CUDA GPU unavailable\n";
            return 77;
        }
        CUDA_CHECK(available);
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaStreamCreate(&stream));
        GPUContext context;
        context.device_id = 0;
        context.stream = stream;
        run_tests(context);
        CUDA_CHECK(cudaStreamDestroy(stream));
        stream = nullptr;
        std::cout << "GPU sparse Sinkhorn tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        if (stream) {
            (void)cudaStreamSynchronize(stream);
            (void)cudaStreamDestroy(stream);
        }
        std::cerr << error.what() << '\n';
        return 1;
    }
}
