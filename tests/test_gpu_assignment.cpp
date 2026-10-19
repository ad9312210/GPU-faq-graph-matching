
#include "gpu_assignment.hpp"

#include <cmath>
#include <functional>
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

using Rows = std::vector<std::vector<std::pair<int, float>>>;

CSRMatrix make_costs(const Rows& rows, int columns) {
    std::vector<std::vector<Candidate>> candidates(rows.size());

    for (std::size_t row = 0; row < rows.size(); ++row) {
        for (const auto& entry : rows[row]) {
            candidates[row].push_back(
                Candidate{entry.first, 1.0f - entry.second});
        }
    }

    return build_sparse_cost_matrix(
        candidates, static_cast<int>(rows.size()), columns);
}

DeviceCSR upload(const CSRMatrix& host, const GPUContext& context) {
    DeviceCSR device;
    device.num_rows = host.rows;
    device.num_cols = host.cols;
    device.entry_count = host.values.size();

    device.row_ptr =
        DeviceBuffer<int>(host.row_ptr.size(), context.device_id);
    device.col_idx =
        DeviceBuffer<int>(host.col_idx.size(), context.device_id);
    device.values =
        DeviceBuffer<float>(host.values.size(), context.device_id);

    device.row_ptr.upload(
        host.row_ptr.data(), host.row_ptr.size(), context.stream);
    device.col_idx.upload(
        host.col_idx.data(), host.col_idx.size(), context.stream);
    device.values.upload(
        host.values.data(), host.values.size(), context.stream);

    return device;
}

double exhaustive(const CSRMatrix& costs) {
    double best = std::numeric_limits<double>::infinity();
    std::vector<bool> used(costs.cols, false);

    std::function<void(int, double)> visit =
        [&](int row, double objective) {
            if (row == costs.rows) {
                best = std::min(best, objective);
                return;
            }

            for (int p = costs.row_ptr[row];
                 p < costs.row_ptr[row + 1]; ++p) {
                const int column = costs.col_idx[p];

                if (!used[column]) {
                    used[column] = true;
                    visit(row + 1, objective + costs.values[p]);
                    used[column] = false;
                }
            }
        };

    visit(0, 0.0);
    return best;
}

void check_variant(const CSRMatrix& costs, const GPUContext& context,
                   GPUAssignmentExecution execution) {
    const double oracle = exhaustive(costs);
    auto device_costs = upload(costs, context);

    GPUAssignmentOptions options;
    options.execution = execution;
    auto solved = gpu_solve_sparse_assignment(device_costs, context, options);
    require(solved.execution == execution, "Incorrect execution variant reported");
    require(solved.execution_blocks == (execution == GPUAssignmentExecution::Cooperative ? 2 : 1),
            "Incorrect execution block count");
    auto host = gpu_download_assignment(solved, costs, context);

    if (!std::isfinite(oracle)) {
        require(solved.status == GPUAssignmentStatus::Infeasible,
                "Infeasible support was not reported");
        require(!host.assignment.feasible && !host.certificate.optimal,
                "Infeasible support received a success certificate");
        return;
    }

    require(solved.status == GPUAssignmentStatus::Success,
            "GPU solver failed on feasible support");
    require(host.certificate.primal_feasible, "Primal feasibility failed");
    require(host.certificate.dual_feasible, "Dual feasibility failed");
    require(host.certificate.complementary_slackness,
            "Complementary slackness failed");
    require(host.certificate.optimal, "Optimality certificate failed");

    require(std::abs(host.assignment.total_cost - oracle) < 1e-8,
            "GPU objective differs from exhaustive oracle");

    if (costs.rows != 0) {
        const auto cpu = solve_lapjv(costs);
        require(cpu.feasible, "CPU reference rejected feasible support");
        require(std::abs(cpu.total_cost - oracle) < 1e-8,
                "CPU reference differs from exhaustive oracle");
        require(std::abs(cpu.total_cost - host.assignment.total_cost) < 1e-8,
                "CPU/GPU assignment objectives differ");
    }

    // Tied optima may differ from CPU; GPU execution must be deterministic.
    auto repeated = gpu_solve_sparse_assignment(device_costs, context, options);
    const auto again = gpu_download_assignment(repeated, costs, context);

    require(again.certificate.optimal, "Repeated certificate failed");
    require(again.assignment.row_to_col == host.assignment.row_to_col,
            "GPU matching is nondeterministic");

    GPUAssignmentOptions tiny;
    tiny.execution = execution;
    tiny.workspace_budget_bytes = 0;

    bool rejected = false;
    try {
        auto unused = gpu_solve_sparse_assignment(device_costs, context, tiny);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "Assignment workspace budget was ignored");
}

void check(const CSRMatrix& costs, const GPUContext& context) {
    check_variant(costs, context, GPUAssignmentExecution::Block);
    check_variant(costs, context, GPUAssignmentExecution::Warp);
    int supported = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&supported,cudaDevAttrCooperativeLaunch,context.device_id));
    if (supported) check_variant(costs, context, GPUAssignmentExecution::Cooperative);
}

void run_tests(const GPUContext& context) {
    int cooperative_supported = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&cooperative_supported,
        cudaDevAttrCooperativeLaunch,context.device_id));
    if (!cooperative_supported)
        std::cout << "Cooperative variant unavailable; its checks are not executed\n";
    // Requires rerouting the first row through an augmenting path.
    check(make_costs({{{0, 1.0f}, {1, 2.0f}},
                      {{0, 1.5f}, {1, 100.0f}}}, 2), context);

    check(make_costs({{{0, -2.0f}, {1, -1.0f}},
                      {{0, -3.0f}, {1, 2.0f}}}, 2), context);

    check(make_costs({{{4, 2.0f}, {0, 1.0f}},
                      {{0, 0.0f}, {2, 1.0f}},
                      {{2, 0.0f}, {3, 1.0f}}}, 5), context);

    check(make_costs({{{0, 1.0f}, {1, 1.0f}, {2, 1.0f}},
                      {{0, 1.0f}, {1, 1.0f}, {2, 1.0f}},
                      {{0, 1.0f}, {1, 1.0f}, {2, 1.0f}}}, 5), context);

    // Hall deficiency despite every row having candidates.
    check(make_costs({{{0, 1.0f}, {1, 2.0f}},
                      {{0, 2.0f}, {1, 1.0f}},
                      {{0, 0.0f}, {1, 0.0f}}}, 3), context);

    check(make_costs({{}, {{0, 1.0f}}}, 3), context);
    check(make_costs({}, 0), context);
    check(make_costs({}, 3), context);

    std::mt19937 random(20261019u);

    for (int trial = 0; trial < 40; ++trial) {
        const int rows = 1 + static_cast<int>(random() % 5);
        const int columns = rows + static_cast<int>(random() % 3);
        Rows entries(rows);

        for (int row = 0; row < rows; ++row) {
            for (int column = 0; column < columns; ++column) {
                // Diagonal guarantees feasible fixed support.
                if (column == row || random() % 3 != 0) {
                    const float cost =
                        (static_cast<int>(random() % 33) - 16) / 4.0f;
                    entries[row].push_back({column, cost});
                }
            }
        }

        check(make_costs(entries, columns), context);
    }

    // Cross warp/block boundaries without invoking the exponential oracle.
    Rows large(300);
    for (int r=0;r<300;++r) {
        large[r].push_back({r,-1.0f});
        large[r].push_back({(r+1)%300,1.0f});
    }
    const auto costs=make_costs(large,300);
    const auto cpu=solve_lapjv(costs);
    require(cpu.feasible,"Large CPU reference failed");
    auto device=upload(costs,context);
    for (auto execution : {GPUAssignmentExecution::Warp, GPUAssignmentExecution::Block,
                           GPUAssignmentExecution::Cooperative}) {
        if (execution == GPUAssignmentExecution::Cooperative && !cooperative_supported) continue;
        GPUAssignmentOptions options; options.execution=execution;
        auto solved=gpu_solve_sparse_assignment(device,context,options);
        auto host=gpu_download_assignment(solved,costs,context);
        require(host.certificate.optimal,"Large variant certificate failed");
        require(std::abs(cpu.total_cost-host.assignment.total_cost)<1e-8,
                "Large variant objective differs from CPU");
    }

    GPUAssignmentOptions invalid;
    invalid.execution=GPUAssignmentExecution::Cooperative;
    invalid.cooperative_blocks=1;
    bool rejected=false;
    try { auto unused=gpu_solve_sparse_assignment(device,context,invalid); }
    catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"Single-block cooperative request was not rejected");
}

} // namespace

int main() {
    cudaStream_t stream = nullptr;

    try {
        int count = 0;
        const cudaError_t available = cudaGetDeviceCount(&count);

        if (available == cudaErrorNoDevice ||
            available == cudaErrorInsufficientDriver ||
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

        std::cout << "GPU sparse assignment tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        if (stream != nullptr) {
            (void)cudaStreamSynchronize(stream);
            (void)cudaStreamDestroy(stream);
        }

        std::cerr << error.what() << '\n';
        return 1;
    }
}
