#include "gpu_sinkhorn.hpp"

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

constexpr int THREADS = 256;
using BlockReduce = cub::BlockReduce<double, THREADS>;

struct Events {
    cudaEvent_t begin = nullptr, end = nullptr;
    Events() {
        CUDA_CHECK(cudaEventCreate(&begin));
        const auto error = cudaEventCreate(&end);
        if (error != cudaSuccess) {
            cudaEventDestroy(begin);
            begin = nullptr;
            CUDA_CHECK(error);
        }
    }
    ~Events() {
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
    }
};

__global__ void check_positive_support(
    const int* rows, const int* columns, int n, int* status)
{
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<std::size_t>(n) &&
        (rows[i] == rows[i + 1] || columns[i] == columns[i + 1])) {
        atomicMax(status, 2);
    }
}

__global__ void make_log_kernel(
    const float* costs, const float* minimum, std::size_t entries,
    double epsilon, double* log_kernel, int* status)
{
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= entries) return;
    const double value =
        -(static_cast<double>(costs[i]) - static_cast<double>(*minimum)) / epsilon;
    if (!isfinite(value)) {
        atomicMax(status, 3);
        log_kernel[i] = 0.0;
    } else {
        log_kernel[i] = value;
    }
}

// The same operation handles rows and reverse-index columns.
// positions==nullptr means CSR positions are already contiguous.
__global__ void update_log_scaling(
    const int* offsets, const int* neighbors, const int* positions,
    const double* log_kernel, const double* other_scale,
    double* output, int* status)
{
    const int group = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    __shared__ BlockReduce::TempStorage temporary;
    __shared__ double maximum;
    __shared__ int skip;

    // A block-uniform snapshot is required: another block may set status
    // while this block is executing collective operations.
    if (lane == 0) skip = atomicAdd(status, 0);
    __syncthreads();
    if (skip != 0) {
        if (lane == 0) output[group] = 0.0;
        return;
    }

    const int begin = offsets[group], end = offsets[group + 1];
    double local_maximum = -CUDART_INF;
    for (std::size_t p = static_cast<std::size_t>(begin) + threadIdx.x;
         p < static_cast<std::size_t>(end); p += THREADS) {
        const int edge = positions ? positions[p] : static_cast<int>(p);
        const double value = log_kernel[edge] + other_scale[neighbors[p]];
        if (isfinite(value)) local_maximum = fmax(local_maximum, value);
        else atomicMax(status, 3);
    }

    const double reduced_maximum = BlockReduce(temporary).Reduce(
        local_maximum, cub::Max());
    if (lane == 0) maximum = reduced_maximum;
    __syncthreads();

    if (!isfinite(maximum)) {
        if (lane == 0) {
            atomicMax(status, 3);
            output[group] = 0.0;
        }
        return;
    }

    double local_sum = 0.0;
    for (std::size_t p = static_cast<std::size_t>(begin) + threadIdx.x;
         p < static_cast<std::size_t>(end); p += THREADS) {
        const int edge = positions ? positions[p] : static_cast<int>(p);
        const double value = log_kernel[edge] + other_scale[neighbors[p]];
        if (isfinite(value)) local_sum += exp(value - maximum);
    }

    // The barrier above completed all accesses to the previous reduction.
    const double sum = BlockReduce(temporary).Sum(local_sum);
    if (lane == 0) {
        const double value = -(maximum + log(sum));
        if (!(sum > 0.0) || !isfinite(sum) || !isfinite(value)) {
            atomicMax(status, 3);
            output[group] = 0.0;
        } else {
            output[group] = value;
        }
    }
}

__global__ void capture_gauge(
    const double* row_scale, double* gauge, const int* status)
{
    *gauge = *status == 0 ? row_scale[0] : 0.0;
}

__global__ void recenter_scaling(
    double* rows, double* columns, const double* gauge, int n, int* status)
{
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::size_t>(n) || atomicAdd(status, 0) != 0) return;
    const double row = rows[i] - *gauge;
    const double column = columns[i] + *gauge;
    if (!isfinite(row) || !isfinite(column)) atomicMax(status, 3);
    rows[i] = row;
    columns[i] = column;
}

__global__ void write_probabilities(
    const int* offsets, const int* columns, const double* log_kernel,
    const double* row_scale, const double* column_scale,
    double* probabilities, int* status)
{
    const int row = static_cast<int>(blockIdx.x);
    if (atomicAdd(status, 0) != 0) return; // No block collectives here.
    for (std::size_t p = static_cast<std::size_t>(offsets[row]) + threadIdx.x;
         p < static_cast<std::size_t>(offsets[row + 1]); p += THREADS) {
        const double log_probability =
            (log_kernel[p] + row_scale[row]) + column_scale[columns[p]];
        const double probability = exp(log_probability);
        if (!isfinite(log_probability) || !isfinite(probability)) {
            atomicMax(status, 3);
            probabilities[p] = 0.0;
        } else {
            // Underflow to zero is allowed for negligible edges.
            probabilities[p] = probability;
        }
    }
}

__global__ void marginal_errors(
    const int* offsets, const int* positions, const double* probabilities,
    double* errors, int* status)
{
    const int group = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    __shared__ BlockReduce::TempStorage temporary;
    __shared__ int skip;
    if (lane == 0) skip = atomicAdd(status, 0);
    __syncthreads();
    if (skip != 0) {
        if (lane == 0) errors[group] = CUDART_INF;
        return;
    }
    double local_sum = 0.0;
    for (std::size_t p = static_cast<std::size_t>(offsets[group]) + threadIdx.x;
         p < static_cast<std::size_t>(offsets[group + 1]); p += THREADS) {
        const int edge = positions ? positions[p] : static_cast<int>(p);
        const double value = probabilities[edge];
        if (!isfinite(value) || value < 0.0) atomicMax(status, 3);
        else local_sum += value;
    }
    const double sum = BlockReduce(temporary).Sum(local_sum);
    if (lane == 0) {
        errors[group] = fabs(sum - 1.0);
        if (!isfinite(errors[group])) atomicMax(status, 3);
    }
}

GPUSinkhornResult solve_impl(
    DeviceCSR& costs, const GPUContext& context,
    const GPUSinkhornOptions& options)
{
    Timer setup("GPU Sinkhorn setup");
    if (!std::isfinite(options.epsilon) || options.epsilon <= 0.0 ||
        !std::isfinite(options.tolerance) || options.tolerance <= 0.0 ||
        options.max_iterations <= 0) {
        throw std::invalid_argument("Invalid GPU Sinkhorn parameters");
    }
    if (costs.num_rows != costs.num_cols) {
        throw std::invalid_argument("GPU Sinkhorn requires a square matrix");
    }
    gpu_validate_device_csr(costs, context);
    CUDA_CHECK(cudaSetDevice(context.device_id));

    GPUSinkhornResult result;
    auto& info = result.diagnostics;
    const int n = costs.num_rows;
    const std::size_t entries = costs.entry_count;

    if (n == 0) {
        info.status = GPUSinkhornStatus::Converged;
        info.probabilities_valid = true;
        info.max_row_marginal_error = 0.0;
        info.max_column_marginal_error = 0.0;
        info.setup_ms = setup.elapsed_ms();
        return result;
    }
    if (entries == 0) {
        info.status = GPUSinkhornStatus::InvalidSupport;
        info.setup_ms = setup.elapsed_ms();
        return result;
    }

    if (!costs.has_reverse_index) {
        Timer reverse("GPU Sinkhorn reverse index");
        gpu_build_device_reverse_index(costs, context);
        info.reverse_index_build_ms = reverse.elapsed_ms();
    } else {
        gpu_validate_device_reverse_index(costs, context);
    }
    info.reverse_index_bytes = costs.column_ptr.bytes() +
        costs.source_rows.bytes() + costs.csr_positions.bytes();

    // int dimensions/entry counts bound this arithmetic on 64-bit CUDA hosts.
    static_assert(sizeof(std::size_t) >= 8, "64-bit size_t required");
    const std::size_t base_workspace =
        entries * sizeof(double) +
        static_cast<std::size_t>(n) * 2 * sizeof(double) +
        sizeof(float) + sizeof(int) + 3 * sizeof(double);
    if (base_workspace > options.workspace_budget_bytes) {
        throw std::invalid_argument("GPU Sinkhorn workspace exceeds its budget");
    }

    DeviceBuffer<int> status(1, context.device_id);
    DeviceBuffer<double> log_kernel(entries, context.device_id);
    DeviceBuffer<double> row_errors(n, context.device_id);
    DeviceBuffer<double> column_errors(n, context.device_id);
    DeviceBuffer<double> residuals(2, context.device_id);
    DeviceBuffer<double> gauge(1, context.device_id);
    DeviceBuffer<float> minimum(1, context.device_id);
    DeviceBuffer<unsigned char> reduction_workspace;
    Events events;
    info.workspace_bytes = base_workspace;

    try {
        CUDA_CHECK(cudaMemsetAsync(status.data(), 0, sizeof(int), context.stream));
        const unsigned int groups = static_cast<unsigned int>(n);
        const unsigned int vertex_blocks =
            static_cast<unsigned int>(1 + (static_cast<std::size_t>(n) - 1) / THREADS);
        const unsigned int edge_blocks =
            static_cast<unsigned int>(1 + (entries - 1) / THREADS);

        check_positive_support<<<vertex_blocks, THREADS, 0, context.stream>>>(
            costs.row_ptr.data(), costs.column_ptr.data(), n, status.data());
        CUDA_CHECK(cudaGetLastError());
        int host_status = 0;
        status.download(&host_status, 1, context.stream);
        if (host_status != 0) {
            info.status = GPUSinkhornStatus::InvalidSupport;
            info.setup_ms = setup.elapsed_ms();
            return result;
        }

        std::size_t minimum_bytes = 0, maximum_bytes = 0;
        CUDA_CHECK(cub::DeviceReduce::Min(
            nullptr, minimum_bytes, costs.values.data(), minimum.data(),
            static_cast<int>(entries), context.stream));
        CUDA_CHECK(cub::DeviceReduce::Max(
            nullptr, maximum_bytes, row_errors.data(), residuals.data(),
            n, context.stream));
        std::size_t reduction_bytes = std::max<std::size_t>(
            1, std::max(minimum_bytes, maximum_bytes));
        if (reduction_bytes > options.workspace_budget_bytes - base_workspace) {
            throw std::invalid_argument(
                "GPU Sinkhorn reduction workspace exceeds its budget");
        }
        reduction_workspace = DeviceBuffer<unsigned char>(
            reduction_bytes, context.device_id);
        info.workspace_bytes = base_workspace + reduction_bytes;

        result.edge_probabilities = DeviceBuffer<double>(entries, context.device_id);
        result.row_log_scale = DeviceBuffer<double>(n, context.device_id);
        result.column_log_scale = DeviceBuffer<double>(n, context.device_id);
        info.persistent_result_bytes = result.edge_probabilities.bytes() +
            result.row_log_scale.bytes() + result.column_log_scale.bytes();

        CUDA_CHECK(cudaMemsetAsync(result.row_log_scale.data(), 0,
            result.row_log_scale.bytes(), context.stream));
        CUDA_CHECK(cudaMemsetAsync(result.column_log_scale.data(), 0,
            result.column_log_scale.bytes(), context.stream));
        CUDA_CHECK(cub::DeviceReduce::Min(
            reduction_workspace.data(), minimum_bytes,
            costs.values.data(), minimum.data(),
            static_cast<int>(entries), context.stream));
        make_log_kernel<<<edge_blocks, THREADS, 0, context.stream>>>(
            costs.values.data(), minimum.data(), entries, options.epsilon,
            log_kernel.data(), status.data());
        CUDA_CHECK(cudaGetLastError());
        status.download(&host_status, 1, context.stream);
        info.setup_ms = setup.elapsed_ms();
        if (host_status != 0) {
            info.status = GPUSinkhornStatus::NumericalFailure;
            return result;
        }

        for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
            CUDA_CHECK(cudaEventRecord(events.begin, context.stream));
            update_log_scaling<<<groups, THREADS, 0, context.stream>>>(
                costs.row_ptr.data(), costs.col_idx.data(), nullptr,
                log_kernel.data(), result.column_log_scale.data(),
                result.row_log_scale.data(), status.data());
            CUDA_CHECK(cudaGetLastError());
            update_log_scaling<<<groups, THREADS, 0, context.stream>>>(
                costs.column_ptr.data(), costs.source_rows.data(),
                costs.csr_positions.data(), log_kernel.data(),
                result.row_log_scale.data(), result.column_log_scale.data(),
                status.data());
            CUDA_CHECK(cudaGetLastError());

            // Recenter a scaling gauge without changing edge probabilities.
            capture_gauge<<<1, 1, 0, context.stream>>>(
                result.row_log_scale.data(), gauge.data(), status.data());
            CUDA_CHECK(cudaGetLastError());
            recenter_scaling<<<vertex_blocks, THREADS, 0, context.stream>>>(
                result.row_log_scale.data(), result.column_log_scale.data(),
                gauge.data(), n, status.data());
            CUDA_CHECK(cudaGetLastError());

            write_probabilities<<<groups, THREADS, 0, context.stream>>>(
                costs.row_ptr.data(), costs.col_idx.data(), log_kernel.data(),
                result.row_log_scale.data(), result.column_log_scale.data(),
                result.edge_probabilities.data(), status.data());
            CUDA_CHECK(cudaGetLastError());
            marginal_errors<<<groups, THREADS, 0, context.stream>>>(
                costs.row_ptr.data(), nullptr, result.edge_probabilities.data(),
                row_errors.data(), status.data());
            CUDA_CHECK(cudaGetLastError());
            marginal_errors<<<groups, THREADS, 0, context.stream>>>(
                costs.column_ptr.data(), costs.csr_positions.data(),
                result.edge_probabilities.data(), column_errors.data(), status.data());
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cub::DeviceReduce::Max(
                reduction_workspace.data(), maximum_bytes, row_errors.data(),
                residuals.data(), n, context.stream));
            CUDA_CHECK(cub::DeviceReduce::Max(
                reduction_workspace.data(), maximum_bytes, column_errors.data(),
                residuals.data() + 1, n, context.stream));
            CUDA_CHECK(cudaEventRecord(events.end, context.stream));
            CUDA_CHECK(cudaEventSynchronize(events.end));
            float milliseconds = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, events.begin, events.end));
            info.iteration_gpu_ms += milliseconds;

            double errors[2] = {};
            residuals.download(errors, 2, context.stream);
            status.download(&host_status, 1, context.stream);
            info.iterations = iteration + 1;
            info.max_row_marginal_error = errors[0];
            info.max_column_marginal_error = errors[1];

            if (host_status != 0 || !std::isfinite(errors[0]) ||
                !std::isfinite(errors[1])) {
                info.status = GPUSinkhornStatus::NumericalFailure;
                info.probabilities_valid = false;
                break;
            }
            info.probabilities_valid = true;
            if (errors[0] <= options.tolerance && errors[1] <= options.tolerance) {
                info.status = GPUSinkhornStatus::Converged;
                break;
            }
            info.status = GPUSinkhornStatus::IterationLimit;
        }
    } catch (...) {
        (void)cudaStreamSynchronize(context.stream);
        throw;
    }
    return result;
}

} // namespace

GPUSinkhornResult gpu_sparse_sinkhorn(
    DeviceCSR& costs, const GPUContext& context,
    const GPUSinkhornOptions& options)
{
    Timer total("Complete GPU Sinkhorn call");
    auto result = solve_impl(costs, context, options);
    result.diagnostics.total_ms = total.elapsed_ms();
    return result;
}

SparseSinkhornResult gpu_download_sinkhorn(
    const GPUSinkhornResult& result, const GPUContext& context)
{
    if (!result.diagnostics.probabilities_valid) {
        throw std::invalid_argument("GPU Sinkhorn has no valid probabilities to download");
    }
    SparseSinkhornResult host;
    host.edge_probabilities.resize(result.edge_probabilities.size());
    result.edge_probabilities.download(host.edge_probabilities.data(),
        host.edge_probabilities.size(), context.stream);
    host.iterations = result.diagnostics.iterations;
    host.max_row_marginal_error = result.diagnostics.max_row_marginal_error;
    host.max_column_marginal_error = result.diagnostics.max_column_marginal_error;
    host.converged = result.diagnostics.status == GPUSinkhornStatus::Converged;
    return host;
}

const char* gpu_sinkhorn_status_name(GPUSinkhornStatus status) {
    switch (status) {
        case GPUSinkhornStatus::Converged: return "converged";
        case GPUSinkhornStatus::IterationLimit: return "iteration_limit";
        case GPUSinkhornStatus::InvalidSupport: return "invalid_support";
        case GPUSinkhornStatus::NumericalFailure: return "numerical_failure";
    }
    return "unknown";
}
