#include "gpu_assignment.hpp"

#include <math_constants.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

constexpr int THREADS = 256;

struct Events {
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;

    Events() {
        CUDA_CHECK(cudaEventCreate(&begin));
        const cudaError_t error = cudaEventCreate(&end);

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

__global__ void initialize_assignment(
    const int* offsets,
    const float* costs,
    int rows,
    int columns,
    int* row_match,
    int* column_match,
    double* row_dual,
    double* column_dual,
    int* status)
{
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (index < static_cast<std::size_t>(rows)) {
        row_match[index] = -1;

        const int begin = offsets[index];
        const int end = offsets[index + 1];

        double minimum = CUDART_INF;

        for (int p = begin; p < end; ++p) {
            minimum = fmin(minimum, static_cast<double>(costs[p]));
        }

        if (begin == end) {
            row_dual[index] = 0.0;
            atomicExch(status, 1);
        } else {
            // Feasible initial dual, including negative costs.
            row_dual[index] = minimum;
        }
    }

    if (index < static_cast<std::size_t>(columns)) {
        column_match[index] = -1;
        column_dual[index] = 0.0;
    }
}

/*
 * One cooperative block owns all matching updates.
 *
 * Within an augmentation:
 *   - each CSR row has unique target IDs, so relaxation writes are disjoint;
 *   - block reduction chooses one deterministic minimum-slack column;
 *   - used matched columns refer to distinct rows during dual updates;
 *   - one thread commits the alternating path.
 *
 * Augmentations are sequential. No concurrent roots compete for columns.
 */
template<int WIDTH>
__global__ void augment_assignments(
    const int* offsets,
    const int* targets,
    const float* costs,
    int rows,
    int columns,
    int* row_match,
    int* column_match,
    double* row_dual,
    double* column_dual,
    double* slack,
    int* predecessor,
    unsigned char* used,
    int* status,
    unsigned long long* search_steps)
{
    __shared__ double reduction_value[WIDTH];
    __shared__ int reduction_column[WIDTH];

    __shared__ int active_row;
    __shared__ int active_column;
    __shared__ int selected_column;
    __shared__ double delta;
    __shared__ int augmented;

    const int lane = static_cast<int>(threadIdx.x);

    if (*status != 0) return;

    for (int root = 0; root < rows; ++root) {
        for (std::size_t column = static_cast<std::size_t>(lane);
             column < static_cast<std::size_t>(columns);
             column += WIDTH) {
            slack[column] = CUDART_INF;
            predecessor[column] = -1;
            used[column] = 0;
        }

        if (lane == 0) {
            active_row = root;
            active_column = -1; // Root sentinel.
            augmented = 0;
        }

        __syncthreads();

        for (int iteration = 0; iteration < columns; ++iteration) {
            const int begin = offsets[active_row];
            const int end = offsets[active_row + 1];

            // Sparse relaxation of the current alternating-tree row.
            for (std::size_t p =
                     static_cast<std::size_t>(begin) + threadIdx.x;
                 p < static_cast<std::size_t>(end);
                 p += WIDTH) {
                const int column = targets[p];

                if (used[column]) continue;

                const double reduced =
                    static_cast<double>(costs[p]) -
                    row_dual[active_row] -
                    column_dual[column];

                if (!isfinite(reduced)) {
                    atomicExch(status, 2);
                    continue;
                }

                if (reduced < slack[column] ||
                    (reduced == slack[column] &&
                     active_column < predecessor[column])) {
                    slack[column] = reduced;
                    predecessor[column] = active_column;
                }
            }

            __syncthreads();
            if (*status != 0) return;

            double best_value = CUDART_INF;
            int best_column = -1;

            for (std::size_t column = static_cast<std::size_t>(lane);
                 column < static_cast<std::size_t>(columns);
                 column += WIDTH) {
                if (used[column] || !isfinite(slack[column])) continue;

                if (best_column < 0 ||
                    slack[column] < best_value ||
                    (slack[column] == best_value &&
                     static_cast<int>(column) < best_column)) {
                    best_value = slack[column];
                    best_column = static_cast<int>(column);
                }
            }

            reduction_value[lane] = best_value;
            reduction_column[lane] = best_column;
            __syncthreads();

            for (int stride = WIDTH / 2; stride > 0; stride /= 2) {
                if (lane < stride) {
                    const int other = reduction_column[lane + stride];
                    const double value = reduction_value[lane + stride];

                    if (other >= 0 &&
                        (reduction_column[lane] < 0 ||
                         value < reduction_value[lane] ||
                         (value == reduction_value[lane] &&
                          other < reduction_column[lane]))) {
                        reduction_value[lane] = value;
                        reduction_column[lane] = other;
                    }
                }

                __syncthreads();
            }

            if (lane == 0) {
                selected_column = reduction_column[0];
                delta = reduction_value[0];
                ++(*search_steps);

                if (selected_column < 0) {
                    // No reachable unused column: support is infeasible.
                    *status = 1;
                } else if (!isfinite(delta)) {
                    *status = 2;
                }
            }

            __syncthreads();
            if (*status != 0) return;

            // Update duals using the tree before adding selected_column.
            // The free root is distinct from every currently matched row.
            if (lane == 0) {
                row_dual[root] += delta;
            }

            for (std::size_t column = static_cast<std::size_t>(lane);
                 column < static_cast<std::size_t>(columns);
                 column += WIDTH) {
                if (used[column]) {
                    const int row = column_match[column];
                    row_dual[row] += delta;
                    column_dual[column] -= delta;
                } else if (isfinite(slack[column])) {
                    slack[column] -= delta;
                }
            }

            __syncthreads();

            if (lane == 0) {
                used[selected_column] = 1;

                const int next_row = column_match[selected_column];

                if (next_row < 0) {
                    // Commit the alternating path with a single writer.
                    int column = selected_column;
                    int length = 0;

                    while (column >= 0 && length < columns) {
                        const int previous = predecessor[column];

                        if (previous < -1 || previous >= columns) {
                            *status = 2;
                            break;
                        }

                        const int row = previous < 0
                            ? root
                            : column_match[previous];

                        if (row < 0 || row >= rows) {
                            *status = 2;
                            break;
                        }

                        column_match[column] = row;
                        row_match[row] = column;

                        column = previous;
                        ++length;
                    }

                    if (column >= 0) {
                        *status = 2;
                    } else if (*status == 0) {
                        augmented = 1;
                    }
                } else {
                    active_row = next_row;
                    active_column = selected_column;
                }
            }

            __syncthreads();

            if (*status != 0) return;
            if (augmented) break;
        }

        if (lane == 0 && !augmented) {
            *status = 2;
        }

        __syncthreads();
        if (*status != 0) return;
    }
}

struct CooperativeState {
    int active_row, active_column, selected_column, augmented;
    double delta;
};

// Experimental multi-block correctness baseline. Grid barriers separate sparse
// relaxations, per-block reductions, dual updates, and the single path writer.
// All blocks must be concurrently resident: launch checks enforce this.
__global__ void augment_cooperative(
    const int* offsets, const int* targets, const float* costs,
    int rows, int columns, int* row_match, int* column_match,
    double* row_dual, double* column_dual, double* slack,
    int* predecessor, unsigned char* used, int* status,
    unsigned long long* search_steps, CooperativeState* state,
    double* block_values, int* block_columns) {
    const auto grid = cooperative_groups::this_grid();
    const std::size_t rank = grid.thread_rank();
    const std::size_t stride = grid.size();
    __shared__ double values[THREADS];
    __shared__ int choices[THREADS];
    if (*status != 0) return;
    for (int root = 0; root < rows; ++root) {
        for (std::size_t c = rank; c < static_cast<std::size_t>(columns); c += stride) {
            slack[c] = CUDART_INF; predecessor[c] = -1; used[c] = 0;
        }
        if (rank == 0) {
            state->active_row = root; state->active_column = -1; state->augmented = 0;
        }
        grid.sync();
        for (int iteration = 0; iteration < columns; ++iteration) {
            const int active = state->active_row;
            for (std::size_t p = static_cast<std::size_t>(offsets[active]) + rank;
                 p < static_cast<std::size_t>(offsets[active + 1]); p += stride) {
                const int c = targets[p];
                if (used[c]) continue;
                const double reduced = static_cast<double>(costs[p]) - row_dual[active] - column_dual[c];
                if (!isfinite(reduced)) { atomicExch(status, 2); continue; }
                if (reduced < slack[c] ||
                    (reduced == slack[c] && state->active_column < predecessor[c])) {
                    slack[c] = reduced; predecessor[c] = state->active_column;
                }
            }
            grid.sync();
            if (*status != 0) return;
            double best = CUDART_INF;
            int selected = -1;
            for (std::size_t c = rank; c < static_cast<std::size_t>(columns); c += stride) {
                if (!used[c] && isfinite(slack[c]) &&
                    (selected < 0 || slack[c] < best ||
                     (slack[c] == best && static_cast<int>(c) < selected))) {
                    best = slack[c]; selected = static_cast<int>(c);
                }
            }
            values[threadIdx.x] = best; choices[threadIdx.x] = selected;
            __syncthreads();
            for (int step = THREADS / 2; step; step /= 2) {
                if (threadIdx.x < static_cast<unsigned>(step)) {
                    const int other = choices[threadIdx.x + step];
                    const double value = values[threadIdx.x + step];
                    if (other >= 0 && (choices[threadIdx.x] < 0 || value < values[threadIdx.x] ||
                        (value == values[threadIdx.x] && other < choices[threadIdx.x]))) {
                        values[threadIdx.x] = value; choices[threadIdx.x] = other;
                    }
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                block_values[blockIdx.x] = values[0]; block_columns[blockIdx.x] = choices[0];
            }
            grid.sync();
            if (rank == 0) {
                int winner = -1;
                double minimum = CUDART_INF;
                // Small bounded grid reduction; sparse relaxations and dual
                // updates above/below execute across all participating blocks.
                for (unsigned b = 0; b < gridDim.x; ++b) {
                    const int c = block_columns[b];
                    const double v = block_values[b];
                    if (c >= 0 && (winner < 0 || v < minimum || (v == minimum && c < winner))) {
                        winner = c; minimum = v;
                    }
                }
                state->selected_column = winner; state->delta = minimum;
                ++*search_steps;
                if (winner < 0) *status = 1;
                else if (!isfinite(minimum)) *status = 2;
            }
            grid.sync();
            if (*status != 0) return;
            if (rank == 0) row_dual[root] += state->delta;
            for (std::size_t c = rank; c < static_cast<std::size_t>(columns); c += stride) {
                if (used[c]) {
                    row_dual[column_match[c]] += state->delta;
                    column_dual[c] -= state->delta;
                } else if (isfinite(slack[c])) slack[c] -= state->delta;
            }
            grid.sync();
            if (rank == 0) {
                const int selected_column = state->selected_column;
                used[selected_column] = 1;
                const int next = column_match[selected_column];
                if (next >= 0) {
                    state->active_row = next; state->active_column = selected_column;
                } else {
                    int c = selected_column, length = 0;
                    while (c >= 0 && length < columns) {
                        const int previous = predecessor[c];
                        if (previous < -1 || previous >= columns) { *status = 2; break; }
                        const int row = previous < 0 ? root : column_match[previous];
                        if (row < 0 || row >= rows) { *status = 2; break; }
                        column_match[c] = row; row_match[row] = c;
                        c = previous; ++length;
                    }
                    if (c >= 0) *status = 2;
                    else if (*status == 0) state->augmented = 1;
                }
            }
            grid.sync();
            if (*status != 0) return;
            if (state->augmented) break;
        }
        if (rank == 0 && !state->augmented) *status = 2;
        grid.sync();
        if (*status != 0) return;
    }
}

double tolerance(double a, double b, double absolute, double relative) {
    return absolute + relative *
        std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

void validate_host_costs(const CSRMatrix& costs) {
    if (costs.rows < 0 || costs.cols < 0 ||
        costs.row_ptr.size() != static_cast<std::size_t>(costs.rows) + 1 ||
        costs.col_idx.size() != costs.values.size() ||
        costs.col_idx.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        costs.row_ptr.front() != 0 ||
        costs.row_ptr.back() != static_cast<int>(costs.col_idx.size())) {
        throw std::invalid_argument("GPU assignment: invalid host CSR");
    }

    for (int row = 0; row < costs.rows; ++row) {
        const int begin = costs.row_ptr[row];
        const int end = costs.row_ptr[row + 1];

        if (begin < 0 || begin > end ||
            end > static_cast<int>(costs.col_idx.size())) {
            throw std::invalid_argument("GPU assignment: invalid host offsets");
        }

        for (int p = begin; p < end; ++p) {
            if (costs.col_idx[p] < 0 || costs.col_idx[p] >= costs.cols ||
                !std::isfinite(costs.values[p]) ||
                (p > begin && costs.col_idx[p - 1] >= costs.col_idx[p])) {
                throw std::invalid_argument("GPU assignment: invalid host entry");
            }
        }
    }
}

} // namespace

GPUAssignmentResult gpu_solve_sparse_assignment(
    const DeviceCSR& costs,
    const GPUContext& context,
    const GPUAssignmentOptions& options)
{
    gpu_validate_device_csr(costs, context);

    if (costs.num_rows > costs.num_cols) {
        throw std::invalid_argument(
            "GPU assignment requires source rows <= target columns");
    }

    const std::size_t rows = static_cast<std::size_t>(costs.num_rows);
    const std::size_t columns = static_cast<std::size_t>(costs.num_cols);

    // int dimensions bound this arithmetic on the supported 64-bit platform.
    static_assert(sizeof(std::size_t) >= 8, "64-bit size_t required");

    CUDA_CHECK(cudaSetDevice(context.device_id));
    if (options.execution != GPUAssignmentExecution::Warp &&
        options.execution != GPUAssignmentExecution::Block &&
        options.execution != GPUAssignmentExecution::Cooperative)
        throw std::invalid_argument("Unknown assignment execution variant");
    const bool cooperative = options.execution == GPUAssignmentExecution::Cooperative;
    if (cooperative) {
        if (options.cooperative_blocks < 2)
            throw std::invalid_argument("Cooperative assignment requires at least two blocks");
        int supported = 0, multiprocessors = 0, per_sm = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrCooperativeLaunch, context.device_id));
        if (!supported) throw std::invalid_argument("Cooperative assignment launch unsupported on device");
        CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, context.device_id));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, augment_cooperative, THREADS, 0));
        if (static_cast<long long>(options.cooperative_blocks) > static_cast<long long>(per_sm) * multiprocessors)
            throw std::invalid_argument("Cooperative assignment grid exceeds resident occupancy");
    }
    const std::size_t cooperative_bytes = cooperative
        ? sizeof(CooperativeState) + static_cast<std::size_t>(options.cooperative_blocks) *
            (sizeof(double) + sizeof(int)) : 0;
    const std::size_t workspace_bytes =
        columns * (sizeof(double) + sizeof(int) + sizeof(unsigned char)) +
        sizeof(int) + sizeof(unsigned long long) + cooperative_bytes;

    if (workspace_bytes > options.workspace_budget_bytes) {
        throw std::invalid_argument(
            "GPU assignment workspace exceeds its configured budget");
    }

    CUDA_CHECK(cudaSetDevice(context.device_id));

    GPUAssignmentResult result;
    result.num_rows = costs.num_rows;
    result.num_cols = costs.num_cols;
    result.workspace_bytes = workspace_bytes;
    result.execution = options.execution;
    result.execution_blocks = cooperative ? options.cooperative_blocks : 1;

    result.row_to_col = DeviceBuffer<int>(rows, context.device_id);
    result.col_to_row = DeviceBuffer<int>(columns, context.device_id);
    result.row_dual = DeviceBuffer<double>(rows, context.device_id);
    result.col_dual = DeviceBuffer<double>(columns, context.device_id);

    DeviceBuffer<double> slack(columns, context.device_id);
    DeviceBuffer<int> predecessor(columns, context.device_id);
    DeviceBuffer<unsigned char> used(columns, context.device_id);
    DeviceBuffer<int> status(1, context.device_id);
    DeviceBuffer<unsigned long long> steps(1, context.device_id);
    DeviceBuffer<CooperativeState> control(cooperative ? 1 : 0, context.device_id);
    DeviceBuffer<double> block_values(cooperative ? options.cooperative_blocks : 0, context.device_id);
    DeviceBuffer<int> block_columns(cooperative ? options.cooperative_blocks : 0, context.device_id);
    Events events;

    try {
        CUDA_CHECK(cudaEventRecord(events.begin, context.stream));

        CUDA_CHECK(cudaMemsetAsync(
            status.data(), 0, sizeof(int), context.stream));
        CUDA_CHECK(cudaMemsetAsync(
            steps.data(), 0, sizeof(unsigned long long), context.stream));

        const std::size_t extent = std::max(rows, columns);
        const unsigned int blocks = extent == 0
            ? 1u
            : static_cast<unsigned int>(1 + (extent - 1) / THREADS);

        initialize_assignment<<<blocks, THREADS, 0, context.stream>>>(
            costs.row_ptr.data(), costs.values.data(),
            costs.num_rows, costs.num_cols,
            result.row_to_col.data(), result.col_to_row.data(),
            result.row_dual.data(), result.col_dual.data(),
            status.data());

        CUDA_CHECK(cudaGetLastError());

        if (cooperative) {
            const int* offsets = costs.row_ptr.data();
            const int* targets = costs.col_idx.data();
            const float* values = costs.values.data();
            int nr = costs.num_rows, nc = costs.num_cols;
            int* rm = result.row_to_col.data(); int* cm = result.col_to_row.data();
            double* rd = result.row_dual.data(); double* cd = result.col_dual.data();
            double* sl = slack.data(); int* pr = predecessor.data();
            unsigned char* us = used.data(); int* st = status.data();
            unsigned long long* sp = steps.data();
            CooperativeState* ct = control.data();
            double* bv = block_values.data(); int* bc = block_columns.data();
            void* arguments[] = {&offsets,&targets,&values,&nr,&nc,&rm,&cm,
                &rd,&cd,&sl,&pr,&us,&st,&sp,&ct,&bv,&bc};
            CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(augment_cooperative),
                dim3(options.cooperative_blocks),dim3(THREADS),arguments,0,context.stream));
        } else if (options.execution == GPUAssignmentExecution::Warp) {
            augment_assignments<32><<<1,32,0,context.stream>>>(
                costs.row_ptr.data(),costs.col_idx.data(),costs.values.data(),costs.num_rows,costs.num_cols,
                result.row_to_col.data(),result.col_to_row.data(),result.row_dual.data(),result.col_dual.data(),
                slack.data(),predecessor.data(),used.data(),status.data(),steps.data());
        } else {
            augment_assignments<THREADS><<<1,THREADS,0,context.stream>>>(
                costs.row_ptr.data(),costs.col_idx.data(),costs.values.data(),costs.num_rows,costs.num_cols,
                result.row_to_col.data(),result.col_to_row.data(),result.row_dual.data(),result.col_dual.data(),
                slack.data(),predecessor.data(),used.data(),status.data(),steps.data());
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(events.end, context.stream));
        CUDA_CHECK(cudaEventSynchronize(events.end));

        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(
            &milliseconds, events.begin, events.end));
        result.kernel_ms = milliseconds;

        int host_status = 2;
        status.download(&host_status, 1, context.stream);
        steps.download(&result.search_steps, 1, context.stream);

        result.status = host_status == 0
            ? GPUAssignmentStatus::Success
            : host_status == 1
                ? GPUAssignmentStatus::Infeasible
                : GPUAssignmentStatus::NumericalFailure;
    } catch (...) {
        (void)cudaStreamSynchronize(context.stream);
        throw;
    }

    return result;
}

GPUAssignmentHostResult gpu_download_assignment(
    const GPUAssignmentResult& device,
    const CSRMatrix& costs,
    const GPUContext& context,
    double absolute,
    double relative)
{
    validate_host_costs(costs);

    if (!std::isfinite(absolute) || !std::isfinite(relative) ||
        absolute < 0 || relative < 0 ||
        costs.rows != device.num_rows || costs.cols != device.num_cols) {
        throw std::invalid_argument(
            "GPU assignment: invalid certificate request");
    }

    if (device.row_to_col.size() != static_cast<std::size_t>(costs.rows) ||
        device.col_to_row.size() != static_cast<std::size_t>(costs.cols) ||
        device.row_dual.size() != static_cast<std::size_t>(costs.rows) ||
        device.col_dual.size() != static_cast<std::size_t>(costs.cols)) {
        throw std::invalid_argument(
            "GPU assignment: result dimensions disagree");
    }

    GPUAssignmentHostResult host;
    host.status = device.status;
    host.kernel_ms = device.kernel_ms;
    host.search_steps = device.search_steps;
    host.workspace_bytes = device.workspace_bytes;

    auto& assignment = host.assignment;
    assignment.row_to_col.resize(costs.rows);
    assignment.col_to_row.resize(costs.cols);
    host.row_dual.resize(costs.rows);
    host.col_dual.resize(costs.cols);

    device.row_to_col.download(
        assignment.row_to_col.data(), assignment.row_to_col.size(),
        context.stream);
    device.col_to_row.download(
        assignment.col_to_row.data(), assignment.col_to_row.size(),
        context.stream);
    device.row_dual.download(
        host.row_dual.data(), host.row_dual.size(), context.stream);
    device.col_dual.download(
        host.col_dual.data(), host.col_dual.size(), context.stream);

    host.download_bytes =
        device.row_to_col.bytes() + device.col_to_row.bytes() +
        device.row_dual.bytes() + device.col_dual.bytes();

    auto& certificate = host.certificate;
    certificate.primal_feasible =
        device.status == GPUAssignmentStatus::Success;
    certificate.dual_feasible = true;
    certificate.complementary_slackness = true;

    std::vector<int> seen(costs.cols, 0);

    for (double value : host.row_dual) {
        if (!std::isfinite(value)) {
            certificate.dual_feasible = false;
            certificate.complementary_slackness = false;
        }
        certificate.dual_objective += value;
    }

    for (int column = 0; column < costs.cols; ++column) {
        const double value = host.col_dual[column];
        const int row = assignment.col_to_row[column];

        if (!std::isfinite(value) ||
            value > tolerance(value, 0.0, absolute, relative)) {
            certificate.dual_feasible = false;
        }

        certificate.dual_objective += value;

        if (row < -1 || row >= costs.rows) {
            certificate.primal_feasible = false;
        } else if (row >= 0 &&
                   assignment.row_to_col[row] != column) {
            certificate.primal_feasible = false;
        }

        if (row == -1 &&
            (!std::isfinite(value) ||
             std::abs(value) > tolerance(value, 0.0, absolute, relative))) {
            certificate.complementary_slackness = false;
        }
    }

    for (int row = 0; row < costs.rows; ++row) {
        const int matched = assignment.row_to_col[row];
        bool found_match = false;

        if (matched < 0 || matched >= costs.cols) {
            certificate.primal_feasible = false;
        } else {
            if (++seen[matched] != 1 ||
                assignment.col_to_row[matched] != row) {
                certificate.primal_feasible = false;
            }
        }

        for (int p = costs.row_ptr[row]; p < costs.row_ptr[row + 1]; ++p) {
            const int column = costs.col_idx[p];
            const double cost = costs.values[p];
            const double dual = host.row_dual[row] + host.col_dual[column];

            if (!std::isfinite(dual)) {
                certificate.dual_feasible = false;
                certificate.complementary_slackness = false;
                continue;
            }

            certificate.max_dual_violation = std::max(
                certificate.max_dual_violation, std::max(0.0, dual - cost));

            if (dual - cost > tolerance(dual, cost, absolute, relative)) {
                certificate.dual_feasible = false;
            }

            if (column == matched) {
                found_match = true;
                certificate.primal_objective += cost;

                const double slack = std::abs(cost - dual);
                certificate.max_matched_slack = std::max(
                    certificate.max_matched_slack, slack);

                if (slack > tolerance(dual, cost, absolute, relative)) {
                    certificate.complementary_slackness = false;
                }
            }
        }

        if (!found_match) certificate.primal_feasible = false;
    }

    certificate.objective_gap = std::abs(
        certificate.primal_objective - certificate.dual_objective);

    const double gap_tolerance =
        absolute * std::max(1, costs.rows) +
        relative * std::max(
            1.0, std::max(std::abs(certificate.primal_objective),
                          std::abs(certificate.dual_objective)));

    certificate.optimal =
        certificate.primal_feasible &&
        certificate.dual_feasible &&
        certificate.complementary_slackness &&
        std::isfinite(certificate.objective_gap) &&
        certificate.objective_gap <= gap_tolerance;

    assignment.feasible = certificate.primal_feasible;
    assignment.total_cost = certificate.primal_feasible
        ? certificate.primal_objective : 0.0;

    return host;
}
