#include "device_csr.hpp"
#include "common.hpp"

#include <cub/device/device_scan.cuh>

#include <limits>
#include <stdexcept>
#include <utility>

namespace {

__global__ void validate_candidate_storage(
    const std::size_t* storage_offsets,
    const int* row_counts,
    int num_rows,
    std::size_t storage_entries,
    std::size_t* scan_counts,
    int* invalid)
{
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (row == 0) {
        if (storage_offsets[0] != 0 ||
            storage_offsets[num_rows] != storage_entries) {
            atomicExch(invalid, 1);
        }
    }

    if (row >= static_cast<std::size_t>(num_rows)) return;

    const std::size_t begin = storage_offsets[row];
    const std::size_t end = storage_offsets[row + 1];

    if (begin > end || end > storage_entries) {
        atomicExch(invalid, 1);
        return;
    }

    const std::size_t capacity = end - begin;

    if (row_counts != nullptr) {
        const int count = row_counts[row];

        if (count < 0 ||
            static_cast<std::size_t>(count) > capacity) {
            atomicExch(invalid, 1);
            return;
        }

        scan_counts[row] = static_cast<std::size_t>(count);
    }
}

/*
 * Construct a separate CSR representation.
 *
 * Each active candidate counts smaller target IDs in its row.
 * Its rank is its destination within the target-sorted CSR row.
 *
 * Unique IDs yield unique destinations. Duplicate candidates skip their
 * writes and cause construction to fail.
 *
 * This correctness-first implementation performs O(sum(row_length^2))
 * comparisons. Prefix summation and row processing run on GPU.
 */
__global__ void write_candidate_costs(
    const std::size_t* storage_offsets,
    const std::size_t* compact_offsets,
    const int* input_targets,
    const float* input_scores,
    int num_rows,
    int num_cols,
    int* output_offsets,
    int* output_targets,
    float* output_costs,
    int* invalid)
{
    const int row = static_cast<int>(blockIdx.x);

    if (row >= num_rows) {
        if (row == 0 && threadIdx.x == 0) {
            output_offsets[0] = 0;
        }
        return;
    }

    const std::size_t storage_begin = storage_offsets[row];
    const std::size_t output_begin = compact_offsets[row];
    const std::size_t output_end = compact_offsets[row + 1];
    const std::size_t count = output_end - output_begin;

    if (threadIdx.x == 0) {
        // Host checks the final total before this conversion.
        output_offsets[row] = static_cast<int>(output_begin);

        if (row == num_rows - 1) {
            output_offsets[num_rows] = static_cast<int>(output_end);
        }
    }

    for (std::size_t local = static_cast<std::size_t>(threadIdx.x);
         local < count;
         local += blockDim.x) {
        const std::size_t input_position = storage_begin + local;
        const int target = input_targets[input_position];
        const float similarity = input_scores[input_position];

        if (target < 0 || target >= num_cols || !isfinite(similarity)) {
            atomicExch(invalid, 1);
            continue;
        }

        std::size_t rank = 0;
        bool duplicate = false;

        for (std::size_t other = 0; other < count; ++other) {
            const int other_target =
                input_targets[storage_begin + other];

            if (other_target < target) {
                ++rank;
            } else if (other != local && other_target == target) {
                duplicate = true;
            }
        }

        if (duplicate) {
            atomicExch(invalid, 1);
            continue;
        }

        const float cost = 1.0f - similarity;

        if (!isfinite(cost)) {
            atomicExch(invalid, 1);
            continue;
        }

        const std::size_t destination = output_begin + rank;
        output_targets[destination] = target;
        output_costs[destination] = cost;
    }
}

unsigned int validation_blocks(int rows) {
    return rows > 0
        ? static_cast<unsigned int>(
            1 + (static_cast<std::size_t>(rows) - 1) / 256)
        : 1u;
}

} // namespace


namespace {

void validate_csr_storage(
    const DeviceCSR& matrix,
    const GPUContext& context)
{
    DeviceCSR::validate_dimensions(
        matrix.num_rows, matrix.num_cols, matrix.entry_count);

    if (matrix.row_ptr.size() !=
            static_cast<std::size_t>(matrix.num_rows) + 1 ||
        matrix.col_idx.size() != matrix.entry_count ||
        matrix.values.size() != matrix.entry_count) {
        throw std::invalid_argument(
            "GPU CSR: buffer sizes disagree with metadata");
    }

    if (matrix.row_ptr.device_id() != context.device_id ||
        (!matrix.col_idx.empty() &&
         matrix.col_idx.device_id() != context.device_id) ||
        (!matrix.values.empty() &&
         matrix.values.device_id() != context.device_id)) {
        throw std::invalid_argument(
            "GPU CSR: storage belongs to another device");
    }
}

__global__ void validate_csr_contents(
    const int* row_ptr,
    const int* targets,
    const float* costs,
    int rows,
    int cols,
    int entries,
    int* invalid)
{
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (row == 0) {
        if (row_ptr[0] != 0 || row_ptr[rows] != entries) {
            atomicExch(invalid, 1);
        }
    }

    if (row >= static_cast<std::size_t>(rows)) return;

    const int begin = row_ptr[row];
    const int end = row_ptr[row + 1];

    if (begin < 0 || begin > end || end > entries) {
        atomicExch(invalid, 1);
        return;
    }

    for (int position = begin; position < end; ++position) {
        const int target = targets[position];

        if (target < 0 || target >= cols || !isfinite(costs[position])) {
            atomicExch(invalid, 1);
        }

        // Strict ordering also rejects duplicates within a row.
        if (position > begin &&
            targets[position - 1] >= target) {
            atomicExch(invalid, 1);
        }
    }
}

__global__ void count_reverse_columns(
    const int* targets,
    std::size_t entries,
    int* column_counts)
{
    const std::size_t position =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (position < entries) {
        atomicAdd(column_counts + targets[position], 1);
    }
}

__global__ void fill_reverse_columns(
    const int* row_ptr,
    const int* targets,
    int rows,
    int* cursors,
    int* unsorted_rows,
    int* unsorted_positions)
{
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;

    const int begin = row_ptr[row];
    const int end = row_ptr[row + 1];

    for (std::size_t position =
             static_cast<std::size_t>(begin) + threadIdx.x;
         position < static_cast<std::size_t>(end);
         position += blockDim.x) {
        const int column = targets[position];
        const int destination = atomicAdd(cursors + column, 1);

        unsorted_rows[destination] = row;
        unsorted_positions[destination] = static_cast<int>(position);
    }
}

/*
 * Atomics produce an unspecified insertion order.
 * Sort a separate output by source row for deterministic traversal.
 *
 * CSR validation guarantees at most one entry per (row, column),
 * so source rows within a column have unique ranks.
 *
 * Correctness-first work: O(sum(column_degree^2)).
 */
__global__ void sort_reverse_columns(
    const int* column_ptr,
    const int* unsorted_rows,
    const int* unsorted_positions,
    int columns,
    int* sorted_rows,
    int* sorted_positions)
{
    const int column = static_cast<int>(blockIdx.x);
    if (column >= columns) return;

    const int begin = column_ptr[column];
    const int end = column_ptr[column + 1];

    for (std::size_t position =
             static_cast<std::size_t>(begin) + threadIdx.x;
         position < static_cast<std::size_t>(end);
         position += blockDim.x) {
        const int source_row = unsorted_rows[position];
        int rank = 0;

        for (int other = begin; other < end; ++other) {
            if (unsorted_rows[other] < source_row) {
                ++rank;
            }
        }

        const std::size_t destination =
            static_cast<std::size_t>(begin) +
            static_cast<std::size_t>(rank);

        sorted_rows[destination] = source_row;
        sorted_positions[destination] = unsorted_positions[position];
    }
}

__global__ void validate_reverse_contents(
    const int* csr_row_ptr,
    const int* csr_targets,
    const int* column_ptr,
    const int* source_rows,
    const int* csr_positions,
    int rows,
    int columns,
    int entries,
    int* invalid)
{
    const std::size_t column =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (column == 0) {
        if (column_ptr[0] != 0 ||
            column_ptr[columns] != entries) {
            atomicExch(invalid, 1);
        }
    }

    if (column >= static_cast<std::size_t>(columns)) return;

    const int begin = column_ptr[column];
    const int end = column_ptr[column + 1];

    if (begin < 0 || begin > end || end > entries) {
        atomicExch(invalid, 1);
        return;
    }

    for (int position = begin; position < end; ++position) {
        const int row = source_rows[position];
        const int csr_position = csr_positions[position];

        if (row < 0 || row >= rows ||
            csr_position < 0 || csr_position >= entries) {
            atomicExch(invalid, 1);
            continue;
        }

        if (position > begin &&
            source_rows[position - 1] >= row) {
            atomicExch(invalid, 1);
        }

        if (csr_position < csr_row_ptr[row] ||
            csr_position >= csr_row_ptr[row + 1] ||
            csr_targets[csr_position] != static_cast<int>(column)) {
            atomicExch(invalid, 1);
        }
    }

    /*
     * Valid CSR has unique (row, column) pairs.
     * Strict source ordering prevents reverse duplicates.
     * Equal total entry counts therefore establish one-to-one coverage.
     */
}

void check_reverse_storage(
    const DeviceCSR& matrix,
    const GPUContext& context)
{
    if (!matrix.has_reverse_index) {
        throw std::invalid_argument(
            "GPU CSR: reverse index has not been constructed");
    }

    if (matrix.column_ptr.size() !=
            static_cast<std::size_t>(matrix.num_cols) + 1 ||
        matrix.source_rows.size() != matrix.entry_count ||
        matrix.csr_positions.size() != matrix.entry_count) {
        throw std::invalid_argument(
            "GPU CSR: reverse-index buffer sizes disagree");
    }

    if (matrix.column_ptr.device_id() != context.device_id ||
        (!matrix.source_rows.empty() &&
         matrix.source_rows.device_id() != context.device_id) ||
        (!matrix.csr_positions.empty() &&
         matrix.csr_positions.device_id() != context.device_id)) {
        throw std::invalid_argument(
            "GPU CSR: reverse index belongs to another device");
    }
}

} // namespace

void gpu_validate_device_csr(
    const DeviceCSR& matrix,
    const GPUContext& context)
{
    validate_csr_storage(matrix, context);
    CUDA_CHECK(cudaSetDevice(context.device_id));

    DeviceBuffer<int> invalid(1, context.device_id);

    try {
        CUDA_CHECK(cudaMemsetAsync(
            invalid.data(), 0, sizeof(int), context.stream));

        validate_csr_contents
            <<<validation_blocks(matrix.num_rows), 256, 0, context.stream>>>(
                matrix.row_ptr.data(),
                matrix.col_idx.data(),
                matrix.values.data(),
                matrix.num_rows,
                matrix.num_cols,
                DeviceCSR::checked_int_offset(matrix.entry_count),
                invalid.data());

        CUDA_CHECK(cudaGetLastError());

        int host_invalid = 0;
        invalid.download(&host_invalid, 1, context.stream);

        if (host_invalid != 0) {
            throw std::invalid_argument(
                "GPU CSR: invalid offsets, targets, ordering, or costs");
        }
    } catch (...) {
        (void)cudaStreamSynchronize(context.stream);
        throw;
    }
}

void gpu_build_device_reverse_index(
    DeviceCSR& matrix,
    const GPUContext& context)
{
    // Validate before using target IDs as histogram indices.
    gpu_validate_device_csr(matrix, context);

    const std::size_t columns =
        static_cast<std::size_t>(matrix.num_cols);

    // Keep the original reverse index unchanged until successful completion.
    DeviceBuffer<int> column_counts;
    DeviceBuffer<int> column_ptr;
    DeviceBuffer<int> cursors;
    DeviceBuffer<int> unsorted_rows;
    DeviceBuffer<int> unsorted_positions;
    DeviceBuffer<int> sorted_rows;
    DeviceBuffer<int> sorted_positions;
    DeviceBuffer<int> invalid;
    DeviceBuffer<unsigned char> scan_workspace;

    try {
        column_counts = DeviceBuffer<int>(columns, context.device_id);
        column_ptr = DeviceBuffer<int>(columns + 1, context.device_id);
        cursors = DeviceBuffer<int>(columns, context.device_id);

        unsorted_rows =
            DeviceBuffer<int>(matrix.entry_count, context.device_id);
        unsorted_positions =
            DeviceBuffer<int>(matrix.entry_count, context.device_id);
        sorted_rows =
            DeviceBuffer<int>(matrix.entry_count, context.device_id);
        sorted_positions =
            DeviceBuffer<int>(matrix.entry_count, context.device_id);

        invalid = DeviceBuffer<int>(1, context.device_id);

        CUDA_CHECK(cudaMemsetAsync(
            invalid.data(), 0, sizeof(int), context.stream));

        CUDA_CHECK(cudaMemsetAsync(
            column_ptr.data(), 0, sizeof(int), context.stream));

        if (columns != 0) {
            CUDA_CHECK(cudaMemsetAsync(
                column_counts.data(), 0,
                column_counts.bytes(), context.stream));
        }

        if (matrix.entry_count != 0) {
            const unsigned int blocks = static_cast<unsigned int>(
                1 + (matrix.entry_count - 1) / 256);

            count_reverse_columns
                <<<blocks, 256, 0, context.stream>>>(
                    matrix.col_idx.data(),
                    matrix.entry_count,
                    column_counts.data());

            CUDA_CHECK(cudaGetLastError());
        }

        if (matrix.num_cols > 0) {
            std::size_t workspace_bytes = 0;

            CUDA_CHECK(cub::DeviceScan::InclusiveSum(
                nullptr,
                workspace_bytes,
                column_counts.data(),
                column_ptr.data() + 1,
                matrix.num_cols,
                context.stream));

            scan_workspace = DeviceBuffer<unsigned char>(
                workspace_bytes == 0 ? 1 : workspace_bytes,
                context.device_id);

            CUDA_CHECK(cub::DeviceScan::InclusiveSum(
                scan_workspace.data(),
                workspace_bytes,
                column_counts.data(),
                column_ptr.data() + 1,
                matrix.num_cols,
                context.stream));

            // Each cursor starts at its column's first output position.
            CUDA_CHECK(cudaMemcpyAsync(
                cursors.data(),
                column_ptr.data(),
                cursors.bytes(),
                cudaMemcpyDeviceToDevice,
                context.stream));
        }

        if (matrix.entry_count != 0) {
            fill_reverse_columns
                <<<static_cast<unsigned int>(matrix.num_rows),
                   256, 0, context.stream>>>(
                    matrix.row_ptr.data(),
                    matrix.col_idx.data(),
                    matrix.num_rows,
                    cursors.data(),
                    unsorted_rows.data(),
                    unsorted_positions.data());

            CUDA_CHECK(cudaGetLastError());

            sort_reverse_columns
                <<<static_cast<unsigned int>(matrix.num_cols),
                   256, 0, context.stream>>>(
                    column_ptr.data(),
                    unsorted_rows.data(),
                    unsorted_positions.data(),
                    matrix.num_cols,
                    sorted_rows.data(),
                    sorted_positions.data());

            CUDA_CHECK(cudaGetLastError());
        }

        validate_reverse_contents
            <<<validation_blocks(matrix.num_cols), 256, 0, context.stream>>>(
                matrix.row_ptr.data(),
                matrix.col_idx.data(),
                column_ptr.data(),
                sorted_rows.data(),
                sorted_positions.data(),
                matrix.num_rows,
                matrix.num_cols,
                DeviceCSR::checked_int_offset(matrix.entry_count),
                invalid.data());

        CUDA_CHECK(cudaGetLastError());

        int host_invalid = 0;
        invalid.download(&host_invalid, 1, context.stream);

        if (host_invalid != 0) {
            throw std::runtime_error(
                "GPU CSR: constructed reverse index failed validation");
        }

        // Commit only after validation and stream completion.
        matrix.has_reverse_index = false;
        matrix.column_ptr = std::move(column_ptr);
        matrix.source_rows = std::move(sorted_rows);
        matrix.csr_positions = std::move(sorted_positions);
        matrix.has_reverse_index = true;
    } catch (...) {
        (void)cudaStreamSynchronize(context.stream);
        throw;
    }
}

void gpu_validate_device_reverse_index(
    const DeviceCSR& matrix,
    const GPUContext& context)
{
    gpu_validate_device_csr(matrix, context);
    check_reverse_storage(matrix, context);

    DeviceBuffer<int> invalid(1, context.device_id);

    try {
        CUDA_CHECK(cudaMemsetAsync(
            invalid.data(), 0, sizeof(int), context.stream));

        validate_reverse_contents
            <<<validation_blocks(matrix.num_cols), 256, 0, context.stream>>>(
                matrix.row_ptr.data(),
                matrix.col_idx.data(),
                matrix.column_ptr.data(),
                matrix.source_rows.data(),
                matrix.csr_positions.data(),
                matrix.num_rows,
                matrix.num_cols,
                DeviceCSR::checked_int_offset(matrix.entry_count),
                invalid.data());

        CUDA_CHECK(cudaGetLastError());

        int host_invalid = 0;
        invalid.download(&host_invalid, 1, context.stream);

        if (host_invalid != 0) {
            throw std::invalid_argument(
                "GPU CSR: inconsistent reverse column index");
        }
    } catch (...) {
        (void)cudaStreamSynchronize(context.stream);
        throw;
    }
}

DeviceCSR gpu_build_candidate_csr(
    const DeviceCandidateRowsView& input,
    const GPUContext& context)
{
    DeviceCSR::validate_dimensions(
        input.num_rows, input.num_cols, 0);

    if (input.device_id != context.device_id) {
        throw std::invalid_argument(
            "GPU CSR: inputs belong to another device");
    }

    if (input.storage_offsets == nullptr ||
        (input.storage_entries != 0 &&
         (input.target_ids == nullptr || input.similarities == nullptr))) {
        throw std::invalid_argument(
            "GPU CSR: missing candidate storage");
    }

    if (input.storage_entries >
            std::numeric_limits<std::size_t>::max() / sizeof(int) ||
        input.storage_entries >
            std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error(
            "GPU CSR: candidate storage byte count overflow");
    }

    const std::size_t rows =
        static_cast<std::size_t>(input.num_rows);

    CUDA_CHECK(cudaSetDevice(context.device_id));

    // Declare owners outside the try block so failure handling drains the
    // stream before these allocations are released.
    DeviceBuffer<int> invalid;
    DeviceBuffer<std::size_t> scan_counts;
    DeviceBuffer<std::size_t> compact_offsets;
    DeviceBuffer<unsigned char> scan_workspace;

    DeviceCSR output;
    output.num_rows = input.num_rows;
    output.num_cols = input.num_cols;

    try {
        invalid = DeviceBuffer<int>(1, context.device_id);

        if (input.row_counts != nullptr) {
            scan_counts =
                DeviceBuffer<std::size_t>(rows, context.device_id);
        }

        CUDA_CHECK(cudaMemsetAsync(
            invalid.data(), 0, sizeof(int), context.stream));

        validate_candidate_storage
            <<<validation_blocks(input.num_rows), 256, 0, context.stream>>>(
                input.storage_offsets,
                input.row_counts,
                input.num_rows,
                input.storage_entries,
                scan_counts.data(),
                invalid.data());

        CUDA_CHECK(cudaGetLastError());

        int host_invalid = 0;
        invalid.download(&host_invalid, 1, context.stream);

        if (host_invalid != 0) {
            throw std::invalid_argument(
                "GPU CSR: invalid storage offsets or row counts");
        }

        // Compact input already supplies the required offsets.
        const std::size_t* final_offsets = input.storage_offsets;
        std::size_t entry_count = input.storage_entries;

        if (input.row_counts != nullptr) {
            compact_offsets = DeviceBuffer<std::size_t>(
                rows + 1, context.device_id);

            // Inclusive scan writes offsets[1..rows].
            // The first offset is explicitly zero.
            CUDA_CHECK(cudaMemsetAsync(
                compact_offsets.data(), 0,
                sizeof(std::size_t), context.stream));

            if (input.num_rows > 0) {
                std::size_t workspace_bytes = 0;

                CUDA_CHECK(cub::DeviceScan::InclusiveSum(
                    nullptr,
                    workspace_bytes,
                    scan_counts.data(),
                    compact_offsets.data() + 1,
                    input.num_rows,
                    context.stream));

                // A nonnull pointer distinguishes execution from CUB's
                // workspace-size query, including a zero-byte requirement.
                const std::size_t allocation_bytes =
                    workspace_bytes == 0 ? 1 : workspace_bytes;

                scan_workspace = DeviceBuffer<unsigned char>(
                    allocation_bytes, context.device_id);

                CUDA_CHECK(cub::DeviceScan::InclusiveSum(
                    scan_workspace.data(),
                    workspace_bytes,
                    scan_counts.data(),
                    compact_offsets.data() + 1,
                    input.num_rows,
                    context.stream));
            }

            compact_offsets.download(
                &entry_count, 1, context.stream, rows);

            final_offsets = compact_offsets.data();
        }

        // Never cast scanned offsets before checking the final total.
        DeviceCSR::validate_dimensions(
            input.num_rows, input.num_cols, entry_count);

        output.entry_count = entry_count;

        output.row_ptr =
            DeviceBuffer<int>(rows + 1, context.device_id);
        output.col_idx =
            DeviceBuffer<int>(entry_count, context.device_id);
        output.values =
            DeviceBuffer<float>(entry_count, context.device_id);

        const unsigned int row_blocks = input.num_rows > 0
            ? static_cast<unsigned int>(input.num_rows)
            : 1u;

        write_candidate_costs
            <<<row_blocks, 256, 0, context.stream>>>(
                input.storage_offsets,
                final_offsets,
                input.target_ids,
                input.similarities,
                input.num_rows,
                input.num_cols,
                output.row_ptr.data(),
                output.col_idx.data(),
                output.values.data(),
                invalid.data());

        CUDA_CHECK(cudaGetLastError());

        invalid.download(&host_invalid, 1, context.stream);

        if (host_invalid != 0) {
            throw std::invalid_argument(
                "GPU CSR: invalid target, duplicate target, "
                "or nonfinite similarity/cost");
        }
    } catch (...) {
        (void)cudaStreamSynchronize(context.stream);
        throw;
    }

    // Reverse-index buffers remain empty.
    return output;
}