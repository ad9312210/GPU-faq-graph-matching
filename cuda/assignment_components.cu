#include "assignment_components.hpp"
#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>

namespace {
__global__ void initialize_labels(int* labels, int vertices) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < static_cast<std::size_t>(vertices); i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        labels[i] = static_cast<int>(i);
}
__global__ void invert_columns(const int* originals, int count, int* inverse) {
    for (std::size_t i = threadIdx.x; i < static_cast<std::size_t>(count); i += blockDim.x)
        inverse[originals[i]] = static_cast<int>(i);
}
// Correctness baseline: a single writer constructs bounded int offsets.
// Parallel scan is a later optimization; selected rows already have counts.
__global__ void compact_offsets(const int* input_offsets, const int* rows,
    int count, int expected_edges, int* output_offsets, int* error) {
    long long total = 0;
    output_offsets[0] = 0;
    for (int i = 0; i < count; ++i) {
        total += input_offsets[rows[i] + 1] - input_offsets[rows[i]];
        if (total > expected_edges) { *error = 1; return; }
        output_offsets[i + 1] = static_cast<int>(total);
    }
    if (total != expected_edges) *error = 1;
}
__global__ void compact_entries(const int* input_offsets, const int* input_columns,
    const float* input_values, const int* rows, int count, const int* inverse,
    const int* output_offsets, int* columns, float* values, int* error) {
    for (std::size_t local_row = blockIdx.x;
         local_row < static_cast<std::size_t>(count); local_row += gridDim.x) {
        const int original = rows[local_row];
        const int begin = input_offsets[original];
        const int end = input_offsets[original + 1];
        for (std::size_t p = static_cast<std::size_t>(begin) + threadIdx.x;
             p < static_cast<std::size_t>(end); p += blockDim.x) {
            const int column = inverse[input_columns[p]];
            if (column < 0) { atomicExch(error, 1); continue; }
            const auto destination = static_cast<std::size_t>(output_offsets[local_row]) + p - begin;
            columns[destination] = column;
            values[destination] = input_values[p];
        }
    }
}
__global__ void propagate(const int* offsets, const int* columns, int rows,
                          int* labels, int* counts, int* changed) {
    for (std::size_t r = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         r < static_cast<std::size_t>(rows); r += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        counts[r] = offsets[r + 1] - offsets[r];
        for (int p = offsets[r]; p < offsets[r + 1]; ++p) {
            const int c = rows + columns[p];
            // All label accesses during propagation are atomic. Labels only
            // decrease, so convergence gives each connected component its
            // smallest original bipartite vertex ID.
            const int a = atomicAdd(labels + r, 0);
            const int b = atomicAdd(labels + c, 0);
            const int smallest = a < b ? a : b;
            if (atomicMin(labels + r, smallest) > smallest) atomicExch(changed, 1);
            if (atomicMin(labels + c, smallest) > smallest) atomicExch(changed, 1);
        }
    }
}
}

AssignmentComponents gpu_find_assignment_components(
    const DeviceCSR& costs, const GPUContext& context, std::size_t budget) {
    DeviceCSR::validate_dimensions(costs.num_rows, costs.num_cols, costs.entry_count);
    const auto rows = static_cast<std::size_t>(costs.num_rows);
    const auto columns = static_cast<std::size_t>(costs.num_cols);
    const auto vertices = rows + columns;
    if (vertices > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("Bipartite component vertex IDs exceed int capacity");
    const auto bytes = (vertices + rows + 1) * sizeof(int);
    if (bytes > budget) throw std::invalid_argument("Component workspace exceeds budget");
    gpu_validate_device_csr(costs, context);
    CUDA_CHECK(cudaSetDevice(context.device_id));
    AssignmentComponents result;
    result.workspace_bytes = bytes;
    DeviceBuffer<int> labels(vertices, context.device_id);
    DeviceBuffer<int> counts(rows, context.device_id);
    DeviceBuffer<int> changed(1, context.device_id);
    const unsigned blocks = static_cast<unsigned>(std::min<std::size_t>(
        65535, std::max<std::size_t>(1, (vertices + 255) / 256)));
    std::vector<int> host_labels(vertices), host_counts(rows);
    try {
        initialize_labels<<<blocks, 256, 0, context.stream>>>(labels.data(), static_cast<int>(vertices));
        CUDA_CHECK(cudaGetLastError());
        if (rows && costs.entry_count) {
            for (;;) {
                CUDA_CHECK(cudaMemsetAsync(changed.data(), 0, sizeof(int), context.stream));
                propagate<<<blocks, 256, 0, context.stream>>>(costs.row_ptr.data(),
                    costs.col_idx.data(), costs.num_rows, labels.data(), counts.data(), changed.data());
                CUDA_CHECK(cudaGetLastError());
                int flag = 0;
                changed.download(&flag, 1, context.stream);
                ++result.propagation_rounds;
                if (!flag) break;
                if (result.propagation_rounds >= static_cast<int>(vertices))
                    throw std::runtime_error("Component propagation did not converge");
            }
        } else if (rows) {
            CUDA_CHECK(cudaMemsetAsync(counts.data(), 0, rows * sizeof(int), context.stream));
        }
        labels.download(host_labels.data(), vertices, context.stream);
        counts.download(host_counts.data(), rows, context.stream);
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
    } catch (...) {
        (void)cudaStreamSynchronize(context.stream);
        throw;
    }
    result.download_bytes = (vertices + rows) * sizeof(int) +
        static_cast<std::size_t>(result.propagation_rounds) * sizeof(int);
    std::map<int, int> component_ids;
    for (int label : host_labels) component_ids.emplace(label, 0);
    int next = 0;
    for (auto& item : component_ids) item.second = next++;
    result.components.resize(component_ids.size());
    result.row_component.resize(rows);
    result.column_component.resize(columns);
    result.row_local_index.resize(rows);
    result.column_local_index.resize(columns);
    for (std::size_t r = 0; r < rows; ++r) {
        const int id = component_ids.at(host_labels[r]);
        result.row_component[r] = id;
        result.row_local_index[r] = static_cast<int>(result.components[id].rows.size());
        result.components[id].rows.push_back(static_cast<int>(r));
        result.components[id].edges += static_cast<std::size_t>(host_counts[r]);
    }
    for (std::size_t c = 0; c < columns; ++c) {
        const int id = component_ids.at(host_labels[rows + c]);
        result.column_component[c] = id;
        result.column_local_index[c] = static_cast<int>(result.components[id].columns.size());
        result.components[id].columns.push_back(static_cast<int>(c));
    }
    for (auto& component : result.components)
        component.passes_cardinality_check = component.rows.size() <= component.columns.size();
    return result;
}

CompactAssignmentComponent gpu_compact_assignment_component(
    const DeviceCSR& costs, const AssignmentComponent& component,
    const GPUContext& context, std::size_t budget) {
    gpu_validate_device_csr(costs, context);
    auto validate_ids = [](const std::vector<int>& ids, int extent) {
        int previous = -1;
        for (int id : ids) {
            if (id < 0 || id >= extent || id <= previous)
                throw std::invalid_argument("Component IDs must be sorted, unique, and in range");
            previous = id;
        }
    };
    validate_ids(component.rows, costs.num_rows);
    validate_ids(component.columns, costs.num_cols);
    if (component.edges > costs.entry_count)
        throw std::invalid_argument("Component edge count exceeds original support");
    const int rows = static_cast<int>(component.rows.size());
    const int columns = static_cast<int>(component.columns.size());
    DeviceCSR::validate_dimensions(rows, columns, component.edges);
    const auto workspace = (static_cast<std::size_t>(costs.num_cols) + 1) * sizeof(int);
    if (workspace > budget)
        throw std::invalid_argument("Component compaction workspace exceeds budget");
    CUDA_CHECK(cudaSetDevice(context.device_id));
    CompactAssignmentComponent result;
    result.passes_cardinality_check = rows <= columns;
    result.workspace_bytes = workspace;
    result.costs.num_rows = rows;
    result.costs.num_cols = columns;
    result.costs.entry_count = component.edges;
    result.original_rows = DeviceBuffer<int>(component.rows.size(), context.device_id);
    result.original_columns = DeviceBuffer<int>(component.columns.size(), context.device_id);
    result.costs.row_ptr = DeviceBuffer<int>(component.rows.size() + 1, context.device_id);
    result.costs.col_idx = DeviceBuffer<int>(component.edges, context.device_id);
    result.costs.values = DeviceBuffer<float>(component.edges, context.device_id);
    DeviceBuffer<int> inverse(static_cast<std::size_t>(costs.num_cols), context.device_id);
    DeviceBuffer<int> error(1, context.device_id);
    try {
        result.original_rows.upload(component.rows.data(), component.rows.size(), context.stream);
        result.original_columns.upload(component.columns.data(), component.columns.size(), context.stream);
        if (costs.num_cols)
            CUDA_CHECK(cudaMemsetAsync(inverse.data(), 0xff, inverse.bytes(), context.stream));
        CUDA_CHECK(cudaMemsetAsync(error.data(), 0, sizeof(int), context.stream));
        invert_columns<<<1,256,0,context.stream>>>(result.original_columns.data(),columns,inverse.data());
        CUDA_CHECK(cudaGetLastError());
        compact_offsets<<<1,1,0,context.stream>>>(costs.row_ptr.data(),result.original_rows.data(),
            rows,static_cast<int>(component.edges),result.costs.row_ptr.data(),error.data());
        CUDA_CHECK(cudaGetLastError());
        int invalid = 0;
        error.download(&invalid,1,context.stream);
        if (invalid) throw std::invalid_argument("Component edge count disagrees with selected rows");
        if (rows) {
            compact_entries<<<std::min(rows,65535),256,0,context.stream>>>(
                costs.row_ptr.data(),costs.col_idx.data(),costs.values.data(),
                result.original_rows.data(),rows,inverse.data(),result.costs.row_ptr.data(),
                result.costs.col_idx.data(),result.costs.values.data(),error.data());
            CUDA_CHECK(cudaGetLastError());
            error.download(&invalid,1,context.stream);
            if (invalid) throw std::invalid_argument("Candidate edge crosses component boundary");
        }
        gpu_validate_device_csr(result.costs,context);
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
    } catch (...) {
        (void)cudaStreamSynchronize(context.stream);
        throw;
    }
    return result;
}
