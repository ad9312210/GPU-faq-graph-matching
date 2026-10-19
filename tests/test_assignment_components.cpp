#include "assignment_components.hpp"
#include "gpu_assignment.hpp"
#include <iostream>
#include <stdexcept>

void require(bool ok) { if (!ok) throw std::runtime_error("Component check failed"); }
void check(const GPUContext& ctx) {
    // {r0,r2,c1,c3}, {r1,c0}, isolated r3, isolated c2/c4.
    DeviceCSR matrix;
    matrix.num_rows = 4; matrix.num_cols = 5; matrix.entry_count = 5;
    const int offsets[]{0,2,3,5,5};
    const int columns[]{1,3,0,1,3};
    const float values[]{0,1,2,3,4};
    matrix.row_ptr = DeviceBuffer<int>(5,ctx.device_id);
    matrix.col_idx = DeviceBuffer<int>(5,ctx.device_id);
    matrix.values = DeviceBuffer<float>(5,ctx.device_id);
    matrix.row_ptr.upload(offsets,5,ctx.stream);
    matrix.col_idx.upload(columns,5,ctx.stream);
    matrix.values.upload(values,5,ctx.stream);
    auto components = gpu_find_assignment_components(matrix,ctx);
    require(components.components.size() == 5);
    require(components.row_component[0] == components.row_component[2]);
    require(components.row_component[0] == components.column_component[1]);
    require(components.row_component[0] == components.column_component[3]);
    require(components.row_component[1] == components.column_component[0]);
    require(!components.components[components.row_component[3]].passes_cardinality_check);
    require(components.components[components.row_component[0]].edges == 4);
    require(components.row_local_index == std::vector<int>({0,0,1,0}));
    for (int r = 0; r < matrix.num_rows; ++r)
        require(components.components[components.row_component[r]].rows[
            components.row_local_index[r]] == r);
    for (int c = 0; c < matrix.num_cols; ++c)
        require(components.components[components.column_component[c]].columns[
            components.column_local_index[c]] == c);
    // Every original edge must stay wholly within a single component.
    for (int r = 0; r < matrix.num_rows; ++r)
        for (int p = offsets[r]; p < offsets[r + 1]; ++p)
            require(components.row_component[r] == components.column_component[columns[p]]);
    const auto repeated = gpu_find_assignment_components(matrix,ctx);
    require(components.row_component == repeated.row_component);
    require(components.column_component == repeated.column_component);
    require(components.row_local_index == repeated.row_local_index);
    require(components.column_local_index == repeated.column_local_index);
    std::size_t compacted_edges = 0;
    for (const auto& component : components.components) {
        auto compact = gpu_compact_assignment_component(matrix,component,ctx);
        require(compact.costs.num_rows == static_cast<int>(component.rows.size()));
        require(compact.costs.num_cols == static_cast<int>(component.columns.size()));
        require(compact.costs.entry_count == component.edges);
        std::vector<int> row_ptr(component.rows.size()+1), targets(component.edges);
        std::vector<float> costs(component.edges);
        compact.costs.row_ptr.download(row_ptr.data(),row_ptr.size(),ctx.stream);
        compact.costs.col_idx.download(targets.data(),targets.size(),ctx.stream);
        compact.costs.values.download(costs.data(),costs.size(),ctx.stream);
        for (std::size_t r=0;r<component.rows.size();++r) {
            const int original=component.rows[r];
            require(row_ptr[r+1]-row_ptr[r] == offsets[original+1]-offsets[original]);
            for (int p=row_ptr[r];p<row_ptr[r+1];++p) {
                const int old=offsets[original]+p-row_ptr[r];
                require(component.columns[targets[p]] == columns[old]);
                require(costs[p] == values[old]);
            }
        }
        compacted_edges += component.edges;
        if (compact.passes_cardinality_check) {
            auto solved=gpu_solve_sparse_assignment(compact.costs,ctx);
            require(solved.status == GPUAssignmentStatus::Success);
        }
    }
    require(compacted_edges == matrix.entry_count);

    // A connected 3x3 component can still violate Hall's condition.
    DeviceCSR hall;
    hall.num_rows=3; hall.num_cols=3; hall.entry_count=5;
    const int hall_offsets[]{0,1,2,5};
    const int hall_columns[]{0,0,0,1,2};
    hall.row_ptr=DeviceBuffer<int>(4,ctx.device_id);
    hall.col_idx=DeviceBuffer<int>(5,ctx.device_id);
    hall.values=DeviceBuffer<float>(5,ctx.device_id);
    hall.row_ptr.upload(hall_offsets,4,ctx.stream);
    hall.col_idx.upload(hall_columns,5,ctx.stream);
    hall.values.upload(values,5,ctx.stream);
    const auto hall_components=gpu_find_assignment_components(hall,ctx);
    require(hall_components.components.size()==1);
    auto hall_compact=gpu_compact_assignment_component(hall,hall_components.components[0],ctx);
    require(hall_compact.passes_cardinality_check);
    auto hall_solution=gpu_solve_sparse_assignment(hall_compact.costs,ctx);
    require(hall_solution.status == GPUAssignmentStatus::Infeasible);

    bool bad_count=false;
    try {
        auto bad=components.components[components.row_component[0]];
        --bad.edges;
        auto unused=gpu_compact_assignment_component(matrix,bad,ctx);
    } catch (const std::invalid_argument&) { bad_count=true; }
    require(bad_count);
    bool cross_edge=false;
    try {
        auto bad=components.components[components.row_component[0]];
        bad.columns={1};
        auto unused=gpu_compact_assignment_component(matrix,bad,ctx);
    } catch (const std::invalid_argument&) { cross_edge=true; }
    require(cross_edge);
    bool rejected = false;
    try { auto unused = gpu_find_assignment_components(matrix,ctx,0); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);

    DeviceCSR empty_support;
    empty_support.num_rows = 2;
    empty_support.num_cols = 3;
    const int zeros[]{0,0,0};
    empty_support.row_ptr = DeviceBuffer<int>(3,ctx.device_id);
    empty_support.row_ptr.upload(zeros,3,ctx.stream);
    const auto isolated = gpu_find_assignment_components(empty_support,ctx);
    require(isolated.components.size() == 5);
    for (int r = 0; r < 2; ++r) {
        const auto& group = isolated.components[isolated.row_component[r]];
        require(group.rows == std::vector<int>({r}));
        require(group.columns.empty() && group.edges == 0);
        require(!group.passes_cardinality_check);
        require(isolated.row_local_index[r] == 0);
    }
    for (int c = 0; c < 3; ++c) {
        const auto& group = isolated.components[isolated.column_component[c]];
        require(group.columns == std::vector<int>({c}));
        require(group.rows.empty() && group.edges == 0);
        require(group.passes_cardinality_check);
        require(isolated.column_local_index[c] == 0);
    }

    DeviceCSR zero;
    const int origin = 0;
    zero.row_ptr = DeviceBuffer<int>(1,ctx.device_id);
    zero.row_ptr.upload(&origin,1,ctx.stream);
    const auto none = gpu_find_assignment_components(zero,ctx);
    require(none.components.empty());
    require(none.row_component.empty() && none.column_component.empty());
}
int main() {
    cudaStream_t stream = nullptr;
    try {
        int count = 0;
        const auto status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && !count)) return 77;
        CUDA_CHECK(status);
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaStreamCreate(&stream));
        GPUContext ctx; ctx.device_id=0; ctx.stream=stream;
        check(ctx);
        CUDA_CHECK(cudaStreamDestroy(stream)); stream=nullptr;
        std::cout << "Assignment component checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
        std::cerr << error.what() << '\n'; return 1;
    }
}
