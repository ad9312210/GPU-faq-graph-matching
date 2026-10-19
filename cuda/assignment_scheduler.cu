#include "assignment_scheduler.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
__global__ void initialize_global(int rows, int columns, int* rm, int* cm,
                                 double* rd, double* cd) {
    for (std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<static_cast<std::size_t>(rows>columns?rows:columns);
         i+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
        if (i<static_cast<std::size_t>(rows)) { rm[i]=-1; rd[i]=0; }
        if (i<static_cast<std::size_t>(columns)) { cm[i]=-1; cd[i]=0; }
    }
}
__global__ void scatter_component(int rows,int columns,const int* original_rows,
    const int* original_columns,const int* rm,const int* cm,const double* rd,
    const double* cd,int* global_rm,int* global_cm,double* global_rd,double* global_cd) {
    for (std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<static_cast<std::size_t>(rows>columns?rows:columns);
         i+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
        if (i<static_cast<std::size_t>(rows)) {
            global_rm[original_rows[i]]=rm[i]<0?-1:original_columns[rm[i]];
            global_rd[original_rows[i]]=rd[i];
        }
        if (i<static_cast<std::size_t>(columns)) {
            global_cm[original_columns[i]]=cm[i]<0?-1:original_rows[cm[i]];
            global_cd[original_columns[i]]=cd[i];
        }
    }
}
unsigned blocks_for(int rows,int columns) {
    return static_cast<unsigned>(std::min<std::size_t>(65535,
        std::max<std::size_t>(1,(static_cast<std::size_t>(std::max(rows,columns))+255)/256)));
}
}

const char* gpu_assignment_execution_name(GPUAssignmentExecution execution) noexcept {
    switch (execution) {
    case GPUAssignmentExecution::Warp: return "warp";
    case GPUAssignmentExecution::Block: return "block";
    case GPUAssignmentExecution::Cooperative: return "cooperative";
    }
    return "unknown";
}

GPUAssignmentScheduleResult gpu_schedule_sparse_assignment(
    const DeviceCSR& costs,const GPUContext& context,
    const GPUAssignmentSchedulerOptions& options) {
    const auto started=Clock::now();
    const auto requested=options.assignment.execution;
    if (requested!=GPUAssignmentExecution::Warp && requested!=GPUAssignmentExecution::Block &&
        requested!=GPUAssignmentExecution::Cooperative)
        throw std::invalid_argument("Unknown scheduled assignment execution");
    if (requested==GPUAssignmentExecution::Cooperative && options.assignment.cooperative_blocks<2)
        throw std::invalid_argument("Cooperative assignment requires at least two blocks");
    CUDA_CHECK(cudaSetDevice(context.device_id));
    int cooperative_supported=0;
    if (requested==GPUAssignmentExecution::Cooperative)
        CUDA_CHECK(cudaDeviceGetAttribute(&cooperative_supported,cudaDevAttrCooperativeLaunch,context.device_id));
    if (requested==GPUAssignmentExecution::Cooperative && !cooperative_supported &&
        !options.allow_unsupported_cooperative_fallback)
        throw std::invalid_argument("Cooperative scheduler execution unsupported; fallback disabled");
    GPUAssignmentScheduleResult result;
    const auto discovery_start=Clock::now();
    const auto components=gpu_find_assignment_components(costs,context,options.component_workspace_budget_bytes);
    result.discovery_ms=elapsed(discovery_start);
    result.discovery_workspace_bytes=components.workspace_bytes;
    result.workspace_peak_bytes=components.workspace_bytes;
    result.metadata_download_bytes=components.download_bytes;
    auto& global=result.assignment;
    global.num_rows=costs.num_rows; global.num_cols=costs.num_cols;
    global.execution=requested==GPUAssignmentExecution::Cooperative && !cooperative_supported
        ? GPUAssignmentExecution::Block : requested;
    global.execution_blocks=0;
    global.status=GPUAssignmentStatus::Success;
    global.row_to_col=DeviceBuffer<int>(costs.num_rows,context.device_id);
    global.col_to_row=DeviceBuffer<int>(costs.num_cols,context.device_id);
    global.row_dual=DeviceBuffer<double>(costs.num_rows,context.device_id);
    global.col_dual=DeviceBuffer<double>(costs.num_cols,context.device_id);
    try {
        initialize_global<<<blocks_for(costs.num_rows,costs.num_cols),256,0,context.stream>>>(
            costs.num_rows,costs.num_cols,global.row_to_col.data(),global.col_to_row.data(),
            global.row_dual.data(),global.col_dual.data());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
        result.dispatch.reserve(components.components.size());
        for (std::size_t id=0;id<components.components.size();++id) {
            const auto phase=Clock::now();
            const auto& component=components.components[id];
            GPUComponentDispatch info;
            info.component=static_cast<int>(id);
            info.rows=static_cast<int>(component.rows.size());
            info.columns=static_cast<int>(component.columns.size());
            info.edges=component.edges;
            info.requested=info.effective=requested;
            if (component.rows.empty()) {
                info.reason="isolated_columns_no_assignment";
                info.status=GPUAssignmentStatus::Success;
            } else if (!component.passes_cardinality_check) {
                info.reason="component_rows_exceed_columns";
                info.status=GPUAssignmentStatus::Infeasible;
                if (global.status==GPUAssignmentStatus::Success) global.status=info.status;
            } else {
                auto local_options=options.assignment;
                if (requested==GPUAssignmentExecution::Cooperative && !cooperative_supported) {
                    info.effective=local_options.execution=GPUAssignmentExecution::Block;
                    info.reason="cooperative_launch_unsupported";
                }
                const auto compact_start=Clock::now();
                auto compact=gpu_compact_assignment_component(costs,component,context,
                    options.component_workspace_budget_bytes);
                info.compaction_ms=elapsed(compact_start);
                auto solved=gpu_solve_sparse_assignment(compact.costs,context,local_options);
                info.executed=true; info.status=solved.status;
                info.blocks=solved.execution_blocks;
                info.assignment_kernel_ms=solved.kernel_ms;
                info.workspace_peak_bytes=std::max(compact.workspace_bytes,solved.workspace_bytes);
                info.compact_payload_bytes=compact.costs.row_ptr.bytes()+compact.costs.col_idx.bytes()+
                    compact.costs.values.bytes()+compact.original_rows.bytes()+compact.original_columns.bytes();
                const auto payload=info.compact_payload_bytes+solved.row_to_col.bytes()+solved.col_to_row.bytes()+
                    solved.row_dual.bytes()+solved.col_dual.bytes();
                result.component_payload_peak_bytes=std::max(result.component_payload_peak_bytes,payload);
                result.workspace_peak_bytes=std::max(result.workspace_peak_bytes,info.workspace_peak_bytes);
                global.kernel_ms+=solved.kernel_ms;
                global.search_steps+=solved.search_steps;
                if (solved.status==GPUAssignmentStatus::NumericalFailure ||
                    (solved.status==GPUAssignmentStatus::Infeasible && global.status==GPUAssignmentStatus::Success))
                    global.status=solved.status;
                if (solved.status!=GPUAssignmentStatus::NumericalFailure) {
                    scatter_component<<<blocks_for(info.rows,info.columns),256,0,context.stream>>>(
                        info.rows,info.columns,compact.original_rows.data(),compact.original_columns.data(),
                        solved.row_to_col.data(),solved.col_to_row.data(),solved.row_dual.data(),solved.col_dual.data(),
                        global.row_to_col.data(),global.col_to_row.data(),global.row_dual.data(),global.col_dual.data());
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaStreamSynchronize(context.stream));
                }
            }
            info.total_ms=elapsed(phase);
            result.dispatch.push_back(std::move(info));
        }
        global.workspace_bytes=result.workspace_peak_bytes;
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
    } catch (...) {
        (void)cudaStreamSynchronize(context.stream);
        throw;
    }
    result.total_ms=elapsed(started);
    return result;
}
