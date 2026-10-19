#include "assignment_scheduler.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
void require(bool ok,const char* message) { if (!ok) throw std::runtime_error(message); }
DeviceCSR upload(const CSRMatrix& host,const GPUContext& ctx) {
    DeviceCSR d;
    d.num_rows=host.rows; d.num_cols=host.cols; d.entry_count=host.values.size();
    d.row_ptr=DeviceBuffer<int>(host.row_ptr.size(),ctx.device_id);
    d.col_idx=DeviceBuffer<int>(host.col_idx.size(),ctx.device_id);
    d.values=DeviceBuffer<float>(host.values.size(),ctx.device_id);
    d.row_ptr.upload(host.row_ptr.data(),host.row_ptr.size(),ctx.stream);
    d.col_idx.upload(host.col_idx.data(),host.col_idx.size(),ctx.stream);
    d.values.upload(host.values.data(),host.values.size(),ctx.stream);
    return d;
}
void check(const CSRMatrix& costs,const GPUContext& ctx,bool feasible) {
    auto device=upload(costs,ctx);
    const auto components=gpu_find_assignment_components(device,ctx);
    std::vector<int> seen_rows(costs.rows,0),seen_columns(costs.cols,0);
    std::size_t total_edges=0;
    for (std::size_t id=0;id<components.components.size();++id) {
        const auto& component=components.components[id];
        std::size_t actual_edges=0;
        for (std::size_t local=0;local<component.rows.size();++local) {
            const int row=component.rows[local];
            require(++seen_rows[row]==1,"Source occurs in multiple components");
            require(components.row_component[row]==static_cast<int>(id) &&
                components.row_local_index[row]==static_cast<int>(local),"Source remapping failed");
            for (int p=costs.row_ptr[row];p<costs.row_ptr[row+1];++p) {
                require(components.column_component[costs.col_idx[p]]==static_cast<int>(id),
                    "Candidate edge crosses component boundary");
                ++actual_edges;
            }
        }
        for (std::size_t local=0;local<component.columns.size();++local) {
            const int column=component.columns[local];
            require(++seen_columns[column]==1,"Target occurs in multiple components");
            require(components.column_component[column]==static_cast<int>(id) &&
                components.column_local_index[column]==static_cast<int>(local),"Target remapping failed");
        }
        require(actual_edges==component.edges,"Component edge count differs from original CSR");
        total_edges+=actual_edges;
    }
    require(std::all_of(seen_rows.begin(),seen_rows.end(),[](int n){return n==1;}),
        "A source was omitted from decomposition");
    require(std::all_of(seen_columns.begin(),seen_columns.end(),[](int n){return n==1;}),
        "A target was omitted from decomposition");
    require(total_edges==costs.values.size(),"An edge was omitted from decomposition");
    int supported=0;
    CUDA_CHECK(cudaDeviceGetAttribute(&supported,cudaDevAttrCooperativeLaunch,ctx.device_id));
    for (auto mode : {GPUAssignmentExecution::Warp,GPUAssignmentExecution::Block,
                       GPUAssignmentExecution::Cooperative}) {
        GPUAssignmentSchedulerOptions options;
        options.assignment.execution=mode;
        options.allow_unsupported_cooperative_fallback=true;
        auto scheduled=gpu_schedule_sparse_assignment(device,ctx,options);
        const auto host=gpu_download_assignment(scheduled.assignment,costs,ctx);
        require(host.assignment.feasible==feasible,"Scheduler feasibility differs");
        require((scheduled.assignment.status==GPUAssignmentStatus::Success)==feasible,
            "Scheduler status differs");
        std::size_t rows=0,columns=0,edges=0;
        for (const auto& info : scheduled.dispatch) {
            rows+=info.rows; columns+=info.columns; edges+=info.edges;
            if (info.executed) {
                require(info.effective==(mode==GPUAssignmentExecution::Cooperative && !supported
                    ? GPUAssignmentExecution::Block:mode),"Incorrect effective dispatch");
                require(info.blocks==(info.effective==GPUAssignmentExecution::Cooperative?2:1),
                    "Incorrect dispatch blocks");
                require(info.workspace_peak_bytes<=std::max(options.assignment.workspace_budget_bytes,
                    options.component_workspace_budget_bytes),"Workspace exceeded budget");
            }
        }
        require(rows==static_cast<std::size_t>(costs.rows) &&
            columns==static_cast<std::size_t>(costs.cols) && edges==costs.values.size(),
            "Components do not cover original support");
        if (feasible) {
            require(host.certificate.optimal,"Original-matrix certificate failed");
            require(host.certificate.primal_feasible && host.certificate.dual_feasible &&
                host.certificate.complementary_slackness,"Original-matrix primal/dual conditions failed");
            if (costs.rows) {
                const auto cpu=solve_lapjv(costs);
                require(cpu.feasible && std::abs(cpu.total_cost-host.assignment.total_cost)<1e-8,
                    "Scheduler objective differs from CPU");
            }
        }
        auto repeated=gpu_schedule_sparse_assignment(device,ctx,options);
        const auto again=gpu_download_assignment(repeated.assignment,costs,ctx);
        require(host.assignment.row_to_col==again.assignment.row_to_col,"Nondeterministic scheduler");
        require(host.assignment.col_to_row==again.assignment.col_to_row,"Nondeterministic inverse mapping");
        require(host.row_dual==again.row_dual && host.col_dual==again.col_dual,
            "Nondeterministic remapped duals");
        require(scheduled.dispatch.size()==repeated.dispatch.size(),"Nondeterministic dispatch count");
        for (std::size_t i=0;i<scheduled.dispatch.size();++i) {
            const auto& a=scheduled.dispatch[i];
            const auto& b=repeated.dispatch[i];
            require(a.rows==b.rows && a.columns==b.columns && a.edges==b.edges &&
                a.effective==b.effective && a.executed==b.executed && a.reason==b.reason &&
                a.status==b.status,"Nondeterministic dispatch metadata");
        }
        std::vector<int> unchanged_offsets(costs.row_ptr.size()),unchanged_columns(costs.col_idx.size());
        std::vector<float> unchanged_values(costs.values.size());
        device.row_ptr.download(unchanged_offsets.data(),unchanged_offsets.size(),ctx.stream);
        device.col_idx.download(unchanged_columns.data(),unchanged_columns.size(),ctx.stream);
        device.values.download(unchanged_values.data(),unchanged_values.size(),ctx.stream);
        require(unchanged_offsets==costs.row_ptr && unchanged_columns==costs.col_idx &&
            unchanged_values==costs.values,"Scheduler modified the original CSR");
    }
}
void run(const GPUContext& ctx) {
    // Two row-bearing components with noncontiguous original IDs, plus an
    // isolated column. Negative costs exercise dual remapping as well.
    CSRMatrix costs;
    costs.rows=3; costs.cols=5;
    costs.row_ptr={0,2,3,5}; costs.col_idx={1,3,0,1,3}; costs.values={-2,1,3,0,-1};
    check(costs,ctx,true);
    // A single connected component larger than both warp and block widths.
    // Each diagonal is cheapest; the ring edges connect all 300 row/col pairs.
    CSRMatrix large;
    large.rows=300; large.cols=300; large.row_ptr.push_back(0);
    for (int r=0;r<300;++r) {
        const int other=(r+1)%300;
        if (other<r) { large.col_idx.push_back(other); large.values.push_back(1.0f); }
        large.col_idx.push_back(r); large.values.push_back(-1.0f);
        if (other>r) { large.col_idx.push_back(other); large.values.push_back(1.0f); }
        large.row_ptr.push_back(static_cast<int>(large.values.size()));
    }
    {
        auto d=upload(large,ctx);
        const auto c=gpu_find_assignment_components(d,ctx);
        require(c.components.size()==1 && c.components[0].rows.size()==300 &&
            c.components[0].columns.size()==300,"Large support is not one connected component");
    }
    check(large,ctx,true);

    // Deterministic random disconnected rectangular workloads. Components
    // interleave original IDs, so contiguous row partitions cannot pass.
    std::mt19937 random(20261019u);
    for (int trial=0;trial<8;++trial) {
        CSRMatrix sample;
        sample.rows=6; sample.cols=9; sample.row_ptr.push_back(0);
        for (int r=0;r<sample.rows;++r) {
            for (int c=0;c<sample.cols;++c) {
                if (r%3==c%3) {
                    sample.col_idx.push_back(c);
                    sample.values.push_back((static_cast<int>(random()%25)-12)/4.0f);
                }
            }
            sample.row_ptr.push_back(static_cast<int>(sample.values.size()));
        }
        check(sample,ctx,true);
    }

    // Connected tied optima: compare objectives/certificates with CPU, not
    // matching identity, since multiple optimal permutations are valid.
    CSRMatrix ties;
    ties.rows=3; ties.cols=4; ties.row_ptr={0,4,8,12};
    for (int r=0;r<3;++r) for (int c=0;c<4;++c) {
        ties.col_idx.push_back(c); ties.values.push_back(0.0f);
    }
    check(ties,ctx,true);
    // Connected Hall-deficient support passes row/column cardinality.
    costs.rows=3; costs.cols=3;
    costs.row_ptr={0,1,2,5}; costs.col_idx={0,0,0,1,2}; costs.values={0,0,0,1,2};
    check(costs,ctx,false);
    // Row-heavy component is rejected before solving.
    costs.rows=2; costs.cols=2;
    costs.row_ptr={0,1,2}; costs.col_idx={0,0}; costs.values={0,1};
    check(costs,ctx,false);
    costs.rows=0; costs.cols=3; costs.row_ptr={0}; costs.col_idx.clear(); costs.values.clear();
    check(costs,ctx,true);
    costs.cols=0;
    check(costs,ctx,true);
    costs.rows=1; costs.cols=1; costs.row_ptr={0,0};
    check(costs,ctx,false);
    auto device=upload(costs,ctx);
    GPUAssignmentSchedulerOptions tiny;
    tiny.component_workspace_budget_bytes=0;
    bool rejected=false;
    try { auto unused=gpu_schedule_sparse_assignment(device,ctx,tiny); }
    catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"Component workspace limit ignored");

    // Assignment workspace errors must propagate, not trigger a silent
    // execution fallback. Use nonempty feasible support so a solve is required.
    CSRMatrix one;
    one.rows=1; one.cols=1; one.row_ptr={0,1}; one.col_idx={0}; one.values={0};
    auto one_device=upload(one,ctx);
    tiny=GPUAssignmentSchedulerOptions{};
    tiny.assignment.workspace_budget_bytes=0;
    rejected=false;
    try { auto unused=gpu_schedule_sparse_assignment(one_device,ctx,tiny); }
    catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"Assignment workspace failure was swallowed by scheduler");
}
}
int main() {
    cudaStream_t stream=nullptr;
    try {
        int count=0;
        const auto status=cudaGetDeviceCount(&count);
        if (status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver ||
            (status==cudaSuccess && count==0)) return 77;
        CUDA_CHECK(status); CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaStreamCreate(&stream));
        GPUContext ctx; ctx.device_id=0; ctx.stream=stream;
        run(ctx);
        CUDA_CHECK(cudaStreamDestroy(stream)); stream=nullptr;
        std::cout << "Assignment scheduler checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
        std::cerr << error.what() << '\n'; return 1;
    }
}
