#include "multi_gpu_matching.hpp"
#include <iostream>
#include <set>

namespace {
void require(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
void compare_candidates(const MultiGPUCandidateResult& a,const MultiGPUCandidateResult& b) {
    require(a.candidates.size()==b.candidates.size(),"Candidate row count mismatch");
    for (std::size_t r=0;r<a.candidates.size();++r) {
        require(a.candidates[r].size()==b.candidates[r].size(),"Candidate count mismatch");
        for (std::size_t i=0;i<a.candidates[r].size();++i) {
            require(a.candidates[r][i].target==b.candidates[r][i].target,"Multi-GPU target order differs");
            require(std::abs(a.candidates[r][i].similarity-b.candidates[r][i].similarity)<1e-6,
                "Multi-GPU score differs");
        }
    }
}
void assignment_case(const CSRMatrix& costs,bool feasible,bool multiple) {
    const auto result=multi_gpu_solve_assignment(costs,{0,1});
    require(result.assignment.assignment.feasible==feasible,"Multi-GPU feasibility differs");
    std::set<int> devices;
    for (const auto& d:result.dispatch) if (d.component.executed) devices.insert(d.device);
    if (multiple) require(devices.size()==2,"Independent components were not assigned to both GPUs");
    if (feasible) {
        require(result.assignment.certificate.optimal,"Original-CSR global certificate failed");
        if (costs.rows) {
            const auto reference=solve_lapjv(costs);
            require(reference.feasible && std::abs(reference.total_cost-result.assignment.assignment.total_cost)<1e-8,
                "Multi-GPU objective differs from CPU");
        }
        const auto& mapping=result.assignment.assignment.row_to_col;
        require(std::set<int>(mapping.begin(),mapping.end()).size()==mapping.size(),
            "Multi-GPU assignment reused a global target");
    } else require(result.assignment.status==GPUAssignmentStatus::Infeasible,"Missing infeasible status");
    const auto again=multi_gpu_solve_assignment(costs,{0,1});
    require(result.assignment.assignment.row_to_col==again.assignment.assignment.row_to_col,
        "Multi-GPU remapping is nondeterministic");
}
void run() {
    // Exactly tied features isolate target-ID ordering from score differences.
    const NodeFeatures f{1,2,3,4,5,6};
    std::vector<NodeFeatures> source(7,f),target(513,f);
    for (int k:{1,4,256,512}) {
        const auto one=multi_gpu_generate_candidates(source,target,k,{0});
        const auto two=multi_gpu_generate_candidates(source,target,k,{0,1});
        compare_candidates(one,two);
        require(two.workers.size()==2 && two.workers[0].executed && two.workers[1].executed,
            "Source generation did not execute on both GPUs");
        require(two.workers[0].source_end==two.workers[1].source_begin &&
            two.workers[0].source_begin==0 && two.workers[1].source_end==source.size(),
            "Source intervals overlap or omit rows");
        for (const auto& row:two.candidates) for (int i=0;i<k;++i)
            require(row[i].target==i,"Tie target ordering changed");
        if (k==512) for (const auto& w:two.workers)
            require(w.selection.fallback==GPUFallbackReason::UnsupportedK,"K512 fallback not recorded");
    }
    compare_candidates(multi_gpu_generate_candidates(source,target,4,{0,1}),
        multi_gpu_generate_candidates(source,target,4,{1,0}));
    auto varied_source=source;
    std::vector<NodeFeatures> varied_target(31);
    for (std::size_t i=0;i<varied_source.size();++i)
        varied_source[i]=NodeFeatures{static_cast<float>(i),0.25f,static_cast<float>(i%3),
            0.5f,static_cast<float>(i%2),static_cast<float>(i*i)};
    for (std::size_t i=0;i<varied_target.size();++i)
        varied_target[i]=NodeFeatures{static_cast<float>(i),static_cast<float>(i%7)/8,
            static_cast<float>(i%5),0.5f,static_cast<float>(i%3),static_cast<float>(i*i)};
    normalize_features_joint_zscore(varied_source,varied_target);
    compare_candidates(multi_gpu_generate_candidates(varied_source,varied_target,8,{0}),
        multi_gpu_generate_candidates(varied_source,varied_target,8,{0,1}));
    const auto tiny=multi_gpu_generate_candidates(std::vector<NodeFeatures>{f},target,1,{0,1});
    require(tiny.candidates.size()==1,"Fewer sources than devices failed");
    const auto empty=multi_gpu_generate_candidates({},target,1,{0,1});
    require(empty.candidates.empty(),"Empty source input failed");
    bool rejected=false;
    try { auto unused=multi_gpu_generate_candidates(source,target,1,{0,0}); }
    catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"Duplicate GPU IDs accepted");
    rejected=false;
    try { auto unused=multi_gpu_generate_candidates(source,target,1,{-1,0}); }
    catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"Negative GPU ID accepted");
    rejected=false;
    auto invalid_source=source; invalid_source[0].degree=std::numeric_limits<float>::quiet_NaN();
    try { auto unused=multi_gpu_generate_candidates(invalid_source,target,1,{0,1}); }
    catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"Nonfinite feature accepted");
    MultiGPUOptions budget; budget.resident_budget_bytes=1;
    rejected=false;
    try { auto unused=multi_gpu_generate_candidates(source,target,1,{0,1},budget); }
    catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"Worker memory budget ignored");
    // Worker exceptions have joined/drained; following calls must still work.
    compare_candidates(multi_gpu_generate_candidates(source,target,4,{0}),
        multi_gpu_generate_candidates(source,target,4,{0,1}));
    CSRMatrix costs;
    costs.rows=4; costs.cols=5; costs.row_ptr={0,2,4,6,8};
    costs.col_idx={0,2,1,3,0,2,1,3}; costs.values={-2,1,-3,2,0,-1,0,-2};
    assignment_case(costs,true,true);
    rejected=false;
    try { auto unused=multi_gpu_solve_assignment(costs,{0,1},budget); }
    catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"Assignment memory budget ignored");
    // A connected ring must stay on one GPU, despite two available devices.
    costs.cols=4; costs.col_idx={0,1,1,2,2,3,0,3};
    const auto connected=multi_gpu_solve_assignment(costs,{0,1});
    std::size_t executed=0;
    for (const auto& d:connected.dispatch) executed+=d.component.executed;
    require(executed==1 && connected.assignment.certificate.optimal,"Connected component was split");
    costs.rows=3; costs.cols=3; costs.row_ptr={0,1,2,5};
    costs.col_idx={0,0,0,1,2}; costs.values={0,0,0,1,2};
    assignment_case(costs,false,false);
    costs.rows=1; costs.cols=1; costs.row_ptr={0,0}; costs.col_idx.clear(); costs.values.clear();
    assignment_case(costs,false,false);
    costs.rows=0; costs.cols=2; costs.row_ptr={0};
    assignment_case(costs,true,false);
}
}
int main() {
    try {
        int count=0; const auto status=cudaGetDeviceCount(&count);
        if (status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver ||
            (status==cudaSuccess && count<2)) {
            std::cout << "SKIP: two visible CUDA GPUs required\n"; return 77;
        }
        CUDA_CHECK(status); run();
        std::cout << "Multi-GPU correctness checks passed\n"; return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
