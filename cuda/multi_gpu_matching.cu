#include "multi_gpu_matching.hpp"
#include <future>
#include <numeric>

namespace {
std::size_t checked_add(std::size_t a,std::size_t b) {
    if (b>std::numeric_limits<std::size_t>::max()-a) throw std::overflow_error("Multi-GPU size overflow");
    return a+b;
}
std::size_t checked_mul(std::size_t a,std::size_t b) {
    if (b && a>std::numeric_limits<std::size_t>::max()/b) throw std::overflow_error("Multi-GPU size overflow");
    return a*b;
}
std::size_t admit(std::size_t payload,std::size_t scratch,const MultiGPUOptions& options) {
    if (payload>options.resident_budget_bytes) throw std::invalid_argument("Multi-GPU resident payload exceeds worker budget");
    std::size_t available=0,total=0;
    CUDA_CHECK(cudaMemGetInfo(&available,&total));
    if (checked_add(checked_add(payload,scratch),options.free_memory_reserve_bytes)>available)
        throw std::runtime_error("Multi-GPU worker cannot fit payload, workspace, and free-memory reserve");
    return available;
}
DeviceCSR upload(const CSRMatrix& h,const GPUContext& ctx) {
    DeviceCSR d; d.num_rows=h.rows; d.num_cols=h.cols; d.entry_count=h.values.size();
    d.row_ptr=DeviceBuffer<int>(h.row_ptr.size(),ctx.device_id);
    d.col_idx=DeviceBuffer<int>(h.col_idx.size(),ctx.device_id);
    d.values=DeviceBuffer<float>(h.values.size(),ctx.device_id);
    d.row_ptr.upload(h.row_ptr.data(),h.row_ptr.size(),ctx.stream);
    d.col_idx.upload(h.col_idx.data(),h.col_idx.size(),ctx.stream);
    d.values.upload(h.values.data(),h.values.size(),ctx.stream);
    return d;
}
std::size_t csr_bytes(const CSRMatrix& h) {
    return checked_add(checked_mul(h.row_ptr.size(),sizeof(int)),
        checked_mul(h.values.size(),sizeof(int)+sizeof(float)));
}
void validate_host(const CSRMatrix& h) {
    DeviceCSR::validate_dimensions(h.rows,h.cols,h.values.size());
    if (h.row_ptr.size()!=static_cast<std::size_t>(h.rows)+1 || h.col_idx.size()!=h.values.size() ||
        h.row_ptr.front()!=0 || h.row_ptr.back()!=static_cast<int>(h.values.size()))
        throw std::invalid_argument("Invalid multi-GPU CSR layout");
    for (int r=0;r<h.rows;++r) {
        const int begin=h.row_ptr[r],end=h.row_ptr[r+1];
        if (begin<0 || end<begin || static_cast<std::size_t>(end)>h.values.size())
            throw std::invalid_argument("Invalid multi-GPU CSR offsets");
        for (int p=begin;p<end;++p)
            if (h.col_idx[p]<0 || h.col_idx[p]>=h.cols || !std::isfinite(h.values[p]) ||
                (p>begin && h.col_idx[p-1]>=h.col_idx[p]))
                throw std::invalid_argument("Invalid multi-GPU CSR entry");
    }
}
CSRMatrix compact_host(const CSRMatrix& original,const AssignmentComponent& component) {
    CSRMatrix result;
    result.rows=static_cast<int>(component.rows.size());
    result.cols=static_cast<int>(component.columns.size());
    result.row_ptr.push_back(0);
    for (int r:component.rows) {
        for (int p=original.row_ptr[r];p<original.row_ptr[r+1];++p) {
            const auto found=std::lower_bound(component.columns.begin(),component.columns.end(),original.col_idx[p]);
            if (found==component.columns.end() || *found!=original.col_idx[p])
                throw std::runtime_error("Edge crosses multi-GPU component boundary");
            result.col_idx.push_back(static_cast<int>(found-component.columns.begin()));
            result.values.push_back(original.values[p]);
        }
        result.row_ptr.push_back(static_cast<int>(result.values.size()));
    }
    if (result.values.size()!=component.edges) throw std::runtime_error("Component edge count disagrees");
    return result;
}
}

MultiGPUCandidateResult multi_gpu_generate_candidates(
    const std::vector<NodeFeatures>& sources,const std::vector<NodeFeatures>& targets,
    int k,const std::vector<int>& devices,const MultiGPUOptions& options) {
    Timer wall("Multi-GPU candidates");
    validate_gpu_device_ids(devices);
    if ((options.selector!=GPUSelector::Parallel && options.selector!=GPUSelector::SerialHeap) ||
        options.topk.batch_candidates==0)
        throw std::invalid_argument("Invalid multi-GPU selector configuration");
    if (sources.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        targets.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        k<=0 || (!sources.empty() && static_cast<std::size_t>(k)>targets.size()))
        throw std::invalid_argument("Invalid multi-GPU dimensions or K");
    for (const auto* features:{&sources,&targets}) for (const auto& f:*features)
        for (int j=0;j<NUM_FEATURES;++j) if (!std::isfinite(f.as_array(j)))
            throw std::invalid_argument("Nonfinite multi-GPU feature");
    struct Part { MultiGPUCandidateWorker info; std::vector<std::vector<Candidate>> rows; };
    std::vector<std::future<Part>> futures;
    for (std::size_t worker=0;worker<devices.size();++worker) {
        const auto begin=sources.size()*worker/devices.size();
        const auto end=sources.size()*(worker+1)/devices.size();
        futures.push_back(std::async(std::launch::async,[&,worker,begin,end] {
            Timer timer("Multi-GPU worker");
            Part part; part.info.device=devices[worker];
            part.info.source_begin=begin; part.info.source_end=end;
            if (begin==end) { part.info.wall_ms=timer.elapsed_ms(); return part; }
            GPUContextOwner owner(devices[worker]); const auto& ctx=owner.get();
            part.info.target_replica_bytes=checked_mul(targets.size(),sizeof(NodeFeatures));
            const auto entries=checked_mul(end-begin,static_cast<std::size_t>(k));
            part.info.resident_payload_bytes=checked_add(
                checked_mul(checked_add(end-begin,targets.size()),sizeof(NodeFeatures)),
                checked_add(checked_mul(entries,sizeof(int)+sizeof(float)),
                    checked_mul(end-begin+1,sizeof(std::size_t))));
            part.info.free_bytes_at_admission=admit(part.info.resident_payload_bytes,
                checked_add(options.topk.scratch_budget_bytes,sizeof(int)),options);
            DeviceFeatureSet a,b;
            a.num_vertices=static_cast<int>(end-begin); b.num_vertices=static_cast<int>(targets.size());
            a.normalized=DeviceBuffer<NodeFeatures>(end-begin,ctx.device_id);
            b.normalized=DeviceBuffer<NodeFeatures>(targets.size(),ctx.device_id);
            a.normalized.upload(sources.data()+begin,end-begin,ctx.stream);
            b.normalized.upload(targets.data(),targets.size(),ctx.stream);
            auto selected=gpu_score_exhaustive_device_pools(a,b,k,ctx,options.selector,options.topk);
            part.info.executed=true;
            part.info.selection=selected.selection; part.info.similarity_ms=selected.similarity_ms;
            std::vector<int> ids(entries); std::vector<float> scores(entries);
            selected.candidates.target_ids.download(ids.data(),entries,ctx.stream);
            selected.candidates.similarities.download(scores.data(),entries,ctx.stream);
            part.info.candidate_download_bytes=checked_mul(entries,sizeof(int)+sizeof(float));
            part.rows.resize(end-begin);
            for (std::size_t s=0;s<end-begin;++s) {
                part.rows[s].reserve(k);
                for (int i=0;i<k;++i) part.rows[s].push_back({ids[s*k+i],scores[s*k+i]});
            }
            part.info.wall_ms=timer.elapsed_ms(); return part;
        }));
    }
    MultiGPUCandidateResult result; result.candidates.resize(sources.size());
    std::exception_ptr error;
    for (auto& future:futures) {
        try {
            auto part=future.get();
            for (std::size_t i=0;i<part.rows.size();++i)
                result.candidates[part.info.source_begin+i]=std::move(part.rows[i]);
            result.workers.push_back(part.info);
        } catch (...) { if (!error) error=std::current_exception(); }
    }
    if (error) std::rethrow_exception(error);
    result.wall_ms=wall.elapsed_ms(); return result;
}

MultiGPUAssignmentResult multi_gpu_solve_assignment(
    const CSRMatrix& costs,const std::vector<int>& devices,const MultiGPUOptions& options) {
    Timer wall("Multi-GPU assignment"); validate_gpu_device_ids(devices); validate_host(costs);
    const auto execution=options.scheduler.assignment.execution;
    if (execution!=GPUAssignmentExecution::Warp && execution!=GPUAssignmentExecution::Block &&
        execution!=GPUAssignmentExecution::Cooperative)
        throw std::invalid_argument("Invalid multi-GPU assignment execution");
    if (execution==GPUAssignmentExecution::Cooperative && options.scheduler.assignment.cooperative_blocks<2)
        throw std::invalid_argument("Cooperative assignment requires at least two blocks");
    MultiGPUAssignmentResult result;
    AssignmentComponents components;
    {
        GPUContextOwner owner(devices.front()); const auto& ctx=owner.get();
        admit(csr_bytes(costs),options.scheduler.component_workspace_budget_bytes,options);
        auto d=upload(costs,ctx); Timer discovery("Component discovery");
        components=gpu_find_assignment_components(d,ctx,options.scheduler.component_workspace_budget_bytes);
        result.discovery_ms=discovery.elapsed_ms();
        result.known_h2d_bytes=csr_bytes(costs); result.known_d2h_bytes=components.download_bytes;
    }
    // Stable largest-edge-first placement; entire components remain indivisible.
    std::vector<std::size_t> order(components.components.size()); std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](auto a,auto b) {
        return components.components[a].edges>components.components[b].edges;
    });
    std::vector<std::vector<std::size_t>> work(devices.size());
    std::vector<std::size_t> load(devices.size(),0);
    for (auto id:order) {
        const auto worker=static_cast<std::size_t>(std::min_element(load.begin(),load.end())-load.begin());
        work[worker].push_back(id);
        load[worker]=checked_add(load[worker],std::max<std::size_t>(1,components.components[id].edges));
    }
    struct Solved { std::size_t id; GPUAssignmentHostResult host; MultiGPUComponentDispatch info; };
    struct Worker { std::vector<Solved> solved; std::size_t h2d=0,d2h=0; };
    std::vector<std::future<Worker>> futures;
    for (std::size_t worker=0;worker<devices.size();++worker)
        futures.push_back(std::async(std::launch::async,[&,worker] {
            Worker output;
            GPUContextOwner owner(devices[worker]); const auto& ctx=owner.get();
            for (auto id:work[worker]) {
                Timer timer("Component worker"); const auto& c=components.components[id];
                Solved item; item.id=id; item.info.device=devices[worker];
                auto& info=item.info.component;
                info.component=static_cast<int>(id); info.rows=static_cast<int>(c.rows.size());
                info.columns=static_cast<int>(c.columns.size()); info.edges=c.edges;
                info.requested=info.effective=options.scheduler.assignment.execution;
                if (c.rows.empty()) { info.status=GPUAssignmentStatus::Success; info.reason="isolated_columns_no_assignment"; }
                else if (!c.passes_cardinality_check) {
                    info.status=GPUAssignmentStatus::Infeasible; info.reason="component_rows_exceed_columns";
                } else {
                    Timer compact("Host compaction"); auto local=compact_host(costs,c);
                    info.compaction_ms=compact.elapsed_ms();
                    const auto local_payload=checked_add(csr_bytes(local),checked_mul(
                        checked_add(c.rows.size(),c.columns.size()),sizeof(int)+sizeof(double)));
                    admit(local_payload,options.scheduler.assignment.workspace_budget_bytes,options);
                    auto device=upload(local,ctx);
                    auto parameters=options.scheduler.assignment;
                    if (parameters.execution==GPUAssignmentExecution::Cooperative) {
                        int supported=0;
                        CUDA_CHECK(cudaDeviceGetAttribute(&supported,cudaDevAttrCooperativeLaunch,ctx.device_id));
                        if (!supported && options.scheduler.allow_unsupported_cooperative_fallback) {
                            parameters.execution=info.effective=GPUAssignmentExecution::Block;
                            info.reason="cooperative_launch_unsupported";
                        }
                    }
                    auto solved=gpu_solve_sparse_assignment(device,ctx,parameters);
                    item.host=gpu_download_assignment(solved,local,ctx);
                    if (solved.status==GPUAssignmentStatus::Success && !item.host.certificate.optimal)
                        throw std::runtime_error("Multi-GPU component certificate failed");
                    if (solved.status==GPUAssignmentStatus::NumericalFailure)
                        throw std::runtime_error("Multi-GPU component numerical failure");
                    info.executed=true; info.status=solved.status; info.blocks=solved.execution_blocks;
                    info.assignment_kernel_ms=solved.kernel_ms; info.workspace_peak_bytes=solved.workspace_bytes;
                    info.compact_payload_bytes=csr_bytes(local);
                    output.h2d=checked_add(output.h2d,csr_bytes(local));
                    output.d2h=checked_add(output.d2h,item.host.download_bytes);
                }
                info.total_ms=timer.elapsed_ms(); output.solved.push_back(std::move(item));
            }
            return output;
        }));
    std::vector<int> rm(costs.rows,-1),cm(costs.cols,-1);
    std::vector<double> rd(costs.rows,0),cd(costs.cols,0);
    GPUAssignmentStatus status=GPUAssignmentStatus::Success;
    double kernel_ms=0; unsigned long long steps=0; std::size_t workspace=0;
    result.dispatch.resize(components.components.size());
    std::exception_ptr error;
    for (auto& future:futures) {
        try {
            auto worker=future.get();
            result.known_h2d_bytes=checked_add(result.known_h2d_bytes,worker.h2d);
            result.known_d2h_bytes=checked_add(result.known_d2h_bytes,worker.d2h);
            for (auto& item:worker.solved) {
                const auto& c=components.components[item.id];
                result.dispatch[item.id]=item.info;
                if (item.info.component.status!=GPUAssignmentStatus::Success) status=GPUAssignmentStatus::Infeasible;
                if (!item.info.component.executed) continue;
                const auto& host=item.host;
                kernel_ms+=host.kernel_ms; steps+=host.search_steps; workspace=std::max(workspace,host.workspace_bytes);
                for (std::size_t r=0;r<c.rows.size();++r) {
                    const int local=host.assignment.row_to_col[r];
                    rm[c.rows[r]]=local<0?-1:c.columns[local]; rd[c.rows[r]]=host.row_dual[r];
                }
                for (std::size_t column=0;column<c.columns.size();++column) {
                    const int local=host.assignment.col_to_row[column];
                    cm[c.columns[column]]=local<0?-1:c.rows[local]; cd[c.columns[column]]=host.col_dual[column];
                }
            }
        } catch (...) { if (!error) error=std::current_exception(); }
    }
    if (error) std::rethrow_exception(error);
    {
        // Reuse the existing independent certificate implementation on ORIGINAL
        // support. This explicit host roundtrip avoids a second certificate.
        GPUContextOwner owner(devices.front()); const auto& ctx=owner.get();
        const auto payload=checked_mul(checked_add(rm.size(),cm.size()),sizeof(int)+sizeof(double));
        admit(payload,0,options);
        GPUAssignmentResult assembled; assembled.num_rows=costs.rows; assembled.num_cols=costs.cols;
        assembled.status=status; assembled.kernel_ms=kernel_ms; assembled.search_steps=steps; assembled.workspace_bytes=workspace;
        assembled.row_to_col=DeviceBuffer<int>(rm.size(),ctx.device_id);
        assembled.col_to_row=DeviceBuffer<int>(cm.size(),ctx.device_id);
        assembled.row_dual=DeviceBuffer<double>(rd.size(),ctx.device_id);
        assembled.col_dual=DeviceBuffer<double>(cd.size(),ctx.device_id);
        assembled.row_to_col.upload(rm.data(),rm.size(),ctx.stream);
        assembled.col_to_row.upload(cm.data(),cm.size(),ctx.stream);
        assembled.row_dual.upload(rd.data(),rd.size(),ctx.stream);
        assembled.col_dual.upload(cd.data(),cd.size(),ctx.stream);
        result.assignment=gpu_download_assignment(assembled,costs,ctx);
        result.known_h2d_bytes=checked_add(result.known_h2d_bytes,payload);
        result.known_d2h_bytes=checked_add(result.known_d2h_bytes,result.assignment.download_bytes);
        if (status==GPUAssignmentStatus::Success && !result.assignment.certificate.optimal)
            throw std::runtime_error("Multi-GPU global original-CSR certificate failed");
    }
    result.wall_ms=wall.elapsed_ms(); return result;
}
