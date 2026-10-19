// GPU_SINKHORN_CLI_V1
#include "common.hpp"
#include "graph.hpp"
#include "features.hpp"
#include "cosine_similarity.hpp"
#include "candidate_generation.hpp"
#include "bipartite_matching.hpp"
#include "candidate_repair.hpp"
#include "sparse_sinkhorn.hpp"
#include "csr_matrix.hpp"
#include "lapjv.hpp"
#include "gpu_matching.hpp"
#include "evaluation.hpp"
#include "device_memory_sampler.hpp"
#include "gpu_assignment.hpp"
#include "assignment_scheduler.hpp"
#include "multi_gpu_matching.hpp"
#include "gpu_sinkhorn.hpp"
#include <optional>

struct CandidateOptions {
    std::vector<int> devices{0};
    bool explicit_devices = false;
    bool explicit_multi_limits = false;
    MultiGPUOptions multi;
    std::string backend = "exhaustive";
    StructuralIndexConfig index;
    bool explicit_index_options = false;
    GPUSelector selector = GPUSelector::SerialHeap;
    std::string selection_policy = "serial";
    GPUTopKOptions topk;
    bool explicit_topk_options = false;
    bool explicit_topk_limits = false;
    bool resident_pipeline = false;
    std::string assignment_backend = "cpu";
    bool component_assignment = false;
    GPUAssignmentSchedulerOptions scheduler;
    std::string sinkhorn_backend = "cpu";
    std::string sinkhorn_output;
    bool repair = false;
    bool explicit_repair_limits = false;
    CandidateRepairOptions repair_options;
};

static const char* selector_name(GPUSelector selector) {
    return selector == GPUSelector::Parallel ? "parallel" : "serial_heap";
}
static const char* fallback_name(GPUFallbackReason reason) {
    switch (reason) {
    case GPUFallbackReason::None: return "none";
    case GPUFallbackReason::UnsupportedK: return "unsupported_k";
    case GPUFallbackReason::DeviceAllocationFailure: return "device_allocation_failure";
    case GPUFallbackReason::ScratchBudgetTooSmall: return "scratch_budget_too_small";
    }
    return "unknown";
}

static std::string measured_time_json(double value, bool available) {
    if (!available) return "null";
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    return output.str();
}

static int parse_index_int(const std::string& value) {
    size_t used = 0;
    int result = std::stoi(value, &used);
    if (used != value.size()) throw std::invalid_argument("Invalid index integer: " + value);
    return result;
}

static double parse_index_double(const std::string& value) {
    size_t used = 0;
    double result = std::stod(value, &used);
    if (used != value.size() || !std::isfinite(result))
        throw std::invalid_argument("Invalid index number: " + value);
    return result;
}

// The six-node asymmetric correctness test graph
static void build_toy_graphs(Graph& source, Graph& target,
                              std::vector<int>& ground_truth,
                              std::vector<std::string>& source_labels,
                              std::vector<int>& target_ids) {
    // Source graph vertices: A=0, B=1, C=2, D=3, E=4, F=5
    source_labels = {"A", "B", "C", "D", "E", "F"};
    std::vector<std::pair<int,int>> source_edges = {
        {0, 1},  // A-B
        {0, 5},  // A-F
        {1, 2},  // B-C
        {1, 3},  // B-D
        {1, 4},  // B-E
        {2, 3},  // C-D
        {2, 5},  // C-F
        {3, 4},  // D-E
    };
    source = build_graph(6, source_edges);

    // Ground truth permutation:
    // A(0)->14, B(1)->11, C(2)->16, D(3)->12, E(4)->15, F(5)->13
    // We remap target IDs to 0-based for internal graph:
    // 11->0, 12->1, 13->2, 14->3, 15->4, 16->5
    // So: A(0)->3, B(1)->0, C(2)->5, D(3)->1, E(4)->4, F(5)->2
    target_ids = {11, 12, 13, 14, 15, 16};

    ground_truth = {3, 0, 5, 1, 4, 2};
    // A->14 = idx 3, B->11 = idx 0, C->16 = idx 5, D->12 = idx 1, E->15 = idx 4, F->13 = idx 2

    // Target graph edges (using 0-based internal IDs):
    // Original edges: (14,11)(14,13)(11,16)(11,12)(11,15)(16,12)(16,13)(12,15)
    // Mapped: (3,0)(3,2)(0,5)(0,1)(0,4)(5,1)(5,2)(1,4)
    std::vector<std::pair<int,int>> target_edges = {
        {3, 0},  // 14-11
        {3, 2},  // 14-13
        {0, 5},  // 11-16
        {0, 1},  // 11-12
        {0, 4},  // 11-15
        {5, 1},  // 16-12
        {5, 2},  // 16-13
        {1, 4},  // 12-15
    };
    target = build_graph(6, target_edges);
}

static void run_toy_test(bool use_gpu) {
    std::cout << "SCGM CUDA/CSR/LAPJV CORRECTNESS TEST" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << std::endl;

    Graph source, target;
    std::vector<int> ground_truth;
    std::vector<std::string> source_labels;
    std::vector<int> target_ids;
    build_toy_graphs(source, target, ground_truth, source_labels, target_ids);

    int N = source.num_vertices;
    int M = target.num_vertices;
    int K = M;  // For 6-node test, use K=M to ensure recall

    std::cout << "Source vertices: " << N << std::endl;
    std::cout << "Target vertices: " << M << std::endl;
    std::cout << "K: " << K << std::endl;
    std::cout << std::endl;

    std::cout << "Feature implementation:" << std::endl;
    std::cout << "    Six paper-defined structural descriptors" << std::endl;
    std::cout << std::endl;

    std::cout << "Similarity:" << std::endl;
    std::cout << "    Cosine similarity" << std::endl;
    std::cout << std::endl;

    std::cout << "Cost:" << std::endl;
    std::cout << "    1 - cosine similarity" << std::endl;
    std::cout << std::endl;

    // Compute features (CPU)
    Timer t_feat("Feature extraction");
    auto feats_src = compute_features(source);
    auto feats_tgt = compute_features(target);
    normalize_features_zscore(feats_src);
    normalize_features_zscore(feats_tgt);
    t_feat.report();

    // CPU candidate generation
    Timer t_cand("Candidate generation");
    auto candidates_cpu = generate_top_k_candidates(feats_src, feats_tgt, K);
    t_cand.report();

    // GPU path
    std::vector<std::vector<Candidate>> candidates_gpu;
    std::vector<NodeFeatures> gpu_feats_src, gpu_feats_tgt;
    bool gpu_available = false;

    if (use_gpu) {
        try {
            auto gpus = initialize_gpus({0});
            if (!gpus.empty()) {
                gpu_available = true;
                Timer t_gpu("GPU pipeline");
                auto gpu_result = gpu_compute_candidates(source, target, K, gpus[0]);
                candidates_gpu = gpu_result.candidates;
                gpu_feats_src = gpu_result.feats_source;
                gpu_feats_tgt = gpu_result.feats_target;
                t_gpu.report();
                cleanup_gpus(gpus);
            }
        } catch (const std::exception& e) {
            std::cerr << "GPU init failed: " << e.what() << std::endl;
            gpu_available = false;
        }
    }

    // Use CPU candidates for assignment
    auto& candidates = candidates_cpu;

    // Print candidate pairs
    std::cout << std::endl;
    std::cout << "Candidate pairs:" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << "    " << source_labels[i] << " -> [";
        for (int k = 0; k < (int)candidates[i].size(); k++) {
            if (k > 0) std::cout << ", ";
            std::cout << target_ids[candidates[i][k].target]
                      << "(sim=" << std::fixed << std::setprecision(4)
                      << candidates[i][k].similarity << ")";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;

    // Build CSR
    Timer t_csr("CSR construction");
    CSRMatrix csr = build_sparse_cost_matrix(candidates, N, M);
    t_csr.report();

    print_csr_info(csr);
    std::cout << std::endl;

    // CSR validation
    bool csr_valid = validate_csr(csr, K);

    // Solve LAPJV
    Timer t_lapjv("LAPJV");
    AssignmentResult assignment = solve_lapjv(csr);
    t_lapjv.report();

    // Print permutation convention
    std::cout << std::endl;
    std::cout << "Permutation convention:" << std::endl;
    std::cout << "    source -> target" << std::endl;
    std::cout << std::endl;

    // Print ground truth
    std::cout << "Ground truth:" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << "    " << source_labels[i] << " -> "
                  << target_ids[ground_truth[i]] << std::endl;
    }
    std::cout << std::endl;

    // Print predicted
    std::cout << "Predicted:" << std::endl;
    for (int i = 0; i < N; i++) {
        int pred = assignment.row_to_col[i];
        std::string tgt_str = (pred >= 0 && pred < (int)target_ids.size())
                                  ? std::to_string(target_ids[pred])
                                  : "UNASSIGNED";
        std::cout << "    " << source_labels[i] << " -> " << tgt_str << std::endl;
    }
    std::cout << std::endl;

    // Evaluation
    double recall = candidate_recall(candidates, ground_truth);
    double accuracy = mapping_accuracy(assignment.row_to_col, ground_truth);
    bool one_to_one = is_one_to_one(assignment.row_to_col, M);
    double edge_pres = edge_preservation(source, target, assignment.row_to_col);
    double rev_edge_pres = reverse_edge_preservation(source, target, assignment.row_to_col);

    std::cout << "Candidate Recall@K: " << std::fixed << std::setprecision(6)
              << recall << std::endl;
    std::cout << "Mapping Accuracy:   " << accuracy << std::endl;
    std::cout << "One-to-One:         " << (one_to_one ? "TRUE" : "FALSE") << std::endl;
    std::cout << "Edge Preservation:  " << edge_pres << std::endl;
    std::cout << "Reverse Edge Pres:  " << rev_edge_pres << std::endl;
    std::cout << std::endl;

    // Exhaustive 720-permutation oracle
    std::cout << "LAPJV cost: " << std::fixed << std::setprecision(6)
              << assignment.total_cost << std::endl;

    // Brute force: try all 720 permutations
    std::vector<int> perm = {0, 1, 2, 3, 4, 5};
    double brute_force_best = std::numeric_limits<double>::max();
    std::vector<int> brute_force_perm;
    int perm_count = 0;

    do {
        perm_count++;
        double cost = 0.0;
        bool valid = true;
        for (int i = 0; i < N; i++) {
            float c = get_value(csr, i, perm[i]);
            if (c >= SCGM_INF * 0.5f) {
                valid = false;
                break;
            }
            cost += (double)c;
        }
        if (valid && cost < brute_force_best) {
            brute_force_best = cost;
            brute_force_perm = perm;
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    std::cout << "Brute-force cost: " << brute_force_best << std::endl;
    std::cout << "Permutations evaluated: " << perm_count << std::endl;
    std::cout << std::endl;

    // Comparison
    bool lapjv_vs_brute = std::abs(assignment.total_cost - brute_force_best) < 1e-4;
    bool cpu_gpu_pass = true;
    if (gpu_available) {
        // Compare CPU vs GPU features
        for (int i = 0; i < N; i++) {
            for (int f = 0; f < NUM_FEATURES; f++) {
                float cpu_val = feats_src[i].as_array(f);
                float gpu_val = gpu_feats_src[i].as_array(f);
                if (std::abs(cpu_val - gpu_val) > 1e-3) {
                    std::cout << "CPU/GPU feature mismatch: node " << i
                              << " feature " << f << " CPU=" << cpu_val
                              << " GPU=" << gpu_val << std::endl;
                    cpu_gpu_pass = false;
                }
            }
        }
        // Compare Top-K candidate sets
        for (int i = 0; i < N; i++) {
            std::set<int> cpu_set, gpu_set;
            for (auto& c : candidates_cpu[i]) cpu_set.insert(c.target);
            for (auto& c : candidates_gpu[i]) gpu_set.insert(c.target);
            if (cpu_set != gpu_set) {
                std::cout << "CPU/GPU candidate set mismatch for node " << i << std::endl;
                cpu_gpu_pass = false;
            }
        }
    }

    // Permutation validation
    bool perm_valid = validate_permutation(assignment.row_to_col, N);
    if (perm_valid) {
        auto inv = inverse_permutation(assignment.row_to_col);
        // Verify double inverse
        for (int i = 0; i < N; i++) {
            int j = assignment.row_to_col[i];
            if (inv[j] != i) {
                perm_valid = false;
                break;
            }
        }
    }

    std::cout << "LAPJV vs exhaustive oracle:" << std::endl;
    std::cout << "    " << (lapjv_vs_brute ? "PASS" : "FAIL") << std::endl;
    std::cout << std::endl;

    std::cout << "CPU vs GPU:" << std::endl;
    if (!gpu_available) {
        std::cout << "    SKIPPED (no GPU)" << std::endl;
    } else {
        std::cout << "    " << (cpu_gpu_pass ? "PASS" : "FAIL") << std::endl;
    }
    std::cout << std::endl;

    std::cout << "CSR validation:" << std::endl;
    std::cout << "    " << (csr_valid ? "PASS" : "FAIL") << std::endl;
    std::cout << std::endl;

    std::cout << "Permutation validation:" << std::endl;
    std::cout << "    " << (perm_valid ? "PASS" : "FAIL") << std::endl;
    std::cout << std::endl;

    // Memory reporting
    std::cout << "Memory:" << std::endl;
    std::cout << "    CSR NNZ: " << csr.nnz() << std::endl;
    std::cout << "    CSR memory: " << csr_memory_bytes(csr) << " bytes" << std::endl;
    std::cout << "    Feature memory: " << N * NUM_FEATURES * sizeof(float)
              << " bytes (source)" << std::endl;
    std::cout << "    GPU feature memory: " << gpu_feature_memory_bytes(N)
              << " bytes per graph" << std::endl;
    std::cout << "    GPU candidate memory: " << gpu_candidate_memory_bytes(N, K)
              << " bytes" << std::endl;
    std::cout << std::endl;

    bool overall = assignment.feasible && lapjv_vs_brute && csr_valid && perm_valid;
    if (gpu_available) overall = overall && cpu_gpu_pass;

    std::cout << "OVERALL CORRECTNESS:" << std::endl;
    std::cout << "    " << (overall ? "PASS" : "FAIL") << std::endl;
    std::cout << std::endl;

    std::cout << "WARNING:" << std::endl;
    std::cout << "Toy timing is for correctness/debugging only." << std::endl;
    std::cout << "It is not a performance benchmark." << std::endl;
}

static int run_from_files(const std::string& source_file,
                            const std::string& target_file,
                            int K, bool use_gpu, bool verify, bool verbose,
                            const std::string& output_file,
                            const std::string& candidates_file,
                            const std::string& candidate_input_file,
                            const std::string& matching_output_file,
                            bool joint_normalize,
                            unsigned int feature_mask,
                            const std::vector<float>& feature_weights,
                            float neighbor_profile_weight,
                            const std::string& timings_file,
                            const CandidateOptions& options) {
    Graph source = load_graph(source_file);
    Graph target = load_graph(target_file);

    validate_graph(source);
    validate_graph(target);

    int N = source.num_vertices;
    int M = target.num_vertices;

    const bool requested_gpu = use_gpu;
    const bool multi_gpu = options.devices.size() > 1;
    const bool indexed = options.backend == "structural-index";
    if (options.sinkhorn_backend == "gpu" && N != M) {
        throw std::invalid_argument("GPU Sinkhorn requires a square matrix");
    }
    if (indexed && (K <= 0 || K > M))
        throw std::invalid_argument("indexed generation requires 1 <= K <= target count");
    if (K <= 0) K = 20;
    if (K > M) K = M;
    if (options.repair && (K <= 0 || options.repair_options.k_limit < K ||
                          options.repair_options.k_limit > M)) {
        throw std::invalid_argument("Repair requires 1 <= K <= repair-k-limit <= target count");
    }
    if (options.repair) {
        const auto rows = static_cast<std::size_t>(N);
        const auto width = static_cast<std::size_t>(K);
        const auto entry_limit = std::min<std::size_t>(
            options.repair_options.max_candidate_edges,
            static_cast<std::size_t>(std::numeric_limits<int>::max()));
        if (rows > entry_limit / width ||
            rows > (options.repair_options.max_payload_bytes / sizeof(Candidate)) / width)
            throw std::invalid_argument("Initial candidates exceed repair limits");
    }

    std::cout << "Source: " << source_file << " (" << N << " vertices)" << std::endl;
    std::cout << "Target: " << target_file << " (" << M << " vertices)" << std::endl;
    std::cout << "K: " << K << std::endl;
    std::cout << std::endl;

    std::vector<std::vector<Candidate>> candidates;
    std::vector<NodeFeatures> feats_src, feats_tgt;
    StructuralIndex target_index;
    std::vector<NodeFeatures> raw_sources;
    std::vector<StructuralRetrievalStats> index_stats;
    double index_build_ms = 0.0;
    double index_retrieval_ms = 0.0;
    double feature_ms = 0.0;
    double candidate_ms = 0.0;
    double similarity_ms = 0.0;
    double topk_selection_ms = 0.0;
    double csr_ms = 0.0;
    double lapjv_ms = 0.0;
    double transfer_ms = 0.0;
    size_t peak_device_application_allocated_bytes = 0;
    bool device_allocation_tracking_complete = true;
    GPUSelectionInfo gpu_selection;
    CSRMatrix downloaded_gpu_csr;
    bool gpu_costs_ready = false;
    bool resident_pipeline_used = false;
    GPUResidentDiagnostics resident_evidence;
    GPUAssignmentHostResult gpu_assignment_result;
    bool gpu_assignment_ready = false;
    std::optional<MultiGPUCandidateResult> multi_candidates;
    std::optional<MultiGPUAssignmentResult> multi_assignment;
    std::optional<GPUAssignmentScheduleResult> assignment_schedule;
    constexpr double sinkhorn_epsilon = 0.1;
    constexpr int sinkhorn_max_iterations = 500;
    constexpr double sinkhorn_tolerance = 1e-3;
    SparseSinkhornResult sinkhorn_result;
    GPUSinkhornDiagnostics gpu_sinkhorn_diagnostics;
    std::optional<CandidateSupportStats> cached_support;
    std::optional<CandidateRepairResult> repair_result;
    double repair_ms = 0.0;
    bool gpu_sinkhorn_executed = false;
    bool cpu_sinkhorn_executed = false;
    bool sinkhorn_confidence_exported = false;
    std::size_t sinkhorn_download_bytes = 0;
    double sinkhorn_ms = 0.0;
    std::string sinkhorn_status = "not_run";
    DeviceMemorySampler device_memory_sampler(use_gpu && !multi_gpu ? options.devices.front() : -1);

    if (use_gpu && neighbor_profile_weight > 0.0f) {
        throw std::invalid_argument(
            "--neighbor-profile-weight is currently supported only with --cpu");
    }

    if (!candidate_input_file.empty()) {
        std::ifstream input(candidate_input_file);
        if (!input) throw std::runtime_error("Cannot open candidate input: " + candidate_input_file);
        std::string line;
        if (!std::getline(input, line) || line.find("source") == std::string::npos ||
            line.find("target") == std::string::npos || line.find("similarity") == std::string::npos) {
            throw std::runtime_error("Candidate input must have source, target, and similarity columns");
        }
        candidates.resize(N);
        int line_number = 1;
        while (std::getline(input, line)) {
            ++line_number;
            std::stringstream fields(line);
            std::string source_text, second, third, fourth;
            std::getline(fields, source_text, ',');
            std::getline(fields, second, ',');
            std::getline(fields, third, ',');
            std::getline(fields, fourth, ',');
            try {
                const int source_vertex = std::stoi(source_text);
                const int target_vertex = fourth.empty() ? std::stoi(second) : std::stoi(third);
                const float similarity = std::stof(fourth.empty() ? third : fourth);
                if (source_vertex < 0 || source_vertex >= N || target_vertex < 0 || target_vertex >= M ||
                    !std::isfinite(similarity)) throw std::runtime_error("candidate endpoint or score out of range");
                candidates[source_vertex].push_back({target_vertex, similarity});
            } catch (const std::exception& error) {
                throw std::runtime_error("Invalid candidate input line " + std::to_string(line_number) + ": " + error.what());
            }
        }
        std::cout << "Loaded fixed candidate support and costs from: " << candidate_input_file << std::endl;
    } else if (use_gpu && multi_gpu) {
        // First multi-GPU version uses one global CPU feature/normalization
        // reference, then distributes scoring. Never normalize source shards.
        Timer features("Multi-GPU global CPU features and normalization");
        feats_src=compute_features(source); feats_tgt=compute_features(target);
        if (joint_normalize) normalize_features_joint_zscore(feats_src,feats_tgt);
        else { normalize_features_zscore(feats_src); normalize_features_zscore(feats_tgt); }
        feature_ms=features.elapsed_ms();
        auto parameters=options.multi;
        parameters.selector=options.selector; parameters.topk=options.topk;
        parameters.scheduler=options.scheduler;
        multi_candidates=multi_gpu_generate_candidates(feats_src,feats_tgt,K,options.devices,parameters);
        candidates=std::move(multi_candidates->candidates);
        candidate_ms=multi_candidates->wall_ms;
        // Kernel totals across GPUs are cumulative work, not elapsed wall time.
        for (const auto& worker:multi_candidates->workers) {
            similarity_ms+=worker.similarity_ms;
            topk_selection_ms+=worker.selection.selection_ms;
        }
        device_allocation_tracking_complete=false;
    } else if (use_gpu) {
        if (options.explicit_devices) validate_gpu_device_ids(options.devices);
        auto gpus = initialize_gpus(options.devices);
        if (!gpus.empty()) {
            try {
                if (options.resident_pipeline) {
                    // Destroy the device owner before cleanup_gpus() below.
                    auto device = gpu_compute_device_candidates(
                        source, target, K, gpus[0],
                        joint_normalize, options.selector, options.topk,
                        indexed ? &options.index : nullptr);

                    // CPU support analysis/export needs candidates; assignment
                    // and Sinkhorn need costs. Neither needs feature arrays.
                    candidates = gpu_download_candidates(device, gpus[0]);
                    downloaded_gpu_csr = gpu_download_costs(device, gpus[0]);
                    gpu_costs_ready = true;

                    if (options.assignment_backend == "gpu") {
                        Timer assignment_timer("GPU assignment and certificate");
                        {
                            // Keep the CSR owner and stream alive through solving,
                            // explicit downloads, and primal/dual certification.
                            GPUAssignmentResult solved;
                            if (options.component_assignment) {
                                assignment_schedule = gpu_schedule_sparse_assignment(
                                    device.costs, gpus[0], options.scheduler);
                                solved = std::move(assignment_schedule->assignment);
                            } else {
                                solved = gpu_solve_sparse_assignment(device.costs, gpus[0]);
                            }
                            gpu_assignment_result = gpu_download_assignment(
                                solved, downloaded_gpu_csr, gpus[0]);

                            if (solved.status ==
                                GPUAssignmentStatus::NumericalFailure) {
                                throw std::runtime_error(
                                    "GPU sparse assignment encountered a numerical failure");
                            }
                            if (solved.status == GPUAssignmentStatus::Success &&
                                !gpu_assignment_result.certificate.optimal) {
                                throw std::runtime_error(
                                    "GPU assignment failed primal/dual optimality validation");
                            }
                        }
                        // Includes device-state cleanup and CPU certification.
                        lapjv_ms = assignment_timer.elapsed_ms();
                        gpu_assignment_ready = true;
                    }

                    if (indexed && !timings_file.empty()) {
                        gpu_download_index_diagnostics(
                            device, target_index, index_stats, gpus[0]);
                    }

                    if (options.sinkhorn_backend == "gpu") {
                        cached_support = analyze_candidate_support(candidates, M);
                        if (cached_support->maximum_matching_size == N) {
                            Timer phase("GPU Sinkhorn and requested confidence handoff");
                            {
                                GPUSinkhornOptions parameters;
                                parameters.epsilon = sinkhorn_epsilon;
                                parameters.max_iterations = sinkhorn_max_iterations;
                                parameters.tolerance = sinkhorn_tolerance;
                                auto confidence = gpu_sparse_sinkhorn(device.costs, gpus[0], parameters);
                                gpu_sinkhorn_diagnostics = confidence.diagnostics;
                                sinkhorn_result.iterations = confidence.diagnostics.iterations;
                                sinkhorn_result.max_row_marginal_error =
                                    confidence.diagnostics.max_row_marginal_error;
                                sinkhorn_result.max_column_marginal_error =
                                    confidence.diagnostics.max_column_marginal_error;
                                sinkhorn_result.converged = confidence.diagnostics.status ==
                                    GPUSinkhornStatus::Converged;
                                sinkhorn_status = gpu_sinkhorn_status_name(confidence.diagnostics.status);
                                if (!options.sinkhorn_output.empty() && confidence.diagnostics.probabilities_valid) {
                                    sinkhorn_result = gpu_download_sinkhorn(confidence, gpus[0]);
                                    sinkhorn_download_bytes = confidence.edge_probabilities.bytes();
                                }
                            }
                            // Owned confidence/scaling buffers are gone before context cleanup.
                            sinkhorn_ms = phase.elapsed_ms();
                            gpu_sinkhorn_executed = true;
                        }
                    }

                    gpu_selection = device.selection;
                    feature_ms = device.timings.feature_extraction_ms +
                                 device.residency.normalization_ms;
                    candidate_ms = device.timings.candidate_generation_ms;
                    similarity_ms = device.timings.similarity_ms;
                    topk_selection_ms = device.timings.topk_selection_ms;
                    transfer_ms = device.timings.host_device_transfer_ms;
                    csr_ms = device.residency.cost_construction_ms;
                    index_build_ms = device.residency.index_build_ms;
                    index_retrieval_ms = device.residency.retrieval_ms;
                    resident_pipeline_used = true;
                    resident_evidence = device.residency;
                    device_allocation_tracking_complete =
                        device.device_allocation_tracking_complete;
                    peak_device_application_allocated_bytes =
                        device.timings.peak_device_application_allocated_bytes;
                } else {
                    GPUMatchingResult result;
                    if (indexed) {
                        auto indexed_result = gpu_compute_indexed_candidates(
                            source, target, K, options.index, gpus[0],
                            joint_normalize, options.selector, options.topk);
                        target_index = std::move(indexed_result.index_diagnostics);
                        index_stats = std::move(indexed_result.retrieval_stats);
                        index_build_ms = indexed_result.index_build_ms;
                        index_retrieval_ms = indexed_result.retrieval_ms;
                        result = std::move(indexed_result.matching);
                    } else {
                        result = gpu_compute_candidates(
                            source, target, K, gpus[0],
                            joint_normalize, options.selector, options.topk);
                    }

                    candidates = std::move(result.candidates);
                    feats_src = std::move(result.feats_source);
                    feats_tgt = std::move(result.feats_target);
                    gpu_selection = result.selection;
                    feature_ms = result.timings.feature_extraction_ms +
                        (result.device_resident
                            ? result.residency.normalization_ms : 0.0);
                    candidate_ms = result.timings.candidate_generation_ms;
                    similarity_ms = result.timings.similarity_ms;
                    topk_selection_ms = result.timings.topk_selection_ms;
                    transfer_ms = result.timings.host_device_transfer_ms;
                    device_allocation_tracking_complete =
                        result.device_allocation_tracking_complete;
                    peak_device_application_allocated_bytes =
                        result.timings.peak_device_application_allocated_bytes;
                    resident_pipeline_used = result.device_resident;
                    resident_evidence = result.residency;
                }
            } catch (...) {
                cleanup_gpus(gpus);
                throw;
            }

            // GPU work and downloads are complete, and device owners have
            // unwound. Remaining CPU consumers use independent host copies.
            cleanup_gpus(gpus);
        } else {
            if (options.assignment_backend == "gpu" || options.sinkhorn_backend == "gpu") {
                throw std::runtime_error(
                    "A GPU stage was requested, but no GPU is available");
            }
            std::cerr << "No GPU available, falling back to CPU." << std::endl;
            use_gpu = false;
        }
    }

    if (!use_gpu && candidate_input_file.empty()) {
        Timer feature_timer("CPU feature extraction");
        feats_src = compute_features(source);
        feats_tgt = compute_features(target);
        if (indexed) {
            raw_sources = feats_src;
            Timer index_timer("CPU structural index construction");
            target_index = build_structural_index(feats_tgt, options.index);
            index_build_ms = index_timer.elapsed_ms();
        }
        if (joint_normalize) {
            normalize_features_joint_zscore(feats_src, feats_tgt);
        } else {
            normalize_features_zscore(feats_src);
            normalize_features_zscore(feats_tgt);
        }
        NeighborDegreeProfiles source_profiles;
        NeighborDegreeProfiles target_profiles;
        if (neighbor_profile_weight > 0.0f) {
            source_profiles = compute_neighbor_degree_profiles(source);
            target_profiles = compute_neighbor_degree_profiles(target);
        }
        feature_ms = feature_timer.elapsed_ms() - index_build_ms;
        if (indexed) {
            Timer candidate_timer("CPU indexed retrieval and scoring");
            candidates = generate_indexed_top_k_candidates(
                raw_sources, feats_src, feats_tgt, target_index, K, &index_stats);
            candidate_ms = candidate_timer.elapsed_ms();
        } else {
        candidates = generate_top_k_candidates(
            feats_src, feats_tgt, K, feature_mask, feature_weights,
            neighbor_profile_weight > 0.0f ? &source_profiles : nullptr,
            neighbor_profile_weight > 0.0f ? &target_profiles : nullptr,
            neighbor_profile_weight, &similarity_ms, &topk_selection_ms);
        candidate_ms = similarity_ms + topk_selection_ms;
        }
    }

    // CPU baseline: use the same normalization and exact exhaustive scorer
    // as initial generation. Score only the affected source, without N*M storage.
    if (options.repair) {
        Timer repair_timer("Native CPU candidate repair");
        auto parameters = options.repair_options;
        parameters.initial_k = K;
        repair_result = repair_candidate_support(candidates, M,
            [&](int source_vertex, int prefix_k) {
                std::vector<NodeFeatures> one_source{feats_src.at(source_vertex)};
                auto rows = generate_top_k_candidates(one_source, feats_tgt, prefix_k);
                return std::move(rows.at(0));
            }, parameters);
        repair_ms = repair_timer.elapsed_ms();
        cached_support = repair_result->support;
        std::cout << "Candidate repair: "
                  << candidate_repair_stop_name(repair_result->stop)
                  << "; rounds=" << repair_result->rounds
                  << "; edges added=" << repair_result->edges_added << std::endl;
    }

    if (!candidates_file.empty()) {
        std::ofstream out(candidates_file);
        if (!out.is_open()) {
            throw std::runtime_error("Cannot open candidates output: " + candidates_file);
        }
        out << std::setprecision(std::numeric_limits<float>::max_digits10);
        out << "source,rank,target,similarity\n";
        for (int source_vertex = 0; source_vertex < N; source_vertex++) {
            for (int rank = 0; rank < (int)candidates[source_vertex].size(); rank++) {
                const Candidate& candidate = candidates[source_vertex][rank];
                out << source_vertex << "," << rank << ","
                    << candidate.target << "," << candidate.similarity << "\n";
            }
        }
        std::cout << "Candidates written to: " << candidates_file << std::endl;
    }

    const CandidateSupportStats support = cached_support
        ? std::move(*cached_support) : analyze_candidate_support(candidates, M);
    if (!matching_output_file.empty()) {
        std::ofstream matching_out(matching_output_file);
        if (!matching_out) throw std::runtime_error("Cannot open matching output: " + matching_output_file);
        matching_out << "source,target\n";
        for (int source_vertex = 0; source_vertex < N; ++source_vertex)
            matching_out << source_vertex << "," << support.source_to_target[source_vertex] << "\n";
    }
    const int zero_incoming_targets = static_cast<int>(std::count(
        support.target_incoming_counts.begin(), support.target_incoming_counts.end(), 0));
    const bool feasible_before_repair = support.maximum_matching_size == N;
    // Existing control flow uses final support; preserve the actual initial
    // feasibility separately for reporting.
    const bool initial_support_feasible = repair_result
        ? repair_result->history.front().maximum_matching_size == N
        : feasible_before_repair;
    std::cout << "Candidate support: " << support.maximum_matching_size << "/" << N
              << " source vertices matched; " << zero_incoming_targets
              << " targets have zero incoming candidates." << std::endl;

    CSRMatrix csr;
    if (gpu_costs_ready) {
        csr = std::move(downloaded_gpu_csr);
    } else {
        Timer csr_timer("CPU CSR construction");
        csr = build_sparse_cost_matrix(candidates, N, M);
        csr_ms = csr_timer.elapsed_ms();
    }
    if (multi_gpu && options.assignment_backend=="gpu" && feasible_before_repair) {
        Timer assignment_timer("Multi-GPU component assignment and certification");
        auto parameters=options.multi;
        parameters.scheduler=options.scheduler;
        multi_assignment=multi_gpu_solve_assignment(csr,options.devices,parameters);
        gpu_assignment_result=std::move(multi_assignment->assignment);
        gpu_assignment_ready=true;
        lapjv_ms=assignment_timer.elapsed_ms();
    }
    if (!gpu_assignment_ready) {
        lapjv_ms = 0.0;
    }
    print_csr_info(csr);

    AssignmentResult assignment;
    assignment.feasible = false;
    assignment.total_cost = 0.0;
    assignment.row_to_col.assign(N, -1);
    assignment.col_to_row.assign(M, -1);
    if (feasible_before_repair) {
        if (gpu_assignment_ready) {
            if (gpu_assignment_result.status != GPUAssignmentStatus::Success ||
                !gpu_assignment_result.certificate.optimal) {
                throw std::runtime_error(
                    "GPU assignment failed on support marked feasible");
            }
            assignment = std::move(gpu_assignment_result.assignment);
        } else {
            if (options.assignment_backend == "gpu") {
                throw std::runtime_error(
                    "GPU assignment was requested but did not execute");
            }
            Timer assignment_timer("CPU LAPJV");
            assignment = solve_lapjv(csr);
            lapjv_ms = assignment_timer.elapsed_ms();
        }
    } else {
        if (gpu_assignment_ready &&
            gpu_assignment_result.status == GPUAssignmentStatus::Success) {
            throw std::runtime_error(
                "GPU assignment success disagrees with candidate support analysis");
        }
        std::cerr << "Candidate support is infeasible: maximum matching covers "
                  << support.maximum_matching_size << " of " << N
                  << " source vertices. CPU assignment was not run." << std::endl;
    }

    if (assignment.feasible) {
        std::cout << "Assignment feasible: YES" << std::endl;
        std::cout << "Total cost: " << assignment.total_cost << std::endl;
        if (is_one_to_one(assignment.row_to_col, M)) {
            std::cout << "One-to-one: YES" << std::endl;
        } else {
            std::cout << "One-to-one: NO" << std::endl;
        }

        if (verbose) {
            for (int i = 0; i < N; i++) {
                std::cout << "  " << i << " -> " << assignment.row_to_col[i] << std::endl;
            }
        }
    } else {
        std::cout << "Assignment feasible: NO" << std::endl;
    }

    if (!output_file.empty()) {
        std::ofstream out(output_file);
        if (out.is_open()) {
            out << std::setprecision(std::numeric_limits<float>::max_digits10);
            out << "source,target,cost" << std::endl;
            for (int i = 0; i < N; i++) {
                int j = assignment.row_to_col[i];
                float c = (j >= 0) ? get_value(csr, i, j) : -1.0f;
                out << i << "," << j << "," << c << std::endl;
            }
            std::cout << "Results written to: " << output_file << std::endl;
        }
    }

    if (assignment.feasible && is_one_to_one(assignment.row_to_col, M)) {
        if (options.sinkhorn_backend == "gpu") {
            if (!gpu_sinkhorn_executed) {
                throw std::runtime_error("GPU Sinkhorn was requested but did not execute");
            }
        } else {
            Timer timer("CPU sparse Sinkhorn");
            sinkhorn_result = sparse_sinkhorn(
                csr, sinkhorn_epsilon, sinkhorn_max_iterations, sinkhorn_tolerance);
            sinkhorn_ms = timer.elapsed_ms();
            cpu_sinkhorn_executed = true;
            sinkhorn_status = sinkhorn_result.converged ? "converged" : "iteration_limit";
        }
        std::cout << "Sparse Sinkhorn (" << options.sinkhorn_backend << "): "
                  << sinkhorn_status << " after " << sinkhorn_result.iterations
                  << " iterations; max row error=" << sinkhorn_result.max_row_marginal_error
                  << ", max column error=" << sinkhorn_result.max_column_marginal_error
                  << std::endl;
    } else if (!gpu_sinkhorn_executed) {
        sinkhorn_status = "skipped_infeasible_assignment";
    }

    // Confidence output never modifies the hard assignment.
    if (!options.sinkhorn_output.empty() &&
        (cpu_sinkhorn_executed ||
         (gpu_sinkhorn_executed && gpu_sinkhorn_diagnostics.probabilities_valid))) {
        if (sinkhorn_result.edge_probabilities.size() != csr.values.size()) {
            throw std::runtime_error("Sinkhorn confidence count disagrees with CSR");
        }
        std::ofstream confidence_out(options.sinkhorn_output);
        if (!confidence_out) {
            throw std::runtime_error("Cannot open Sinkhorn output: " + options.sinkhorn_output);
        }
        confidence_out << std::setprecision(std::numeric_limits<double>::max_digits10);
        confidence_out << "source,target,probability\n";
        for (int row = 0; row < csr.rows; ++row) {
            for (int p = csr.row_ptr[row]; p < csr.row_ptr[row + 1]; ++p) {
                confidence_out << row << ',' << csr.col_idx[p] << ','
                    << sinkhorn_result.edge_probabilities[p] << '\n';
            }
        }
        confidence_out.close();
        if (!confidence_out) throw std::runtime_error("Failed to write Sinkhorn output");
        sinkhorn_confidence_exported = true;
    }

    std::string pipeline_description = !candidate_input_file.empty()
        ? "Fixed candidate support/costs"
        : (multi_gpu ? "CPU features/normalization with multi-GPU candidate processing" :
           (use_gpu ? "GPU feature/candidate processing" : "CPU feature/candidate processing"));
    pipeline_description += "; assignment requested: " + options.assignment_backend;
    pipeline_description += "; Sinkhorn requested: " + options.sinkhorn_backend;
    if (multi_gpu) pipeline_description += "; global CPU features/normalization, multi-GPU scoring, host staging";

    device_memory_sampler.stop();
    const std::string peak_device_process_memory_json =
        device_memory_sampler.available()
            ? std::to_string(device_memory_sampler.peak_process_bytes())
            : "null";

    if (!timings_file.empty()) {
        std::ofstream timings_out(timings_file);
        if (!timings_out.is_open()) {
            throw std::runtime_error("Cannot open timings output: " + timings_file);
        }
        timings_out << std::setprecision(std::numeric_limits<double>::max_digits10);
        size_t fallback_sources = 0, scored_pairs = 0;
        size_t pool_min = index_stats.empty() ? 0 : std::numeric_limits<size_t>::max();
        size_t pool_max = 0, bucket_pool_sum = 0;
        int max_radius = -1;
        for (const auto& stats : index_stats) {
            const size_t pool = stats.used_exhaustive_fallback ? static_cast<size_t>(M) : stats.bucket_pool_size;
            fallback_sources += stats.used_exhaustive_fallback;
            scored_pairs += pool;
            bucket_pool_sum += stats.bucket_pool_size;
            pool_min = std::min(pool_min, pool);
            pool_max = std::max(pool_max, pool);
            max_radius = std::max(max_radius, stats.final_radius);
        }
        size_t occupancy_min = target_index.bucket_keys.empty() ? 0 : std::numeric_limits<size_t>::max();
        size_t occupancy_max = 0;
        for (size_t bucket = 0; bucket < target_index.bucket_keys.size(); ++bucket) {
            size_t count = target_index.bucket_offsets[bucket+1] - target_index.bucket_offsets[bucket];
            occupancy_min = std::min(occupancy_min, count);
            occupancy_max = std::max(occupancy_max, count);
        }
        timings_out << "{\n";
        timings_out << "  \"multi_gpu\": ";
        if (multi_candidates) {
            timings_out << "{\"communication\":\"host_staging_cuda_runtime\""
                << ",\"feature_normalization_backend\":\"cpu_global_reference\""
                << ",\"assignment_policy\":\"whole_independent_components\""
                << ",\"split_component_supported\":false"
                << ",\"resident_budget_bytes_per_worker\":" << options.multi.resident_budget_bytes
                << ",\"free_memory_reserve_bytes\":" << options.multi.free_memory_reserve_bytes
                << ",\"candidate_wall_ms\":" << multi_candidates->wall_ms
                << ",\"kernel_timing_scope\":\"cumulative_across_devices\""
                << ",\"transfer_tracking_complete\":false,\"workers\":[";
            for (std::size_t i=0;i<multi_candidates->workers.size();++i) {
                const auto& w=multi_candidates->workers[i]; if (i) timings_out << ',';
                timings_out << "{\"device\":" << w.device << ",\"source_begin\":" << w.source_begin
                    << ",\"executed\":" << (w.executed?"true":"false")
                    << ",\"source_end\":" << w.source_end
                    << ",\"target_replica_bytes\":" << w.target_replica_bytes
                    << ",\"resident_payload_bytes\":" << w.resident_payload_bytes
                    << ",\"free_bytes_at_admission\":" << w.free_bytes_at_admission
                    << ",\"candidate_download_bytes\":" << w.candidate_download_bytes
                    << ",\"selector_used\":\"" << selector_name(w.selection.used) << "\""
                    << ",\"fallback\":\"" << fallback_name(w.selection.fallback) << "\""
                    << ",\"scratch_peak_bytes\":" << w.selection.scratch_peak_bytes
                    << ",\"selection_ms\":" << w.selection.selection_ms
                    << ",\"similarity_ms\":" << w.similarity_ms
                    << ",\"wall_ms\":" << w.wall_ms << '}';
            }
            timings_out << "],\"assignment\":";
            if (multi_assignment) {
                const auto& a=*multi_assignment;
                timings_out << "{\"wall_ms\":" << a.wall_ms << ",\"discovery_ms\":" << a.discovery_ms
                    << ",\"known_h2d_bytes\":" << a.known_h2d_bytes << ",\"known_d2h_bytes\":" << a.known_d2h_bytes
                    << ",\"components\":[";
                for (std::size_t i=0;i<a.dispatch.size();++i) {
                    const auto& d=a.dispatch[i]; const auto& c=d.component; if (i) timings_out << ',';
                    timings_out << "{\"component\":" << c.component << ",\"device\":" << d.device
                        << ",\"rows\":" << c.rows << ",\"columns\":" << c.columns << ",\"edges\":" << c.edges
                        << ",\"executed\":" << (c.executed?"true":"false") << ",\"execution\":";
                    if (c.executed) timings_out << '"' << gpu_assignment_execution_name(c.effective) << '"';
                    else timings_out << "null";
                    timings_out << ",\"reason\":\"" << c.reason << "\",\"workspace_bytes\":" << c.workspace_peak_bytes
                        << ",\"requested_execution\":\"" << gpu_assignment_execution_name(c.requested) << "\""
                        << ",\"status\":\"" << (c.status==GPUAssignmentStatus::Success?"success":
                            c.status==GPUAssignmentStatus::Infeasible?"infeasible":"numerical_failure") << "\""
                        << ",\"kernel_ms\":" << c.assignment_kernel_ms << ",\"total_ms\":" << c.total_ms << '}';
                }
                timings_out << "]}";
            } else timings_out << "null";
            timings_out << "},\n";
        } else timings_out << "null,\n";
        timings_out << "  \"assignment_scheduler\": ";
        if (assignment_schedule) {
            const auto& schedule = *assignment_schedule;
            timings_out << "{\"policy\":\"explicit_sequential_components\""
                << ",\"requested_execution\":\""
                << gpu_assignment_execution_name(options.scheduler.assignment.execution) << "\""
                << ",\"component_count\":" << schedule.dispatch.size()
                << ",\"discovery_ms\":" << schedule.discovery_ms
                << ",\"total_ms\":" << schedule.total_ms
                << ",\"workspace_peak_bytes\":" << schedule.workspace_peak_bytes
                << ",\"component_payload_peak_bytes\":" << schedule.component_payload_peak_bytes
                << ",\"metadata_download_bytes\":" << schedule.metadata_download_bytes
                << ",\"components\":[";
            for (std::size_t i=0; i<schedule.dispatch.size(); ++i) {
                const auto& c=schedule.dispatch[i];
                if (i) timings_out << ',';
                timings_out << "{\"component\":" << c.component
                    << ",\"rows\":" << c.rows << ",\"columns\":" << c.columns
                    << ",\"edges\":" << c.edges
                    << ",\"executed\":" << (c.executed ? "true" : "false")
                    << ",\"effective_execution\":";
                if (c.executed) timings_out << '"' << gpu_assignment_execution_name(c.effective) << '"';
                else timings_out << "null";
                timings_out << ",\"reason\":\"" << c.reason << "\""
                    << ",\"status\":\"" << (c.status==GPUAssignmentStatus::Success ? "success" :
                        c.status==GPUAssignmentStatus::Infeasible ? "infeasible" : "numerical_failure") << "\""
                    << ",\"blocks\":" << c.blocks
                    << ",\"workspace_peak_bytes\":" << c.workspace_peak_bytes
                    << ",\"compact_payload_bytes\":" << c.compact_payload_bytes
                    << ",\"compaction_ms\":" << c.compaction_ms
                    << ",\"assignment_kernel_ms\":" << c.assignment_kernel_ms
                    << ",\"total_ms\":" << c.total_ms << '}';
            }
            timings_out << "]},\n";
        } else timings_out << "null,\n";
        timings_out << "  \"candidate_repair\": ";
        if (repair_result) {
            const auto& r = *repair_result;
            timings_out << "{\"backend\":\"cpu\",\"policy\":\"alternating_conflict_ranked_prefix\""
                << ",\"stop_reason\":\"" << candidate_repair_stop_name(r.stop) << "\""
                << ",\"elapsed_ms\":" << repair_ms
                << ",\"initial_k\":" << K
                << ",\"k_limit\":" << options.repair_options.k_limit
                << ",\"max_rounds\":" << options.repair_options.max_rounds
                << ",\"max_candidate_edges\":" << options.repair_options.max_candidate_edges
                << ",\"max_payload_bytes\":" << options.repair_options.max_payload_bytes
                << ",\"memory_scope\":\"logical_candidate_payload_only\""
                << ",\"initial_candidate_edges\":" << r.initial_candidate_edges
                << ",\"final_candidate_edges\":" << r.final_candidate_edges
                << ",\"history\":[";
            for (std::size_t i = 0; i < r.history.size(); ++i) {
                const auto& h = r.history[i];
                if (i) timings_out << ',';
                timings_out << "{\"k\":" << h.k
                    << ",\"maximum_matching_size\":" << h.maximum_matching_size
                    << ",\"candidate_edges\":" << h.candidate_edges
                    << ",\"conflict_sources\":" << h.conflict_sources
                    << ",\"edges_added\":" << h.edges_added << '}';
            }
            timings_out << "]},\n";
        } else timings_out << "null,\n";
        timings_out << "  \"requested_sinkhorn_backend\": \"" << options.sinkhorn_backend
                    << "\",\n  \"sinkhorn_backend\": \""
                    << (gpu_sinkhorn_executed ? "gpu" : (cpu_sinkhorn_executed ? "cpu" : "none"))
                    << "\",\n  \"sinkhorn_status\": \"" << sinkhorn_status
                    << "\",\n  \"sinkhorn_confidence_exported\": "
                    << (sinkhorn_confidence_exported ? "true" : "false") << ",\n";
        if (gpu_sinkhorn_executed) {
            const auto& s = gpu_sinkhorn_diagnostics;
            timings_out << "  \"gpu_sinkhorn\": {\"status\":\""
                << gpu_sinkhorn_status_name(s.status)
                << "\",\"probabilities_valid\":" << (s.probabilities_valid ? "true" : "false")
                << ",\"setup_ms\":" << s.setup_ms
                << ",\"reverse_index_build_ms\":" << s.reverse_index_build_ms
                << ",\"iteration_gpu_ms\":" << s.iteration_gpu_ms
                << ",\"solver_total_ms\":" << s.total_ms
                << ",\"workspace_bytes\":" << s.workspace_bytes
                << ",\"workspace_budget_bytes\":" << GPUSinkhornOptions{}.workspace_budget_bytes
                << ",\"persistent_result_bytes\":" << s.persistent_result_bytes
                << ",\"reverse_index_bytes\":" << s.reverse_index_bytes
                << ",\"confidence_download_bytes\":" << sinkhorn_download_bytes
                << ",\"marginals\":\"unit_rows_and_columns\"},\n";
        } else {
            timings_out << "  \"gpu_sinkhorn\": null,\n";
        }
        timings_out
            << "  \"requested_assignment_backend\": \""
            << options.assignment_backend << "\",\n"
            << "  \"assignment_backend\": \""
            << (gpu_assignment_ready ? "gpu" :
                (feasible_before_repair ? "cpu" : "none")) << "\",\n"
            << "  \"assignment_ms\": " << lapjv_ms << ",\n";
        if (gpu_assignment_ready) {
            const auto& gpu = gpu_assignment_result;
            const auto& certificate = gpu.certificate;
            timings_out
                << "  \"gpu_assignment\": {"
                << "\"algorithm\":\"" << (multi_gpu ? "independent_components_sparse_primal_dual" : "sparse_primal_dual_augmenting_path") << "\""
                << ",\"status\":\""
                << (gpu.status == GPUAssignmentStatus::Success
                    ? "success" : "infeasible") << "\""
                << ",\"kernel_ms\":" << gpu.kernel_ms
                << ",\"search_steps\":" << gpu.search_steps
                << ",\"workspace_bytes\":" << gpu.workspace_bytes
                << ",\"result_download_bytes\":" << gpu.download_bytes
                << ",\"result_download_scope\":\"matching_and_dual_arrays\""
                << ",\"primal_feasible\":"
                    << (certificate.primal_feasible ? "true" : "false")
                << ",\"dual_feasible\":"
                    << (certificate.dual_feasible ? "true" : "false")
                << ",\"complementary_slackness\":"
                    << (certificate.complementary_slackness ? "true" : "false")
                << ",\"optimal\":" << (certificate.optimal ? "true" : "false")
                << ",\"primal_objective\":" << certificate.primal_objective
                << ",\"dual_objective\":" << certificate.dual_objective
                << ",\"objective_gap\":" << certificate.objective_gap
                << ",\"max_dual_violation\":" << certificate.max_dual_violation
                << ",\"max_matched_slack\":" << certificate.max_matched_slack
                << "},\n";
        } else {
            timings_out << "  \"gpu_assignment\": null,\n";
        }
        if (resident_pipeline_used) {
            const auto& r = resident_evidence;
            timings_out
                << "  \"gpu_residency\": {"
                << "\"enabled\":true"
                << ",\"transfer_scope\":\"feature_candidate_cost_stages\""
                << ",\"gpu_costs_consumed\":" << (gpu_costs_ready ? "true" : "false")
                << ",\"adjacency_bytes\":" << r.adjacency_bytes
                << ",\"raw_feature_bytes\":" << r.raw_feature_bytes
                << ",\"normalized_feature_bytes\":" << r.normalized_feature_bytes
                << ",\"candidate_bytes\":" << r.candidate_bytes
                << ",\"cost_bytes\":" << r.cost_bytes
                << ",\"index_payload_bytes\":" << r.index_payload_bytes
                << ",\"query_payload_bytes\":" << r.query_payload_bytes
                << ",\"scoring_selection_scratch_budget_bytes\":"
                    << options.topk.scratch_budget_bytes
                << ",\"scoring_selection_scratch_peak_bytes\":"
                    << r.scoring_selection_scratch_peak_bytes
                << ",\"normalization_workspace_peak_bytes\":"
                    << r.normalization_workspace_peak_bytes
                << ",\"cost_workspace_peak_bytes\":"
                    << r.cost_workspace_peak_bytes
                << ",\"transfer_tracking_complete\":"
                    << (r.transfer_tracking_complete ? "true" : "false")
                << ",\"known_h2d_bytes\":" << r.known_h2d_bytes
                << ",\"known_d2h_bytes\":" << r.known_d2h_bytes
                << ",\"h2d_bytes\":"
                    << (r.transfer_tracking_complete
                        ? std::to_string(r.known_h2d_bytes) : "null")
                << ",\"d2h_bytes\":"
                    << (r.transfer_tracking_complete
                        ? std::to_string(r.known_d2h_bytes) : "null")
                << ",\"normalization_ms\":" << r.normalization_ms
                << ",\"cpu_handoff_ms\":" << r.cpu_handoff_ms
                << ",\"timing_scope\":\"synchronized_stage_wall_time\""
                << "},\n";
        } else {
            timings_out << "  \"gpu_residency\": null,\n";
        }
        if (use_gpu && candidate_input_file.empty() && !multi_gpu) {
            const bool bounded = indexed || options.selector == GPUSelector::Parallel || options.topk.force_streaming_serial;
            const char* implementation = !bounded ? "legacy_tiled_heap" :
                (gpu_selection.used == GPUSelector::Parallel ? "streaming_parallel_merge" : "streaming_serial_heap");
            size_t payload_bytes = 0;
            for (const auto& row : candidates) {
                if (row.size() > (std::numeric_limits<size_t>::max() - payload_bytes) / sizeof(Candidate))
                    throw std::overflow_error("Candidate payload byte count overflow");
                payload_bytes += row.size() * sizeof(Candidate);
            }
            // Resident execution intentionally leaves host feature vectors empty.
            const auto feature_count = static_cast<unsigned long long>(N) +
                                       static_cast<unsigned long long>(M);
            timings_out << "  \"gpu_topk\": {\"schema_version\":1,\"requested_policy\":\"" << options.selection_policy
                << "\",\"implementation\":\"" << implementation
                << "\",\"requested\":\"" << selector_name(gpu_selection.requested)
                << "\",\"used\":\"" << selector_name(gpu_selection.used)
                << "\",\"fallback\":\"" << fallback_name(gpu_selection.fallback)
                << "\",\"budget_enforced\":" << (bounded ? "true" : "false")
                << ",\"chunk_size\":" << GPU_TOPK_CHUNK_SIZE
                << ",\"parallel_k_min\":1,\"parallel_k_max\":" << GPU_TOPK_PARALLEL_MAX_K
                << ",\"requested_batch_candidates\":" << options.topk.batch_candidates
                << ",\"selection_ms\":" << gpu_selection.selection_ms
                << ",\"candidate_stage_wall_ms\":" << gpu_selection.candidate_stage_wall_ms
                << ",\"selection_candidate_transfer_ms\":" << (bounded ? std::to_string(gpu_selection.transfer_ms) : "null")
                << ",\"scratch_budget_bytes\":" << options.topk.scratch_budget_bytes
                << ",\"scratch_peak_bytes\":" << (bounded ? std::to_string(gpu_selection.scratch_peak_bytes) : "null")
                << ",\"resident_feature_bytes\":" << feature_count * NUM_FEATURES * sizeof(float)
                << ",\"final_candidate_payload_bytes\":" << payload_bytes
                << ",\"batch_capacity\":" << gpu_selection.batch_capacity
                << ",\"batches\":" << gpu_selection.batches
                << ",\"shared_bytes_per_block\":" << gpu_selection.shared_bytes_per_block << "},\n";
        } else timings_out << "  \"gpu_topk\": null,\n";
        timings_out
                    << "  \"requested_mode\": \"" << (requested_gpu ? "gpu" : "cpu") << "\",\n"
                    << "  \"effective_mode\": \"" << (use_gpu ? "gpu" : "cpu") << "\",\n"
                    << "  \"candidate_backend\": \"" << (!candidate_input_file.empty() ? "fixed-support" : options.backend) << "\",\n"
                    << "  \"structural_index\": ";
        if (indexed) {
            timings_out << "{\"coordinates\":[\"raw_degree\",\"raw_avg_neighbor_degree\"],"
                << "\"transform\":\"natural_log1p\",\"neighborhood\":\"chebyshev_complete_shells\","
                << "\"tie_break\":\"score_descending_target_id_ascending\","
                << "\"degree_bin_width\":" << options.index.degree_bin_width
                << ",\"neighbor_bin_width\":" << options.index.neighbor_bin_width
                << ",\"minimum_radius\":" << options.index.minimum_radius
                << ",\"maximum_radius\":" << options.index.maximum_radius
                << ",\"pool_multiplier\":" << options.index.pool_multiplier
                << ",\"fallback\":\"exhaustive\"},\n"
                << "  \"index_diagnostics\": {\"index_build_ms\":" << index_build_ms
                << ",\"occupied_buckets\":" << target_index.bucket_keys.size()
                << ",\"bucket_occupancy_min\":" << occupancy_min
                << ",\"bucket_occupancy_max\":" << occupancy_max
                << ",\"fallback_sources\":" << fallback_sources
                << ",\"actual_scored_pairs\":" << scored_pairs
                << ",\"pool_size_min\":" << pool_min << ",\"pool_size_max\":" << pool_max
                << ",\"pool_size_mean\":" << (N ? static_cast<double>(scored_pairs)/N : 0)
                << ",\"bucket_pool_size_mean\":" << (N ? static_cast<double>(bucket_pool_sum)/N : 0)
                << ",\"maximum_observed_radius\":" << max_radius
                << ",\"index_storage_bytes\":" << target_index.bucket_keys.size()*sizeof(StructuralBucketKey)
                    + target_index.bucket_offsets.size()*sizeof(size_t) + target_index.target_ids.size()*sizeof(int)
                << ",\"retrieval_ms\":" << measured_time_json(index_retrieval_ms, use_gpu)
                << ",\"scratch_peak_bytes\":null,\"host_peak_bytes\":null},\n";
        } else timings_out << "null,\n  \"index_diagnostics\": null,\n";
        timings_out << "  \"index_build_ms\": " << index_build_ms << ",\n"

                    << "  \"mode\": \"" << (use_gpu ? "gpu" : "cpu") << "\",\n"
                    << "  \"vertices_source\": " << N << ",\n"
                    << "  \"vertices_target\": " << M << ",\n"
                    << "  \"k\": " << K << ",\n"
                    << "  \"normalization_mode\": \"" << (joint_normalize ? "joint_graph_pair" : "separate_per_graph") << "\",\n"
                    << "  \"pipeline_description\": \"" << pipeline_description << "\",\n"
                    << "  \"feature_extraction_ms\": " << feature_ms << ",\n"
                    << "  \"similarity_ms\": " << measured_time_json(similarity_ms, !indexed || use_gpu) << ",\n"
                    << "  \"topk_selection_ms\": " << measured_time_json(topk_selection_ms, !indexed || use_gpu) << ",\n"
                    << "  \"candidate_generation_ms\": " << candidate_ms << ",\n"
                    << "  \"host_device_transfer_ms\": " << measured_time_json(transfer_ms,!multi_gpu) << ",\n"
                    << "  \"csr_construction_ms\": " << csr_ms << ",\n"
                    << "  \"lapjv_ms\": " << measured_time_json(lapjv_ms, !gpu_assignment_ready) << ",\n"
                    << "  \"sinkhorn_ms\": " << sinkhorn_ms << ",\n"
                    << "  \"sinkhorn_epsilon\": " << sinkhorn_epsilon << ",\n"
                    << "  \"sinkhorn_max_iterations\": " << sinkhorn_max_iterations << ",\n"
                    << "  \"sinkhorn_tolerance\": " << sinkhorn_tolerance << ",\n"
                    << "  \"sinkhorn_iterations\": " << sinkhorn_result.iterations << ",\n"
                    << "  \"sinkhorn_max_row_marginal_error\": " << measured_time_json(sinkhorn_result.max_row_marginal_error, std::isfinite(sinkhorn_result.max_row_marginal_error)) << ",\n"
                    << "  \"sinkhorn_max_column_marginal_error\": " << measured_time_json(sinkhorn_result.max_column_marginal_error, std::isfinite(sinkhorn_result.max_column_marginal_error)) << ",\n"
                    << "  \"sinkhorn_converged\": " << (sinkhorn_result.converged ? "true" : "false") << ",\n"
                    << "  \"csr_memory_bytes\": " << csr_memory_bytes(csr) << ",\n"
                    << "  \"peak_device_application_allocated_bytes\": " << (use_gpu && device_allocation_tracking_complete ? std::to_string(peak_device_application_allocated_bytes) : "null") << ",\n"
                    << "  \"peak_device_process_memory_bytes\": " << (use_gpu ? peak_device_process_memory_json : "null") << ",\n"
                    << "  \"device_memory_sampling_interval_ms\": " << (use_gpu && !multi_gpu ? 5 : 0) << ",\n"
                    << "  \"target_incoming_count_histogram\": {";
        std::map<int, int> incoming_histogram;
        for (int count : support.target_incoming_counts) incoming_histogram[count]++;
        bool first_histogram_entry = true;
        for (const auto& entry : incoming_histogram) {
            timings_out << (first_histogram_entry ? "" : ",")
                        << "\n    \"" << entry.first << "\": " << entry.second;
            first_histogram_entry = false;
        }
        timings_out << "\n  },\n";
        timings_out
                    << "  \"zero_incoming_targets\": " << zero_incoming_targets << ",\n"
                    << "  \"maximum_matching_size\": " << support.maximum_matching_size << ",\n"
                    << "  \"feasible_before_repair\": " << (initial_support_feasible ? "true" : "false") << ",\n"
                    << "  \"repair_rounds\": " << (repair_result ? repair_result->rounds : 0) << ",\n"
                    << "  \"edges_added_by_repair\": " << (repair_result ? repair_result->edges_added : 0) << ",\n"
                    << "  \"feasible_after_repair\": " << (feasible_before_repair ? "true" : "false") << ",\n"
                    << "  \"assignment_status\": \"" << (feasible_before_repair ? (assignment.feasible ? "success" : "solver_failure") : "infeasible_candidate_support") << "\",\n"
                    << "  \"assignment_feasible\": " << (assignment.feasible ? "true" : "false") << ",\n"
                    << "  \"assignment_one_to_one\": "
                    << (is_one_to_one(assignment.row_to_col, M) ? "true" : "false") << "\n"
                    << "}\n";
        std::cout << "Timings written to: " << timings_file << std::endl;
    }
    if (!feasible_before_repair) return 2;
    if (!assignment.feasible || !is_one_to_one(assignment.row_to_col, M)) return 3;
    // Preserve the hard mapping, but report a failed requested confidence stage.
    if (gpu_sinkhorn_executed &&
        gpu_sinkhorn_diagnostics.status != GPUSinkhornStatus::Converged) return 4;
    return 0;
}

int main(int argc, char** argv) {
    std::string mode;
    std::string source_file, target_file, output_file, candidates_file, candidate_input_file, matching_output_file, timings_file;
    int K = 20;
    bool use_gpu = true;
    bool verify = false;
    bool verbose = false;
    bool joint_normalize = true;
    unsigned int feature_mask = ALL_FEATURES_MASK;
    std::vector<float> feature_weights;
    float neighbor_profile_weight = 0.0f;

    CandidateOptions options;
    try {
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--devices") {
            if (i+1>=argc) throw std::invalid_argument("Missing --devices list");
            const std::string list=argv[++i];
            if (list.empty() || list.front()==',' || list.back()==',')
                throw std::invalid_argument("Invalid GPU device list");
            options.devices.clear(); std::stringstream input(list); std::string field;
            while (std::getline(input,field,',')) {
                if (field.empty()) throw std::invalid_argument("Empty GPU device ID");
                const int id=parse_index_int(field);
                if (id<0 || std::find(options.devices.begin(),options.devices.end(),id)!=options.devices.end())
                    throw std::invalid_argument("GPU device IDs must be nonnegative and unique");
                options.devices.push_back(id);
            }
            options.explicit_devices=true;
        } else if (arg == "--multi-gpu-resident-mib" || arg == "--multi-gpu-reserve-mib") {
            if (i+1>=argc) throw std::invalid_argument("Missing multi-GPU memory limit");
            const int value=parse_index_int(argv[++i]);
            if (value<0 || (value==0 && arg=="--multi-gpu-resident-mib"))
                throw std::invalid_argument("Invalid multi-GPU memory limit");
            constexpr std::size_t mib=1024u*1024u;
            if (static_cast<std::size_t>(value)>std::numeric_limits<std::size_t>::max()/mib)
                throw std::overflow_error("Multi-GPU memory limit overflow");
            if (arg=="--multi-gpu-resident-mib") options.multi.resident_budget_bytes=static_cast<std::size_t>(value)*mib;
            else options.multi.free_memory_reserve_bytes=static_cast<std::size_t>(value)*mib;
            options.explicit_multi_limits=true;
        } else if (arg == "--repair") {
            options.repair = true;
        } else if (arg == "--repair-k-limit" || arg == "--repair-max-rounds" ||
                   arg == "--repair-max-edges" || arg == "--repair-payload-mib") {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + arg);
            const int value = parse_index_int(argv[++i]);
            if (value < 0 || (value == 0 && arg != "--repair-max-rounds"))
                throw std::invalid_argument("Invalid repair limit: " + arg);
            options.explicit_repair_limits = true;
            if (arg == "--repair-k-limit") options.repair_options.k_limit = value;
            else if (arg == "--repair-max-rounds") options.repair_options.max_rounds = value;
            else if (arg == "--repair-max-edges") options.repair_options.max_candidate_edges = value;
            else {
                constexpr std::size_t mib = 1024u * 1024u;
                if (static_cast<std::size_t>(value) > std::numeric_limits<std::size_t>::max() / mib)
                    throw std::overflow_error("Repair payload limit overflow");
                options.repair_options.max_payload_bytes = static_cast<std::size_t>(value) * mib;
            }
        } else if (arg == "--sinkhorn" || arg == "--sinkhorn-output") {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + arg);
            const std::string value = argv[++i];
            if (arg == "--sinkhorn") {
                if (value != "cpu" && value != "gpu") {
                    throw std::invalid_argument("--sinkhorn must be cpu or gpu");
                }
                options.sinkhorn_backend = value;
            } else {
                if (value.empty()) throw std::invalid_argument("Empty Sinkhorn output path");
                options.sinkhorn_output = value;
            }
        } else if (arg == "--assignment-execution" || arg == "--assignment-cooperative-blocks" ||
                   arg == "--assignment-component-fallback") {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + arg);
            const std::string value=argv[++i];
            options.component_assignment=true;
            if (arg == "--assignment-execution") {
                if (value == "warp") options.scheduler.assignment.execution=GPUAssignmentExecution::Warp;
                else if (value == "block") options.scheduler.assignment.execution=GPUAssignmentExecution::Block;
                else if (value == "cooperative") options.scheduler.assignment.execution=GPUAssignmentExecution::Cooperative;
                else throw std::invalid_argument("Assignment execution must be warp, block, or cooperative");
            } else if (arg == "--assignment-cooperative-blocks") {
                options.scheduler.assignment.cooperative_blocks=parse_index_int(value);
                if (options.scheduler.assignment.cooperative_blocks<2)
                    throw std::invalid_argument("Cooperative assignment needs at least two blocks");
            } else {
                if (value != "error" && value != "block")
                    throw std::invalid_argument("Component fallback must be error or block");
                options.scheduler.allow_unsupported_cooperative_fallback=value == "block";
            }
        } else if (arg == "--assignment") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for --assignment");
            }
            options.assignment_backend = argv[++i];
            if (options.assignment_backend != "cpu" &&
                options.assignment_backend != "gpu") {
                throw std::invalid_argument("--assignment must be cpu or gpu");
            }
        } else if (arg == "--gpu-resident") {
            options.resident_pipeline = true;
        } else if (arg == "--gpu-selector" || arg == "--gpu-topk-scratch-mib" ||
            arg == "--gpu-topk-batch-candidates") {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + arg);
            const std::string value = argv[++i];
            options.explicit_topk_options = true;
            if (arg == "--gpu-selector") {
                options.selection_policy = value;
                options.topk.force_streaming_serial = value == "streaming-serial";
                if (value == "parallel") options.selector = GPUSelector::Parallel;
                else if (value == "serial" || value == "streaming-serial") options.selector = GPUSelector::SerialHeap;
                else throw std::invalid_argument("GPU selector must be serial, streaming-serial, or parallel");
            } else {
                options.explicit_topk_limits = true;
                const int number = parse_index_int(value);
                if (number <= 0) throw std::invalid_argument(arg + " must be positive");
                if (arg == "--gpu-topk-scratch-mib") {
                    constexpr size_t mib = 1024u * 1024u;
                    if (static_cast<size_t>(number) > std::numeric_limits<size_t>::max() / mib)
                        throw std::overflow_error("GPU scratch budget overflow");
                    options.topk.scratch_budget_bytes = static_cast<size_t>(number) * mib;
                } else options.topk.batch_candidates = static_cast<size_t>(number);
            }
        } else if (arg == "--candidate-backend" || arg.rfind("--index-", 0) == 0) {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + arg);
            const std::string value = argv[++i];
            if (arg == "--candidate-backend") options.backend = value;
            else {
                options.explicit_index_options = true;
                if (arg == "--index-degree-width") options.index.degree_bin_width = parse_index_double(value);
                else if (arg == "--index-neighbor-width") options.index.neighbor_bin_width = parse_index_double(value);
                else if (arg == "--index-min-radius") options.index.minimum_radius = parse_index_int(value);
                else if (arg == "--index-max-radius") options.index.maximum_radius = parse_index_int(value);
                else if (arg == "--index-pool-multiplier") options.index.pool_multiplier = parse_index_int(value);
                else if (arg == "--index-fallback") {
                    if (value != "exhaustive") throw std::invalid_argument("Only exhaustive index fallback is implemented");
                } else throw std::invalid_argument("Unknown index option: " + arg);
            }
        } else if (arg == "--toy") {
            mode = "toy";
        } else if (arg == "--source" && i + 1 < argc) {
            source_file = argv[++i];
            mode = "file";
        } else if (arg == "--target" && i + 1 < argc) {
            target_file = argv[++i];
        } else if (arg == "--k" && i + 1 < argc) {
            K = std::stoi(argv[++i]);
        } else if (arg == "--gpu") {
            use_gpu = true;
        } else if (arg == "--cpu") {
            use_gpu = false;
        } else if (arg == "--verify") {
            verify = true;
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--candidates" && i + 1 < argc) {
            candidates_file = argv[++i];
        } else if (arg == "--candidate-input" && i + 1 < argc) {
            candidate_input_file = argv[++i];
        } else if (arg == "--matching-output" && i + 1 < argc) {
            matching_output_file = argv[++i];
        } else if (arg == "--timings" && i + 1 < argc) {
            timings_file = argv[++i];
        } else if (arg == "--joint-normalize") {
            joint_normalize = true;
        } else if (arg == "--separate-normalize") {
            joint_normalize = false;
        } else if (arg == "--feature-set" && i + 1 < argc) {
            std::string feature_set = argv[++i];
            if (feature_set == "all") {
                feature_mask = ALL_FEATURES_MASK;
            } else if (feature_set == "degree") {
                feature_mask = 1u << 0;
            } else if (feature_set == "no-degree-squared") {
                feature_mask = ALL_FEATURES_MASK & ~(1u << 5);
            } else {
                throw std::invalid_argument(
                    "Unknown feature set: " + feature_set +
                    " (use all, degree, or no-degree-squared)");
            }
        } else if (arg == "--feature-weights" && i + 1 < argc) {
            std::string weights_text = argv[++i];
            std::stringstream weights_stream(weights_text);
            std::string value;
            while (std::getline(weights_stream, value, ',')) {
                feature_weights.push_back(std::stof(value));
            }
            if (feature_weights.size() != NUM_FEATURES) {
                throw std::invalid_argument(
                    "--feature-weights requires six comma-separated values");
            }
        } else if (arg == "--neighbor-profile-weight" && i + 1 < argc) {
            neighbor_profile_weight = std::stof(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "  --devices 0,1        Distribute GPU scoring and whole assignment components\n"
                      << "  --multi-gpu-resident-mib N  Per-worker payload admission (default 1024)\n"
                      << "  --multi-gpu-reserve-mib N   Free-memory reserve (default 64)\n"
                      << "  Multi-GPU: CPU global features/normalization and Sinkhorn; host staging.\n";
            std::cout << "  --assignment-execution warp|block|cooperative  Enable component scheduling\n"
                      << "  --assignment-cooperative-blocks N  Experimental grid size (default 2)\n"
                      << "  --assignment-component-fallback error|block  Unsupported cooperative device only\n";
            std::cout << "  --repair             Native CPU exhaustive support repair (opt-in)\n"
                      << "  --repair-k-limit N   Maximum ranked prefix (default 512)\n"
                      << "  --repair-max-rounds N  Expansion limit, zero allowed (default 3)\n"
                      << "  --repair-max-edges N   Total accepted edge limit (default 16777216)\n"
                      << "  --repair-payload-mib N Logical candidate payload limit (default 256)\n";
            std::cout << "  --candidate-backend exhaustive|structural-index (default exhaustive)\n"
                      << "  --index-degree-width W --index-neighbor-width W (default 0.5)\n"
                      << "  --index-min-radius R --index-max-radius R (defaults 1,3)\n"
                      << "  --index-pool-multiplier N (default 4) --index-fallback exhaustive\n"
                      << "  Indexed scoring supports CPU/GPU standard six-feature mode.\n";
            std::cout << "  --sinkhorn cpu|gpu    Confidence backend (default cpu)\n"
                      << "  --sinkhorn-output FILE  Export edge confidence CSV\n";
            std::cout << "Usage: scgm_cuda [OPTIONS]" << std::endl;
            std::cout << "  --toy                Run 6-node asymmetric correctness test" << std::endl;
            std::cout << "  --source FILE        Source graph file" << std::endl;
            std::cout << "  --target FILE        Target graph file" << std::endl;
            std::cout << "  --k K                Top-K candidates (default 20)" << std::endl;
            std::cout << "  --gpu                Use GPU (default)" << std::endl;
            std::cout << "  --assignment cpu|gpu  Assignment backend (default cpu)\n";
            std::cout << "  --gpu-resident       Keep intermediate GPU stages resident; requires streaming-serial or parallel\n";
            std::cout << "  --gpu-selector serial|streaming-serial|parallel (default serial reference)\n"
                      << "  --gpu-topk-scratch-mib N (default 64; streaming path only)\n"
                      << "  --gpu-topk-batch-candidates N (default 65536; streaming path only)\n";
            std::cout << "  --cpu                Use CPU only" << std::endl;
            std::cout << "  --verify             Run verification checks" << std::endl;
            std::cout << "  --verbose            Print detailed output" << std::endl;
            std::cout << "  --output FILE        Write results to CSV" << std::endl;
            std::cout << "  --candidates FILE    Write Top-K candidates to CSV" << std::endl;
            std::cout << "  --candidate-input FILE  Reuse fixed source,rank,target,similarity CSV" << std::endl;
            std::cout << "  --matching-output FILE  Write maximum-cardinality source,target matching CSV" << std::endl;
            std::cout << "  --timings FILE       Write stage timings and memory to JSON" << std::endl;
            std::cout << "  --joint-normalize    Joint z-score normalization (default)" << std::endl;
            std::cout << "  --feature-set NAME   all, degree, or no-degree-squared" << std::endl;
            std::cout << "  --feature-weights W  six CPU weights, comma-separated" << std::endl;
            std::cout << "  --neighbor-profile-weight W  CPU neighbor-degree profile weight" << std::endl;
            return 0;
        } else {
            throw std::invalid_argument("Unknown option or missing value: " + arg);
        }
    }

    if (mode.empty()) {
        mode = "toy";  // Default to toy test
    }

    const bool multi_gpu_requested=options.devices.size()>1;
    if (options.explicit_devices && (mode!="file" || !use_gpu || !candidate_input_file.empty()))
        throw std::invalid_argument("--devices requires GPU file generation without candidate-input");
    if (options.explicit_multi_limits && !multi_gpu_requested)
        throw std::invalid_argument("Multi-GPU limits require at least two devices");
    if (multi_gpu_requested && (options.backend!="exhaustive" || options.resident_pipeline ||
        options.sinkhorn_backend!="cpu" || feature_mask!=ALL_FEATURES_MASK ||
        !feature_weights.empty() || neighbor_profile_weight!=0.0f))
        throw std::invalid_argument(
            "Multi-GPU currently requires exhaustive standard scoring, CPU Sinkhorn, and no --gpu-resident");

    if (options.explicit_repair_limits && !options.repair)
        throw std::invalid_argument("Repair limits require --repair");
    if (options.repair && (mode != "file" || use_gpu ||
        !candidate_input_file.empty() || options.backend != "exhaustive" ||
        feature_mask != ALL_FEATURES_MASK || !feature_weights.empty() ||
        neighbor_profile_weight != 0.0f))
        throw std::invalid_argument(
            "--repair currently requires --cpu file generation, exhaustive backend, "
            "standard six-feature scoring, and no candidate-input");

    if (options.sinkhorn_backend == "gpu" &&
        (mode != "file" || !use_gpu || !options.resident_pipeline ||
         !candidate_input_file.empty())) {
        throw std::invalid_argument(
            "--sinkhorn gpu requires GPU file generation with --gpu-resident");
    }
    if (!options.sinkhorn_output.empty() && mode != "file") {
        throw std::invalid_argument("--sinkhorn-output requires file mode");
    }
    if (options.component_assignment && options.assignment_backend != "gpu")
        throw std::invalid_argument("Component execution options require --assignment gpu");
    if (options.assignment_backend == "gpu" &&
        (mode != "file" || !use_gpu || (!options.resident_pipeline && !multi_gpu_requested) ||
         !candidate_input_file.empty())) {
        throw std::invalid_argument(
            "--assignment gpu requires GPU file generation with --gpu-resident or multiple --devices");
    }
    if (options.resident_pipeline &&
        (mode != "file" || !use_gpu || !candidate_input_file.empty())) {
        throw std::invalid_argument(
            "--gpu-resident requires GPU file generation");
    }
    if (options.resident_pipeline && options.selection_policy == "serial") {
        throw std::invalid_argument(
            "--gpu-resident requires --gpu-selector streaming-serial or parallel; "
            "serial retains the legacy reference");
    }
    if (options.resident_pipeline &&
        (feature_mask != ALL_FEATURES_MASK || !feature_weights.empty())) {
        throw std::invalid_argument(
            "Resident GPU scoring supports standard six-feature scoring only");
    }
    if (options.explicit_topk_options && (mode != "file" || !use_gpu || !candidate_input_file.empty()))
        throw std::invalid_argument("GPU Top-K options require GPU file generation without candidate-input");
    if (options.explicit_topk_limits && !multi_gpu_requested && options.backend == "exhaustive" && options.selector == GPUSelector::SerialHeap && !options.topk.force_streaming_serial)
        throw std::invalid_argument("Exhaustive scratch/batch controls require parallel or streaming-serial; serial preserves the legacy reference");
    if (options.backend != "exhaustive" && options.backend != "structural-index")
        throw std::invalid_argument("Unknown candidate backend: " + options.backend);
    if (options.backend == "exhaustive" && options.explicit_index_options)
        throw std::invalid_argument("Index options require structural-index backend");
    if (options.backend == "structural-index") {
        validate_structural_index_config(options.index);
        if (mode != "file") throw std::invalid_argument("Indexing requires file mode");
        if (!candidate_input_file.empty()) throw std::invalid_argument("Indexing cannot be combined with --candidate-input");
        if (feature_mask != ALL_FEATURES_MASK || !feature_weights.empty() || neighbor_profile_weight != 0.0f)
            throw std::invalid_argument("Indexed backend currently supports standard six-feature scoring only");
    }
        if (mode == "toy") {
            run_toy_test(use_gpu);
        } else if (mode == "file") {
            if (source_file.empty() || target_file.empty()) {
                std::cerr << "Error: --source and --target required for file mode." << std::endl;
                return 1;
            }
            return run_from_files(source_file, target_file, K, use_gpu, verify, verbose,
                                  output_file, candidates_file, candidate_input_file, matching_output_file, joint_normalize, feature_mask,
                                  feature_weights, neighbor_profile_weight, timings_file, options);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
