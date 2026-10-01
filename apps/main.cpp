#include "common.hpp"
#include "graph.hpp"
#include "features.hpp"
#include "cosine_similarity.hpp"
#include "candidate_generation.hpp"
#include "bipartite_matching.hpp"
#include "sparse_sinkhorn.hpp"
#include "csr_matrix.hpp"
#include "lapjv.hpp"
#include "gpu_matching.hpp"
#include "evaluation.hpp"
#include "device_memory_sampler.hpp"

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
                            const std::string& timings_file) {
    Graph source = load_graph(source_file);
    Graph target = load_graph(target_file);

    validate_graph(source);
    validate_graph(target);

    int N = source.num_vertices;
    int M = target.num_vertices;

    if (K <= 0) K = 20;
    if (K > M) K = M;

    std::cout << "Source: " << source_file << " (" << N << " vertices)" << std::endl;
    std::cout << "Target: " << target_file << " (" << M << " vertices)" << std::endl;
    std::cout << "K: " << K << std::endl;
    std::cout << std::endl;

    std::vector<std::vector<Candidate>> candidates;
    std::vector<NodeFeatures> feats_src, feats_tgt;
    double feature_ms = 0.0;
    double candidate_ms = 0.0;
    double similarity_ms = 0.0;
    double topk_selection_ms = 0.0;
    double csr_ms = 0.0;
    double lapjv_ms = 0.0;
    double transfer_ms = 0.0;
    size_t peak_device_application_allocated_bytes = 0;
    DeviceMemorySampler device_memory_sampler(use_gpu ? 0 : -1);

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
    } else if (use_gpu) {
        auto gpus = initialize_gpus({0});
        if (!gpus.empty()) {
            auto result = gpu_compute_candidates(source, target, K, gpus[0], joint_normalize);
            candidates = result.candidates;
            feats_src = result.feats_source;
            feats_tgt = result.feats_target;
            feature_ms = result.timings.feature_extraction_ms;
            candidate_ms = result.timings.candidate_generation_ms;
            similarity_ms = result.timings.similarity_ms;
            topk_selection_ms = result.timings.topk_selection_ms;
            transfer_ms = result.timings.host_device_transfer_ms;
            peak_device_application_allocated_bytes =
                result.timings.peak_device_application_allocated_bytes;
            cleanup_gpus(gpus);
        } else {
            std::cerr << "No GPU available, falling back to CPU." << std::endl;
            use_gpu = false;
        }
    }

    if (!use_gpu && candidate_input_file.empty()) {
        Timer feature_timer("CPU feature extraction");
        feats_src = compute_features(source);
        feats_tgt = compute_features(target);
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
        feature_ms = feature_timer.elapsed_ms();
        candidates = generate_top_k_candidates(
            feats_src, feats_tgt, K, feature_mask, feature_weights,
            neighbor_profile_weight > 0.0f ? &source_profiles : nullptr,
            neighbor_profile_weight > 0.0f ? &target_profiles : nullptr,
            neighbor_profile_weight, &similarity_ms, &topk_selection_ms);
        candidate_ms = similarity_ms + topk_selection_ms;
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

    const CandidateSupportStats support = analyze_candidate_support(candidates, M);
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
    std::cout << "Candidate support: " << support.maximum_matching_size << "/" << N
              << " source vertices matched; " << zero_incoming_targets
              << " targets have zero incoming candidates." << std::endl;

    Timer csr_timer("CSR construction");
    CSRMatrix csr = build_sparse_cost_matrix(candidates, N, M);
    csr_ms = csr_timer.elapsed_ms();
    double lapjv_ms_local = 0.0;
    lapjv_ms = 0.0;
    print_csr_info(csr);

    AssignmentResult assignment;
    assignment.feasible = false;
    assignment.total_cost = 0.0;
    assignment.row_to_col.assign(N, -1);
    assignment.col_to_row.assign(M, -1);
    if (feasible_before_repair) {
        Timer lapjv_timer("LAPJV");
        assignment = solve_lapjv(csr);
        lapjv_ms = lapjv_timer.elapsed_ms();
    } else {
        std::cerr << "Candidate support is infeasible: maximum matching covers "
                  << support.maximum_matching_size << " of " << N
                  << " source vertices. LAPJV was not run." << std::endl;
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

    constexpr double sinkhorn_epsilon = 0.1;
    constexpr int sinkhorn_max_iterations = 500;
    constexpr double sinkhorn_tolerance = 1e-3;
    SparseSinkhornResult sinkhorn_result;
    double sinkhorn_ms = 0.0;
    if (assignment.feasible && is_one_to_one(assignment.row_to_col, M)) {
        Timer sinkhorn_timer("Sparse Sinkhorn");
        sinkhorn_result = sparse_sinkhorn(
            csr, sinkhorn_epsilon, sinkhorn_max_iterations, sinkhorn_tolerance);
        sinkhorn_ms = sinkhorn_timer.elapsed_ms();
        std::cout << "Sparse Sinkhorn: "
                  << (sinkhorn_result.converged ? "converged" : "did not converge")
                  << " after " << sinkhorn_result.iterations << " iterations; max row error="
                  << sinkhorn_result.max_row_marginal_error
                  << ", max column error=" << sinkhorn_result.max_column_marginal_error
                  << std::endl;
    }

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
        timings_out << "{\n"
                    << "  \"mode\": \"" << (use_gpu ? "gpu" : "cpu") << "\",\n"
                    << "  \"vertices_source\": " << N << ",\n"
                    << "  \"vertices_target\": " << M << ",\n"
                    << "  \"k\": " << K << ",\n"
                    << "  \"normalization_mode\": \"" << (joint_normalize ? "joint_graph_pair" : "separate_per_graph") << "\",\n"
                    << "  \"pipeline_description\": \"" << (!candidate_input_file.empty() ? "Fixed candidate support/costs with CPU assignment and Sinkhorn" : (use_gpu ? "GPU feature/candidate processing with CPU assignment and Sinkhorn" : "CPU feature/candidate processing with CPU assignment and Sinkhorn")) << "\",\n"
                    << "  \"feature_extraction_ms\": " << feature_ms << ",\n"
                    << "  \"similarity_ms\": " << similarity_ms << ",\n"
                    << "  \"topk_selection_ms\": " << topk_selection_ms << ",\n"
                    << "  \"candidate_generation_ms\": " << candidate_ms << ",\n"
                    << "  \"host_device_transfer_ms\": " << transfer_ms << ",\n"
                    << "  \"csr_construction_ms\": " << csr_ms << ",\n"
                    << "  \"lapjv_ms\": " << lapjv_ms << ",\n"
                    << "  \"sinkhorn_ms\": " << sinkhorn_ms << ",\n"
                    << "  \"sinkhorn_epsilon\": " << sinkhorn_epsilon << ",\n"
                    << "  \"sinkhorn_max_iterations\": " << sinkhorn_max_iterations << ",\n"
                    << "  \"sinkhorn_tolerance\": " << sinkhorn_tolerance << ",\n"
                    << "  \"sinkhorn_iterations\": " << sinkhorn_result.iterations << ",\n"
                    << "  \"sinkhorn_max_row_marginal_error\": " << sinkhorn_result.max_row_marginal_error << ",\n"
                    << "  \"sinkhorn_max_column_marginal_error\": " << sinkhorn_result.max_column_marginal_error << ",\n"
                    << "  \"sinkhorn_converged\": " << (sinkhorn_result.converged ? "true" : "false") << ",\n"
                    << "  \"csr_memory_bytes\": " << csr_memory_bytes(csr) << ",\n"
                    << "  \"peak_device_application_allocated_bytes\": " << (use_gpu ? std::to_string(peak_device_application_allocated_bytes) : "null") << ",\n"
                    << "  \"peak_device_process_memory_bytes\": " << (use_gpu ? peak_device_process_memory_json : "null") << ",\n"
                    << "  \"device_memory_sampling_interval_ms\": " << (use_gpu ? 5 : 0) << ",\n"
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
                    << "  \"feasible_before_repair\": " << (feasible_before_repair ? "true" : "false") << ",\n"
                    << "  \"repair_rounds\": 0,\n"
                    << "  \"edges_added_by_repair\": 0,\n"
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

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--toy") {
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
            std::cout << "Usage: scgm_cuda [OPTIONS]" << std::endl;
            std::cout << "  --toy                Run 6-node asymmetric correctness test" << std::endl;
            std::cout << "  --source FILE        Source graph file" << std::endl;
            std::cout << "  --target FILE        Target graph file" << std::endl;
            std::cout << "  --k K                Top-K candidates (default 20)" << std::endl;
            std::cout << "  --gpu                Use GPU (default)" << std::endl;
            std::cout << "  --cpu                Use CPU only" << std::endl;
            std::cout << "  --verify             Run verification checks" << std::endl;
            std::cout << "  --verbose            Print detailed output" << std::endl;
            std::cout << "  --output FILE        Write results to CSV" << std::endl;
            std::cout << "  --candidates FILE    Write Top-K candidates to CSV" << std::endl;
            std::cout << "  --candidate-input FILE  Reuse fixed source,rank,target,similarity CSV" << std::endl;
            std::cout << "  --matching-output FILE  Write maximum-cardinality source,target matching CSV" << std::endl;
            std::cout << "  --timings FILE       Write stage timings and memory to JSON" << std::endl;
            std::cout << "  --joint-normalize    Joint z-score normalization in CPU mode" << std::endl;
            std::cout << "  --feature-set NAME   all, degree, or no-degree-squared" << std::endl;
            std::cout << "  --feature-weights W  six CPU weights, comma-separated" << std::endl;
            std::cout << "  --neighbor-profile-weight W  CPU neighbor-degree profile weight" << std::endl;
            return 0;
        }
    }

    if (mode.empty()) {
        mode = "toy";  // Default to toy test
    }

    try {
        if (mode == "toy") {
            run_toy_test(use_gpu);
        } else if (mode == "file") {
            if (source_file.empty() || target_file.empty()) {
                std::cerr << "Error: --source and --target required for file mode." << std::endl;
                return 1;
            }
            return run_from_files(source_file, target_file, K, use_gpu, verify, verbose,
                                  output_file, candidates_file, candidate_input_file, matching_output_file, joint_normalize, feature_mask,
                                  feature_weights, neighbor_profile_weight, timings_file);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}