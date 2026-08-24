#include "common.hpp"
#include "graph.hpp"
#include "features.hpp"
#include "cosine_similarity.hpp"
#include "candidate_generation.hpp"
#include "csr_matrix.hpp"
#include "lapjv.hpp"
#include "gpu_matching.hpp"
#include "evaluation.hpp"

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

static void run_from_files(const std::string& source_file,
                            const std::string& target_file,
                            int K, bool use_gpu, bool verify, bool verbose,
                            const std::string& output_file) {
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

    if (use_gpu) {
        auto gpus = initialize_gpus({0});
        if (!gpus.empty()) {
            auto result = gpu_compute_candidates(source, target, K, gpus[0]);
            candidates = result.candidates;
            feats_src = result.feats_source;
            feats_tgt = result.feats_target;
            cleanup_gpus(gpus);
        } else {
            std::cerr << "No GPU available, falling back to CPU." << std::endl;
            use_gpu = false;
        }
    }

    if (!use_gpu) {
        feats_src = compute_features(source);
        feats_tgt = compute_features(target);
        normalize_features_zscore(feats_src);
        normalize_features_zscore(feats_tgt);
        candidates = generate_top_k_candidates(feats_src, feats_tgt, K);
    }

    CSRMatrix csr = build_sparse_cost_matrix(candidates, N, M);
    print_csr_info(csr);

    AssignmentResult assignment = solve_lapjv(csr);

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
            out << "source,target,cost" << std::endl;
            for (int i = 0; i < N; i++) {
                int j = assignment.row_to_col[i];
                float c = (j >= 0) ? get_value(csr, i, j) : -1.0f;
                out << i << "," << j << "," << c << std::endl;
            }
            std::cout << "Results written to: " << output_file << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    std::string mode;
    std::string source_file, target_file, output_file;
    int K = 20;
    bool use_gpu = true;
    bool verify = false;
    bool verbose = false;

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
            run_from_files(source_file, target_file, K, use_gpu, verify, verbose, output_file);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}