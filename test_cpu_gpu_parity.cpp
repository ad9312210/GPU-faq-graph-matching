#include "common.hpp"
#include "graph.hpp"
#include "features.hpp"
#include "cosine_similarity.hpp"
#include "candidate_generation.hpp"
#include "csr_matrix.hpp"
#include "gpu_matching.hpp"

static int run_tests() {
    int failures = 0;

    // Check GPU availability
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cout << "test_cpu_gpu_parity: SKIPPED (no GPU)" << std::endl;
        return 0;
    }

    auto gpus = initialize_gpus({0});
    if (gpus.empty()) {
        std::cout << "test_cpu_gpu_parity: SKIPPED (GPU init failed)" << std::endl;
        return 0;
    }

    // Build the six-node asymmetric test graph
    std::vector<std::pair<int,int>> src_edges = {
        {0,1}, {0,5}, {1,2}, {1,3}, {1,4}, {2,3}, {2,5}, {3,4}
    };
    Graph source = build_graph(6, src_edges);

    std::vector<std::pair<int,int>> tgt_edges = {
        {3,0}, {3,2}, {0,5}, {0,1}, {0,4}, {5,1}, {5,2}, {1,4}
    };
    Graph target = build_graph(6, tgt_edges);

    int N = 6, M = 6, K = M;

    // CPU path
    auto cpu_feats_src = compute_features(source);
    auto cpu_feats_tgt = compute_features(target);
    normalize_features_zscore(cpu_feats_src);
    normalize_features_zscore(cpu_feats_tgt);
    auto cpu_candidates = generate_top_k_candidates(cpu_feats_src, cpu_feats_tgt, K);

    // GPU path
    auto gpu_result = gpu_compute_candidates(source, target, K, gpus[0]);
    auto& gpu_feats_src = gpu_result.feats_source;
    auto& gpu_feats_tgt = gpu_result.feats_target;
    auto& gpu_candidates = gpu_result.candidates;

    float tol = 1e-3f;  // Tolerance for float comparison

    // Test 1: Feature values approximately equal
    std::cout << "Checking feature parity..." << std::endl;
    for (int i = 0; i < N; i++) {
        for (int f = 0; f < NUM_FEATURES; f++) {
            float cpu_val = cpu_feats_src[i].as_array(f);
            float gpu_val = gpu_feats_src[i].as_array(f);
            float err_val = std::abs(cpu_val - gpu_val);
            if (err_val > tol) {
                std::cerr << "FAIL: Source feature mismatch: node=" << i
                          << " feature=" << f
                          << " CPU=" << cpu_val
                          << " GPU=" << gpu_val
                          << " abs_error=" << err_val << std::endl;
                failures++;
            }
        }
    }

    for (int i = 0; i < M; i++) {
        for (int f = 0; f < NUM_FEATURES; f++) {
            float cpu_val = cpu_feats_tgt[i].as_array(f);
            float gpu_val = gpu_feats_tgt[i].as_array(f);
            float err_val = std::abs(cpu_val - gpu_val);
            if (err_val > tol) {
                std::cerr << "FAIL: Target feature mismatch: node=" << i
                          << " feature=" << f
                          << " CPU=" << cpu_val
                          << " GPU=" << gpu_val
                          << " abs_error=" << err_val << std::endl;
                failures++;
            }
        }
    }

    // Test 2: Similarity values approximately equal
    std::cout << "Checking similarity parity..." << std::endl;
    for (int i = 0; i < N; i++) {
        for (auto& cpu_c : cpu_candidates[i]) {
            float cpu_sim = cpu_c.similarity;
            // Find same target in GPU candidates
            float gpu_sim = -999.0f;
            for (auto& gpu_c : gpu_candidates[i]) {
                if (gpu_c.target == cpu_c.target) {
                    gpu_sim = gpu_c.similarity;
                    break;
                }
            }
            if (gpu_sim > -998.0f) {
                float err_val = std::abs(cpu_sim - gpu_sim);
                if (err_val > tol) {
                    std::cerr << "FAIL: Similarity mismatch: node=" << i
                              << " candidate=" << cpu_c.target
                              << " CPU=" << cpu_sim
                              << " GPU=" << gpu_sim
                              << " abs_error=" << err_val << std::endl;
                    failures++;
                }
            }
        }
    }

    // Test 3: Top-K candidate sets equal
    std::cout << "Checking top-K candidate set parity..." << std::endl;
    for (int i = 0; i < N; i++) {
        std::set<int> cpu_set, gpu_set;
        for (auto& c : cpu_candidates[i]) cpu_set.insert(c.target);
        for (auto& c : gpu_candidates[i]) gpu_set.insert(c.target);

        if (cpu_set != gpu_set) {
            std::cerr << "FAIL: Candidate set mismatch for node " << i << std::endl;
            std::cerr << "  CPU: {";
            for (int t : cpu_set) std::cerr << t << " ";
            std::cerr << "}" << std::endl;
            std::cerr << "  GPU: {";
            for (int t : gpu_set) std::cerr << t << " ";
            std::cerr << "}" << std::endl;
            failures++;
        }
    }

    // Test 4: CSR construction parity
    std::cout << "Checking CSR parity..." << std::endl;
    auto cpu_csr = build_sparse_cost_matrix(cpu_candidates, N, M);
    auto gpu_csr = build_sparse_cost_matrix(gpu_candidates, N, M);

    // row_ptr equal
    if (cpu_csr.row_ptr != gpu_csr.row_ptr) {
        std::cerr << "FAIL: CSR row_ptr mismatch" << std::endl;
        failures++;
    }

    // col_idx equal (after both are sorted)
    if (cpu_csr.col_idx != gpu_csr.col_idx) {
        std::cerr << "FAIL: CSR col_idx mismatch" << std::endl;
        failures++;
    }

    // values approximately equal
    if (cpu_csr.values.size() == gpu_csr.values.size()) {
        for (int j = 0; j < (int)cpu_csr.values.size(); j++) {
            float err_val = std::abs(cpu_csr.values[j] - gpu_csr.values[j]);
            if (err_val > tol) {
                std::cerr << "FAIL: CSR values[" << j << "] CPU=" << cpu_csr.values[j]
                          << " GPU=" << gpu_csr.values[j]
                          << " abs_error=" << err_val << std::endl;
                failures++;
            }
        }
    } else {
        std::cerr << "FAIL: CSR values size mismatch" << std::endl;
        failures++;
    }

    cleanup_gpus(gpus);

    if (failures == 0) {
        std::cout << "test_cpu_gpu_parity: ALL PASSED" << std::endl;
    } else {
        std::cout << "test_cpu_gpu_parity: " << failures << " FAILURES" << std::endl;
    }

    return failures;
}

int main() {
    return run_tests();
}