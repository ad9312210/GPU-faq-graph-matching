#include "common.hpp"
#include "graph.hpp"
#include "features.hpp"
#include "cosine_similarity.hpp"
#include "candidate_generation.hpp"
#include "csr_matrix.hpp"

static int run_tests() {
    int failures = 0;

    // Build toy graphs
    std::vector<std::pair<int,int>> src_edges = {
        {0,1}, {0,5}, {1,2}, {1,3}, {1,4}, {2,3}, {2,5}, {3,4}
    };
    Graph source = build_graph(6, src_edges);

    std::vector<std::pair<int,int>> tgt_edges = {
        {3,0}, {3,2}, {0,5}, {0,1}, {0,4}, {5,1}, {5,2}, {1,4}
    };
    Graph target = build_graph(6, tgt_edges);

    auto feats_src = compute_features(source);
    auto feats_tgt = compute_features(target);
    normalize_features_zscore(feats_src);
    normalize_features_zscore(feats_tgt);

    int N = 6, M = 6, K = 4;
    auto candidates = generate_top_k_candidates(feats_src, feats_tgt, K);

    CSRMatrix csr = build_sparse_cost_matrix(candidates, N, M);

    // Test 1: row_ptr[0] == 0
    if (csr.row_ptr[0] != 0) {
        std::cerr << "FAIL: row_ptr[0] = " << csr.row_ptr[0] << " (expected 0)" << std::endl;
        failures++;
    }

    // Test 2: row_ptr[N] == NNZ
    if (csr.row_ptr[N] != csr.nnz()) {
        std::cerr << "FAIL: row_ptr[N] = " << csr.row_ptr[N]
                  << " != nnz = " << csr.nnz() << std::endl;
        failures++;
    }

    // Test 3: row_ptr monotonically increasing
    for (int i = 0; i < N; i++) {
        if (csr.row_ptr[i+1] < csr.row_ptr[i]) {
            std::cerr << "FAIL: row_ptr not monotonic at row " << i << std::endl;
            failures++;
        }
    }

    // Test 4: sizes match
    if ((int)csr.col_idx.size() != csr.nnz()) {
        std::cerr << "FAIL: col_idx.size() = " << csr.col_idx.size()
                  << " != nnz = " << csr.nnz() << std::endl;
        failures++;
    }
    if ((int)csr.values.size() != csr.nnz()) {
        std::cerr << "FAIL: values.size() = " << csr.values.size()
                  << " != nnz = " << csr.nnz() << std::endl;
        failures++;
    }

    // Test 5: NNZ <= N*K
    if (csr.nnz() > N * K) {
        std::cerr << "FAIL: nnz = " << csr.nnz() << " > N*K = " << N*K << std::endl;
        failures++;
    }

    // Test 6: Each row has at most K entries
    for (int i = 0; i < N; i++) {
        int row_nnz = csr.row_ptr[i+1] - csr.row_ptr[i];
        if (row_nnz > K) {
            std::cerr << "FAIL: row " << i << " has " << row_nnz
                      << " entries (K=" << K << ")" << std::endl;
            failures++;
        }
    }

    // Test 7: col_idx in range [0, M)
    for (int j = 0; j < csr.nnz(); j++) {
        if (csr.col_idx[j] < 0 || csr.col_idx[j] >= M) {
            std::cerr << "FAIL: col_idx[" << j << "] = " << csr.col_idx[j]
                      << " out of range" << std::endl;
            failures++;
        }
    }

    // Test 8: values are cost = 1 - similarity, should be in [0, 2] range
    for (int j = 0; j < csr.nnz(); j++) {
        if (csr.values[j] < -0.01f || csr.values[j] > 2.01f) {
            std::cerr << "FAIL: values[" << j << "] = " << csr.values[j]
                      << " out of reasonable range" << std::endl;
            failures++;
        }
    }

    // Test 9: validate_csr utility
    if (!validate_csr(csr, K)) {
        std::cerr << "FAIL: validate_csr returned false" << std::endl;
        failures++;
    }

    // Test 10: Reconstruct candidates from CSR and compare
    for (int i = 0; i < N; i++) {
        std::set<int> csr_targets;
        for (int idx = csr.row_ptr[i]; idx < csr.row_ptr[i+1]; idx++) {
            csr_targets.insert(csr.col_idx[idx]);
        }
        std::set<int> cand_targets;
        for (auto& c : candidates[i]) {
            cand_targets.insert(c.target);
        }
        if (csr_targets != cand_targets) {
            std::cerr << "FAIL: CSR targets for row " << i
                      << " don't match candidate targets" << std::endl;
            failures++;
        }
    }

    // Test 11: get_value for stored vs missing entries
    {
        // A stored entry should return a finite value
        int test_row = 0;
        int test_col = csr.col_idx[csr.row_ptr[0]];  // first candidate of row 0
        float val = get_value(csr, test_row, test_col);
        if (val >= SCGM_INF * 0.5f) {
            std::cerr << "FAIL: stored entry returned INF" << std::endl;
            failures++;
        }

        // A missing entry should return SCGM_INF
        // Find a column not in row 0's candidates
        std::set<int> row0_cols;
        for (int idx = csr.row_ptr[0]; idx < csr.row_ptr[0+1]; idx++) {
            row0_cols.insert(csr.col_idx[idx]);
        }
        if (K < M) {
            for (int c = 0; c < M; c++) {
                if (row0_cols.find(c) == row0_cols.end()) {
                    float missing_val = get_value(csr, 0, c);
                    if (missing_val < SCGM_INF * 0.5f) {
                        std::cerr << "FAIL: missing entry returned " << missing_val
                                  << " (expected INF)" << std::endl;
                        failures++;
                    }
                    break;
                }
            }
        }
    }

    // Test 12: memory_bytes sanity
    size_t mem = csr_memory_bytes(csr);
    size_t expected_mem = (N + 1) * sizeof(int) + csr.nnz() * sizeof(int) + csr.nnz() * sizeof(float);
    if (mem != expected_mem) {
        std::cerr << "FAIL: memory bytes = " << mem << " expected " << expected_mem << std::endl;
        failures++;
    }

    if (failures == 0) {
        std::cout << "test_csr: ALL PASSED" << std::endl;
    } else {
        std::cout << "test_csr: " << failures << " FAILURES" << std::endl;
    }

    return failures;
}

int main() {
    return run_tests();
}