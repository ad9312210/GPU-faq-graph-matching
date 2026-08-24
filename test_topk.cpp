#include "common.hpp"
#include "graph.hpp"
#include "features.hpp"
#include "cosine_similarity.hpp"
#include "candidate_generation.hpp"

static int run_tests() {
    int failures = 0;

    // Build the six-node asymmetric test graph
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

    int N = 6, M = 6;

    // Test 1: K = M (all targets as candidates)
    {
        int K = M;
        auto candidates = generate_top_k_candidates(feats_src, feats_tgt, K);

        if ((int)candidates.size() != N) {
            std::cerr << "FAIL: candidates.size() = " << candidates.size()
                      << " expected " << N << std::endl;
            failures++;
        }

        for (int i = 0; i < N; i++) {
            if ((int)candidates[i].size() != K) {
                std::cerr << "FAIL: candidates[" << i << "].size() = "
                          << candidates[i].size() << " expected " << K << std::endl;
                failures++;
            }
        }
    }

    // Test 2: K = 2
    {
        int K = 2;
        auto candidates = generate_top_k_candidates(feats_src, feats_tgt, K);

        for (int i = 0; i < N; i++) {
            if ((int)candidates[i].size() != K) {
                std::cerr << "FAIL K=2: candidates[" << i << "].size() = "
                          << candidates[i].size() << std::endl;
                failures++;
            }
            // Verify sorted by similarity (descending)
            for (int k = 1; k < (int)candidates[i].size(); k++) {
                if (candidates[i][k].similarity > candidates[i][k-1].similarity + 1e-7f) {
                    std::cerr << "FAIL K=2: candidates[" << i
                              << "] not sorted by similarity" << std::endl;
                    failures++;
                }
            }
        }
    }

    // Test 3: K = 1
    {
        int K = 1;
        auto candidates = generate_top_k_candidates(feats_src, feats_tgt, K);

        for (int i = 0; i < N; i++) {
            if ((int)candidates[i].size() != 1) {
                std::cerr << "FAIL K=1: candidates[" << i << "].size() = "
                          << candidates[i].size() << std::endl;
                failures++;
            }

            // The single candidate should be the most similar target
            float best_sim = -1e30f;
            int best_tgt = -1;
            for (int j = 0; j < M; j++) {
                float sim = cosine_similarity(feats_src[i], feats_tgt[j]);
                if (sim > best_sim) {
                    best_sim = sim;
                    best_tgt = j;
                }
            }

            if (candidates[i][0].target != best_tgt) {
                std::cerr << "FAIL K=1: node " << i << " best candidate = "
                          << candidates[i][0].target << " expected " << best_tgt << std::endl;
                failures++;
            }
        }
    }

    // Test 4: Invalid K
    {
        try {
            generate_top_k_candidates(feats_src, feats_tgt, 0);
            std::cerr << "FAIL: K=0 should throw" << std::endl;
            failures++;
        } catch (...) {
            // Expected
        }

        try {
            generate_top_k_candidates(feats_src, feats_tgt, M + 1);
            std::cerr << "FAIL: K>M should throw" << std::endl;
            failures++;
        } catch (...) {
            // Expected
        }
    }

    // Test 5: Top-K similarity values match CPU computation
    {
        int K = 3;
        auto candidates = generate_top_k_candidates(feats_src, feats_tgt, K);

        for (int i = 0; i < N; i++) {
            for (auto& c : candidates[i]) {
                float expected_sim = cosine_similarity(feats_src[i], feats_tgt[c.target]);
                if (std::abs(c.similarity - expected_sim) > 1e-5f) {
                    std::cerr << "FAIL: similarity mismatch node " << i
                              << " target " << c.target
                              << " expected " << expected_sim
                              << " got " << c.similarity << std::endl;
                    failures++;
                }
            }
        }
    }

    // Test 6: Ground truth recall with K=M (should be 1.0)
    {
        std::vector<int> ground_truth = {3, 0, 5, 1, 4, 2};
        int K = M;
        auto candidates = generate_top_k_candidates(feats_src, feats_tgt, K);

        for (int i = 0; i < N; i++) {
            bool found = false;
            for (auto& c : candidates[i]) {
                if (c.target == ground_truth[i]) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cerr << "FAIL: ground truth target " << ground_truth[i]
                          << " not in K=" << K << " candidates for node " << i << std::endl;
                failures++;
            }
        }
    }

    if (failures == 0) {
        std::cout << "test_topk: ALL PASSED" << std::endl;
    } else {
        std::cout << "test_topk: " << failures << " FAILURES" << std::endl;
    }

    return failures;
}

int main() {
    return run_tests();
}