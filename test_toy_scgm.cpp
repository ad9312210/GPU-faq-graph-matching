/*
 * Exhaustive 6-node correctness oracle.
 *
 * The toy graph has 6 vertices -> 6! = 720 possible one-to-one assignments.
 * This test verifies that:
 *   native LAPJV minimum cost == exhaustive minimum cost
 * within numerical tolerance.
 *
 * PERMUTATION CONVENTION:
 *   row_to_col[i] = j means source node i -> target node j.
 */

#include "common.hpp"
#include "graph.hpp"
#include "features.hpp"
#include "cosine_similarity.hpp"
#include "candidate_generation.hpp"
#include "csr_matrix.hpp"
#include "lapjv.hpp"
#include "evaluation.hpp"

static int run_tests() {
    int failures = 0;

    // Build source graph
    std::vector<std::pair<int,int>> src_edges = {
        {0,1}, {0,5}, {1,2}, {1,3}, {1,4}, {2,3}, {2,5}, {3,4}
    };
    Graph source = build_graph(6, src_edges);

    // Ground truth: A(0)->3, B(1)->0, C(2)->5, D(3)->1, E(4)->4, F(5)->2
    std::vector<int> ground_truth = {3, 0, 5, 1, 4, 2};

    // Build target graph
    std::vector<std::pair<int,int>> tgt_edges = {
        {3,0}, {3,2}, {0,5}, {0,1}, {0,4}, {5,1}, {5,2}, {1,4}
    };
    Graph target = build_graph(6, tgt_edges);

    int N = 6, M = 6;
    int K = M;  // Use all targets to ensure recall

    // Compute features
    auto feats_src = compute_features(source);
    auto feats_tgt = compute_features(target);
    normalize_features_zscore(feats_src);
    normalize_features_zscore(feats_tgt);

    // Generate candidates
    auto candidates = generate_top_k_candidates(feats_src, feats_tgt, K);

    // Test 1: Candidate recall must be 1.0 for K=M
    double recall = candidate_recall(candidates, ground_truth);
    if (std::abs(recall - 1.0) > 1e-9) {
        std::cerr << "FAIL: Candidate recall@" << K << " = " << recall
                  << " (expected 1.0)" << std::endl;
        failures++;
    }

    // Build CSR
    CSRMatrix csr = build_sparse_cost_matrix(candidates, N, M);

    // Test 2: CSR validity
    if (!validate_csr(csr, K)) {
        std::cerr << "FAIL: CSR validation failed" << std::endl;
        failures++;
    }

    // Test 3: Solve LAPJV
    AssignmentResult result = solve_lapjv(csr);
    if (!result.feasible) {
        std::cerr << "FAIL: LAPJV returned infeasible" << std::endl;
        failures++;
    }

    // Test 4: Exhaustive 720-permutation oracle
    std::vector<int> perm = {0, 1, 2, 3, 4, 5};
    double brute_force_best = std::numeric_limits<double>::max();
    std::vector<int> brute_force_perm;
    int perm_count = 0;
    int feasible_count = 0;

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
        if (valid) {
            feasible_count++;
            if (cost < brute_force_best) {
                brute_force_best = cost;
                brute_force_perm = perm;
            }
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    if (perm_count != 720) {
        std::cerr << "FAIL: evaluated " << perm_count << " permutations (expected 720)" << std::endl;
        failures++;
    }

    std::cout << "  Permutations evaluated: " << perm_count << std::endl;
    std::cout << "  Feasible assignments: " << feasible_count << std::endl;
    std::cout << "  LAPJV cost: " << std::fixed << std::setprecision(6)
              << result.total_cost << std::endl;
    std::cout << "  Brute-force cost: " << brute_force_best << std::endl;

    if (result.feasible) {
        double diff = std::abs(result.total_cost - brute_force_best);
        if (diff > 1e-4) {
            std::cerr << "FAIL: LAPJV cost " << result.total_cost
                      << " != brute force cost " << brute_force_best
                      << " (diff=" << diff << ")" << std::endl;
            failures++;
        } else {
            std::cout << "  LAPJV vs exhaustive oracle: MATCH (diff=" << diff << ")" << std::endl;
        }
    }

    // Test 5: Verify LAPJV assignment matches brute-force assignment
    if (result.feasible && !brute_force_perm.empty()) {
        bool assignments_match = true;
        for (int i = 0; i < N; i++) {
            if (result.row_to_col[i] != brute_force_perm[i]) {
                assignments_match = false;
            }
        }
        if (!assignments_match) {
            // Different assignment but same cost is OK
            double lapjv_cost = result.total_cost;
            double bf_cost = brute_force_best;
            if (std::abs(lapjv_cost - bf_cost) < 1e-4) {
                std::cout << "  Note: Different optimal assignment with same cost (OK)"
                          << std::endl;
            } else {
                std::cerr << "FAIL: Different assignment AND different cost" << std::endl;
                failures++;
            }
        }
    }

    // Test 6: One-to-one
    if (result.feasible && !is_one_to_one(result.row_to_col, M)) {
        std::cerr << "FAIL: LAPJV result not one-to-one" << std::endl;
        failures++;
    }

    // Test 7: Permutation validation
    if (result.feasible && !validate_permutation(result.row_to_col, N)) {
        std::cerr << "FAIL: LAPJV result not a valid permutation" << std::endl;
        failures++;
    }

    // Test 8: Edge preservation for the correct mapping
    {
        double ep = edge_preservation(source, target, ground_truth);
        if (std::abs(ep - 1.0) > 1e-9) {
            std::cerr << "FAIL: Edge preservation for ground truth = " << ep
                      << " (expected 1.0)" << std::endl;
            failures++;
        }

        double rep = reverse_edge_preservation(source, target, ground_truth);
        if (std::abs(rep - 1.0) > 1e-9) {
            std::cerr << "FAIL: Reverse edge preservation for ground truth = " << rep
                      << " (expected 1.0)" << std::endl;
            failures++;
        }
    }

    // Test 9: Mapping accuracy for LAPJV result
    if (result.feasible) {
        double acc = mapping_accuracy(result.row_to_col, ground_truth);
        std::cout << "  Mapping accuracy: " << acc << std::endl;

        double ep = edge_preservation(source, target, result.row_to_col);
        std::cout << "  Edge preservation: " << ep << std::endl;
    }

    // Test 10: Inverse permutation
    if (result.feasible) {
        auto inv = inverse_permutation(result.row_to_col);
        for (int i = 0; i < N; i++) {
            int j = result.row_to_col[i];
            if (inv[j] != i) {
                std::cerr << "FAIL: inverse permutation at j=" << j
                          << " expected " << i << " got " << inv[j] << std::endl;
                failures++;
            }
        }
    }

    if (failures == 0) {
        std::cout << "test_toy_scgm: ALL PASSED" << std::endl;
    } else {
        std::cout << "test_toy_scgm: " << failures << " FAILURES" << std::endl;
    }

    return failures;
}

int main() {
    return run_tests();
}