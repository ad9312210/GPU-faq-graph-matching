#include "common.hpp"
#include "csr_matrix.hpp"
#include "lapjv.hpp"
#include "evaluation.hpp"
#include "candidate_generation.hpp"

static int run_tests() {
    int failures = 0;

    // Test 1: Simple 3x3 identity-like assignment
    {
        // Cost matrix (sparse):
        //     0    1    2
        // 0 [0.1  0.9  INF]
        // 1 [INF  0.2  0.8]
        // 2 [0.7  INF  0.3]
        std::vector<std::vector<Candidate>> cands = {
            {{0, 0.9f}, {1, 0.1f}},   // row 0: targets 0(sim=0.9), 1(sim=0.1)
            {{1, 0.8f}, {2, 0.2f}},   // row 1: targets 1(sim=0.8), 2(sim=0.2)
            {{0, 0.3f}, {2, 0.7f}},   // row 2: targets 0(sim=0.3), 2(sim=0.7)
        };

        CSRMatrix csr = build_sparse_cost_matrix(cands, 3, 3);
        AssignmentResult result = solve_lapjv(csr);

        if (!result.feasible) {
            std::cerr << "FAIL test1: assignment infeasible" << std::endl;
            failures++;
        } else {
            // Optimal: 0->0(0.1), 1->1(0.2), 2->2(0.3) = 0.6
            if (result.row_to_col[0] != 0 || result.row_to_col[1] != 1 || result.row_to_col[2] != 2) {
                std::cerr << "FAIL test1: wrong assignment: "
                          << result.row_to_col[0] << " "
                          << result.row_to_col[1] << " "
                          << result.row_to_col[2] << std::endl;
                failures++;
            }
            if (std::abs(result.total_cost - 0.6) > 1e-4) {
                std::cerr << "FAIL test1: wrong cost " << result.total_cost << " expected 0.6" << std::endl;
                failures++;
            }
        }
    }

    // Test 2: 2x3 rectangular sparse assignment
    {
        // 2 sources, 3 targets, K=2
        std::vector<std::vector<Candidate>> cands = {
            {{0, 0.9f}, {2, 0.5f}},   // row 0: targets 0, 2
            {{1, 0.8f}, {2, 0.3f}},   // row 1: targets 1, 2
        };

        CSRMatrix csr = build_sparse_cost_matrix(cands, 2, 3);
        AssignmentResult result = solve_lapjv(csr);

        if (!result.feasible) {
            std::cerr << "FAIL test2: assignment infeasible" << std::endl;
            failures++;
        } else {
            // Optimal: 0->0(0.1), 1->1(0.2) = 0.3
            if (std::abs(result.total_cost - 0.3) > 1e-4) {
                std::cerr << "FAIL test2: wrong cost " << result.total_cost << std::endl;
                failures++;
            }
        }
    }

    // Test 3: Infeasible assignment
    {
        // Row 0 and row 1 both can only go to target 0
        std::vector<std::vector<Candidate>> cands = {
            {{0, 0.9f}},
            {{0, 0.8f}},
        };

        CSRMatrix csr = build_sparse_cost_matrix(cands, 2, 2);
        AssignmentResult result = solve_lapjv(csr);

        if (result.feasible) {
            std::cerr << "FAIL test3: should be infeasible" << std::endl;
            failures++;
        }
    }

    // Test 4: One-to-one check
    {
        std::vector<std::vector<Candidate>> cands = {
            {{0, 0.5f}, {1, 0.3f}},
            {{0, 0.4f}, {1, 0.6f}},
        };

        CSRMatrix csr = build_sparse_cost_matrix(cands, 2, 2);
        AssignmentResult result = solve_lapjv(csr);

        if (!result.feasible) {
            std::cerr << "FAIL test4: infeasible" << std::endl;
            failures++;
        } else {
            if (!is_one_to_one(result.row_to_col, 2)) {
                std::cerr << "FAIL test4: not one-to-one" << std::endl;
                failures++;
            }
        }
    }

    // Test 5: Verify col_to_row is inverse of row_to_col
    {
        std::vector<std::vector<Candidate>> cands = {
            {{0, 0.9f}, {1, 0.1f}, {2, 0.05f}},
            {{0, 0.1f}, {1, 0.8f}, {2, 0.2f}},
            {{0, 0.05f}, {1, 0.2f}, {2, 0.7f}},
        };

        CSRMatrix csr = build_sparse_cost_matrix(cands, 3, 3);
        AssignmentResult result = solve_lapjv(csr);

        if (result.feasible) {
            for (int i = 0; i < 3; i++) {
                int j = result.row_to_col[i];
                if (result.col_to_row[j] != i) {
                    std::cerr << "FAIL test5: col_to_row[" << j << "] = "
                              << result.col_to_row[j] << " expected " << i << std::endl;
                    failures++;
                }
            }
        }
    }

    // Test 6: 4x4 with specific structure
    {
        std::vector<std::vector<Candidate>> cands = {
            {{1, 0.9f}, {3, 0.1f}},
            {{0, 0.8f}, {2, 0.3f}},
            {{1, 0.2f}, {3, 0.7f}},
            {{0, 0.1f}, {2, 0.85f}},
        };

        CSRMatrix csr = build_sparse_cost_matrix(cands, 4, 4);
        AssignmentResult result = solve_lapjv(csr);

        if (!result.feasible) {
            std::cerr << "FAIL test6: infeasible" << std::endl;
            failures++;
        } else {
            // Brute force check
            std::vector<int> perm = {0, 1, 2, 3};
            double best = 1e30;
            do {
                double cost = 0;
                bool valid = true;
                for (int i = 0; i < 4; i++) {
                    float c = get_value(csr, i, perm[i]);
                    if (c >= SCGM_INF * 0.5f) { valid = false; break; }
                    cost += c;
                }
                if (valid) best = std::min(best, cost);
            } while (std::next_permutation(perm.begin(), perm.end()));

            if (std::abs(result.total_cost - best) > 1e-4) {
                std::cerr << "FAIL test6: LAPJV cost " << result.total_cost
                          << " != brute force " << best << std::endl;
                failures++;
            }
        }
    }

    if (failures == 0) {
        std::cout << "test_lapjv: ALL PASSED" << std::endl;
    } else {
        std::cout << "test_lapjv: " << failures << " FAILURES" << std::endl;
    }

    return failures;
}

int main() {
    return run_tests();
}