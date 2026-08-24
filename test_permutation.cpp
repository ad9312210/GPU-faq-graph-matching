#include "common.hpp"
#include "evaluation.hpp"

static int run_tests() {
    int failures = 0;

    // Test 1: Identity permutation
    {
        std::vector<int> identity = {0, 1, 2, 3, 4, 5};
        if (!validate_permutation(identity, 6)) {
            std::cerr << "FAIL: identity permutation not valid" << std::endl;
            failures++;
        }

        auto inv = inverse_permutation(identity);
        for (int i = 0; i < 6; i++) {
            if (inv[i] != i) {
                std::cerr << "FAIL: inverse of identity[" << i << "] = "
                          << inv[i] << std::endl;
                failures++;
            }
        }
    }

    // Test 2: Known permutation (ground truth from toy example)
    // source -> target convention: A(0)->3, B(1)->0, C(2)->5, D(3)->1, E(4)->4, F(5)->2
    {
        std::vector<int> perm = {3, 0, 5, 1, 4, 2};
        // This maps source indices to target indices, but target indices go up to 5
        // so we need to validate as a permutation of [0,6)
        if (!validate_permutation(perm, 6)) {
            std::cerr << "FAIL: toy permutation not valid" << std::endl;
            failures++;
        }

        auto inv = inverse_permutation(perm);
        // inv[3]=0, inv[0]=1, inv[5]=2, inv[1]=3, inv[4]=4, inv[2]=5
        // Check double inverse: inv[perm[i]] == i
        for (int i = 0; i < 6; i++) {
            if (inv[perm[i]] != i) {
                std::cerr << "FAIL: double inverse at i=" << i << std::endl;
                failures++;
            }
        }

        // Verify convention: perm[i] = j means source i -> target j
        // A(0) -> 3 which represents target ID 14
        // B(1) -> 0 which represents target ID 11
        if (perm[0] != 3 || perm[1] != 0) {
            std::cerr << "FAIL: permutation convention check" << std::endl;
            failures++;
        }
    }

    // Test 3: Random permutation
    {
        std::vector<int> perm = {4, 2, 0, 3, 1, 5};
        if (!validate_permutation(perm, 6)) {
            std::cerr << "FAIL: random permutation not valid" << std::endl;
            failures++;
        }

        auto inv = inverse_permutation(perm);
        for (int i = 0; i < 6; i++) {
            if (inv[perm[i]] != i) {
                std::cerr << "FAIL: random double inverse at i=" << i << std::endl;
                failures++;
            }
        }
    }

    // Test 4: Invalid permutations
    {
        std::vector<int> dup = {0, 1, 1, 3, 4, 5};
        if (validate_permutation(dup, 6)) {
            std::cerr << "FAIL: duplicate permutation should be invalid" << std::endl;
            failures++;
        }

        std::vector<int> oor = {0, 1, 2, 3, 4, 6};
        if (validate_permutation(oor, 6)) {
            std::cerr << "FAIL: out-of-range permutation should be invalid" << std::endl;
            failures++;
        }

        std::vector<int> wrong_size = {0, 1, 2};
        if (validate_permutation(wrong_size, 6)) {
            std::cerr << "FAIL: wrong size permutation should be invalid" << std::endl;
            failures++;
        }
    }

    // Test 5: mapping_accuracy
    {
        std::vector<int> pred = {3, 0, 5, 1, 4, 2};
        std::vector<int> gt   = {3, 0, 5, 1, 4, 2};
        double acc = mapping_accuracy(pred, gt);
        if (std::abs(acc - 1.0) > 1e-9) {
            std::cerr << "FAIL: perfect mapping accuracy = " << acc << std::endl;
            failures++;
        }

        // One wrong
        std::vector<int> pred2 = {3, 0, 5, 1, 2, 4};
        acc = mapping_accuracy(pred2, gt);
        if (std::abs(acc - 4.0/6.0) > 1e-9) {
            std::cerr << "FAIL: 4/6 accuracy = " << acc << std::endl;
            failures++;
        }
    }

    // Test 6: is_one_to_one
    {
        std::vector<int> good = {3, 0, 5, 1, 4, 2};
        if (!is_one_to_one(good, 6)) {
            std::cerr << "FAIL: good mapping not one-to-one" << std::endl;
            failures++;
        }

        std::vector<int> bad = {3, 0, 5, 1, 4, 4};
        if (is_one_to_one(bad, 6)) {
            std::cerr << "FAIL: bad mapping should not be one-to-one" << std::endl;
            failures++;
        }
    }

    if (failures == 0) {
        std::cout << "test_permutation: ALL PASSED" << std::endl;
    } else {
        std::cout << "test_permutation: " << failures << " FAILURES" << std::endl;
    }

    return failures;
}

int main() {
    return run_tests();
}