#include "common.hpp"
#include "csr_matrix.hpp"
#include "lapjv.hpp"
#include "evaluation.hpp"
#include "candidate_generation.hpp"
#include "bipartite_matching.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <fstream>
#include <random>
#include <limits>
using namespace std;

struct ReferenceResults{
    bool feasible;
    double cost;
};

//Explore every allowed one-to one assignment , row by row.
static void search_assignments(
    const CSRMatrix& cost,
    int row,
    double accumulated_cost,
    vector<bool>& target_used,
    ReferenceResults& best
){
    //every source has received a distinct allowed target
    if(row==cost.rows){
        if(!best.feasible || accumulated_cost <best.cost){
            best.feasible = true;
            best.cost = accumulated_cost;
        }
        return;
    }

    // Try each stored candidate in this source row
    for(int edge = cost.row_ptr[row]; edge < cost.row_ptr[row+1]; edge++){

        const int target = cost.col_idx[edge];

        if(target_used[target]){
            continue;
        }

        target_used[target] = true;

        search_assignments(cost, row+1, accumulated_cost + cost.values[edge], target_used, best);

        //restore the state before trying another candidate
        target_used[target] = false;
    }
}


static ReferenceResults brute_force(const CSRMatrix& cost){
    ReferenceResults best{
        false,
        numeric_limits<double>::infinity()
    };

    //impossible to give every source a distinct target.
    if(cost.rows > cost.cols){
        return best;
    }

    vector<bool> target_used(cost.cols, false);

    search_assignments(cost, 0, 0.0, target_used, best);

    return best;
}



static bool close_cost(double a, double b){
    return isfinite(a) && isfinite(b) && fabs(a-b)<1e-6 + 1e-6 * max(fabs(a), fabs(b));
}

static bool check_assignment(
    const CSRMatrix& cost,
    const AssignmentResult& result
){
    //result.feasible is true
    if (!result.feasible) {
        cerr << "check_assignment: result.feasible is false" << endl;
        return false;
    }

    if(cost.rows <0 || cost.cols <0){
        cerr << "check_assignment: cost matrix has negative dimensions" << endl;
        return false;
    }
    //row_to_col.size() == cost.rows
    //col_to_row.size() == cost.cols

    if (result.row_to_col.size() != cost.rows) {
        cerr << "check_assignment: row_to_col.size() = "
                  << result.row_to_col.size() << " != cost.rows = "
                  << cost.rows << endl;
        return false;
    }
    if (result.col_to_row.size() != cost.cols) {
        cerr << "check_assignment: col_to_row.size() = "
                  << result.col_to_row.size() << " != cost.cols = "
                  << cost.cols << endl;
        return false;
    }
    //every assignedd target is withn bounds

    vector<bool> target_used(cost.cols, false);
    double recomputed_cost = 0.0;

    for (int i=0;i<cost.rows;i++){
        int j = result.row_to_col[i];

        //valoidate the target before using it as index
        if (j<0 || j>=cost.cols){
            cerr << "check_assignment: row " << i << " assigned to column "
                      << j << " out of bounds [0," << cost.cols << ")" << endl;
            return false;
        }

        //each target can be assigned to at most one source.
        if(target_used[j]){
            cerr << "check_assignment: target " << j << " assigned to multiple sources" << endl;
            return false;
        }
        target_used[j] = true;

        //forward/reverse consistency check
        if (result.col_to_row[j] != i){
            cerr << "check_assignment: forward/reverse mismatch for row "
              << i << ", target " << j << '\n';
            return false;
        }

        //find the selected edge and accumulate its stored cot
        bool found = false;
        for(int k=cost.row_ptr[i]; k < cost.row_ptr[i+1]; k++){
            if(cost.col_idx[k] == j){
                recomputed_cost += cost.values[k];
                found = true;
                break;
            }
        }

        if(!found){
            cerr << "check_assignment: edge (" << i << "," << j << ") is selected but not in CSR entries" << endl;
            return false;
        }
    }

    //rectangular problems may leave some targets unused.
    for(int j = 0; j < cost.cols; j++){
        if(!target_used[j] && result.col_to_row[j] != -1){
            cerr << "check_assignment: target " << j << " is unused but col_to_row[" << j << "] = "
                      << result.col_to_row[j] << endl;
            return false;
        }
    }

    //verify the solver's reported total against its selected edges

    if(!close_cost(recomputed_cost, result.total_cost)){
        cerr << "check_assignment: recomputed cost " << recomputed_cost
                  << " != reported total_cost " << result.total_cost << endl;
        return false;
    }

    

    return true;
}

//check_assignment is a utility function to verify if result.feasible is true, and if so, whether the assignment is valid and has the correct cost.


//Add Random CSR Generator
static CSRMatrix generate_random_case(
    unsigned int seed,
    bool guaranteed_feasible
){
    mt19937 rng(seed);
    uniform_int_distribution<int> dimension(1, 7);
    uniform_int_distribution<int> random_cost(-5, 5);
    bernoulli_distribution include_edge(0.35);

    const int rows = dimension(rng);

    const int cols = guaranteed_feasible
        ? uniform_int_distribution<int>(rows, 7)(rng)
        : dimension(rng);

    //tiny dense mask is only for constructing test inputs

    vector<vector<bool>> allowed(
        rows, vector<bool>(cols, false)
    );

    if(guaranteed_feasible){
        vector<int> targets(cols);

        for(int j=0; j < cols; j++){
            targets[j] = j;
        }

        shuffle(targets.begin(), targets.end(), rng);

        //plant a complete matching with distinct targets
        for(int i=0; i < rows; i++){
            allowed[i][targets[i]] = true;
        }
    }

    //add random edges. A boolean mask prevents duplicates

    for(int i=0; i< rows; i++){
        for(int j =0; j < cols; j++){
            if(include_edge(rng)){
                allowed[i][j] = true;
            }
        }
    }

    CSRMatrix costs;
    costs.rows = rows;
    costs.cols = cols;
    costs.row_ptr.push_back(0);

    //increasing j produces sorted, unique columns within each row

    for(int i =0;i<rows;i++){
        for(int j=0;j<cols;j++){
            if(allowed[i][j]){
                costs.col_idx.push_back(j);
                costs.values.push_back(static_cast<float>(random_cost(rng)));
            }
        }
        costs.row_ptr.push_back(costs.col_idx.size());
    }
    return costs;
}


static void report_random_failure(
    int case_number,
    unsigned int seed,
    bool guaranteed_feasible,
    const CSRMatrix& costs,
    const ReferenceResults& expected,
    const AssignmentResult& actual
) {
    cerr << "\nRandom assignment failure"
              << "\ncase: " << case_number
              << "\nseed: " << seed
              << "\nguaranteed_feasible: " << guaranteed_feasible
              << "\nrows: " << costs.rows
              << "\ncols: " << costs.cols << '\n';

    auto print_array = [](const char* name, const auto& values) {
        cerr << name << ":";
        for (const auto& value : values) {
            std::cerr << ' ' << value;
        }
        cerr << '\n';
    };

    print_array("row_ptr", costs.row_ptr);
    print_array("col_idx", costs.col_idx);
    print_array("values", costs.values);

    std::cerr << "reference feasible: " << expected.feasible
              << ", cost: " << expected.cost
              << "\nnative feasible: " << actual.feasible
              << ", cost: " << actual.total_cost << '\n';

    print_array("row_to_col", actual.row_to_col);
    print_array("col_to_row", actual.col_to_row);

    // Save the actual input, independently of the generator.
    const string filename =
        "lapjv_failure_case_" + to_string(case_number)
        + "_seed_" + to_string(seed) + ".txt";

    ofstream out(filename);

    if (!out) {
        cerr << "Could not save fixture: " << filename << '\n';
        return;
    }

    out << setprecision(numeric_limits<float>::max_digits10);
    out << costs.rows << ' ' << costs.cols
        << ' ' << costs.values.size() << '\n';

    for (int i = 0; i < costs.rows; ++i) {
        for (int edge = costs.row_ptr[i];
             edge < costs.row_ptr[i + 1];
             ++edge) {
            out << i << ' '
                << costs.col_idx[edge] << ' '
                << costs.values[edge] << '\n';
        }
    }

    out.close();

    if (!out) {
        cerr << "Failed while writing fixture: "
                  << filename << '\n';
    } else {
        cerr << "Saved fixture: " << filename << '\n';
    }
}





static int run_tests() {
    int failures = 0;

    // Test 1: Simple 3x3 identity-like assignment, optimal cost =0.6
    {
        // Cost matrix (sparse):
        //     0    1    2
        // 0 [0.1  0.9  INF]
        // 1 [INF  0.2  0.8]
        // 2 [0.7  INF  0.3]
        vector<vector<Candidate>> cands = {
            {{0, 0.9f}, {1, 0.1f}},   // row 0: targets 0(sim=0.9), 1(sim=0.1)
            {{1, 0.8f}, {2, 0.2f}},   // row 1: targets 1(sim=0.8), 2(sim=0.2)
            {{0, 0.3f}, {2, 0.7f}},   // row 2: targets 0(sim=0.3), 2(sim=0.7)
        };

        CSRMatrix csr = build_sparse_cost_matrix(cands, 3, 3);
        const ReferenceResults reference = brute_force(csr);
        const AssignmentResult actual = solve_lapjv(csr);

        if(actual.feasible != reference.feasible){
            cerr << "FAIL test1: expected feasible=" << reference.feasible
                      << ", got " << actual.feasible << endl;
            failures++;
        } else if (actual.feasible) {
            if(!check_assignment(csr, actual)){
                cerr << "FAIL test1: assignment check failed" << endl;
                failures++;
            } else if (!close_cost(actual.total_cost, 0.6)) {
                cerr << "FAIL test1: total cost " << actual.total_cost
                          << " != expected 0.6" << endl;
                failures++;
            }
        }

    }


    // Test 2: 2x3 rectangular sparse assignment, optimal cost = 0.3
    {
        // 2 sources, 3 targets, K=2
        vector<vector<Candidate>> cands = {
            {{0, 0.9f}, {2, 0.5f}},   // row 0: targets 0, 2
            {{1, 0.8f}, {2, 0.3f}},   // row 1: targets 1, 2
        };

        CSRMatrix csr = build_sparse_cost_matrix(cands, 2, 3);
        const ReferenceResults expected = brute_force(csr);
        const AssignmentResult actual = solve_lapjv(csr);

        // Confirm the reference agrees with the manually known answer.
        if (!expected.feasible || !close_cost(expected.cost, 0.3)) {
            cerr << "FAIL test2: reference should be feasible with cost 0.3\n";
            ++failures;
        } else if (actual.feasible != expected.feasible) {
            cerr << "FAIL test2: feasibility disagrees with reference\n";
            ++failures;
        } else if (!check_assignment(csr, actual)) {
            cerr << "FAIL test2: invalid assignment\n";
            ++failures;
        } else if (!close_cost(actual.total_cost, expected.cost)) {
            cerr << "FAIL test2: native cost " << actual.total_cost
                << " != reference cost " << expected.cost << '\n';
            ++failures;
        }
    }

    // Test 3: Infeasible assignment
    {
        // Row 0 and row 1 both can only go to target 0
        vector<vector<Candidate>> cands = {
            {{0, 0.9f}},
            {{0, 0.8f}},
        };

        CSRMatrix csr = build_sparse_cost_matrix(cands, 2, 2);
        const ReferenceResults expected = brute_force(csr);
        const AssignmentResult actual = solve_lapjv(csr);

        if (expected.feasible) {
            cerr << "FAIL test3: reference incorrectly accepted infeasible support\n";
            ++failures;
        } else if (actual.feasible != expected.feasible) {
            cerr << "FAIL test3: native solver accepted infeasible support\n";
            ++failures;
        }
    }

    // Test 4: One-to-one check
    {
        vector<vector<Candidate>> cands = {
            {{0, 0.5f}, {1, 0.3f}},
            {{0, 0.4f}, {1, 0.6f}},
        };

        CSRMatrix csr = build_sparse_cost_matrix(cands, 2, 2);
        const ReferenceResults expected = brute_force(csr);
        const AssignmentResult actual = solve_lapjv(csr);

        if (!expected.feasible || !close_cost(expected.cost, 0.9)) {
            cerr << "FAIL test4: reference should be feasible with cost 0.9\n";
            ++failures;
        } else if (actual.feasible != expected.feasible) {
            cerr << "FAIL test4: feasibility disagrees with reference\n";
            ++failures;
        } else if (!check_assignment(csr, actual)) {
            cerr << "FAIL test4: invalid assignment\n";
            ++failures;
        } else if (!close_cost(actual.total_cost, expected.cost)) {
            cerr << "FAIL test4: native cost " << actual.total_cost
                << " != reference cost " << expected.cost << '\n';
            ++failures;
        }
    }

    // Test 5: Verify col_to_row is inverse of row_to_col
    {
        vector<vector<Candidate>> cands = {
            {{0, 0.9f}, {1, 0.1f}, {2, 0.05f}},
            {{0, 0.1f}, {1, 0.8f}, {2, 0.2f}},
            {{0, 0.05f}, {1, 0.2f}, {2, 0.7f}},
        };

        CSRMatrix csr = build_sparse_cost_matrix(cands, 3, 3);
        const ReferenceResults expected = brute_force(csr);
        const AssignmentResult actual = solve_lapjv(csr);

        if (!expected.feasible || !close_cost(expected.cost, 0.6)) {
            cerr << "FAIL test5: reference should be feasible with cost 0.6\n";
            ++failures;
        } else if (actual.feasible != expected.feasible) {
            cerr << "FAIL test5: feasibility disagrees with reference\n";
            ++failures;
        } else if (!check_assignment(csr, actual)) {
            cerr << "FAIL test5: invalid assignment or inconsistent mappings\n";
            ++failures;
        } else if (!close_cost(actual.total_cost, expected.cost)) {
            cerr << "FAIL test5: native cost " << actual.total_cost
                << " != reference cost " << expected.cost << '\n';
            ++failures;
        }
    }

    // Test 6: 4x4 with specific structure
    {
        vector<vector<Candidate>> cands = {
            {{1, 0.9f}, {3, 0.1f}},
            {{0, 0.8f}, {2, 0.3f}},
            {{1, 0.2f}, {3, 0.7f}},
            {{0, 0.1f}, {2, 0.85f}},
        };

        CSRMatrix csr = build_sparse_cost_matrix(cands, 4, 4);
        const ReferenceResults expected = brute_force(csr);
        const AssignmentResult actual = solve_lapjv(csr);

        if (!expected.feasible || !close_cost(expected.cost, 0.75)) {
            cerr << "FAIL test6: reference should be feasible with cost 0.75\n";
            ++failures;
        } else if (actual.feasible != expected.feasible) {
            cerr << "FAIL test6: feasibility disagrees with reference\n";
            ++failures;
        } else if (!check_assignment(csr, actual)) {
            cerr << "FAIL test6: invalid assignment\n";
            ++failures;
        } else if (!close_cost(actual.total_cost, expected.cost)) {
            cerr << "FAIL test6: native cost " << actual.total_cost
                << " != reference cost " << expected.cost << '\n';
            ++failures;
        }
    }

    // Test 7: support can fail Hall's condition even when every target has
    // at least one incoming edge. Maximum matching must detect this case.
    {
        vector<vector<Candidate>> cands = {
            {{0, 0.9f}, {1, 0.8f}},
            {{0, 0.9f}, {1, 0.8f}},
            {{0, 0.9f}, {1, 0.8f}},
            {{2, 0.9f}, {3, 0.8f}},
        };

        const auto support = analyze_candidate_support(cands, 4);
        
        if (support.maximum_matching_size != 3) {
            cerr << "FAIL test7: expected maximum matching 3, got "
                      << support.maximum_matching_size << endl;
            failures++;
        }

        for (int target = 0; target < 4; ++target) {
            if (support.target_incoming_counts[target] == 0) {
                cerr << "FAIL test7: expected all targets to have incoming edges" << endl;
                failures++;
                break;
            }
        }

        
        const CSRMatrix csr = build_sparse_cost_matrix(cands, 4, 4);
        const ReferenceResults expected = brute_force(csr);
        const AssignmentResult actual = solve_lapjv(csr);

        if (expected.feasible) {
            cerr << "FAIL test7: reference accepted Hall-deficient support\n";
            ++failures;
        } else if (actual.feasible != expected.feasible) {
            cerr << "FAIL test7: native solver accepted Hall-deficient support\n";
            ++failures;
        }
    }
    {
        //Test 8: Direct CSR input with negative costs, rectangular 2x3.
        CSRMatrix costs;
        costs.rows=2;
        costs.cols=3;
        costs.row_ptr = {0, 2, 4};
        costs.col_idx = {0, 2, 1, 2};
        costs.values = {-3.0f, 2.0f, -2.0f, 4.0f};

        const ReferenceResults expected = brute_force(costs);
        const AssignmentResult actual = solve_lapjv(costs);

        if (!expected.feasible || !close_cost(expected.cost, -5.0)) {
            cerr << "FAIL test8: reference should be feasible with cost -5\n";
            ++failures;
        } else if (actual.feasible != expected.feasible) {
            cerr << "FAIL test8: feasibility disagrees with reference\n";
            ++failures;
        } else if (!check_assignment(costs, actual)) {
            cerr << "FAIL test8: invalid assignment\n";
            ++failures;
        } else if (!close_cost(actual.total_cost, expected.cost)) {
            cerr << "FAIL test8: native cost " << actual.total_cost
                << " != reference cost " << expected.cost << '\n';
            ++failures;
        }
    }
    //Test 9: Tied costs, either permutation is optimal
    {
        CSRMatrix costs;
        costs.rows = 2;
        costs.cols = 2;
        costs.row_ptr = {0, 2, 4};
        costs.col_idx = {0, 1, 0, 1};
        costs.values = {1.0f, 1.0f, 1.0f, 1.0f};

        const ReferenceResults expected = brute_force(costs);
        const AssignmentResult actual = solve_lapjv(costs);

        if (!expected.feasible || !close_cost(expected.cost, 2.0)) {
            cerr << "FAIL test9: reference should be feasible with cost 2\n";
            ++failures;
        } else if (actual.feasible != expected.feasible) {
            cerr << "FAIL test9: feasibility disagrees with reference\n";
            ++failures;
        } else if (!check_assignment(costs, actual)) {
            cerr << "FAIL test9: invalid assignment\n";
            ++failures;
        } else if (!close_cost(actual.total_cost, expected.cost)) {
            cerr << "FAIL test9: native cost " << actual.total_cost
                << " != reference cost " << expected.cost << '\n';
            ++failures;
        }
    }

    // Test 10: Two allowed zero-cost edges form a complete assignment.
    {
        CSRMatrix costs;
        costs.rows = 2;
        costs.cols = 2;
        costs.row_ptr = {0, 1, 2};
        costs.col_idx = {0, 1};
        costs.values = {0.0f, 0.0f};

        const ReferenceResults expected = brute_force(costs);
        const AssignmentResult actual = solve_lapjv(costs);

        if (!expected.feasible || !close_cost(expected.cost, 0.0)) {
            cerr << "FAIL test10: reference should be feasible with cost 0\n";
            ++failures;
        } else if (actual.feasible != expected.feasible) {
            cerr << "FAIL test10: feasibility disagrees with reference\n";
            ++failures;
        } else if (!check_assignment(costs, actual)) {
            cerr << "FAIL test10: invalid assignment\n";
            ++failures;
        } else if (!close_cost(actual.total_cost, expected.cost)) {
            cerr << "FAIL test10: native cost " << actual.total_cost
                << " != reference cost " << expected.cost << '\n';
            ++failures;
        }
    }
    //Test 11: Row 1 has no allowed targets, infeasible.
    {
        CSRMatrix costs;
        costs.rows = 2;
        costs.cols = 2;
        costs.row_ptr = {0, 1, 1};
        costs.col_idx = {0};
        costs.values = {1.0f};

        const ReferenceResults expected = brute_force(costs);
        const AssignmentResult actual = solve_lapjv(costs);

        if (expected.feasible) {
            cerr << "FAIL test11: reference accepted an empty source row\n";
            ++failures;
        } else if (actual.feasible != expected.feasible) {
            cerr << "FAIL test11: native solver accepted an empty source row\n";
            ++failures;
        }
    }

    //Test 12: Three sources cannot recieve two disticnt targets, infeasible.
    {
        CSRMatrix costs;
        costs.rows = 3;
        costs.cols = 2;
        costs.row_ptr = {0, 2, 4, 6};
        costs.col_idx = {0, 1, 0, 1, 0, 1};
        costs.values = {
            1.0f, 2.0f,
            2.0f, 1.0f,
            3.0f, 4.0f
        };

        const ReferenceResults expected = brute_force(costs);
        const AssignmentResult actual = solve_lapjv(costs);

        if (expected.feasible) {
            cerr << "FAIL test12: reference accepted more sources than targets\n";
            ++failures;
        } else if (actual.feasible != expected.feasible) {
            cerr << "FAIL test12: native solver accepted more sources than targets\n";
            ++failures;
        }
    }


    //Test 13: Rectangular 2x4 assignment with competing preferences
    {
        CSRMatrix costs;
        costs.rows = 2;
        costs.cols = 4;
        costs.row_ptr = {0, 3, 6};
        costs.col_idx = {0, 1, 2, 0, 1, 3};
        costs.values = {
            1.0f,  2.0f, 8.0f,
            1.0f, 10.0f, 9.0f
        };

        const ReferenceResults expected = brute_force(costs);
        const AssignmentResult actual = solve_lapjv(costs);

        if (!expected.feasible || !close_cost(expected.cost, 3.0)) {
            cerr << "FAIL test13: reference should be feasible with cost 3\n";
            ++failures;
        } else if (actual.feasible != expected.feasible) {
            cerr << "FAIL test13: feasibility disagrees with reference\n";
            ++failures;
        } else if (!check_assignment(costs, actual)) {
            cerr << "FAIL test13: invalid assignment\n";
            ++failures;
        } else if (!close_cost(actual.total_cost, expected.cost)) {
            cerr << "FAIL test13: native cost " << actual.total_cost
                << " != reference cost " << expected.cost << '\n';
            ++failures;
        }
    }

    //Deterministic randomized tests;
    //100 guaranteed-feasible and 100 unrestricted sparse problems
    {
        constexpr int number_of_cases = 200;
        constexpr unsigned int base_seed = 20261016u;

        int passed = 0;
        int feasible_cases = 0;
        int infeasible_cases = 0;

        for (int case_number = 0;
            case_number < number_of_cases;
            ++case_number) {
            const unsigned int seed =
                base_seed + static_cast<unsigned int>(case_number);

            const bool guaranteed_feasible =
                case_number < number_of_cases / 2;

            const CSRMatrix costs =
                generate_random_case(seed, guaranteed_feasible);

            const ReferenceResults expected = brute_force(costs);
            const AssignmentResult actual = solve_lapjv(costs);

            bool valid = true;

            // Also check the generator's planted-matching guarantee.
            if (guaranteed_feasible && !expected.feasible) {
                cerr << "Generator/reference disagreement\n";
                valid = false;
            }

            if (expected.feasible) {
                ++feasible_cases;
            } else {
                ++infeasible_cases;
            }

            if (actual.feasible != expected.feasible) {
                valid = false;
            } else if (expected.feasible) {
                if (!check_assignment(costs, actual)) {
                    valid = false;
                } else if (!close_cost(actual.total_cost, expected.cost)) {
                    valid = false;
                }
            }

            if (!valid) {
                ++failures;

                report_random_failure(
                    case_number,
                    seed,
                    guaranteed_feasible,
                    costs,
                    expected,
                    actual);
            } else {
                ++passed;
            }
        }


        cerr << "Random cases passed: "
                << passed << '/' << number_of_cases
                << " (reference feasible: " << feasible_cases
                << ", infeasible: " << infeasible_cases << ")\n";
    }

    if (failures == 0) {
        cerr << "test_lapjv: ALL PASSED" << endl;
    } else {
        cerr << "test_lapjv: " << failures << " FAILURES" << endl;
    }

    return failures;
}

int main() {
    return run_tests();
}