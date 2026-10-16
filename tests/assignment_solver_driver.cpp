#include "lapjv.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace std;

// stdin: rows cols nnz, followed by nnz records: source target cost.
// Edges may be unordered; duplicates are rejected. Costs use float32,
// matching CSRMatrix::values. Missing edges are forbidden.
// stdout: one JSON object. Infeasibility is a normal result (exit 0),
// with total_cost=null. Input/runtime errors go to stderr (exit 1).
namespace {
struct Edge {
    int row;
    int column;
    float cost;
};

int read_nonnegative_int(const char* field) {
    long long value;
    if (!(cin >> value) || value < 0 ||
        value > numeric_limits<int>::max()) {
        throw invalid_argument(field);
    }
    return static_cast<int>(value);
}

CSRMatrix read_problem() {
    CSRMatrix costs;
    costs.rows = read_nonnegative_int("invalid row count");
    costs.cols = read_nonnegative_int("invalid column count");
    const int nnz = read_nonnegative_int("invalid edge count");
    if (static_cast<long long>(nnz) >
        static_cast<long long>(costs.rows) * costs.cols) {
        throw invalid_argument("edge count exceeds matrix capacity");
    }

    vector<Edge> edges;
    edges.reserve(nnz);
    for (int index = 0; index < nnz; ++index) {
        const int row = read_nonnegative_int("invalid source index");
        const int column = read_nonnegative_int("invalid target index");
        double value;
        if (row >= costs.rows || column >= costs.cols) {
            throw std::invalid_argument("edge endpoint out of range");
        }
        if (!(cin >> value) || !isfinite(value) ||
            abs(value) > numeric_limits<float>::max()) {
            throw invalid_argument("cost must be finite and fit float32");
        }
        edges.push_back({row, column, static_cast<float>(value)});
    }

    cin >> ws;
    if (!cin.eof()) {
        throw invalid_argument("unexpected data after final edge");
    }

    sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        return a.row < b.row || (a.row == b.row && a.column < b.column);
    });

    costs.row_ptr.assign(static_cast<size_t>(costs.rows) + 1, 0);
    costs.col_idx.reserve(nnz);
    costs.values.reserve(nnz);
    for (size_t index = 0; index < edges.size(); ++index) {
        const Edge& edge = edges[index];
        if (index > 0 && edge.row == edges[index - 1].row &&
            edge.column == edges[index - 1].column) {
            throw invalid_argument("duplicate source-target edge");
        }
        ++costs.row_ptr[edge.row + 1];
        costs.col_idx.push_back(edge.column);
        costs.values.push_back(edge.cost);
    }
    for (int row = 0; row < costs.rows; ++row) {
        costs.row_ptr[row + 1] += costs.row_ptr[row];
    }
    return costs;
}

void write_array(const vector<int>& values) {
    cout << '[';
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) cout << ',';
        cout << values[index];
    }
    cout << ']';
}
}  // namespace

int main() {
    try {
        const CSRMatrix costs = read_problem();
        const AssignmentResult result = solve_lapjv(costs);
        if (result.feasible && !isfinite(result.total_cost)) {
            throw runtime_error("solver returned a nonfinite total cost");
        }

        cout << setprecision(numeric_limits<double>::max_digits10);
        cout << "{\"feasible\":" << (result.feasible ? "true" : "false")
                  << ",\"total_cost\":";
        if (result.feasible) cout << result.total_cost;
        else cout << "null";
        cout << ",\"row_to_col\":";
        write_array(result.row_to_col);
        cout << ",\"col_to_row\":";
        write_array(result.col_to_row);
        cout << "}\n";
        return 0;
    } catch (const std::exception& error) {
        cerr << "assignment_solver_driver: " << error.what() << '\n';
        return 1;
    }
}
