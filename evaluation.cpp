#include "evaluation.hpp"

/*
 * PERMUTATION CONVENTION:
 *   predicted[i] = j means source node i is mapped to target node j.
 *   ground_truth[i] = target node corresponding to source node i.
 */

double mapping_accuracy(
    const std::vector<int>& predicted,
    const std::vector<int>& ground_truth)
{
    if (predicted.size() != ground_truth.size()) {
        throw std::invalid_argument("mapping_accuracy: size mismatch");
    }
    int n = (int)predicted.size();
    if (n == 0) return 1.0;

    int correct = 0;
    for (int i = 0; i < n; i++) {
        if (predicted[i] == ground_truth[i]) {
            correct++;
        }
    }
    return (double)correct / (double)n;
}

bool is_one_to_one(const std::vector<int>& mapping, int num_targets) {
    std::set<int> seen;
    for (int j : mapping) {
        if (j < 0 || j >= num_targets) return false;
        if (seen.count(j)) return false;
        seen.insert(j);
    }
    return true;
}

double candidate_recall(
    const std::vector<std::vector<Candidate>>& candidates,
    const std::vector<int>& ground_truth)
{
    int n = (int)ground_truth.size();
    if (n == 0) return 1.0;
    if ((int)candidates.size() != n) {
        throw std::invalid_argument("candidate_recall: size mismatch");
    }

    int found = 0;
    for (int i = 0; i < n; i++) {
        int gt = ground_truth[i];
        for (const auto& c : candidates[i]) {
            if (c.target == gt) {
                found++;
                break;
            }
        }
    }
    return (double)found / (double)n;
}

double edge_preservation(
    const Graph& source,
    const Graph& target,
    const std::vector<int>& mapping)
{
    int num_source_edges = 0;
    int preserved = 0;

    for (int u = 0; u < source.num_vertices; u++) {
        for (int idx = source.row_ptr[u]; idx < source.row_ptr[u + 1]; idx++) {
            int v = source.col_idx[idx];
            if (u < v) {
                num_source_edges++;
                int mu = mapping[u];
                int mv = mapping[v];
                if (has_edge(target, mu, mv)) {
                    preserved++;
                }
            }
        }
    }

    if (num_source_edges == 0) return 1.0;
    return (double)preserved / (double)num_source_edges;
}

double reverse_edge_preservation(
    const Graph& source,
    const Graph& target,
    const std::vector<int>& mapping)
{
    // Build inverse mapping
    std::vector<int> inv = inverse_permutation(mapping);

    int num_target_edges = 0;
    int reverse_preserved = 0;

    for (int u = 0; u < target.num_vertices; u++) {
        for (int idx = target.row_ptr[u]; idx < target.row_ptr[u + 1]; idx++) {
            int v = target.col_idx[idx];
            if (u < v) {
                num_target_edges++;
                // Check if inv[u] and inv[v] are valid source nodes with an edge
                if (u < (int)inv.size() && v < (int)inv.size() &&
                    inv[u] >= 0 && inv[v] >= 0) {
                    if (has_edge(source, inv[u], inv[v])) {
                        reverse_preserved++;
                    }
                }
            }
        }
    }

    if (num_target_edges == 0) return 1.0;
    return (double)reverse_preserved / (double)num_target_edges;
}

std::vector<int> inverse_permutation(const std::vector<int>& p) {
    if (p.empty()) return {};

    int max_val = *std::max_element(p.begin(), p.end());
    std::vector<int> inv(max_val + 1, -1);
    for (int i = 0; i < (int)p.size(); i++) {
        if (p[i] >= 0 && p[i] <= max_val) {
            inv[p[i]] = i;
        }
    }
    return inv;
}

bool validate_permutation(const std::vector<int>& p, int n) {
    if ((int)p.size() != n) return false;
    std::vector<bool> seen(n, false);
    for (int v : p) {
        if (v < 0 || v >= n) return false;
        if (seen[v]) return false;
        seen[v] = true;
    }
    return true;
}