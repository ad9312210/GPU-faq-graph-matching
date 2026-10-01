#include "bipartite_matching.hpp"
#include <queue>

CandidateSupportStats analyze_candidate_support(
    const std::vector<std::vector<Candidate>>& candidates,
    int num_targets) {
    const int num_sources = static_cast<int>(candidates.size());
    CandidateSupportStats stats;
    stats.target_incoming_counts.assign(num_targets, 0);

    std::vector<int> source_match(num_sources, -1);
    std::vector<int> target_match(num_targets, -1);
    std::vector<int> distance(num_sources, -1);

    for (const auto& row : candidates) {
        for (const auto& candidate : row) {
            if (candidate.target < 0 || candidate.target >= num_targets) {
                throw std::invalid_argument("candidate target is outside the target graph");
            }
            stats.target_incoming_counts[candidate.target]++;
        }
    }

    int shortest_free_target = std::numeric_limits<int>::max();
    auto bfs = [&]() {
        std::queue<int> pending;
        for (int source = 0; source < num_sources; ++source) {
            if (source_match[source] < 0) {
                distance[source] = 0;
                pending.push(source);
            } else {
                distance[source] = -1;
            }
        }
        shortest_free_target = std::numeric_limits<int>::max();
        while (!pending.empty()) {
            const int source = pending.front();
            pending.pop();
            if (distance[source] >= shortest_free_target) continue;
            for (const auto& candidate : candidates[source]) {
                const int next_source = target_match[candidate.target];
                if (next_source < 0) {
                    shortest_free_target = distance[source] + 1;
                } else if (distance[next_source] < 0) {
                    distance[next_source] = distance[source] + 1;
                    pending.push(next_source);
                }
            }
        }
        return shortest_free_target != std::numeric_limits<int>::max();
    };

    std::function<bool(int)> dfs = [&](int source) {
        for (const auto& candidate : candidates[source]) {
            const int next_source = target_match[candidate.target];
            if ((next_source < 0 && distance[source] + 1 == shortest_free_target) ||
                (next_source >= 0 && distance[next_source] == distance[source] + 1 && dfs(next_source))) {
                source_match[source] = candidate.target;
                target_match[candidate.target] = source;
                return true;
            }
        }
        distance[source] = -1;
        return false;
    };

    while (bfs()) {
        for (int source = 0; source < num_sources; ++source) {
            if (source_match[source] < 0 && dfs(source)) {
                stats.maximum_matching_size++;
            }
        }
    }
    stats.source_to_target = std::move(source_match);
    stats.target_to_source = std::move(target_match);
    return stats;
}
