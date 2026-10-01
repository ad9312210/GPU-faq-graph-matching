#include "candidate_generation.hpp"
#include "cosine_similarity.hpp"
#include <queue>

NeighborDegreeProfiles compute_neighbor_degree_profiles(const Graph& graph, int bins) {
    if (bins <= 0) throw std::invalid_argument("profile bins must be positive");

    int max_degree = 0;
    std::vector<int> degrees(graph.num_vertices);
    for (int vertex = 0; vertex < graph.num_vertices; vertex++) {
        degrees[vertex] = degree(graph, vertex);
        max_degree = std::max(max_degree, degrees[vertex]);
    }

    NeighborDegreeProfiles profiles(graph.num_vertices,
                                    std::vector<float>(bins, 0.0f));
    for (int vertex = 0; vertex < graph.num_vertices; vertex++) {
        std::vector<float> neighbor_degrees;
        for (int index = graph.row_ptr[vertex]; index < graph.row_ptr[vertex + 1]; index++) {
            neighbor_degrees.push_back(
                max_degree == 0 ? 0.0f
                                : static_cast<float>(degrees[graph.col_idx[index]]) /
                                      static_cast<float>(max_degree));
        }
        std::sort(neighbor_degrees.begin(), neighbor_degrees.end());
        if (neighbor_degrees.empty()) continue;
        for (int bin = 0; bin < bins; bin++) {
            size_t index = static_cast<size_t>(
                std::llround(static_cast<double>(bin) * (neighbor_degrees.size() - 1) /
                             static_cast<double>(bins - 1 == 0 ? 1 : bins - 1)));
            profiles[vertex][bin] = neighbor_degrees[index];
        }
    }
    return profiles;
}

float neighbor_degree_profile_similarity(
    const std::vector<float>& source_profile,
    const std::vector<float>& target_profile) {
    if (source_profile.size() != target_profile.size() || source_profile.empty()) {
        throw std::invalid_argument("neighbor degree profiles must have equal nonzero size");
    }
    float difference = 0.0f;
    for (size_t index = 0; index < source_profile.size(); index++) {
        difference += std::abs(source_profile[index] - target_profile[index]);
    }
    return std::max(0.0f, 1.0f - difference / static_cast<float>(source_profile.size()));
}

std::vector<std::vector<Candidate>> generate_top_k_candidates(
    const std::vector<NodeFeatures>& feats_source,
    const std::vector<NodeFeatures>& feats_target,
    int K,
    unsigned int feature_mask,
    const std::vector<float>& feature_weights,
    const NeighborDegreeProfiles* source_profiles,
    const NeighborDegreeProfiles* target_profiles,
    float neighbor_profile_weight,
    double* similarity_ms,
    double* topk_selection_ms)
{
    int N = (int)feats_source.size();
    int M = (int)feats_target.size();

    if (K <= 0) {
        throw std::invalid_argument("generate_top_k_candidates: K must be positive");
    }
    if (K > M) {
        throw std::invalid_argument(
            "generate_top_k_candidates: K=" + std::to_string(K) +
            " exceeds number of target vertices M=" + std::to_string(M));
    }
    if (!feature_weights.empty() && feature_weights.size() != NUM_FEATURES) {
        throw std::invalid_argument("feature_weights must contain six values");
    }
    if (neighbor_profile_weight < 0.0f) {
        throw std::invalid_argument("neighbor profile weight must be non-negative");
    }
    if ((source_profiles == nullptr) != (target_profiles == nullptr)) {
        throw std::invalid_argument("source and target profiles must be provided together");
    }
    if (neighbor_profile_weight > 0.0f && source_profiles == nullptr) {
        throw std::invalid_argument("profiles are required when profile weight is positive");
    }

    std::vector<std::vector<Candidate>> candidates(N);
    double similarity_elapsed_ms = 0.0;
    double topk_elapsed_ms = 0.0;
    auto better_candidate = [](const Candidate& a, const Candidate& b) {
        if (a.similarity != b.similarity) return a.similarity > b.similarity;
        return a.target < b.target;
    };

    for (int i = 0; i < N; i++) {
        // Reuse O(M) row scratch; the algorithm never materializes an N*M matrix.
        std::vector<Candidate> scored(M);
        auto similarity_start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < M; j++) {
            float similarity = cosine_similarity(
                feats_source[i], feats_target[j], feature_mask, feature_weights);
            if (neighbor_profile_weight > 0.0f) {
                similarity += neighbor_profile_weight *
                    neighbor_degree_profile_similarity(
                        (*source_profiles)[i], (*target_profiles)[j]);
            }
            scored[j] = {j, similarity};
        }
        similarity_elapsed_ms += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - similarity_start).count();

        auto topk_start = std::chrono::high_resolution_clock::now();
        std::priority_queue<Candidate, std::vector<Candidate>, decltype(better_candidate)>
            top_k(better_candidate);
        for (const Candidate& candidate : scored) {
            if (static_cast<int>(top_k.size()) < K) {
                top_k.push(candidate);
            } else if (better_candidate(candidate, top_k.top())) {
                top_k.pop();
                top_k.push(candidate);
            }
        }
        candidates[i].reserve(K);
        while (!top_k.empty()) {
            candidates[i].push_back(top_k.top());
            top_k.pop();
        }
        std::sort(candidates[i].begin(), candidates[i].end(), better_candidate);
        topk_elapsed_ms += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - topk_start).count();
    }

    if (similarity_ms) *similarity_ms = similarity_elapsed_ms;
    if (topk_selection_ms) *topk_selection_ms = topk_elapsed_ms;
    return candidates;
}
