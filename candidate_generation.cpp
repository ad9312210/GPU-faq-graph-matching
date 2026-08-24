#include "candidate_generation.hpp"
#include "cosine_similarity.hpp"

std::vector<std::vector<Candidate>> generate_top_k_candidates(
    const std::vector<NodeFeatures>& feats_source,
    const std::vector<NodeFeatures>& feats_target,
    int K)
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

    std::vector<std::vector<Candidate>> candidates(N);

    for (int i = 0; i < N; i++) {
        // Compute similarities to all targets
        std::vector<Candidate> all_cands(M);
        for (int j = 0; j < M; j++) {
            all_cands[j].target = j;
            all_cands[j].similarity = cosine_similarity(feats_source[i], feats_target[j]);
        }

        // Partial sort to get top-K by similarity (descending)
        std::partial_sort(all_cands.begin(),
                          all_cands.begin() + K,
                          all_cands.end(),
                          [](const Candidate& a, const Candidate& b) {
                              return a.similarity > b.similarity;
                          });

        candidates[i].assign(all_cands.begin(), all_cands.begin() + K);
    }

    return candidates;
}