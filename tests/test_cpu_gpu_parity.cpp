#include "gpu_matching.hpp"
#include "cosine_similarity.hpp"
#include "gpu_test_utils.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <set>

namespace {
constexpr float FEATURE_ABS = 2e-5f, FEATURE_REL = 2e-4f, SCORE_TOL = 1e-4f;
bool close(float a, float b, float absolute, float relative = 0) {
    return std::isfinite(a) && std::isfinite(b) &&
        std::abs(a-b) <= absolute + relative * std::max(std::abs(a), std::abs(b));
}
void feature_parity(const std::vector<NodeFeatures>& a, const std::vector<NodeFeatures>& b) {
    gpu_test_require(a.size() == b.size(), "Feature row count mismatch");
    for (std::size_t row = 0; row < a.size(); ++row)
        for (int f = 0; f < NUM_FEATURES; ++f)
            gpu_test_require(close(a[row].as_array(f), b[row].as_array(f), FEATURE_ABS, FEATURE_REL),
                             "CPU/GPU feature error exceeds declared tolerance");
}
void exact_rows(const std::vector<std::vector<Candidate>>& a,
                const std::vector<std::vector<Candidate>>& b) {
    gpu_test_require(a.size() == b.size(), "Exact parity row count mismatch");
    for (std::size_t row = 0; row < a.size(); ++row) {
        gpu_test_require(a[row].size() == b[row].size(), "Exact parity length mismatch");
        for (std::size_t rank = 0; rank < a[row].size(); ++rank) {
            gpu_test_require(a[row][rank].target == b[row][rank].target,
                             "Shared GPU scoring kernel yielded different selected IDs");
            gpu_test_require(std::memcmp(&a[row][rank].similarity, &b[row][rank].similarity, sizeof(float)) == 0,
                             "Shared GPU scoring kernel yielded different score bits");
        }
    }
}
void scoring_parity(const std::vector<NodeFeatures>& cpu_source,
                    const std::vector<NodeFeatures>& cpu_target,
                    const std::vector<std::vector<Candidate>>& actual, int k) {
    gpu_test_require(actual.size() == cpu_source.size(), "Wrong candidate row count");
    double maximum_error = 0;
    for (std::size_t source = 0; source < actual.size(); ++source) {
        const auto& row = actual[source];
        gpu_test_require(row.size() == static_cast<std::size_t>(k), "Wrong K");
        std::vector<float> scores(cpu_target.size());
        for (std::size_t target = 0; target < cpu_target.size(); ++target) {
            scores[target] = cosine_similarity(cpu_source[source], cpu_target[target]);
            gpu_test_require(std::isfinite(scores[target]), "Nonfinite CPU reference score");
        }
        std::vector<float> sorted = scores;
        std::sort(sorted.begin(), sorted.end(), std::greater<float>());
        const float cutoff = sorted[k-1];
        std::set<int> selected;
        for (std::size_t rank = 0; rank < row.size(); ++rank) {
            const auto candidate = row[rank];
            gpu_test_require(candidate.target >= 0 && static_cast<std::size_t>(candidate.target) < scores.size(),
                             "Target outside graph");
            gpu_test_require(selected.insert(candidate.target).second, "Duplicate output target");
            gpu_test_require(close(candidate.similarity, scores[candidate.target], SCORE_TOL),
                             "CPU/GPU scoring error exceeds declared tolerance");
            maximum_error = std::max(maximum_error, static_cast<double>(std::abs(candidate.similarity-scores[candidate.target])));
            gpu_test_require(scores[candidate.target] >= cutoff - SCORE_TOL,
                             "Selected target is strictly below CPU cutoff tolerance");
            if (rank) {
                const auto previous = row[rank-1];
                gpu_test_require(previous.similarity > candidate.similarity ||
                    (previous.similarity == candidate.similarity && previous.target < candidate.target),
                    "Native output violates exact score/ID ordering");
            }
        }
        // Near-cutoff substitutions can follow float32 scoring differences;
        // omitting a clearly better CPU candidate cannot.
        for (std::size_t target = 0; target < scores.size(); ++target)
            if (scores[target] > cutoff + SCORE_TOL)
                gpu_test_require(selected.count(static_cast<int>(target)) == 1,
                                 "Omitted CPU candidate strictly above cutoff tolerance");
    }
    std::cout << "maximum CPU/GPU selected-score error=" << maximum_error << '\n';
}
Graph random_graph(int n, unsigned seed) {
    std::mt19937 rng(seed); std::vector<std::pair<int,int>> edges;
    for (int a = 0; a < n; ++a) for (int b = a+1; b < n; ++b)
        if (rng()%5 == 0) edges.emplace_back(a,b);
    return build_graph(n, edges);
}
void graph_case(const Graph& source, const Graph& target, int k, const GPUContext& context) {
    auto cpu_source = compute_features(source), cpu_target = compute_features(target);
    const auto gpu_raw_source = gpu_compute_features(source, context, nullptr, false);
    const auto gpu_raw_target = gpu_compute_features(target, context, nullptr, false);
    feature_parity(cpu_source, gpu_raw_source); feature_parity(cpu_target, gpu_raw_target);
    normalize_features_joint_zscore(cpu_source, cpu_target);
    GPUTopKOptions options; options.batch_candidates = 256; options.scratch_budget_bytes = 8196;
    const auto parallel = gpu_compute_candidates(source, target, k, context, true, GPUSelector::Parallel, options);
    feature_parity(cpu_source, parallel.feats_source); feature_parity(cpu_target, parallel.feats_target);
    scoring_parity(cpu_source, cpu_target, parallel.candidates, k);
    options.force_streaming_serial = true;
    const auto serial = gpu_compute_candidates(source, target, k, context, true, GPUSelector::SerialHeap, options);
    // Both paths use the identical GPU score_span producer; this is exact parity.
    exact_rows(serial.candidates, parallel.candidates);
    const auto legacy = gpu_compute_topk(parallel.feats_source, parallel.feats_target, k, context);
    scoring_parity(cpu_source, cpu_target, legacy, k);
    const auto indexed = gpu_compute_indexed_candidates(source, target, k, StructuralIndexConfig{},
        context, true, GPUSelector::Parallel, options);
    const auto indexed_serial = gpu_compute_indexed_candidates(source, target, k, StructuralIndexConfig{},
        context, true, GPUSelector::SerialHeap, options);
    exact_rows(indexed.matching.candidates, indexed_serial.matching.candidates);
    // Indexed pools need not equal exhaustive pools. The independent indexed
    // pool oracle is covered by test_gpu_topk_streaming and structural-index tests.
}
} // namespace
int main() {
    if (const int available = gpu_test_availability()) return available;
    try {
        GPUTestContext context;
        const auto toy = build_graph(6, {{0,1},{0,5},{1,2},{1,3},{1,4},{2,3},{2,5},{3,4}});
        for (int k : {1,3,6}) graph_case(toy, toy, k, context.value);
        graph_case(build_graph(4, {}), build_graph(17, {}), 8, context.value);
        for (unsigned seed : {173u,9173u})
            graph_case(random_graph(17, seed), random_graph(23, seed+1), 8, context.value);
        std::mt19937 rng(173); std::uniform_real_distribution<float> random(-1,1);
        std::vector<NodeFeatures> sources(4), targets(513);
        for (auto* features : {&sources, &targets}) for (auto& f : *features)
            f = NodeFeatures{random(rng),random(rng),random(rng),random(rng),random(rng),random(rng)};
        GPUTopKOptions options; options.batch_candidates = 256; options.scratch_budget_bytes = 8196;
        GPUSelectionInfo info;
        const auto result = gpu_compute_topk(sources, targets, 512, context.value,
            nullptr, nullptr, nullptr, nullptr, GPUSelector::Parallel, options, &info);
        gpu_test_require(info.used == GPUSelector::SerialHeap && info.fallback == GPUFallbackReason::UnsupportedK,
                         "K512/513-target fallback not preserved");
        scoring_parity(sources, targets, result, 512);
        std::cout << "test_cpu_gpu_parity: PASSED; exact selection and scoring-tolerance checks are separate\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "test_cpu_gpu_parity: " << e.what() << '\n'; return 1; }
}
