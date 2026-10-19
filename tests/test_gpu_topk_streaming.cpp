#include "gpu_matching.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>

namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class Function> void rejects(Function f) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Expected invalid_argument");
}
GPUSelectedPools reference(const GPUScoredPools& pools, int k) {
    GPUSelectedPools result;
    for (std::size_t source = 0; source + 1 < pools.offsets.size(); ++source) {
        const auto begin = pools.offsets[source], end = pools.offsets[source + 1];
        std::set<int> ids(pools.target_ids.begin() + begin, pools.target_ids.begin() + end);
        require(ids.size() == end - begin, "Input pool has duplicate targets");
        std::vector<std::size_t> order(end - begin);
        std::iota(order.begin(), order.end(), begin);
        std::sort(order.begin(), order.end(), [&](auto a, auto b) {
            return pools.scores[a] > pools.scores[b] ||
                (pools.scores[a] == pools.scores[b] && pools.target_ids[a] < pools.target_ids[b]);
        });
        order.resize(std::min<std::size_t>(order.size(), k));
        for (auto i : order) { result.target_ids.push_back(pools.target_ids[i]); result.scores.push_back(pools.scores[i]); }
        result.offsets.push_back(result.target_ids.size());
    }
    return result;
}
void same(const GPUSelectedPools& a, const GPUSelectedPools& b) {
    require(a.offsets == b.offsets && a.target_ids == b.target_ids && a.scores == b.scores,
            "Selection differs from independent full sort");
    for (std::size_t i = 0; i < a.scores.size(); ++i)
        require(std::signbit(a.scores[i]) == std::signbit(b.scores[i]), "Score sign changed");
}
void check_budget(const GPUSelectionInfo& info, const GPUTopKOptions& options, int k,
                  std::size_t additional_scratch_bytes = 0) {
    require(info.scratch_peak_bytes <= options.scratch_budget_bytes, "Scratch budget exceeded");
    require(info.batch_capacity <= options.batch_candidates && info.batch_capacity > 0,
            "Invalid effective batch capacity");
    const auto expected = info.used == GPUSelector::Parallel
        ? 2 * (info.batch_capacity + 256) * sizeof(GPUScoredCandidate) + sizeof(int)
        : (info.batch_capacity + static_cast<std::size_t>(k)) * sizeof(GPUScoredCandidate) + sizeof(int);
    require(info.scratch_peak_bytes == expected + additional_scratch_bytes,
            "Incomplete workspace accounting");
}
void scored_pool_tests(const GPUContext& context) {
    GPUScoredPools pools;
    std::mt19937 rng(173);
    for (int size : {0, 1, 255, 256, 257, 513, 8193, 0}) {
        std::vector<int> ids(size); std::iota(ids.begin(), ids.end(), 0);
        std::shuffle(ids.begin(), ids.end(), rng);
        for (int id : ids) {
            pools.target_ids.push_back(id);
            const int value = static_cast<int>(rng() % 17) - 8;
            pools.scores.push_back(value == 0 ? (id % 2 ? -0.0f : 0.0f) : value / 8.0f);
        }
        pools.offsets.push_back(pools.target_ids.size());
    }
    for (int k : {1, 64, 256, 257, 512}) {
        const auto expected = reference(pools, k);
        for (std::size_t budget : {8196u, 16388u, 1048576u}) {
            for (std::size_t batch : {256u, 768u, 4096u}) {
                GPUTopKOptions options; options.scratch_budget_bytes = budget; options.batch_candidates = batch;
                for (auto selector : {GPUSelector::SerialHeap, GPUSelector::Parallel}) {
                    GPUSelectionInfo info;
                    same(gpu_select_topk(pools, 10000, k, context, selector, &info, options), expected);
                    check_budget(info, options, k);
                    require(info.batches > 1, "Large pool did not span multiple processing batches");
                    if (selector == GPUSelector::Parallel && k > 256)
                        require(info.used == GPUSelector::SerialHeap && info.fallback == GPUFallbackReason::UnsupportedK,
                                "K fallback was not recorded");
                    else require(info.used == selector && info.fallback == GPUFallbackReason::None,
                                 "Unexpected selector fallback");
                }
            }
        }
    }
    GPUTopKOptions small; small.scratch_budget_bytes = 4096; small.batch_candidates = 256;
    GPUSelectionInfo info;
    same(gpu_select_topk(pools, 10000, 64, context, GPUSelector::Parallel, &info, small), reference(pools, 64));
    require(info.fallback == GPUFallbackReason::ScratchBudgetTooSmall && info.used == GPUSelector::SerialHeap,
            "Budget fallback not recorded");
    check_budget(info, small, 64);
    small.scratch_budget_bytes = 8;
    rejects([&] { gpu_select_topk(pools, 10000, 64, context, GPUSelector::Parallel, nullptr, small); });
    auto invalid = pools; invalid.scores.back() = std::numeric_limits<float>::infinity();
    rejects([&] { gpu_select_topk(invalid, 10000, 512, context, GPUSelector::Parallel); });
}
struct DeviceFeatures {
    NodeFeatures* p = nullptr;
    explicit DeviceFeatures(const std::vector<NodeFeatures>& features) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p), features.size() * sizeof(NodeFeatures)));
        auto error = cudaMemcpy(p, features.data(), features.size() * sizeof(NodeFeatures), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) { cudaFree(p); p = nullptr; CUDA_CHECK(error); }
    }
    ~DeviceFeatures() { if (p) cudaFree(p); }
    DeviceFeatures(const DeviceFeatures&) = delete;
    DeviceFeatures& operator=(const DeviceFeatures&) = delete;
};
NodeFeatures raw(float degree) { return NodeFeatures{degree, 0, degree, 0, 0, degree * degree}; }
void candidate_tests(const GPUContext& context) {
    constexpr int count = 1025, k = 64;
    std::vector<NodeFeatures> raw_targets, scoring_targets;
    for (int id = 0; id < count; ++id) {
        raw_targets.push_back(raw(id < 600 ? 1.0f : 100000.0f));
        // Exactly representable cosine scores -1, 0, +1; no tolerance needed.
        scoring_targets.push_back(NodeFeatures{static_cast<float>(id % 3 - 1), 0, 0, 0, 0, 0});
    }
    std::vector<NodeFeatures> raw_sources{raw(1), raw(100000), raw(1000)};
    std::vector<NodeFeatures> scoring_sources(3, NodeFeatures{1, 0, 0, 0, 0, 0});
    StructuralIndexConfig config; config.minimum_radius = config.maximum_radius = 0;
    const auto cpu_index = build_structural_index(raw_targets, config);
    DeviceFeatures d_targets(raw_targets), d_sources(raw_sources);
    auto index = build_gpu_structural_index(d_targets.p, count, config, context);
    auto queries = retrieve_gpu_structural_candidates(index, d_sources.p, raw_sources.size(), k, context);
    const auto decisions = download_structural_queries(queries);
    require(!decisions[0].used_exhaustive_fallback && !decisions[1].used_exhaustive_fallback &&
            decisions[2].used_exhaustive_fallback, "Test did not exercise both indexed retrieval cases");
    GPUTopKOptions options; options.scratch_budget_bytes = 8196; options.batch_candidates = 256;
    auto exhaustive = gpu_score_exhaustive_pools(scoring_sources, scoring_targets, k, context,
                                                GPUSelector::Parallel, options);
    auto indexed = gpu_score_indexed_pools(scoring_sources, scoring_targets, index, queries, k, context,
                                          GPUSelector::Parallel, options);
    check_budget(exhaustive.selection, options, k); check_budget(indexed.selection, options, k);
    for (std::size_t source = 0; source < raw_sources.size(); ++source) {
        auto pool = retrieve_structural_candidates(cpu_index, raw_sources[source], k);
        for (bool indexed_mode : {false, true}) {
            std::vector<int> ids = pool.target_ids;
            if (!indexed_mode) { ids.resize(count); std::iota(ids.begin(), ids.end(), 0); }
            std::set<int> unique(ids.begin(), ids.end());
            require(unique.size() == ids.size(), "Retrieved pool has duplicate IDs");
            std::sort(ids.begin(), ids.end(), [&](int a, int b) {
                return scoring_targets[a].degree > scoring_targets[b].degree ||
                    (scoring_targets[a].degree == scoring_targets[b].degree && a < b);
            });
            ids.resize(k);
            const auto& actual = indexed_mode ? indexed.candidates[source] : exhaustive.candidates[source];
            require(actual.size() == ids.size(), "Wrong candidate count");
            for (int rank = 0; rank < k; ++rank)
                require(actual[rank].target == ids[rank] && actual[rank].similarity == scoring_targets[ids[rank]].degree,
                        "Candidate path differs from full-sort reference");
        }
    }
    auto overflow_sources = scoring_sources;
    overflow_sources[0].degree = std::numeric_limits<float>::max();
    rejects([&] { gpu_score_exhaustive_pools(overflow_sources, scoring_targets, k, context,
                                            GPUSelector::Parallel, options); });
    auto nonfinite_sources = scoring_sources;
    nonfinite_sources[0].degree = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { gpu_score_indexed_pools(nonfinite_sources, scoring_targets, index, queries, k, context,
                                         GPUSelector::Parallel, options); });
    // Complete graph-level API integration, including normalization and forwarding.
    auto graph = build_graph(6, {{0,1}, {0,5}, {1,2}, {1,3}, {1,4}, {2,3}, {2,5}, {3,4}});
    // The resident scorer reserves its own validation flag in addition to
    // GPUStreamingTopK's workspace/error flag: 8196 + sizeof(int) bytes here.
    auto graph_options = options;
    graph_options.scratch_budget_bytes += sizeof(int);
    const auto all = gpu_compute_candidates(graph, graph, 6, context, true, GPUSelector::Parallel, graph_options);
    const auto indexed_graph = gpu_compute_indexed_candidates(graph, graph, 6, StructuralIndexConfig{},
        context, true, GPUSelector::Parallel, graph_options);
    const auto legacy = gpu_compute_topk(all.feats_source, all.feats_target, 6, context);
    for (const auto* result : {&all, &indexed_graph.matching}) {
        require(result->selection.requested == GPUSelector::Parallel,
                "Graph pipeline dropped requested selector");
        require(result->selection.used == GPUSelector::Parallel &&
                result->selection.fallback == GPUFallbackReason::None,
                "Graph pipeline unexpectedly fell back with sufficient scratch");
        check_budget(result->selection, graph_options, 6, sizeof(int));
    }
    // Keeping the original tighter budget must trigger a recorded fallback,
    // rather than exceed the cap or silently discard the request.
    const auto bounded = gpu_compute_candidates(graph, graph, 6, context, true,
        GPUSelector::Parallel, options);
    const auto indexed_bounded = gpu_compute_indexed_candidates(graph, graph, 6,
        StructuralIndexConfig{}, context, true, GPUSelector::Parallel, options);
    for (const auto* result : {&bounded, &indexed_bounded.matching}) {
        require(result->selection.requested == GPUSelector::Parallel &&
                result->selection.used == GPUSelector::SerialHeap &&
                result->selection.fallback == GPUFallbackReason::ScratchBudgetTooSmall,
                "Graph pipeline failed to record its scratch-budget fallback");
        check_budget(result->selection, options, 6, sizeof(int));
    }
    for (std::size_t source = 0; source < legacy.size(); ++source) {
        require(indexed_graph.matching.candidates[source].size() == legacy[source].size(), "Indexed graph row size");
        for (std::size_t rank = 0; rank < legacy[source].size(); ++rank) {
            require(all.candidates[source][rank].target == legacy[source][rank].target,
                    "Exhaustive graph path differs from preserved reference");
            require(indexed_graph.matching.candidates[source][rank].target == legacy[source][rank].target,
                    "Indexed graph path differs when its pool is exhaustive");
            require(bounded.candidates[source][rank].target == legacy[source][rank].target &&
                    indexed_bounded.matching.candidates[source][rank].target == legacy[source][rank].target,
                    "Graph budget fallback differs from preserved reference");
        }
    }
}
} // namespace
int main() {
    int devices = 0; const auto error = cudaGetDeviceCount(&devices);
    if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
        (error == cudaSuccess && devices == 0)) { std::cout << "SKIPPED (no usable GPU)\n"; return 77; }
    if (error != cudaSuccess) { std::cerr << cudaGetErrorString(error) << '\n'; return 1; }
    GPUContext context;
    try {
        CUDA_CHECK(cudaSetDevice(0)); CUDA_CHECK(cudaStreamCreate(&context.stream));
        scored_pool_tests(context); candidate_tests(context);
        CUDA_CHECK(cudaStreamDestroy(context.stream)); context.stream = nullptr;
        std::cout << "test_gpu_topk_streaming: ALL PASSED\n"; return 0;
    } catch (const std::exception& e) {
        if (context.stream) cudaStreamDestroy(context.stream);
        std::cerr << "test_gpu_topk_streaming: FAILED: " << e.what() << '\n'; return 1;
    }
}
