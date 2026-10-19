#include "gpu_topk.hpp"
#include "gpu_test_utils.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <set>

namespace {
using Pools = GPUScoredPools;
std::uint32_t bits(float value) {
    std::uint32_t result; std::memcpy(&result, &value, sizeof(result)); return result;
}
GPUSelectedPools oracle(const Pools& p, int k) {
    GPUSelectedPools out;
    for (std::size_t row = 0; row + 1 < p.offsets.size(); ++row) {
        const auto begin = p.offsets[row], end = p.offsets[row + 1];
        const std::set<int> unique(p.target_ids.begin() + begin, p.target_ids.begin() + end);
        gpu_test_require(unique.size() == end - begin, "Producer violated unique-ID precondition");
        std::vector<std::size_t> indices(end - begin);
        std::iota(indices.begin(), indices.end(), begin);
        std::sort(indices.begin(), indices.end(), [&](auto a, auto b) {
            if (p.scores[a] != p.scores[b]) return p.scores[a] > p.scores[b];
            return p.target_ids[a] < p.target_ids[b];
        });
        indices.resize(std::min<std::size_t>(indices.size(), k));
        for (auto i : indices) { out.target_ids.push_back(p.target_ids[i]); out.scores.push_back(p.scores[i]); }
        out.offsets.push_back(out.target_ids.size());
    }
    return out;
}
void check(const Pools& p, int targets, int k, const GPUContext& context) {
    const auto expected = oracle(p, k);
    for (auto selector : {GPUSelector::SerialHeap, GPUSelector::Parallel}) {
        GPUTopKOptions options; options.batch_candidates = 768; options.scratch_budget_bytes = 16388;
        GPUSelectionInfo info;
        const auto actual = gpu_select_topk(p, targets, k, context, selector, &info, options);
        gpu_test_require(actual.offsets == expected.offsets && actual.target_ids == expected.target_ids,
                         "Exact IDs/order differ from full-sort oracle over identical scores");
        gpu_test_require(actual.scores.size() == expected.scores.size(), "Wrong score count");
        for (std::size_t i = 0; i < actual.scores.size(); ++i)
            gpu_test_require(bits(actual.scores[i]) == bits(expected.scores[i]), "Input score bits changed");
        gpu_test_require(info.scratch_peak_bytes <= options.scratch_budget_bytes, "Budget exceeded");
        gpu_test_require(info.requested == selector, "Wrong requested selector diagnostic");
        if (selector == GPUSelector::Parallel && k > GPU_TOPK_PARALLEL_MAX_K) {
            gpu_test_require(info.used == GPUSelector::SerialHeap && info.fallback == GPUFallbackReason::UnsupportedK,
                             "Unsupported K was not recorded as serial fallback");
        } else {
            gpu_test_require(info.used == selector && info.fallback == GPUFallbackReason::None,
                             "Unexpected fallback in correctness fixture");
        }
    }
}
template<class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    gpu_test_require(rejected, "Expected invalid_argument");
}
void invalid_inputs(const GPUContext& context) {
    Pools good; good.offsets = {0, 2}; good.target_ids = {0, 1}; good.scores = {1, 0};
    for (auto selector : {GPUSelector::SerialHeap, GPUSelector::Parallel}) {
        for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
             std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}) {
            auto p = good; p.scores[1] = invalid;
            // Reject invalid scores even if they would lose selection or trigger K fallback.
            for (int k : {1, 512}) rejects([&] { gpu_select_topk(p, 2, k, context, selector); });
        }
        for (int k : {0, -1}) rejects([&] { gpu_select_topk(good, 2, k, context, selector); });
        for (int id : {-1, 2}) { auto p = good; p.target_ids[1] = id;
            rejects([&] { gpu_select_topk(p, 2, 1, context, selector); }); }
        for (const auto& offsets : std::vector<std::vector<std::size_t>>{
             {}, {1,2}, {0,1}, {0,2,1,2}, {0,std::numeric_limits<std::size_t>::max(),2}}) {
            auto p = good; p.offsets = offsets;
            rejects([&] { gpu_select_topk(p, 2, 1, context, selector); });
        }
        auto p = good; p.scores.pop_back();
        rejects([&] { gpu_select_topk(p, 2, 1, context, selector); });
    }
}
} // namespace
int main() {
    if (const int available = gpu_test_availability()) return available;
    try {
        GPUTestContext context;
        Pools explicit_pool;
        explicit_pool.offsets = {0, 0, 10, 10};
        explicit_pool.target_ids = {9, 1, 7, 2, 8, 0, 6, 4, 3, 5};
        explicit_pool.scores = {1, 1, std::nextafter(1.0f, 0.0f), -0.0f, 0.0f, -1,
            std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
            std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::denorm_min()};
        for (int k : {1,2,7,10,256,257,512}) check(explicit_pool, 10, k, context.value);
        Pools large_ids; large_ids.offsets = {0,2};
        large_ids.target_ids = {std::numeric_limits<int>::max()-1, 0}; large_ids.scores = {0,0};
        check(large_ids, std::numeric_limits<int>::max(), 1, context.value);
        Pools empty; check(empty, 0, 512, context.value);
        for (unsigned seed : {173u, 20261018u, 9173u}) {
            std::mt19937 rng(seed); Pools p;
            for (int size : {0,1,255,256,257,513,1025,0}) {
                std::vector<int> ids(size); std::iota(ids.begin(), ids.end(), 0);
                std::shuffle(ids.begin(), ids.end(), rng);
                for (int id : ids) {
                    p.target_ids.push_back(id);
                    p.scores.push_back((static_cast<int>(rng()%33)-16)/16.0f);
                }
                p.offsets.push_back(p.target_ids.size());
            }
            for (int k : {1,2,64,256,257,512}) check(p, 2048, k, context.value);
            // All exact ties exercise ID ordering across chunks and batches.
            std::fill(p.scores.begin(), p.scores.end(), -0.5f);
            check(p, 2048, 64, context.value);
        }
        invalid_inputs(context.value);
        std::cout << "test_gpu_topk: exact selection oracle PASSED (scores supplied, no feature extraction)\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "test_gpu_topk: " << e.what() << '\n'; return 1; }
}
