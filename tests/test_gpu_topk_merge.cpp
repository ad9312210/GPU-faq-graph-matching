#include "gpu_topk.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int NUM_TARGETS = 10000;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void rejects(Function function) {
    bool rejected = false;

    try {
        function();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }

    require(rejected, "expected invalid_argument");
}

GPUScoredPools make_pools(int mode) {
    GPUScoredPools pools;
    std::mt19937 generator(173 + mode);

    // Include odd chunk counts, many merge levels, and empty rows
    // interleaved with nonempty rows.
    const std::vector<int> sizes = {
        0, 1, 2, 255, 256, 257, 0,
        512, 513, 769, 1025, 4097, 8193, 0
    };

    for (int size : sizes) {
        std::vector<int> ids(size);
        std::iota(ids.begin(), ids.end(), 0);
        std::shuffle(ids.begin(), ids.end(), generator);

        for (int target : ids) {
            float score;

            if (mode == 0) {
                const int value = static_cast<int>(generator() % 33);
                score = static_cast<float>(value - 16) / 16.0f;
            } else if (mode == 1) {
                score = -0.5f;
            } else if (mode == 2) {
                score = target % 2 == 0 ? 0.0f : -0.0f;
            } else {
                // Exercise finite extremes without doing score arithmetic.
                switch (target % 5) {
                case 0:
                    score = std::numeric_limits<float>::max();
                    break;
                case 1:
                    score = -std::numeric_limits<float>::max();
                    break;
                case 2:
                    score = 0.125f;
                    break;
                case 3:
                    score = -0.125f;
                    break;
                default:
                    score = 0.0f;
                    break;
                }
            }

            pools.target_ids.push_back(target);
            pools.scores.push_back(score);
        }

        pools.offsets.push_back(pools.target_ids.size());
    }

    return pools;
}

// Independent reference: full-sort each complete source pool,
// rather than reproducing the GPU chunk or merge algorithm.
GPUSelectedPools full_sort_reference(
    const GPUScoredPools& pools,
    int k)
{
    GPUSelectedPools output;

    for (std::size_t source = 0;
         source + 1 < pools.offsets.size();
         ++source) {
        const auto begin = pools.offsets[source];
        const auto end = pools.offsets[source + 1];

        std::set<int> unique(
            pools.target_ids.begin() + begin,
            pools.target_ids.begin() + end);
        require(unique.size() == end - begin,
                "input target IDs are not unique");

        std::vector<std::size_t> order(end - begin);
        std::iota(order.begin(), order.end(), begin);

        std::sort(
            order.begin(), order.end(),
            [&](std::size_t a, std::size_t b) {
                if (pools.scores[a] != pools.scores[b]) {
                    return pools.scores[a] > pools.scores[b];
                }
                return pools.target_ids[a] < pools.target_ids[b];
            });

        order.resize(std::min<std::size_t>(
            static_cast<std::size_t>(k), order.size()));

        for (auto index : order) {
            output.target_ids.push_back(pools.target_ids[index]);
            output.scores.push_back(pools.scores[index]);
        }

        output.offsets.push_back(output.target_ids.size());
    }

    return output;
}

void compare(
    const GPUScoredPools& pools,
    int k,
    const GPUContext& context)
{
    const auto expected = full_sort_reference(pools, k);

    for (auto selector :
         {GPUSelector::SerialHeap, GPUSelector::Parallel}) {
        GPUSelectionInfo info;

        const auto actual = gpu_select_topk(
            pools, NUM_TARGETS, k, context, selector, &info);

        require(actual.offsets == expected.offsets,
                "final source offsets differ from CPU reference");
        require(actual.target_ids == expected.target_ids,
                "final target IDs/order differ from CPU reference");
        require(actual.scores.size() == expected.scores.size(),
                "wrong final score count");

        for (std::size_t i = 0; i < expected.scores.size(); ++i) {
            require(actual.scores[i] == expected.scores[i],
                    "final score differs from input");
            require(
                std::signbit(actual.scores[i]) ==
                    std::signbit(expected.scores[i]),
                "score sign changed");
        }

        require(info.requested == selector,
                "requested selector was not recorded");

        const bool unsupported =
            selector == GPUSelector::Parallel &&
            k > GPU_TOPK_PARALLEL_MAX_K;

        require(
            info.used ==
                (unsupported ? GPUSelector::SerialHeap : selector),
            "unexpected selector used");

        require(
            info.fallback ==
                (unsupported
                     ? GPUFallbackReason::UnsupportedK
                     : GPUFallbackReason::None),
            "unexpected fallback reason");
    }

    // Existing four-argument calls must still select the serial default.
    const auto default_result =
        gpu_select_topk(pools, NUM_TARGETS, k, context);

    require(default_result.offsets == expected.offsets,
            "default selector returned wrong offsets");
    require(default_result.target_ids == expected.target_ids,
            "default selector returned wrong targets");
}

void check_invalid_inputs(const GPUContext& context) {
    GPUScoredPools valid;
    valid.offsets = {0, 1};
    valid.target_ids = {0};
    valid.scores = {0.5f};

    for (auto selector :
         {GPUSelector::SerialHeap, GPUSelector::Parallel}) {
        rejects([&] {
            gpu_select_topk(valid, 1, 0, context, selector);
        });
        rejects([&] {
            gpu_select_topk(valid, 1, -1, context, selector);
        });
        rejects([&] {
            gpu_select_topk(valid, -1, 1, context, selector);
        });

        for (float value : {
                 std::numeric_limits<float>::quiet_NaN(),
                 std::numeric_limits<float>::infinity(),
                 -std::numeric_limits<float>::infinity()}) {
            auto bad = valid;
            bad.scores[0] = value;

            // Invalid scores must be rejected even when K would fall back.
            rejects([&] {
                gpu_select_topk(bad, 1, 512, context, selector);
            });
        }

        for (int target : {-1, 1}) {
            auto bad = valid;
            bad.target_ids[0] = target;
            rejects([&] {
                gpu_select_topk(bad, 1, 1, context, selector);
            });
        }

        for (const auto& offsets :
             std::vector<std::vector<std::size_t>>{
                 {}, {1, 1}, {0, 2}, {0, 1, 0, 1}}) {
            auto bad = valid;
            bad.offsets = offsets;
            rejects([&] {
                gpu_select_topk(bad, 1, 1, context, selector);
            });
        }

        auto bad = valid;
        bad.scores.clear();
        rejects([&] {
            gpu_select_topk(bad, 1, 1, context, selector);
        });
    }

    rejects([&] {
        gpu_select_topk(
            valid, 1, 1, context, static_cast<GPUSelector>(99));
    });
}

} // namespace

int main() {
    int device_count = 0;
    const auto availability = cudaGetDeviceCount(&device_count);

    if (availability == cudaErrorNoDevice ||
        availability == cudaErrorInsufficientDriver ||
        (availability == cudaSuccess && device_count == 0)) {
        std::cout << "test_gpu_topk_merge: SKIPPED (no usable GPU)\n";
        return 77;
    }

    if (availability != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: "
                  << cudaGetErrorString(availability) << '\n';
        return 1;
    }

    GPUContext context;

    try {
        CUDA_CHECK(cudaSetDevice(context.device_id));
        CUDA_CHECK(cudaStreamCreate(&context.stream));

        GPUScoredPools empty;
        compare(empty, 1, context);

        GPUScoredPools empty_rows;
        empty_rows.offsets = {0, 0, 0, 0};
        compare(empty_rows, 256, context);

        for (int mode = 0; mode < 4; ++mode) {
            const auto pools = make_pools(mode);

            for (int k : {1, 2, 63, 64, 127, 128, 255, 256, 257, 512}) {
                compare(pools, k, context);
            }
        }

        check_invalid_inputs(context);

        CUDA_CHECK(cudaStreamDestroy(context.stream));
        context.stream = nullptr;

        std::cout << "test_gpu_topk_merge: ALL PASSED\n";
        return 0;
    } catch (const std::exception& error) {
        if (context.stream) cudaStreamDestroy(context.stream);

        std::cerr << "test_gpu_topk_merge: FAILED: "
                  << error.what() << '\n';
        return 1;
    }
}