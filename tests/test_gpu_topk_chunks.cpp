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

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
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
    std::mt19937 generator(173);

    const std::vector<int> sizes = {
        0, 1, 2, 255, 256, 257, 512, 513, 1025
    };

    for (int size : sizes) {
        std::vector<int> ids(size);
        std::iota(ids.begin(), ids.end(), 0);
        std::shuffle(ids.begin(), ids.end(), generator);

        for (int target : ids) {
            float score;

            if (mode == 0) {
                // Repeated negative, zero, and positive scores.
                const int value = static_cast<int>(generator() % 17);
                score = static_cast<float>(value - 8) / 8.0f;
            } else if (mode == 1) {
                // Every score ties, so target IDs determine all ranks.
                score = -0.5f;
            } else {
                // Signed zeros must tie while retaining their input sign.
                score = (target % 2 == 0) ? 0.0f : -0.0f;
            }

            pools.target_ids.push_back(target);
            pools.scores.push_back(score);
        }

        pools.offsets.push_back(pools.target_ids.size());
    }

    return pools;
}

void compare_chunks(
    const GPUScoredPools& pools,
    int k,
    const GPUContext& context)
{
    // Verify the producer's unique-target invariant.
    for (std::size_t source = 0;
         source + 1 < pools.offsets.size();
         ++source) {
        const auto begin = pools.offsets[source];
        const auto end = pools.offsets[source + 1];

        std::set<int> unique(
            pools.target_ids.begin() + begin,
            pools.target_ids.begin() + end);

        require(unique.size() == end - begin,
                "test input contains duplicate targets");
    }

    const auto actual =
        gpu_select_chunk_topk(pools, 2048, k, context);

    require(actual.source_chunk_offsets.size() == pools.offsets.size(),
            "wrong source offset count");
    require(actual.source_chunk_offsets.front() == 0,
            "source chunk offsets must start at zero");
    require(actual.chunks.offsets.front() == 0,
            "partial offsets must start at zero");

    std::size_t chunk = 0;
    std::size_t output = 0;

    for (std::size_t source = 0;
         source + 1 < pools.offsets.size();
         ++source) {
        require(actual.source_chunk_offsets[source] == chunk,
                "wrong source chunk start");

        const std::size_t end = pools.offsets[source + 1];

        for (std::size_t begin = pools.offsets[source]; begin < end;) {
            const std::size_t length =
                std::min<std::size_t>(GPU_TOPK_CHUNK_SIZE, end - begin);

            // Full-sort reference over this exact chunk's input candidates.
            std::vector<std::size_t> order(length);
            std::iota(order.begin(), order.end(), begin);

            std::sort(
                order.begin(), order.end(),
                [&](std::size_t a, std::size_t b) {
                    if (pools.scores[a] != pools.scores[b]) {
                        return pools.scores[a] > pools.scores[b];
                    }
                    return pools.target_ids[a] < pools.target_ids[b];
                });

            const std::size_t keep =
                std::min<std::size_t>(static_cast<std::size_t>(k), length);

            require(chunk + 1 < actual.chunks.offsets.size(),
                    "missing chunk offsets");
            require(actual.chunks.offsets[chunk] == output,
                    "wrong chunk output start");
            require(actual.chunks.offsets[chunk + 1] == output + keep,
                    "wrong partial length");

            for (std::size_t rank = 0; rank < keep; ++rank) {
                const auto expected = order[rank];
                const auto position = output + rank;

                require(position < actual.chunks.target_ids.size(),
                        "missing output target");
                require(position < actual.chunks.scores.size(),
                        "missing output score");
                require(
                    actual.chunks.target_ids[position] ==
                        pools.target_ids[expected],
                    "target/order differs from CPU full sort");
                require(
                    actual.chunks.scores[position] == pools.scores[expected],
                    "score differs from input");
                require(
                    std::signbit(actual.chunks.scores[position]) ==
                        std::signbit(pools.scores[expected]),
                    "score sign was changed");
            }

            output += keep;
            ++chunk;
            begin += length;
        }

        require(actual.source_chunk_offsets[source + 1] == chunk,
                "wrong source chunk end");
    }

    require(actual.chunks.offsets.size() == chunk + 1,
            "extra chunk offsets");
    require(actual.chunks.target_ids.size() == output,
            "wrong output target count");
    require(actual.chunks.scores.size() == output,
            "wrong output score count");
}

void check_rejections(const GPUContext& context) {
    GPUScoredPools valid;
    valid.offsets = {0, 1};
    valid.target_ids = {0};
    valid.scores = {0.5f};

    rejects([&] {
        gpu_select_chunk_topk(valid, 1, 0, context);
    });
    rejects([&] {
        gpu_select_chunk_topk(valid, 1, -1, context);
    });
    rejects([&] {
        gpu_select_chunk_topk(valid, -1, 1, context);
    });

    for (float invalid : {
             std::numeric_limits<float>::quiet_NaN(),
             std::numeric_limits<float>::infinity(),
             -std::numeric_limits<float>::infinity()}) {
        auto bad = valid;
        bad.scores[0] = invalid;

        rejects([&] {
            gpu_select_chunk_topk(bad, 1, 1, context);
        });
    }

    for (int invalid : {-1, 1}) {
        auto bad = valid;
        bad.target_ids[0] = invalid;

        rejects([&] {
            gpu_select_chunk_topk(bad, 1, 1, context);
        });
    }

    for (const std::vector<std::size_t>& offsets :
         std::vector<std::vector<std::size_t>>{
             {}, {1, 1}, {0, 2}, {0, 1, 0, 1}}) {
        auto bad = valid;
        bad.offsets = offsets;

        rejects([&] {
            gpu_select_chunk_topk(bad, 1, 1, context);
        });
    }

    auto mismatched = valid;
    mismatched.scores.clear();

    rejects([&] {
        gpu_select_chunk_topk(mismatched, 1, 1, context);
    });
}

} // namespace

int main() {
    int device_count = 0;
    const auto availability = cudaGetDeviceCount(&device_count);

    if (availability == cudaErrorNoDevice ||
        availability == cudaErrorInsufficientDriver ||
        (availability == cudaSuccess && device_count == 0)) {
        std::cout << "test_gpu_topk_chunks: SKIPPED (no usable GPU)\n";
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

        // Zero sources.
        GPUScoredPools empty;
        compare_chunks(empty, 1, context);

        // Several sources, all empty.
        GPUScoredPools empty_rows;
        empty_rows.offsets = {0, 0, 0, 0};
        compare_chunks(empty_rows, 1, context);

        for (int mode = 0; mode < 3; ++mode) {
            const auto pools = make_pools(mode);

            for (int k : {1, 2, 64, 255, 256, 257, 512}) {
                compare_chunks(pools, k, context);
            }
        }

        check_rejections(context);

        CUDA_CHECK(cudaStreamDestroy(context.stream));
        context.stream = nullptr;

        std::cout << "test_gpu_topk_chunks: ALL PASSED\n";
        return 0;
    } catch (const std::exception& error) {
        if (context.stream) {
            cudaStreamDestroy(context.stream);
        }

        std::cerr << "test_gpu_topk_chunks: FAILED: "
                  << error.what() << '\n';
        return 1;
    }
}