#include "gpu_structural_index.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace std;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw runtime_error(message);
}

void checked(cudaError_t status) {
    if (status != cudaSuccess)
        throw runtime_error(cudaGetErrorString(status));
}

NodeFeatures feature(float degree, float mean) {
    NodeFeatures value{};
    value.degree = degree;
    value.avg_neighbor_degree = mean;
    return value;
}

struct DeviceFeatures {
    NodeFeatures* data = nullptr;

    explicit DeviceFeatures(const vector<NodeFeatures>& features) {
        if (features.empty()) return;

        checked(cudaMalloc(
            reinterpret_cast<void**>(&data),
            features.size() * sizeof(NodeFeatures)));

        const cudaError_t status = cudaMemcpy(
            data, features.data(),
            features.size() * sizeof(NodeFeatures),
            cudaMemcpyHostToDevice);

        if (status != cudaSuccess) {
            cudaFree(data);
            data = nullptr;
            checked(status);
        }
    }

    ~DeviceFeatures() {
        if (data) cudaFree(data);
    }

    DeviceFeatures(const DeviceFeatures&) = delete;
    DeviceFeatures& operator=(const DeviceFeatures&) = delete;
};

void compare(
    const StructuralIndex& cpu,
    const StructuralIndex& gpu
) {
    require(cpu.target_count == gpu.target_count, "target count mismatch");
    require(cpu.bucket_keys == gpu.bucket_keys, "bucket keys mismatch");
    require(cpu.bucket_offsets == gpu.bucket_offsets, "offset mismatch");
    require(cpu.target_ids == gpu.target_ids, "target membership/order mismatch");
}

void parity(
    const vector<NodeFeatures>& features,
    const StructuralIndexConfig& config,
    const GPUContext& context
) {
    const auto cpu = build_structural_index(features, config);
    DeviceFeatures input(features);

    auto gpu = build_gpu_structural_index(
        input.data, features.size(), config, context);

    compare(cpu, download_structural_index(gpu));

    // Verify ownership transfer and replacement of an existing allocation.
    GPUStructuralIndex moved(std::move(gpu));
    require(gpu.d_bucket_offsets == nullptr, "move constructor kept ownership");

    auto destination = build_gpu_structural_index(
        nullptr, 0, config, context);

    destination = std::move(moved);

    require(moved.d_bucket_offsets == nullptr, "move assignment kept ownership");
    compare(cpu, download_structural_index(destination));
}

template<class Exception, class Function>
void rejects(Function function) {
    bool rejected = false;

    try {
        function();
    } catch (const Exception&) {
        rejected = true;
    }

    require(rejected, "expected exception was not thrown");
}

} // namespace

vector<int> expand_query_for_test(
    const StructuralIndex& index,
    const GPUStructuralQuery& query
) {
    vector<int> targets;

    if (query.used_exhaustive_fallback) {
        for (size_t target = 0; target < index.target_count; ++target) {
            targets.push_back(static_cast<int>(target));
        }

        return targets;
    }

    // Reconstruct the GPU query's implied traversal order.
    for (int64_t radius = 0;
         radius <= query.final_radius;
         ++radius) {
        for (size_t bucket = 0;
             bucket < index.bucket_keys.size();
             ++bucket) {
            const auto key = index.bucket_keys[bucket];

            const auto dx =
                key.degree_bin >= query.source_key.degree_bin
                    ? key.degree_bin - query.source_key.degree_bin
                    : query.source_key.degree_bin - key.degree_bin;

            const auto dy =
                key.neighbor_bin >= query.source_key.neighbor_bin
                    ? key.neighbor_bin - query.source_key.neighbor_bin
                    : query.source_key.neighbor_bin - key.neighbor_bin;

            if ((dx > dy ? dx : dy) != radius) continue;

            for (size_t entry = index.bucket_offsets[bucket];
                 entry < index.bucket_offsets[bucket + 1];
                 ++entry) {
                targets.push_back(index.target_ids[entry]);
            }
        }
    }

    return targets;
}

void retrieval_parity(
    const vector<NodeFeatures>& targets,
    const vector<NodeFeatures>& sources,
    const StructuralIndexConfig& config,
    int k,
    const GPUContext& context
) {
    const auto cpu_index = build_structural_index(targets, config);

    DeviceFeatures device_targets(targets);
    DeviceFeatures device_sources(sources);

    auto gpu_index = build_gpu_structural_index(
        device_targets.data, targets.size(), config, context);

    const auto downloaded_index =
        download_structural_index(gpu_index);

    compare(cpu_index, downloaded_index);

    auto gpu_queries = retrieve_gpu_structural_candidates(
        gpu_index,
        device_sources.data,
        sources.size(),
        k,
        context);

    const auto queries = download_structural_queries(gpu_queries);

    require(queries.size() == sources.size(), "query count mismatch");

    for (size_t source = 0; source < sources.size(); ++source) {
        const auto expected = retrieve_structural_candidates(
            cpu_index, sources[source], k);

        const auto& actual = queries[source];

        require(
            actual.source_key ==
                quantize_structural_features(sources[source], config),
            "source key mismatch");

        require(
            actual.desired_pool_size == expected.stats.desired_pool_size,
            "desired pool mismatch");

        require(
            actual.bucket_pool_size == expected.stats.bucket_pool_size,
            "bucket pool mismatch");

        require(
            actual.final_radius == expected.stats.final_radius,
            "final radius mismatch");

        require(
            actual.used_exhaustive_fallback ==
                expected.stats.used_exhaustive_fallback,
            "fallback mismatch");

        require(
            actual.effective_pool_size == expected.target_ids.size(),
            "effective pool mismatch");

        require(
            expand_query_for_test(downloaded_index, actual) ==
                expected.target_ids,
            "retrieved target order/membership mismatch");
    }

    // Check query ownership transfer.
    GPUStructuralQueries moved(std::move(gpu_queries));

    require(
        gpu_queries.d_queries == nullptr,
        "moved-from queries retained ownership");

    require(
        download_structural_queries(moved).size() == sources.size(),
        "moved queries lost records");
}

int main() {
    int device_count = 0;

    if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
        device_count == 0) {
        cout << "test_gpu_structural_index: SKIPPED (no CUDA device)\n";
        return 77;
    }

    try {
        checked(cudaSetDevice(0));

        GPUContext context;
        context.device_id = 0;
        context.stream = nullptr; // Default stream for this test.

        StructuralIndexConfig config;

        parity({}, config, context);
        parity({feature(0, 0)}, config, context);
        parity(vector<NodeFeatures>(100, feature(2, 4)), config, context);

        parity({
            feature(20, 3),
            feature(0, 0),
            feature(20, 3),
            feature(1, 10),
            feature(0, 0)
        }, config, context);

        // Boundary plus adjacent representable raw float inputs.
        auto boundary_config = config;
        boundary_config.degree_bin_width = log1p(1.0);

        parity({
            feature(nextafter(1.0f, 0.0f), 0),
            feature(1.0f, 0),
            feature(nextafter(1.0f, 2.0f), 0)
        }, boundary_config, context);

        // Wider corpus: positives, zeros, varied bucket occupancy.
        mt19937 rng(20261016);
        uniform_real_distribution<float> value(0.0f, 1000.0f);

        for (int trial = 0; trial < 100; ++trial) {
            vector<NodeFeatures> features;
            const int count = 1 + trial * 3;

            for (int i = 0; i < count; ++i)
                features.push_back(feature(value(rng), value(rng)));

            parity(features, config, context);
        }

        rejects<invalid_argument>([&] {
            auto invalid = config;
            invalid.degree_bin_width = 0;
            build_gpu_structural_index(nullptr, 0, invalid, context);
        });

        rejects<invalid_argument>([&] {
            build_gpu_structural_index(nullptr, 1, config, context);
        });

        for (float invalid_value : {
            -1.0f,
            numeric_limits<float>::infinity(),
            numeric_limits<float>::quiet_NaN()
        }) {
            DeviceFeatures input({feature(invalid_value, 0)});

            rejects<invalid_argument>([&] {
                build_gpu_structural_index(input.data, 1, config, context);
            });
        }

        DeviceFeatures input({feature(1, 0)});
        auto overflow_config = config;
        overflow_config.degree_bin_width =
            numeric_limits<double>::denorm_min();

        rejects<overflow_error>([&] {
            build_gpu_structural_index(
                input.data, 1, overflow_config, context);
        });
        //Own-bucket retrieval, minimum-radius behaviour, oversized bucket, 
        //and exhaustive fallback behaviour are all exercised in this test.
        retrieval_parity(
            vector<NodeFeatures>(100, feature(0, 0)),
            {feature(0,0), feature(100000, 100000)},
            config,
            3,
            context);
        
        // Complete-shell behavior: both neighboring occupied buckets must remain,
        // even if the first already provides enough targets.
        {
            auto c = config;
            c.degree_bin_width = 1.0;
            c.neighbor_bin_width = 1.0;
            c.minimum_radius = 0;
            c.maximum_radius = 2;
            c.pool_multiplier = 1;

            retrieval_parity(
                {feature(2, 0), feature(0, 2)},
                {feature(0, 0)},
                c,
                1,
                context);
        }

        // Pool has enough for K, but not multiplier*K:
        // fallback must follow the documented desired-pool rule.
        {
            auto c = config;
            c.minimum_radius = 0;
            c.maximum_radius = 0;
            c.pool_multiplier = 4;

            retrieval_parity(
                {
                    feature(0, 0),
                    feature(0, 0),
                    feature(1000, 1000),
                    feature(2000, 2000)
                },
                {feature(0, 0)},
                c,
                1,
                context);
        }

        // Deterministic random source/target retrieval comparisons.
        {
            mt19937 retrieval_rng(20261017);
            uniform_real_distribution<float> coordinate(0.0f, 100.0f);

            for (int trial = 0; trial < 50; ++trial) {
                vector<NodeFeatures> targets;
                vector<NodeFeatures> sources;

                for (int i = 0; i < 30; ++i) {
                    targets.push_back(feature(
                        coordinate(retrieval_rng),
                        coordinate(retrieval_rng)));
                }

                for (int i = 0; i < 10; ++i) {
                    sources.push_back(feature(
                        coordinate(retrieval_rng),
                        coordinate(retrieval_rng)));
                }

                auto c = config;
                c.minimum_radius = trial % 2;
                c.maximum_radius = 3;
                c.pool_multiplier = 1 + trial % 4;

                retrieval_parity(
                    targets, sources, c, 1 + trial % 5, context);
            }
        }

        // An empty source batch produces no queries.
        retrieval_parity(
            {feature(0, 0)}, {}, config, 1, context);

        // Reject invalid source values and invalid K.
        {
            DeviceFeatures targets({feature(0, 0)});
            DeviceFeatures sources({feature(-1, 0)});

            auto index = build_gpu_structural_index(
                targets.data, 1, config, context);

            rejects<invalid_argument>([&] {
                retrieve_gpu_structural_candidates(
                    index, sources.data, 1, 1, context);
            });

            rejects<invalid_argument>([&] {
                retrieve_gpu_structural_candidates(
                    index, sources.data, 1, 0, context);
            });

            rejects<invalid_argument>([&] {
                retrieve_gpu_structural_candidates(
                    index, sources.data, 1, 2, context);
            });
        }



        cout << "test_gpu_structural_index: ALL PASSED\n";
        return 0;
    } catch (const exception& error) {
        cerr << "test_gpu_structural_index: FAIL: "
             << error.what() << '\n';
        return 1;
    }
}