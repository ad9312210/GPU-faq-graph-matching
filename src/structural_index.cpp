#include "structural_index.hpp"

#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
using namespace std;

//implement config validation and quantization of the structural features

void validate_structural_index_config(
    const StructuralIndexConfig& config
) {
    if (!isfinite(config.degree_bin_width) ||
        config.degree_bin_width <= 0.0) {
        throw invalid_argument(
            "degree_bin_width must be finite and positive");
    }

    if (!isfinite(config.neighbor_bin_width) ||
        config.neighbor_bin_width <= 0.0) {
        throw invalid_argument(
            "neighbor_bin_width must be finite and positive");
    }

    if (config.minimum_radius < 0 ||
        config.maximum_radius < config.minimum_radius) {
        throw invalid_argument(
            "require 0 <= minimum_radius <= maximum_radius");
    }

    if (config.pool_multiplier < 1) {
        throw invalid_argument(
            "pool_multiplier must be at least 1");
    }
}


//k is not checked because it is a query parameter

namespace {

// Caller must first validate the configuration's bin widths.
int64_t quantize_coordinate(float raw_value, double width) {
    if (!isfinite(raw_value) || raw_value < 0.0f) {
        throw invalid_argument(
            "indexed features must be finite and nonnegative");
    }

    const double transformed =
        log1p(static_cast<double>(raw_value));

    const double coordinate = floor(transformed / width);

    // Signed int64 coordinates must be strictly smaller than 2^63.
    // Do not compare against double(INT64_MAX): it may round to 2^63.
    const double exclusive_upper_bound =
        ldexp(1.0, numeric_limits<int64_t>::digits);

    if (!isfinite(coordinate) ||
        coordinate < 0.0 ||
        coordinate >= exclusive_upper_bound) {
        throw std::overflow_error(
            "quantized bucket coordinate exceeds int64 range");
    }

    return static_cast<std::int64_t>(coordinate);
}

} // namespace

// raw fearture -> log1p -> divide by bin width -> floor -> int64_t

StructuralBucketKey quantize_structural_features(
    const NodeFeatures& raw_features,
    const StructuralIndexConfig& config
) {
    validate_structural_index_config(config);

    StructuralBucketKey key;

    key.degree_bin = quantize_coordinate(
        raw_features.degree,
        config.degree_bin_width);

    key.neighbor_bin = quantize_coordinate(
        raw_features.avg_neighbor_degree,
        config.neighbor_bin_width);

    return key;
}

//pass raw features before normalization. This function cannot reliably detect normalized inputs merely by inspecting their values

StructuralIndex build_structural_index(
    const std::vector<NodeFeatures>& raw_target_features,
    const StructuralIndexConfig& config
) {
    validate_structural_index_config(config);

    // Conservative count limit, consistent with existing int-based code.
    if (raw_target_features.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "target count exceeds supported int range");
    }

    struct IndexedTarget {
        StructuralBucketKey key;
        int target_id;
    };

    std::vector<IndexedTarget> entries;
    entries.reserve(raw_target_features.size());

    // Exactly one entry is created for every target.
    for (std::size_t target = 0;
         target < raw_target_features.size();
         ++target) {
        entries.push_back({
            quantize_structural_features(
                raw_target_features[target], config),
            static_cast<int>(target)
        });
    }

    // Primary order: bucket key.
    // Secondary order: target ID within that bucket.
    sort(
        entries.begin(),
        entries.end(),
        [](const IndexedTarget& a, const IndexedTarget& b) {
            if (a.key == b.key) {
                return a.target_id < b.target_id;
            }

            return a.key < b.key;
        });

    StructuralIndex index;
    index.config = config;
    index.target_count = raw_target_features.size();
    index.target_ids.reserve(entries.size());

    // The empty index must still have offsets {0}.
    index.bucket_offsets.push_back(0);

    for (const IndexedTarget& entry : entries) {
        const bool starts_new_bucket =
            index.bucket_keys.empty() ||
            !(index.bucket_keys.back() == entry.key);

        if (starts_new_bucket) {
            // Close the preceding bucket before starting the next one.
            if (!index.bucket_keys.empty()) {
                index.bucket_offsets.push_back(
                    index.target_ids.size());
            }

            index.bucket_keys.push_back(entry.key);
        }

        index.target_ids.push_back(entry.target_id);
    }

    // Close the final bucket, if one exists.
    if (!index.bucket_keys.empty()) {
        index.bucket_offsets.push_back(index.target_ids.size());
    }

    return index;
}
StructuralCandidatePool retrieve_structural_candidates(
    const StructuralIndex& index,
    const NodeFeatures& raw_source_features,
    int k
) {
    validate_structural_index_config(index.config);
    if (k <= 0 || static_cast<std::size_t>(k) > index.target_count) {
        throw std::invalid_argument("retrieval requires 1 <= k <= target_count");
    }
    // Contract: index is an unmodified result of build_structural_index.
    const auto key = quantize_structural_features(raw_source_features, index.config);
    StructuralCandidatePool pool;
    const auto multiplier = static_cast<std::size_t>(index.config.pool_multiplier);
    const auto count = static_cast<std::size_t>(k);
    pool.stats.desired_pool_size = count > index.target_count / multiplier
        ? index.target_count : count * multiplier;

    auto visit = [&](std::int64_t x, std::int64_t y) {
        if (pool.stats.bucket_keys_examined == std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error("bucket lookup counter overflow");
        }
        ++pool.stats.bucket_keys_examined;
        StructuralBucketKey wanted{x, y};
        auto it = std::lower_bound(index.bucket_keys.begin(), index.bucket_keys.end(), wanted);
        if (it == index.bucket_keys.end() || !(*it == wanted)) return;
        ++pool.stats.occupied_buckets_found;
        auto bucket = static_cast<std::size_t>(it - index.bucket_keys.begin());
        // Complete buckets are retained, even when larger than the desired pool.
        pool.target_ids.insert(pool.target_ids.end(),
            index.target_ids.begin() + index.bucket_offsets[bucket],
            index.target_ids.begin() + index.bucket_offsets[bucket + 1]);
    };
    const auto limit = std::numeric_limits<std::int64_t>::max();
    for (std::int64_t radius = 0; radius <= index.config.maximum_radius; ++radius) {
        const auto low_x = key.degree_bin < radius ? 0 : key.degree_bin - radius;
        const auto high_x = key.degree_bin > limit - radius ? limit : key.degree_bin + radius;
        const auto low_y = key.neighbor_bin < radius ? 0 : key.neighbor_bin - radius;
        const auto high_y = key.neighbor_bin > limit - radius ? limit : key.neighbor_bin + radius;
        // Enumerate shell perimeter only, preserving lexicographic key order.
        for (auto x = low_x;; ++x) {
            const auto dx = x >= key.degree_bin ? x - key.degree_bin : key.degree_bin - x;
            if (dx == radius) {
                for (auto y = low_y;; ++y) {
                    visit(x, y);
                    if (y == high_y) break;
                }
            } else {
                if (key.neighbor_bin >= radius) visit(x, key.neighbor_bin - radius);
                if (radius > 0 && key.neighbor_bin <= limit - radius)
                    visit(x, key.neighbor_bin + radius);
            }
            if (x == high_x) break;
        }
        pool.stats.final_radius = static_cast<int>(radius);
        if (radius >= index.config.minimum_radius &&
            pool.target_ids.size() >= pool.stats.desired_pool_size) break;
    }
    // Disjoint shells and unique bucket membership guarantee unique targets.
    pool.stats.bucket_pool_size = pool.target_ids.size();
    if (pool.target_ids.size() < pool.stats.desired_pool_size) {
        pool.stats.used_exhaustive_fallback = true;
        pool.target_ids.resize(index.target_count);
        for (std::size_t i = 0; i < index.target_count; ++i)
            pool.target_ids[i] = static_cast<int>(i);
    }
    return pool;
}
