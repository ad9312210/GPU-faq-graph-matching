#ifndef SCGM_STRUCTURAL_INDEX_HPP
#define SCGM_STRUCTURAL_INDEX_HPP

#include "features.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

//introduce to node features and hybrid indexing

// Structural indexing groups target IDs by quantized feature signatures.
struct StructuralIndexConfig{
    double degree_bin_width=0.5;
    double neighbor_bin_width=0.5;

    int minimum_radius=1;
    int maximum_radius=3;

    int pool_multiplier=4;
};

// Initial experimental parameters controlling bucket widths and retrieval.
// They do not change the six-feature vector's size.
struct StructuralBucketKey{
    std::int64_t degree_bin = 0;
    std::int64_t neighbor_bin = 0;

    bool operator==(const StructuralBucketKey& other) const {
        return degree_bin == other.degree_bin && neighbor_bin == other.neighbor_bin;
    }

    bool operator<(const StructuralBucketKey& other) const {
        if (degree_bin != other.degree_bin) {
            return degree_bin < other.degree_bin;
        }

        return neighbor_bin < other.neighbor_bin;
    }

};

//defing the target index

struct StructuralIndex {
    StructuralIndexConfig config;
    std::size_t target_count = 0;

    // Sorted, unique occupied bucket keys.
    std::vector<StructuralBucketKey> bucket_keys;

    // Bucket b occupies:
    // target_ids[bucket_offsets[b] ... bucket_offsets[b + 1]).
    // Size is bucket_keys.size() + 1.
    // An empty index has offsets {0}.
    std::vector<std::size_t> bucket_offsets;

    // Every target ID appears exactly once.
    // IDs are increasing within each bucket.
    std::vector<int> target_ids;
};

//defining retrieved pool and its statistics

struct StructuralRetrievalStats {
    // Unique targets requested before accepting bucket retrieval.
    std::size_t desired_pool_size = 0;

    // Number of bucket-key lookups, including empty buckets.
    std::size_t bucket_keys_examined = 0;

    // Number of occupied buckets found.
    std::size_t occupied_buckets_found = 0;

    // Unique targets found before exhaustive fallback.
    std::size_t bucket_pool_size = 0;

    // Last complete shell processed.
    int final_radius = -1;

    bool used_exhaustive_fallback = false;
};

struct StructuralCandidatePool {
    // Unique targets to score; these are not ranked candidates.
    //
    // Without fallback:
    // shell order, then lexicographic bucket order,
    // then increasing target ID within each bucket.
    //
    // With fallback: all target IDs in increasing order.
    std::vector<int> target_ids;

    StructuralRetrievalStats stats;
};

//retrieval is separate from scoring. The structure contains target IDs.

// Throws std::invalid_argument for invalid configuration.
void validate_structural_index_config(
    const StructuralIndexConfig& config
);

// Uses raw, nonnegative degree and average-neighbor-degree.
//
// Throws std::invalid_argument for invalid configuration or
// negative/nonfinite indexed features.
// Throws std::overflow_error if a bucket coordinate cannot be represented.
StructuralBucketKey quantize_structural_features(
    const NodeFeatures& raw_features,
    const StructuralIndexConfig& config
);

// Builds an occupied-bucket index over targets.
//
// Target IDs correspond to positions in raw_target_features.
// Rejects a target count that cannot be represented by int IDs.
// An empty target vector produces an empty index.
StructuralIndex build_structural_index(
    const std::vector<NodeFeatures>& raw_target_features,
    const StructuralIndexConfig& config = StructuralIndexConfig{}
);

// Retrieves targets for one source using the stored configuration.
//
// Requires a valid index produced by build_structural_index.
// Requires 1 <= k <= target_count.
//
// Searches complete shells through maximum_radius.
// Desired pool = min(target_count, pool_multiplier * k),
// calculated without overflow.
//
// If the desired pool is not reached, returns all targets.
// Does not compute similarities or select Top-K.
//
// This operation cannot query an empty target index with positive k.
StructuralCandidatePool retrieve_structural_candidates(
    const StructuralIndex& index,
    const NodeFeatures& raw_source_features,
    int k
);

#endif // SCGM_STRUCTURAL_INDEX_HPP