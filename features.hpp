#ifndef SCGM_FEATURES_HPP
#define SCGM_FEATURES_HPP

#include "common.hpp"
#include "graph.hpp"

/*
 * The six primary structural descriptors defined by the paper:
 *   1. Degree
 *   2. Clustering coefficient
 *   3. Average neighbour degree
 *   4. Neighbour degree standard deviation
 *   5. Triangle count
 *   6. Degree squared
 *
 * Additional/extended features are NOT included here.
 * If ablation features (PageRank, k-core, WL, etc.) are added later,
 * they must be placed behind an optional configuration flag and clearly
 * labelled as ablation/extended features.
 */

struct NodeFeatures {
    float degree;
    float clustering;
    float avg_neighbor_degree;
    float neighbor_degree_std;
    float triangle_count;
    float degree_squared;

    float as_array(int idx) const {
        switch (idx) {
            case 0: return degree;
            case 1: return clustering;
            case 2: return avg_neighbor_degree;
            case 3: return neighbor_degree_std;
            case 4: return triangle_count;
            case 5: return degree_squared;
            default: return 0.0f;
        }
    }
};

static constexpr int NUM_FEATURES = 6;

std::vector<NodeFeatures> compute_features(const Graph& g);

/*
 * Feature normalization.
 *
 * We use z-score normalization: z = (x - mean) / std
 *
 * IMPORTANT: z-score normalization does NOT guarantee non-negative values.
 * Negative values are expected and valid.  Cosine similarity handles them
 * correctly because the formula
 *     S(a,b) = dot(a,b) / (||a|| ||b||)
 * works for any real-valued vectors.
 */
void normalize_features_zscore(std::vector<NodeFeatures>& feats);

void features_to_flat_array(const std::vector<NodeFeatures>& feats,
                            std::vector<float>& out);

void flat_array_to_features(const std::vector<float>& arr, int n,
                            std::vector<NodeFeatures>& out);

#endif // SCGM_FEATURES_HPP