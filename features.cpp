#include "features.hpp"

static float compute_clustering_coefficient(const Graph& g, int v) {
    int deg = degree(g, v);
    if (deg < 2) return 0.0f;

    auto nbrs = neighbors(g, v);
    std::set<int> nbr_set(nbrs.begin(), nbrs.end());

    int triangles = 0;
    for (int i = 0; i < (int)nbrs.size(); i++) {
        for (int j = i + 1; j < (int)nbrs.size(); j++) {
            if (has_edge(g, nbrs[i], nbrs[j])) {
                triangles++;
            }
        }
    }

    float max_triangles = (float)deg * (float)(deg - 1) / 2.0f;
    return (float)triangles / max_triangles;
}

static int count_triangles(const Graph& g, int v) {
    auto nbrs = neighbors(g, v);
    std::set<int> nbr_set(nbrs.begin(), nbrs.end());

    int count = 0;
    for (int i = 0; i < (int)nbrs.size(); i++) {
        for (int j = i + 1; j < (int)nbrs.size(); j++) {
            if (has_edge(g, nbrs[i], nbrs[j])) {
                count++;
            }
        }
    }
    return count;
}

std::vector<NodeFeatures> compute_features(const Graph& g) {
    validate_graph(g);

    int N = g.num_vertices;
    std::vector<NodeFeatures> feats(N);

    // Precompute degrees
    std::vector<int> degrees(N);
    for (int v = 0; v < N; v++) {
        degrees[v] = degree(g, v);
    }

    for (int v = 0; v < N; v++) {
        int deg = degrees[v];
        feats[v].degree = (float)deg;
        feats[v].degree_squared = (float)(deg * deg);

        // Clustering coefficient
        feats[v].clustering = compute_clustering_coefficient(g, v);

        // Triangle count
        feats[v].triangle_count = (float)count_triangles(g, v);

        // Average neighbour degree and std
        auto nbrs = neighbors(g, v);
        if (nbrs.empty()) {
            feats[v].avg_neighbor_degree = 0.0f;
            feats[v].neighbor_degree_std = 0.0f;
        } else {
            float sum = 0.0f;
            for (int nb : nbrs) {
                sum += (float)degrees[nb];
            }
            float avg = sum / (float)nbrs.size();
            feats[v].avg_neighbor_degree = avg;

            float var_sum = 0.0f;
            for (int nb : nbrs) {
                float diff = (float)degrees[nb] - avg;
                var_sum += diff * diff;
            }
            feats[v].neighbor_degree_std = std::sqrt(var_sum / (float)nbrs.size());
        }
    }

    return feats;
}

/*
 * Z-score normalization: z = (x - mean) / std
 *
 * NOTE: This does NOT guarantee non-negative values.
 * Negative values are expected and valid for cosine similarity.
 */
void normalize_features_zscore(std::vector<NodeFeatures>& feats) {
    int N = (int)feats.size();
    if (N == 0) return;

    for (int f = 0; f < NUM_FEATURES; f++) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += feats[i].as_array(f);
        }
        float mean = sum / (float)N;

        float var_sum = 0.0f;
        for (int i = 0; i < N; i++) {
            float diff = feats[i].as_array(f) - mean;
            var_sum += diff * diff;
        }
        float std_dev = std::sqrt(var_sum / (float)N);
        if (std_dev < SCGM_EPS) std_dev = 1.0f;

        for (int i = 0; i < N; i++) {
            float val = feats[i].as_array(f);
            float z = (val - mean) / std_dev;
            switch (f) {
                case 0: feats[i].degree = z; break;
                case 1: feats[i].clustering = z; break;
                case 2: feats[i].avg_neighbor_degree = z; break;
                case 3: feats[i].neighbor_degree_std = z; break;
                case 4: feats[i].triangle_count = z; break;
                case 5: feats[i].degree_squared = z; break;
            }
        }
    }
}

void features_to_flat_array(const std::vector<NodeFeatures>& feats,
                            std::vector<float>& out) {
    int N = (int)feats.size();
    out.resize(N * NUM_FEATURES);
    for (int i = 0; i < N; i++) {
        for (int f = 0; f < NUM_FEATURES; f++) {
            out[i * NUM_FEATURES + f] = feats[i].as_array(f);
        }
    }
}

void flat_array_to_features(const std::vector<float>& arr, int n,
                            std::vector<NodeFeatures>& out) {
    out.resize(n);
    for (int i = 0; i < n; i++) {
        out[i].degree = arr[i * NUM_FEATURES + 0];
        out[i].clustering = arr[i * NUM_FEATURES + 1];
        out[i].avg_neighbor_degree = arr[i * NUM_FEATURES + 2];
        out[i].neighbor_degree_std = arr[i * NUM_FEATURES + 3];
        out[i].triangle_count = arr[i * NUM_FEATURES + 4];
        out[i].degree_squared = arr[i * NUM_FEATURES + 5];
    }
}