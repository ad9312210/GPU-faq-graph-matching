#include "features.hpp"

std::vector<NodeFeatures> compute_features(const Graph& g) {
    validate_graph(g);

    const int n = g.num_vertices;
    std::vector<NodeFeatures> feats(n);
    std::vector<int> degrees(n);
    std::vector<int> triangle_counts(n, 0);

    for (int v = 0; v < n; ++v) {
        degrees[v] = degree(g, v);
    }

    // Enumerate each undirected triangle exactly once using u < w < z.
    // Increment each of its three vertices once; no divide-by-two correction
    // is needed for per-vertex triangle participation counts.
    for (int u = 0; u < n; ++u) {
        const int u_start = g.row_ptr[u];
        const int u_end = g.row_ptr[u + 1];
        for (int u_idx = u_start; u_idx < u_end; ++u_idx) {
            const int w = g.col_idx[u_idx];
            if (w <= u) continue;
            for (int w_idx = g.row_ptr[w]; w_idx < g.row_ptr[w + 1]; ++w_idx) {
                const int z = g.col_idx[w_idx];
                if (z <= w) continue;
                if (has_edge(g, u, z)) {
                    ++triangle_counts[u];
                    ++triangle_counts[w];
                    ++triangle_counts[z];
                }
            }
        }
    }

    for (int v = 0; v < n; ++v) {
        const int d = degrees[v];
        feats[v].degree = static_cast<float>(d);
        feats[v].degree_squared = static_cast<float>(d * d);
        feats[v].triangle_count = static_cast<float>(triangle_counts[v]);
        const float possible_triangles = static_cast<float>(d) * (d - 1) / 2.0f;
        feats[v].clustering = d >= 2
            ? static_cast<float>(triangle_counts[v]) / possible_triangles
            : 0.0f;

        const auto nbrs = neighbors(g, v);
        if (nbrs.empty()) {
            feats[v].avg_neighbor_degree = 0.0f;
            feats[v].neighbor_degree_std = 0.0f;
            continue;
        }

        float sum = 0.0f;
        for (int nb : nbrs) sum += static_cast<float>(degrees[nb]);
        const float mean = sum / static_cast<float>(nbrs.size());
        feats[v].avg_neighbor_degree = mean;

        // The paper uses sample standard deviation; degree-one vertices have
        // zero deviation by definition.
        if (nbrs.size() <= 1) {
            feats[v].neighbor_degree_std = 0.0f;
        } else {
            float squared_deviation_sum = 0.0f;
            for (int nb : nbrs) {
                const float delta = static_cast<float>(degrees[nb]) - mean;
                squared_deviation_sum += delta * delta;
            }
            feats[v].neighbor_degree_std = std::sqrt(
                squared_deviation_sum / static_cast<float>(nbrs.size() - 1));
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

void normalize_features_joint_zscore(std::vector<NodeFeatures>& source,
                                     std::vector<NodeFeatures>& target) {
    const int total = static_cast<int>(source.size() + target.size());
    if (total == 0) return;

    for (int f = 0; f < NUM_FEATURES; f++) {
        float sum = 0.0f;
        for (const auto& feature : source) sum += feature.as_array(f);
        for (const auto& feature : target) sum += feature.as_array(f);
        float mean = sum / static_cast<float>(total);

        float var_sum = 0.0f;
        for (const auto& feature : source) {
            float diff = feature.as_array(f) - mean;
            var_sum += diff * diff;
        }
        for (const auto& feature : target) {
            float diff = feature.as_array(f) - mean;
            var_sum += diff * diff;
        }
        float std_dev = std::sqrt(var_sum / static_cast<float>(total));
        if (std_dev < SCGM_EPS) std_dev = 1.0f;

        auto normalize_feature = [f, mean, std_dev](NodeFeatures& feature) {
            float z = (feature.as_array(f) - mean) / std_dev;
            switch (f) {
                case 0: feature.degree = z; break;
                case 1: feature.clustering = z; break;
                case 2: feature.avg_neighbor_degree = z; break;
                case 3: feature.neighbor_degree_std = z; break;
                case 4: feature.triangle_count = z; break;
                case 5: feature.degree_squared = z; break;
            }
        };
        for (auto& feature : source) normalize_feature(feature);
        for (auto& feature : target) normalize_feature(feature);
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