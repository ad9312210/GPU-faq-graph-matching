#include "common.hpp"
#include "graph.hpp"
#include "features.hpp"

static bool approx_eq(float a, float b, float tol = 1e-4f) {
    return std::abs(a - b) < tol;
}

static int run_tests() {
    int failures = 0;

    // Build the six-node asymmetric test graph
    std::vector<std::pair<int,int>> edges = {
        {0,1}, {0,5}, {1,2}, {1,3}, {1,4}, {2,3}, {2,5}, {3,4}
    };
    Graph g = build_graph(6, edges);

    auto feats = compute_features(g);

    // Independent tiny-graph reference: construct a dense adjacency matrix,
    // enumerate each triangle exactly once by u < w < z, and calculate all
    // six raw descriptors without calling production feature helpers.
    {
        const int n = g.num_vertices;
        std::vector<std::vector<int>> adjacency(n, std::vector<int>(n, 0));
        for (const auto& edge : edges) {
            adjacency[edge.first][edge.second] = 1;
            adjacency[edge.second][edge.first] = 1;
        }
        std::vector<int> ref_degree(n, 0), ref_triangles(n, 0);
        for (int v = 0; v < n; ++v) {
            for (int u = 0; u < n; ++u) ref_degree[v] += adjacency[v][u];
        }
        for (int u = 0; u < n; ++u) {
            for (int w = u + 1; w < n; ++w) {
                if (!adjacency[u][w]) continue;
                for (int z = w + 1; z < n; ++z) {
                    if (adjacency[u][z] && adjacency[w][z]) {
                        ++ref_triangles[u];
                        ++ref_triangles[w];
                        ++ref_triangles[z];
                    }
                }
            }
        }
        for (int v = 0; v < n; ++v) {
            float mean_neighbor_degree = 0.0f;
            int neighbor_count = 0;
            for (int u = 0; u < n; ++u) {
                if (adjacency[v][u]) {
                    mean_neighbor_degree += static_cast<float>(ref_degree[u]);
                    ++neighbor_count;
                }
            }
            if (neighbor_count) mean_neighbor_degree /= neighbor_count;
            float squared_deviation_sum = 0.0f;
            for (int u = 0; u < n; ++u) {
                if (adjacency[v][u]) {
                    const float delta = static_cast<float>(ref_degree[u]) - mean_neighbor_degree;
                    squared_deviation_sum += delta * delta;
                }
            }
            const float neighbor_std = neighbor_count > 1
                ? std::sqrt(squared_deviation_sum / static_cast<float>(neighbor_count - 1))
                : 0.0f;
            const float possible = static_cast<float>(ref_degree[v]) *
                                   (ref_degree[v] - 1) / 2.0f;
            const float clustering = ref_degree[v] >= 2
                ? static_cast<float>(ref_triangles[v]) / possible
                : 0.0f;
            const float expected[] = {
                static_cast<float>(ref_degree[v]),
                clustering,
                mean_neighbor_degree,
                neighbor_std,
                static_cast<float>(ref_triangles[v]),
                static_cast<float>(ref_degree[v] * ref_degree[v])
            };
            for (int feature = 0; feature < NUM_FEATURES; ++feature) {
                if (!approx_eq(feats[v].as_array(feature), expected[feature])) {
                    std::cerr << "FAIL: reference mismatch node=" << v
                              << " feature=" << feature
                              << " expected=" << expected[feature]
                              << " got=" << feats[v].as_array(feature) << std::endl;
                    failures++;
                }
            }
        }
    }

    // Expected degrees: A=2, B=4, C=3, D=3, E=2, F=2
    float expected_deg[] = {2, 4, 3, 3, 2, 2};
    for (int i = 0; i < 6; i++) {
        if (!approx_eq(feats[i].degree, expected_deg[i])) {
            std::cerr << "FAIL: node " << i << " degree expected "
                      << expected_deg[i] << " got " << feats[i].degree << std::endl;
            failures++;
        }
    }

    // Expected degree_squared
    for (int i = 0; i < 6; i++) {
        float expected = expected_deg[i] * expected_deg[i];
        if (!approx_eq(feats[i].degree_squared, expected)) {
            std::cerr << "FAIL: node " << i << " degree_squared expected "
                      << expected << " got " << feats[i].degree_squared << std::endl;
            failures++;
        }
    }

    // Verify clustering coefficient for node B (deg=4, neighbors={A,C,D,E})
    // Edges among neighbors: (C,D), (D,E) => 2 edges out of C(4,2)=6 possible
    // CC(B) = 2/6 = 0.3333...
    if (!approx_eq(feats[1].clustering, 2.0f / 6.0f)) {
        std::cerr << "FAIL: node B clustering expected " << 2.0f/6.0f
                  << " got " << feats[1].clustering << std::endl;
        failures++;
    }

    // Node A (deg=2, neighbors={B,F}): edge (B,F)? B=1, F=5 -> no edge
    // CC(A) = 0/1 = 0
    if (!approx_eq(feats[0].clustering, 0.0f)) {
        std::cerr << "FAIL: node A clustering expected 0 got "
                  << feats[0].clustering << std::endl;
        failures++;
    }

    // Node C (deg=3, neighbors={B,D,F}): edges among neighbors: (B,D)=yes -> 1 out of 3
    // CC(C) = 1/3
    if (!approx_eq(feats[2].clustering, 1.0f / 3.0f)) {
        std::cerr << "FAIL: node C clustering expected " << 1.0f/3.0f
                  << " got " << feats[2].clustering << std::endl;
        failures++;
    }

    // Triangle count for node B:
    // Neighbors: {A=0, C=2, D=3, E=4}
    // Pairs: (A,C)no (A,D)no (A,E)no (C,D)yes (C,E)no (D,E)yes = 2
    if (!approx_eq(feats[1].triangle_count, 2.0f)) {
        std::cerr << "FAIL: node B triangle_count expected 2 got "
                  << feats[1].triangle_count << std::endl;
        failures++;
    }

    // Average neighbor degree for A: neighbors={B(4), F(2)} -> avg = 3.0
    if (!approx_eq(feats[0].avg_neighbor_degree, 3.0f)) {
        std::cerr << "FAIL: node A avg_neighbor_degree expected 3.0 got "
                  << feats[0].avg_neighbor_degree << std::endl;
        failures++;
    }

    // Test z-score normalization
    auto feats_norm = feats;  // copy
    normalize_features_zscore(feats_norm);

    // After z-score, mean should be ~0 for each feature
    for (int f = 0; f < NUM_FEATURES; f++) {
        float sum = 0;
        for (int i = 0; i < 6; i++) {
            sum += feats_norm[i].as_array(f);
        }
        float mean = sum / 6.0f;
        if (!approx_eq(mean, 0.0f, 1e-3f)) {
            std::cerr << "FAIL: z-score mean for feature " << f
                      << " expected ~0 got " << mean << std::endl;
            failures++;
        }
    }

    // Verify z-score can produce negative values (do NOT claim non-negative)
    bool has_negative = false;
    for (int i = 0; i < 6; i++) {
        for (int f = 0; f < NUM_FEATURES; f++) {
            if (feats_norm[i].as_array(f) < 0) {
                has_negative = true;
            }
        }
    }
    if (!has_negative) {
        std::cerr << "WARNING: z-score produced no negative values (unexpected for asymmetric data)"
                  << std::endl;
        // Not necessarily a failure, but unexpected
    }

    // Test on empty-ish graph
    Graph g1 = build_graph(1, {});
    auto f1 = compute_features(g1);
    if (!approx_eq(f1[0].degree, 0.0f)) {
        std::cerr << "FAIL: isolated vertex degree" << std::endl;
        failures++;
    }

    if (failures == 0) {
        std::cout << "test_features: ALL PASSED" << std::endl;
    } else {
        std::cout << "test_features: " << failures << " FAILURES" << std::endl;
    }

    return failures;
}

int main() {
    return run_tests();
}