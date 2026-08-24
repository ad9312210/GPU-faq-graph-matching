/*
 * Toy SCGM example: Six-node asymmetric graph isomorphism.
 *
 * PERMUTATION CONVENTION:
 *   row_to_col[i] = j means source node i -> target node j.
 *
 * Source graph (6 vertices, 8 edges):
 *   (A,B) (A,F) (B,C) (B,D) (B,E) (C,D) (C,F) (D,E)
 *
 * Known permutation:
 *   A->14, B->11, C->16, D->12, E->15, F->13
 *
 * Target graph edges:
 *   (14,11) (14,13) (11,16) (11,12) (11,15) (16,12) (16,13) (12,15)
 */

#include "common.hpp"
#include "graph.hpp"
#include "features.hpp"
#include "cosine_similarity.hpp"
#include "candidate_generation.hpp"
#include "csr_matrix.hpp"
#include "lapjv.hpp"
#include "evaluation.hpp"

int main() {
    std::cout << "=== Toy SCGM Example ===" << std::endl;
    std::cout << std::endl;

    // Build source graph: A=0, B=1, C=2, D=3, E=4, F=5
    std::vector<std::pair<int,int>> src_edges = {
        {0,1}, {0,5}, {1,2}, {1,3}, {1,4}, {2,3}, {2,5}, {3,4}
    };
    Graph source = build_graph(6, src_edges);

    // Ground truth: A->14(3), B->11(0), C->16(5), D->12(1), E->15(4), F->13(2)
    // Internal target IDs: 11=0, 12=1, 13=2, 14=3, 15=4, 16=5
    std::vector<int> ground_truth = {3, 0, 5, 1, 4, 2};
    std::vector<int> target_ids = {11, 12, 13, 14, 15, 16};
    std::vector<std::string> src_labels = {"A", "B", "C", "D", "E", "F"};

    // Build target graph
    std::vector<std::pair<int,int>> tgt_edges = {
        {3,0}, {3,2}, {0,5}, {0,1}, {0,4}, {5,1}, {5,2}, {1,4}
    };
    Graph target = build_graph(6, tgt_edges);

    // Verify graphs are valid
    validate_graph(source);
    validate_graph(target);

    int N = source.num_vertices;
    int M = target.num_vertices;
    int K = M;

    // Step 1: Compute features
    auto feats_src = compute_features(source);
    auto feats_tgt = compute_features(target);

    std::cout << "Raw features (before normalization):" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << "  " << src_labels[i] << ": deg=" << feats_src[i].degree
                  << " cc=" << feats_src[i].clustering
                  << " and=" << feats_src[i].avg_neighbor_degree
                  << " nds=" << feats_src[i].neighbor_degree_std
                  << " tri=" << feats_src[i].triangle_count
                  << " ds=" << feats_src[i].degree_squared << std::endl;
    }
    std::cout << std::endl;

    // Step 2: Normalize
    normalize_features_zscore(feats_src);
    normalize_features_zscore(feats_tgt);

    std::cout << "Normalized features (z-score, may contain negative values):" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << "  " << src_labels[i] << ": "
                  << feats_src[i].degree << " "
                  << feats_src[i].clustering << " "
                  << feats_src[i].avg_neighbor_degree << " "
                  << feats_src[i].neighbor_degree_std << " "
                  << feats_src[i].triangle_count << " "
                  << feats_src[i].degree_squared << std::endl;
    }
    std::cout << std::endl;

    // Step 3: Top-K candidates
    auto candidates = generate_top_k_candidates(feats_src, feats_tgt, K);

    // Step 4: Build CSR
    CSRMatrix csr = build_sparse_cost_matrix(candidates, N, M);
    print_csr_info(csr);
    std::cout << std::endl;

    // Step 5: Solve LAPJV
    AssignmentResult result = solve_lapjv(csr);

    std::cout << "LAPJV result:" << std::endl;
    std::cout << "  Feasible: " << (result.feasible ? "YES" : "NO") << std::endl;
    std::cout << "  Total cost: " << result.total_cost << std::endl;
    for (int i = 0; i < N; i++) {
        int j = result.row_to_col[i];
        std::cout << "  " << src_labels[i] << " -> "
                  << target_ids[j] << std::endl;
    }
    std::cout << std::endl;

    // Step 6: Evaluate
    double acc = mapping_accuracy(result.row_to_col, ground_truth);
    double ep = edge_preservation(source, target, result.row_to_col);
    double rep = reverse_edge_preservation(source, target, result.row_to_col);
    bool oto = is_one_to_one(result.row_to_col, M);

    std::cout << "Evaluation:" << std::endl;
    std::cout << "  Mapping accuracy: " << acc << std::endl;
    std::cout << "  Edge preservation: " << ep << std::endl;
    std::cout << "  Reverse edge preservation: " << rep << std::endl;
    std::cout << "  One-to-one: " << (oto ? "YES" : "NO") << std::endl;

    return 0;
}