# 01_cpu_sequential_sparse_matching.py
"""
SCGM-Seq: CPU Sequential Sparse Candidate Graph Matching
=========================================================
Single-threaded baseline implementation.
Purpose: Correctness verification and sequential baseline.

Research Claim:
    Dense (n x n) assignment matrix and O(n^3) dense LAP bottleneck
    avoided by solving graph matching in sparse (n x k) candidate space,
    where k << n.

Algorithm:
    1. Graph-aware feature extraction (degree, clustering, WL, etc.)
    2. Top-k candidate generation per source node
    3. Sparse masked cost matrix construction -> O(nk)
    4. Sequential sparse Sinkhorn normalization
    5. Dynamic graph-aware candidate refinement
    6. Conflict-aware hard assignment extraction
    7. Reduced LAP only on ambiguous nodes U where |U| << n

Complexity:
    Memory: O(nk) vs O(n^2) dense
    Time:   O(nkT) + O(|U|^3) vs O(n^3) dense
"""

import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
import time
import tracemalloc
import warnings
warnings.filterwarnings('ignore')

from utils_graph_matching import (
    generate_ba_graph, generate_er_graph, generate_ws_graph,
    generate_sbm_graph, permute_graph,
    compute_graph_features, compute_node_similarity,
    select_top_k_candidates, build_sparse_cost_vectorized,
    sparse_sinkhorn_vectorized,
    compute_neighborhood_consistency_fast, refine_candidates,
    conflict_aware_extraction, confidence_threshold_extraction,
    reduced_lap_refinement,
    compute_accuracy, compute_candidate_recall, compute_speedup,
    scipy_hungarian_matching, run_scipy_baseline,
    save_results_csv, print_result_table, print_complexity_analysis,
    measure_peak_memory_cpu
)


# ============================================================
# MAIN SCGM SEQUENTIAL ALGORITHM
# ============================================================

def scgm_sequential(G_A, G_B, A, B, k=32, sinkhorn_iter=20,
                     sinkhorn_temp=0.1, tau=0.5,
                     alpha=0.4, beta=0.3, lam=0.3,
                     n_refinement_iter=2,
                     extraction='conflict_aware',
                     lap_method='hungarian',
                     verbose=False):
    """
    SCGM Sequential: Single-threaded sparse candidate graph matching.
    
    Parameters:
        G_A, G_B: NetworkX graphs
        A, B: Adjacency matrices (numpy arrays)
        k: Number of candidates per source node
        sinkhorn_iter: Sinkhorn normalization iterations
        sinkhorn_temp: Sinkhorn temperature parameter
        tau: Ambiguity threshold for reduced LAP
        alpha, beta, lam: Weights for candidate refinement scores
        n_refinement_iter: Number of dynamic refinement iterations
        extraction: 'conflict_aware' or 'threshold'
        lap_method: 'hungarian' or 'greedy' for reduced LAP
        verbose: Print stage-wise timing
    
    Returns:
        assignment: (n,) matching array
        timings: dict of stage-wise timings (ms)
    """
    n = A.shape[0]
    n_target = B.shape[0]
    timings = {}
    
    # ----------------------------------------------------------
    # STAGE 1: Graph-aware feature extraction
    # ----------------------------------------------------------
    t0 = time.perf_counter()
    
    FA = compute_graph_features(G_A, A)  # (n, d)
    FB = compute_graph_features(G_B, B)  # (n_target, d)
    
    timings['feature_extraction_ms'] = (time.perf_counter() - t0) * 1000
    if verbose:
        print(f"  [STAGE 1] Feature extraction: "
              f"{timings['feature_extraction_ms']:.2f} ms | "
              f"Feature dim: {FA.shape[1]}")
    
    # ----------------------------------------------------------
    # STAGE 2: Node similarity + Top-k candidate generation
    # ----------------------------------------------------------
    t0 = time.perf_counter()
    
    S_node = compute_node_similarity(FA, FB)  # (n, n_target)
    candidates = select_top_k_candidates(S_node, k)  # (n, k)
    
    timings['candidate_gen_ms'] = (time.perf_counter() - t0) * 1000
    if verbose:
        print(f"  [STAGE 2] Candidate generation: "
              f"{timings['candidate_gen_ms']:.2f} ms | "
              f"k={k}, candidates shape: {candidates.shape}")
    
    # ----------------------------------------------------------
    # STAGE 3: Sparse masked cost matrix construction
    # Memory: O(nk) instead of O(n^2)
    # ----------------------------------------------------------
    t0 = time.perf_counter()
    
    sparse_cost = build_sparse_cost_vectorized(S_node, candidates)
    
    timings['sparse_cost_ms'] = (time.perf_counter() - t0) * 1000
    if verbose:
        print(f"  [STAGE 3] Sparse cost matrix: "
              f"{timings['sparse_cost_ms']:.2f} ms | "
              f"Shape: {sparse_cost.shape} | "
              f"Memory: {sparse_cost.nbytes/1024:.1f} KB")
    
    # ----------------------------------------------------------
    # STAGE 4 + 5: Masked Sinkhorn + Dynamic candidate refinement
    # Iteratively refine candidates using neighborhood consistency
    # ----------------------------------------------------------
    t_sinkhorn_total = 0.0
    t_refine_total = 0.0
    
    P_sparse = sparse_cost.copy()
    
    for ref_iter in range(n_refinement_iter):
        # Sinkhorn normalization on sparse candidate matrix
        t0 = time.perf_counter()
        P_sparse = sparse_sinkhorn_vectorized(
            sparse_cost if ref_iter == 0 else
            build_sparse_cost_vectorized(S_combined_curr
                                          if 'S_combined_curr' in dir()
                                          else S_node,
                                         candidates),
            candidates, n_target,
            temperature=sinkhorn_temp,
            n_iter=sinkhorn_iter
        )
        t_sinkhorn_total += (time.perf_counter() - t0) * 1000
        
        if ref_iter < n_refinement_iter - 1:
            # Dynamic graph-aware candidate refinement
            t0 = time.perf_counter()
            
            # Compute neighborhood consistency: O(nk * d)
            S_struct = compute_neighborhood_consistency_fast(
                A, B, P_sparse, candidates)
            
            # Update candidate scores and refresh top-k
            # S_ij^{t+1} = alpha*S_node + beta*S_struct + lambda*P_sparse
            new_candidates, S_combined_curr = refine_candidates(
                S_node, S_struct, P_sparse, candidates,
                k, alpha=alpha, beta=beta, lam=lam
            )
            candidates = new_candidates
            
            t_refine_total += (time.perf_counter() - t0) * 1000
    
    timings['sinkhorn_ms'] = t_sinkhorn_total
    timings['refinement_ms'] = t_refine_total
    
    if verbose:
        print(f"  [STAGE 4] Sinkhorn ({n_refinement_iter} iters): "
              f"{t_sinkhorn_total:.2f} ms")
        print(f"  [STAGE 5] Candidate refinement: "
              f"{t_refine_total:.2f} ms")
    
    # ----------------------------------------------------------
    # STAGE 6: Conflict-aware hard assignment extraction
    # ----------------------------------------------------------
    t0 = time.perf_counter()
    
    if extraction == 'conflict_aware':
        assignment, ambiguous, available_targets = \
            conflict_aware_extraction(P_sparse, candidates, n, n_target,
                                       tau=tau)
    else:  # threshold
        assignment, ambiguous, available_targets = \
            confidence_threshold_extraction(P_sparse, candidates, n,
                                             n_target, tau=tau)
    
    timings['extraction_ms'] = (time.perf_counter() - t0) * 1000
    if verbose:
        print(f"  [STAGE 6] Conflict-aware extraction: "
              f"{timings['extraction_ms']:.2f} ms | "
              f"Ambiguous nodes: {len(ambiguous)}/{n}")
    
    # ----------------------------------------------------------
    # STAGE 7: Reduced LAP only on ambiguous nodes U
    # Complexity: O(|U|^3) where |U| << n
    # ----------------------------------------------------------
    t0 = time.perf_counter()
    
    assignment = reduced_lap_refinement(
        assignment, ambiguous, available_targets,
        S_node, method=lap_method
    )
    
    timings['reduced_lap_ms'] = (time.perf_counter() - t0) * 1000
    if verbose:
        print(f"  [STAGE 7] Reduced LAP ({len(ambiguous)} nodes): "
              f"{timings['reduced_lap_ms']:.2f} ms")
    
    # Total time
    timings['total_ms'] = sum(timings.values())
    timings['n_ambiguous'] = len(ambiguous)
    timings['ambiguous_fraction'] = len(ambiguous) / n
    
    return assignment, timings, P_sparse, candidates


# ============================================================
# EXPERIMENT 1: BASIC CORRECTNESS TEST
# ============================================================

def experiment_basic_correctness(n=500, k=32, noise=0.0, seed=42):
    """Run basic correctness test on a single graph pair."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: Basic Correctness Test")
    print(f"  n={n}, k={k}, noise={noise}")
    print(f"{'='*60}")
    
    # Generate graphs
    G_A, A = generate_ba_graph(n, m=3, seed=seed)
    B, perm, gt = permute_graph(A, noise_level=noise, seed=seed+1)
    G_B = nx.from_numpy_array(B)
    
    # Run SCGM Sequential
    print("\n[SCGM Sequential]")
    assignment, timings, P_sparse, candidates = scgm_sequential(
        G_A, G_B, A, B, k=k, verbose=True
    )
    
    acc = compute_accuracy(assignment, gt)
    recall = compute_candidate_recall(candidates, gt)
    
    print(f"\n  Accuracy:        {acc:.4f}")
    print(f"  Candidate Recall@{k}: {recall:.4f}")
    print(f"  Ambiguous nodes: {timings['n_ambiguous']}")
    print(f"  Total time:      {timings['total_ms']:.2f} ms")
    
    # Run SciPy baseline
    print("\n[SciPy Hungarian Baseline]")
    FA = compute_graph_features(G_A, A)
    FB = compute_graph_features(G_B, B)
    
    t_s = time.perf_counter()
    S_node = compute_node_similarity(FA, FB)
    scipy_assign = scipy_hungarian_matching(S_node, n, n)
    scipy_time = (time.perf_counter() - t_s) * 1000
    scipy_acc = compute_accuracy(scipy_assign, gt)
    
    print(f"  Accuracy: {scipy_acc:.4f}")
    print(f"  Time:     {scipy_time:.2f} ms")
    
    return {
        'n': n, 'k': k, 'noise': noise,
        'scgm_acc': round(acc, 4),
        'scgm_recall': round(recall, 4),
        'scgm_time_ms': round(timings['total_ms'], 2),
        'scipy_acc': round(scipy_acc, 4),
        'scipy_time_ms': round(scipy_time, 2),
        'speedup': round(compute_speedup(scipy_time, timings['total_ms']), 2),
        'n_ambiguous': timings['n_ambiguous']
    }


# ============================================================
# EXPERIMENT 2: SCALABILITY VS n
# ============================================================

def experiment_scalability(ns=None, k=32, noise=0.0, seed=42):
    """Test scalability for different graph sizes."""
    if ns is None:
        ns = [100, 200, 500, 1000, 2000]
    
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: Scalability (Runtime vs n)")
    print(f"  k={k}, noise={noise}")
    print(f"{'='*60}")
    
    results = []
    
    for n in ns:
        G_A, A = generate_ba_graph(n, m=3, seed=seed)
        B, perm, gt = permute_graph(A, noise_level=noise, seed=seed+1)
        G_B = nx.from_numpy_array(B)
        
        # SCGM Sequential
        t0 = time.perf_counter()
        assignment, timings, P_sparse, candidates = scgm_sequential(
            G_A, G_B, A, B, k=k, verbose=False
        )
        total_time = (time.perf_counter() - t0) * 1000
        
        acc = compute_accuracy(assignment, gt)
        recall = compute_candidate_recall(candidates, gt)
        
        # SciPy (only for smaller n due to memory)
        if n <= 2000:
            FA = compute_graph_features(G_A, A)
            FB = compute_graph_features(G_B, B)
            S_node = compute_node_similarity(FA, FB)
            
            t_s = time.perf_counter()
            scipy_assign = scipy_hungarian_matching(S_node, n, n)
            scipy_time = (time.perf_counter() - t_s) * 1000
            scipy_acc = compute_accuracy(scipy_assign, gt)
        else:
            scipy_time = float('nan')
            scipy_acc = float('nan')
        
        row = {
            'n': n,
            'k': k,
            'scgm_acc': round(acc, 4),
            'recall@k': round(recall, 4),
            'scgm_time_ms': round(total_time, 2),
            'scipy_time_ms': round(scipy_time, 2) if not np.isnan(scipy_time) else 'N/A',
            'speedup': round(compute_speedup(scipy_time, total_time), 2)
                       if not np.isnan(scipy_time) else 'N/A',
            'n_ambiguous': timings['n_ambiguous'],
            'feature_ms': round(timings['feature_extraction_ms'], 2),
            'sinkhorn_ms': round(timings['sinkhorn_ms'], 2),
            'lap_ms': round(timings['reduced_lap_ms'], 2)
        }
        results.append(row)
        print(f"  n={n:5d} | SCGM: {total_time:8.2f}ms | "
              f"SciPy: {scipy_time if not np.isnan(scipy_time) else 'N/A':>8} ms | "
              f"Acc: {acc:.3f} | Recall: {recall:.3f}")
    
    return results


# ============================================================
# EXPERIMENT 3: K-SENSITIVITY
# ============================================================

def experiment_k_sensitivity(n=500, ks=None, noise=0.0, seed=42):
    """Test accuracy and runtime for different k values."""
    if ks is None:
        ks = [8, 16, 32, 64, 128]
    
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: k-Sensitivity")
    print(f"  n={n}, noise={noise}")
    print(f"{'='*60}")
    
    G_A, A = generate_ba_graph(n, m=3, seed=seed)
    B, perm, gt = permute_graph(A, noise_level=noise, seed=seed+1)
    G_B = nx.from_numpy_array(B)
    
    # Precompute features once
    FA = compute_graph_features(G_A, A)
    FB = compute_graph_features(G_B, B)
    S_node = compute_node_similarity(FA, FB)
    
    results = []
    
    for k in ks:
        k = min(k, n)
        
        t0 = time.perf_counter()
        assignment, timings, P_sparse, candidates = scgm_sequential(
            G_A, G_B, A, B, k=k, verbose=False
        )
        total_time = (time.perf_counter() - t0) * 1000
        
        acc = compute_accuracy(assignment, gt)
        recall = compute_candidate_recall(candidates, gt)
        
        row = {
            'k': k,
            'accuracy': round(acc, 4),
            'recall@k': round(recall, 4),
            'time_ms': round(total_time, 2),
            'memory_kb': round(n * k * 4 / 1024, 1),
            'n_ambiguous': timings['n_ambiguous']
        }
        results.append(row)
        print(f"  k={k:4d} | Acc: {acc:.4f} | Recall: {recall:.4f} | "
              f"Time: {total_time:.2f}ms | Mem: {n*k*4/1024:.1f}KB")
    
    return results


# ============================================================
# EXPERIMENT 4: NOISE ROBUSTNESS
# ============================================================

def experiment_noise_robustness(n=500, k=32, 
                                  noise_levels=None, seed=42):
    """Test accuracy under different noise levels."""
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30]
    
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: Noise Robustness")
    print(f"  n={n}, k={k}")
    print(f"{'='*60}")
    
    G_A, A = generate_ba_graph(n, m=3, seed=seed)
    results = []
    
    for noise in noise_levels:
        B, perm, gt = permute_graph(A, noise_level=noise, seed=seed+1)
        G_B = nx.from_numpy_array(B)
        
        # SCGM Sequential
        assignment, timings, P_sparse, candidates = scgm_sequential(
            G_A, G_B, A, B, k=k, verbose=False
        )
        acc = compute_accuracy(assignment, gt)
        recall = compute_candidate_recall(candidates, gt)
        
        # SciPy
        FA = compute_graph_features(G_A, A)
        FB = compute_graph_features(G_B, B)
        S_node = compute_node_similarity(FA, FB)
        t_s = time.perf_counter()
        scipy_assign = scipy_hungarian_matching(S_node, n, n)
        scipy_time = (time.perf_counter() - t_s) * 1000
        scipy_acc = compute_accuracy(scipy_assign, gt)
        
        row = {
            'noise': noise,
            'scgm_acc': round(acc, 4),
            'scipy_acc': round(scipy_acc, 4),
            'recall@k': round(recall, 4),
            'scgm_time_ms': round(timings['total_ms'], 2),
            'scipy_time_ms': round(scipy_time, 2)
        }
        results.append(row)
        print(f"  noise={noise:.0%} | SCGM Acc: {acc:.4f} | "
              f"SciPy Acc: {scipy_acc:.4f} | Recall: {recall:.4f}")
    
    return results


# ============================================================
# EXPERIMENT 5: ABLATION STUDY
# ============================================================

def experiment_ablation(n=500, k=32, noise=0.05, seed=42):
    """Ablation study: contribution of each component."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: Ablation Study")
    print(f"  n={n}, k={k}, noise={noise}")
    print(f"{'='*60}")
    
    G_A, A = generate_ba_graph(n, m=3, seed=seed)
    B, perm, gt = permute_graph(A, noise_level=noise, seed=seed+1)
    G_B = nx.from_numpy_array(B)
    
    FA = compute_graph_features(G_A, A)
    FB = compute_graph_features(G_B, B)
    S_node = compute_node_similarity(FA, FB)
    
    results = []
    
    # V1: Only greedy (no Sinkhorn, no refinement, no LAP)
    t0 = time.perf_counter()
    candidates_v1 = select_top_k_candidates(S_node, k)
    sparse_cost_v1 = build_sparse_cost_vectorized(S_node, candidates_v1)
    assignment_v1, amb_v1, avail_v1 = conflict_aware_extraction(
        sparse_cost_v1, candidates_v1, n, n, tau=0.5)
    # Fill remaining greedily
    for i in range(n):
        if assignment_v1[i] == -1 and avail_v1:
            assignment_v1[i] = avail_v1.pop(0)
    t_v1 = (time.perf_counter() - t0) * 1000
    acc_v1 = compute_accuracy(assignment_v1, gt)
    results.append({
        'version': 'V1: Greedy only',
        'sinkhorn': 'No', 'refinement': 'No', 'reduced_lap': 'No',
        'accuracy': round(acc_v1, 4), 'time_ms': round(t_v1, 2)
    })
    
    # V2: Sinkhorn + Greedy (no dynamic refinement, no LAP)
    t0 = time.perf_counter()
    candidates_v2 = select_top_k_candidates(S_node, k)
    sparse_cost_v2 = build_sparse_cost_vectorized(S_node, candidates_v2)
    P_v2 = sparse_sinkhorn_vectorized(sparse_cost_v2, candidates_v2,
                                       n, n_iter=20)
    assignment_v2, amb_v2, avail_v2 = conflict_aware_extraction(
        P_v2, candidates_v2, n, n, tau=0.5)
    for i in range(n):
        if assignment_v2[i] == -1 and avail_v2:
            assignment_v2[i] = avail_v2.pop(0)
    t_v2 = (time.perf_counter() - t0) * 1000
    acc_v2 = compute_accuracy(assignment_v2, gt)
    results.append({
        'version': 'V2: +Sinkhorn',
        'sinkhorn': 'Yes', 'refinement': 'No', 'reduced_lap': 'No',
        'accuracy': round(acc_v2, 4), 'time_ms': round(t_v2, 2)
    })
    
    # V3: Full SCGM (Sinkhorn + Refinement + Reduced LAP)
    t0 = time.perf_counter()
    assignment_v3, timings_v3, P_v3, cands_v3 = scgm_sequential(
        G_A, G_B, A, B, k=k, n_refinement_iter=2, verbose=False
    )
    t_v3 = (time.perf_counter() - t0) * 1000
    acc_v3 = compute_accuracy(assignment_v3, gt)
    results.append({
        'version': 'V3: Full SCGM',
        'sinkhorn': 'Yes', 'refinement': 'Yes', 'reduced_lap': 'Yes',
        'accuracy': round(acc_v3, 4), 'time_ms': round(t_v3, 2)
    })
    
    # V4: No dynamic refinement
    t0 = time.perf_counter()
    assignment_v4, timings_v4, P_v4, cands_v4 = scgm_sequential(
        G_A, G_B, A, B, k=k, n_refinement_iter=1, verbose=False
    )
    t_v4 = (time.perf_counter() - t0) * 1000
    acc_v4 = compute_accuracy(assignment_v4, gt)
    results.append({
        'version': 'V4: No dyn. refinement',
        'sinkhorn': 'Yes', 'refinement': 'No', 'reduced_lap': 'Yes',
        'accuracy': round(acc_v4, 4), 'time_ms': round(t_v4, 2)
    })
    
    print(f"\n  {'Version':<30} {'Sinkhorn':^10} {'Refine':^10} "
          f"{'RedLAP':^10} {'Accuracy':^10} {'Time(ms)':^10}")
    print(f"  {'-'*80}")
    for r in results:
        print(f"  {r['version']:<30} {r['sinkhorn']:^10} "
              f"{r['refinement']:^10} {r['reduced_lap']:^10} "
              f"{r['accuracy']:^10} {r['time_ms']:^10}")
    
    return results


# ============================================================
# EXPERIMENT 6: GRAPH TYPE COMPARISON
# ============================================================

def experiment_graph_types(n=300, k=32, noise=0.05, seed=42):
    """Compare performance across different graph types."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: Graph Type Comparison")
    print(f"  n={n}, k={k}, noise={noise}")
    print(f"{'='*60}")
    
    graph_generators = {
        'BA (scale-free)': lambda: generate_ba_graph(n, m=3, seed=seed),
        'ER (random)': lambda: generate_er_graph(n, p=0.05, seed=seed),
        'WS (small-world)': lambda: generate_ws_graph(n, k=6, p=0.3,
                                                        seed=seed),
        'SBM (community)': lambda: generate_sbm_graph(n, seed=seed),
    }
    
    results = []
    
    for graph_name, gen_func in graph_generators.items():
        try:
            G_A, A = gen_func()
            B, perm, gt = permute_graph(A, noise_level=noise, seed=seed+1)
            G_B = nx.from_numpy_array(B)
            
            assignment, timings, P_sparse, candidates = scgm_sequential(
                G_A, G_B, A, B, k=k, verbose=False
            )
            
            acc = compute_accuracy(assignment, gt)
            recall = compute_candidate_recall(candidates, gt)
            
            row = {
                'graph_type': graph_name,
                'n': n,
                'n_edges': int(A.sum() / 2),
                'avg_degree': round(A.sum() / n, 2),
                'accuracy': round(acc, 4),
                'recall@k': round(recall, 4),
                'time_ms': round(timings['total_ms'], 2)
            }
            results.append(row)
            print(f"  {graph_name:<25} | Acc: {acc:.4f} | "
                  f"Recall: {recall:.4f} | Time: {timings['total_ms']:.2f}ms")
        
        except Exception as e:
            print(f"  {graph_name}: Error - {e}")
    
    return results


# ============================================================
# EXPERIMENT 7: MEMORY MEASUREMENT
# ============================================================

def experiment_memory(ns=None, k=32, seed=42):
    """Measure peak memory usage: sparse vs dense."""
    if ns is None:
        ns = [100, 500, 1000, 2000]
    
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: Memory Comparison")
    print(f"  k={k}")
    print(f"{'='*60}")
    
    results = []
    
    for n in ns:
        G_A, A = generate_ba_graph(n, m=3, seed=seed)
        B, perm, gt = permute_graph(A, noise_level=0.0, seed=seed+1)
        G_B = nx.from_numpy_array(B)
        
        # SCGM memory
        def run_scgm():
            return scgm_sequential(G_A, G_B, A, B, k=k, verbose=False)
        
        _, scgm_peak_mb = measure_peak_memory_cpu(run_scgm)
        
        # Theoretical dense memory (n x n float32)
        dense_mem_mb = n * n * 4 / 1024 / 1024
        sparse_mem_mb = n * k * 4 / 1024 / 1024
        
        row = {
            'n': n,
            'k': k,
            'dense_theory_mb': round(dense_mem_mb, 2),
            'sparse_theory_mb': round(sparse_mem_mb, 3),
            'scgm_peak_mb': round(scgm_peak_mb, 2),
            'memory_reduction': round(dense_mem_mb / max(sparse_mem_mb, 0.001), 1)
        }
        results.append(row)
        print(f"  n={n:5d} | Dense: {dense_mem_mb:8.2f}MB | "
              f"Sparse(theory): {sparse_mem_mb:6.3f}MB | "
              f"SCGM Peak: {scgm_peak_mb:6.2f}MB | "
              f"Reduction: {row['memory_reduction']}x")
    
    return results


# ============================================================
# MAIN RUNNER
# ============================================================

def main():
    """Run all CPU Sequential experiments."""
    print("\n" + "="*70)
    print("  SCGM-Seq: CPU Sequential Sparse Candidate Graph Matching")
    print("="*70)
    print_complexity_analysis()
    
    all_results = {}
    
    # --- Correctness Test ---
    r1 = experiment_basic_correctness(n=300, k=32, noise=0.0)
    all_results['basic'] = [r1]
    
    # --- Scalability ---
    r2 = experiment_scalability(ns=[100, 200, 500, 1000], k=32)
    all_results['scalability'] = r2
    save_results_csv('results/seq_scalability.csv', r2)
    print_result_table(r2, "Scalability Results")
    
    # --- k-Sensitivity ---
    r3 = experiment_k_sensitivity(n=300, ks=[8, 16, 32, 64])
    all_results['k_sensitivity'] = r3
    save_results_csv('results/seq_k_sensitivity.csv', r3)
    print_result_table(r3, "k-Sensitivity Results")
    
    # --- Noise Robustness ---
    r4 = experiment_noise_robustness(n=300, k=32,
                                      noise_levels=[0.0, 0.05, 0.1, 0.2])
    all_results['noise'] = r4
    save_results_csv('results/seq_noise_robustness.csv', r4)
    print_result_table(r4, "Noise Robustness Results")
    
    # --- Ablation Study ---
    r5 = experiment_ablation(n=300, k=32, noise=0.05)
    all_results['ablation'] = r5
    save_results_csv('results/seq_ablation.csv', r5)
    
    # --- Graph Types ---
    r6 = experiment_graph_types(n=200, k=32, noise=0.05)
    all_results['graph_types'] = r6
    save_results_csv('results/seq_graph_types.csv', r6)
    print_result_table(r6, "Graph Type Results")
    
    # --- Memory ---
    r7 = experiment_memory(ns=[100, 500, 1000], k=32)
    all_results['memory'] = r7
    save_results_csv('results/seq_memory.csv', r7)
    print_result_table(r7, "Memory Results")
    
    print("\n[SCGM-Seq] All experiments completed.")
    print("Results saved in results/ directory.")
    
    return all_results


if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)
    main()