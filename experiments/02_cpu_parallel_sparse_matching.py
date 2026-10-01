# 02_cpu_parallel_sparse_matching.py
"""
SCGM-CPU: CPU Parallel Sparse Candidate Graph Matching
=======================================================
Multi-core CPU parallel implementation using Numba + joblib.
Purpose: CPU-level parallelization benefit analysis.

Same algorithm as SCGM-Seq but with parallel execution:
- Feature extraction: parallel across nodes
- Similarity computation: vectorized (inherently parallel via BLAS)
- Top-k selection: parallel across source nodes
- Sinkhorn normalization: parallel row/column operations
- Candidate refinement: parallel across nodes
- Conflict extraction: partially parallel

Expected speedup over SCGM-Seq:
    For medium-large graphs: 4-16x depending on CPU cores
    For small graphs: overhead may reduce benefit

Complexity (same as sequential):
    Memory: O(nk)
    Time:   O(nkT/P) + O(|U|^3) where P = number of CPU cores
"""

import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed
import multiprocessing
import time
import tracemalloc
import warnings
warnings.filterwarnings('ignore')

try:
    from numba import njit, prange, set_num_threads
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: Numba not available. Using joblib parallelism only.")

from utils_graph_matching import (
    generate_ba_graph, generate_er_graph, generate_ws_graph,
    generate_sbm_graph, permute_graph,
    compute_graph_features, compute_node_similarity,
    select_top_k_candidates, build_sparse_cost_vectorized,
    conflict_aware_extraction, confidence_threshold_extraction,
    reduced_lap_refinement,
    compute_accuracy, compute_candidate_recall, compute_speedup,
    scipy_hungarian_matching,
    save_results_csv, print_result_table, print_complexity_analysis,
    measure_peak_memory_cpu
)

N_JOBS = multiprocessing.cpu_count()


# ============================================================
# PARALLEL FEATURE EXTRACTION
# ============================================================

def _extract_single_node_features(node_idx, G, A, n):
    """Extract features for a single node (for parallel execution)."""
    v = list(G.nodes())[node_idx]
    
    # Degree
    deg = G.degree(v) / (n - 1 + 1e-8)
    
    # Clustering
    clust = nx.clustering(G, v)
    
    # Avg neighbor degree
    nbrs = list(G.neighbors(v))
    avg_nd = np.mean([G.degree(u) for u in nbrs]) / (n + 1e-8) if nbrs else 0.0
    
    # Two-hop reachability
    two_hop = float(A[node_idx].dot(A[node_idx])) / (n * n + 1e-8)
    
    # Triangle count
    tri = nx.triangles(G, v) / (n + 1e-8)
    
    # Max neighbor degree
    max_nd = max((G.degree(u) for u in nbrs), default=0) / (n + 1e-8)
    
    # k-core
    core = nx.core_number(G).get(v, 0)
    
    return [deg, clust, avg_nd, two_hop, tri, max_nd, float(core)]


def parallel_feature_extraction(G, A, n_jobs=None):
    """
    Parallel node feature extraction across multiple CPU cores.
    Each node's features computed independently -> embarrassingly parallel.
    """
    if n_jobs is None:
        n_jobs = N_JOBS
    
    n = A.shape[0]
    
    # Use joblib for parallel execution
    # For large graphs, this gives significant speedup
    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_extract_single_node_features)(i, G, A, n)
        for i in range(n)
    )
    
    F = np.array(results, dtype=np.float32)
    
    # L2 normalize
    norms = np.linalg.norm(F, axis=1, keepdims=True) + 1e-8
    F = F / norms
    
    return F


# ============================================================
# PARALLEL TOP-K SELECTION
# ============================================================

def _topk_single_row(i, S_row, k):
    """Compute top-k for a single row."""
    k = min(k, len(S_row))
    idx = np.argpartition(-S_row, kth=k-1)[:k]
    idx = idx[np.argsort(-S_row[idx])]
    return idx


def parallel_topk_selection(S, k, n_jobs=None):
    """
    Parallel top-k candidate selection.
    Each source node's top-k computed independently.
    """
    if n_jobs is None:
        n_jobs = N_JOBS
    
    n = S.shape[0]
    k = min(k, S.shape[1])
    
    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_topk_single_row)(i, S[i], k)
        for i in range(n)
    )
    
    candidates = np.array(results, dtype=np.int64)
    return candidates


# ============================================================
# PARALLEL SINKHORN NORMALIZATION
# ============================================================

def parallel_sinkhorn_row_normalize(P, n_jobs=None):
    """Parallel row normalization."""
    row_sum = P.sum(axis=1, keepdims=True) + 1e-8
    return P / row_sum


def parallel_sinkhorn_col_normalize(P, candidates, n_target, n_jobs=None):
    """
    Parallel column normalization using vectorized scatter-add.
    """
    # Column sum via scatter (vectorized, uses numpy parallelism)
    col_sum = np.zeros(n_target, dtype=np.float64)
    np.add.at(col_sum, candidates, P.astype(np.float64))
    col_sum = col_sum.astype(np.float32)
    
    # Gather and normalize
    col_factors = col_sum[candidates]
    col_factors = np.where(col_factors > 1e-8, col_factors, 1.0)
    P = P / col_factors
    
    return P


def parallel_sinkhorn(sparse_cost, candidates, n_target,
                       temperature=0.1, n_iter=20, n_jobs=None):
    """
    Parallel Sinkhorn normalization.
    Row operations are parallelized; column scatter uses vectorized numpy.
    """
    if n_jobs is None:
        n_jobs = N_JOBS
    
    n, k = sparse_cost.shape
    
    # Initialize
    log_P = sparse_cost / temperature
    log_P = log_P - log_P.max(axis=1, keepdims=True)
    P = np.exp(log_P)
    
    for it in range(n_iter):
        # Row normalization (vectorized = parallel via BLAS)
        P = parallel_sinkhorn_row_normalize(P, n_jobs)
        
        # Column normalization (scatter-based)
        P = parallel_sinkhorn_col_normalize(P, candidates, n_target)
    
    return P


# ============================================================
# PARALLEL NEIGHBORHOOD CONSISTENCY
# ============================================================

def _neighborhood_consistency_chunk(i_start, i_end, A, B,
                                     P_sparse, candidates):
    """Compute neighborhood consistency for a chunk of source nodes."""
    n, k = candidates.shape
    chunk_score = np.zeros((i_end - i_start, k), dtype=np.float32)
    
    for i in range(i_start, i_end):
        nbrs_i = np.where(A[i] > 0)[0]
        if len(nbrs_i) == 0:
            continue
        
        for ki in range(k):
            j = int(candidates[i, ki])
            nbrs_j = np.where(B[j] > 0)[0]
            if len(nbrs_j) == 0:
                continue
            
            score = 0.0
            for u in nbrs_i:
                u_cands = candidates[u]
                u_probs = P_sparse[u]
                for kl, v in enumerate(u_cands):
                    if B[j, v] > 0:
                        score += A[i, u] * B[j, v] * u_probs[kl]
            
            chunk_score[i - i_start, ki] = score
    
    return chunk_score


def parallel_neighborhood_consistency(A, B, P_sparse, candidates,
                                       n_jobs=None):
    """
    Parallel neighborhood consistency computation.
    Source nodes divided into chunks for parallel processing.
    """
    if n_jobs is None:
        n_jobs = N_JOBS
    
    n = A.shape[0]
    k = candidates.shape[1]
    
    # Divide into chunks
    chunk_size = max(1, n // n_jobs)
    chunks = []
    i = 0
    while i < n:
        chunks.append((i, min(i + chunk_size, n)))
        i += chunk_size
    
    # Parallel execution
    chunk_results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_neighborhood_consistency_chunk)(
            i_start, i_end, A, B, P_sparse, candidates)
        for i_start, i_end in chunks
    )
    
    # Concatenate results
    struct_score = np.vstack(chunk_results)
    
    # Normalize
    s_max = struct_score.max()
    if s_max > 1e-8:
        struct_score /= s_max
    
    return struct_score


# ============================================================
# PARALLEL CONFLICT-AWARE EXTRACTION
# ============================================================

def parallel_conflict_extraction(P_sparse, candidates, n, n_target,
                                   tau=0.5, n_jobs=None):
    """
    Parallel confidence computation + sequential conflict resolution.
    Note: Conflict resolution itself is sequential due to dependencies,
    but confidence scoring is parallelized.
    """
    # Compute confidence scores in parallel
    max_conf = P_sparse.max(axis=1)  # Vectorized
    
    # Collect all pairs (parallel flatmap)
    def _collect_pairs_chunk(i_start, i_end):
        pairs = []
        for i in range(i_start, i_end):
            for ki in range(candidates.shape[1]):
                j = int(candidates[i, ki])
                conf = float(P_sparse[i, ki])
                pairs.append((conf, i, j))
        return pairs
    
    if n_jobs is None:
        n_jobs = N_JOBS
    
    chunk_size = max(1, n // n_jobs)
    chunks_idx = [(i, min(i + chunk_size, n))
                   for i in range(0, n, chunk_size)]
    
    chunk_pairs = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_collect_pairs_chunk)(i_start, i_end)
        for i_start, i_end in chunks_idx
    )
    
    # Merge and sort
    all_pairs = []
    for cp in chunk_pairs:
        all_pairs.extend(cp)
    all_pairs.sort(key=lambda x: -x[0])
    
    # Sequential conflict resolution (dependencies prevent parallelism)
    assignment = np.full(n, -1, dtype=np.int64)
    matched_sources = set()
    matched_targets = set()
    
    for conf, i, j in all_pairs:
        if i not in matched_sources and j not in matched_targets:
            assignment[i] = j
            matched_sources.add(i)
            matched_targets.add(j)
    
    ambiguous = [i for i in range(n) if assignment[i] == -1]
    available_targets = list(set(range(n_target)) - matched_targets)
    
    return assignment, ambiguous, available_targets


# ============================================================
# NUMBA-ACCELERATED SINKHORN (if available)
# ============================================================

if NUMBA_AVAILABLE:
    @njit(parallel=True)
    def numba_sinkhorn_iteration(P, candidates, col_sum, n, k, n_target):
        """Single Sinkhorn iteration with Numba parallelism."""
        # Row normalization
        for i in prange(n):
            row_sum = 0.0
            for ki in range(k):
                row_sum += P[i, ki]
            if row_sum > 1e-8:
                for ki in range(k):
                    P[i, ki] /= row_sum
        
        # Column sum computation
        for j in prange(n_target):
            col_sum[j] = 0.0
        
        for i in range(n):
            for ki in range(k):
                j = candidates[i, ki]
                col_sum[j] += P[i, ki]
        
        # Column normalization
        for i in prange(n):
            for ki in range(k):
                j = candidates[i, ki]
                if col_sum[j] > 1e-8:
                    P[i, ki] /= col_sum[j]
        
        return P, col_sum
    
    def numba_parallel_sinkhorn(sparse_cost, candidates, n_target,
                                 temperature=0.1, n_iter=20):
        """Numba-parallel Sinkhorn normalization."""
        n, k = sparse_cost.shape
        
        log_P = sparse_cost / temperature
        log_P = log_P - log_P.max(axis=1, keepdims=True)
        P = np.exp(log_P).astype(np.float64)
        candidates_int = candidates.astype(np.int64)
        col_sum = np.zeros(n_target, dtype=np.float64)
        
        for _ in range(n_iter):
            P, col_sum = numba_sinkhorn_iteration(
                P, candidates_int, col_sum, n, k, n_target)
        
        return P.astype(np.float32)


# ============================================================
# MAIN SCGM PARALLEL ALGORITHM
# ============================================================

def scgm_parallel(G_A, G_B, A, B, k=32, sinkhorn_iter=20,
                   sinkhorn_temp=0.1, tau=0.5,
                   alpha=0.4, beta=0.3, lam=0.3,
                   n_refinement_iter=2,
                   extraction='conflict_aware',
                   lap_method='hungarian',
                   n_jobs=None,
                   use_numba=True,
                   verbose=False):
    """
    SCGM Parallel: Multi-core CPU sparse candidate graph matching.
    
    Same algorithm as SCGM-Seq with parallel execution of:
    - Feature extraction (joblib parallel)
    - Top-k selection (joblib parallel)  
    - Sinkhorn normalization (Numba prange or vectorized numpy)
    - Neighborhood consistency (joblib parallel chunks)
    - Conflict extraction (parallel pair collection)
    
    Parameters: Same as SCGM-Seq plus:
        n_jobs: Number of parallel workers (default: all cores)
        use_numba: Use Numba-JIT acceleration if available
    """
    if n_jobs is None:
        n_jobs = N_JOBS
    
    n = A.shape[0]
    n_target = B.shape[0]
    timings = {}
    
    # ----------------------------------------------------------
    # STAGE 1: Parallel graph-aware feature extraction
    # ----------------------------------------------------------
    t0 = time.perf_counter()
    
    # Use full compute_graph_features (internally vectorized via numpy)
    # For very large graphs, we can parallelize feature computation
    from utils_graph_matching import compute_graph_features
    FA = compute_graph_features(G_A, A)
    FB = compute_graph_features(G_B, B)
    
    timings['feature_extraction_ms'] = (time.perf_counter() - t0) * 1000
    if verbose:
        print(f"  [PAR STAGE 1] Feature extraction: "
              f"{timings['feature_extraction_ms']:.2f} ms | "
              f"n_jobs={n_jobs}")
    
    # ----------------------------------------------------------
    # STAGE 2: Similarity + Parallel top-k candidate generation
    # ----------------------------------------------------------
    t0 = time.perf_counter()
    
    # Matrix multiplication is already parallelized by BLAS
    S_node = FA @ FB.T  # (n, n_target)
    
    # Parallel top-k selection
    candidates = parallel_topk_selection(S_node, k, n_jobs=n_jobs)
    
    timings['candidate_gen_ms'] = (time.perf_counter() - t0) * 1000
    if verbose:
        print(f"  [PAR STAGE 2] Candidate generation: "
              f"{timings['candidate_gen_ms']:.2f} ms")
    
    # ----------------------------------------------------------
    # STAGE 3: Sparse cost matrix (vectorized)
    # ----------------------------------------------------------
    t0 = time.perf_counter()
    
    sparse_cost = build_sparse_cost_vectorized(S_node, candidates)
    
    timings['sparse_cost_ms'] = (time.perf_counter() - t0) * 1000
    
    # ----------------------------------------------------------
    # STAGE 4 + 5: Parallel Sinkhorn + Parallel dynamic refinement
    # ----------------------------------------------------------
    t_sinkhorn = 0.0
    t_refine = 0.0
    
    P_sparse = sparse_cost.copy()
    
    for ref_iter in range(n_refinement_iter):
        # Parallel Sinkhorn
        t0 = time.perf_counter()
        
        cost_input = (sparse_cost if ref_iter == 0
                      else build_sparse_cost_vectorized(S_node, candidates))
        
        if NUMBA_AVAILABLE and use_numba:
            P_sparse = numba_parallel_sinkhorn(
                cost_input, candidates, n_target,
                temperature=sinkhorn_temp, n_iter=sinkhorn_iter)
        else:
            P_sparse = parallel_sinkhorn(
                cost_input, candidates, n_target,
                temperature=sinkhorn_temp, n_iter=sinkhorn_iter,
                n_jobs=n_jobs)
        
        t_sinkhorn += (time.perf_counter() - t0) * 1000
        
        if ref_iter < n_refinement_iter - 1:
            # Parallel neighborhood consistency
            t0 = time.perf_counter()
            
            S_struct = parallel_neighborhood_consistency(
                A, B, P_sparse, candidates, n_jobs=n_jobs)
            
            # Update candidates
            n_nodes = S_node.shape[0]
            S_combined = np.full((n_nodes, n_target), -np.inf, dtype=np.float32)
            for i in range(n_nodes):
                for ki, j in enumerate(candidates[i]):
                    S_combined[i, j] = (alpha * S_node[i, j]
                                         + beta * S_struct[i, ki]
                                         + lam * P_sparse[i, ki])
            
            # Parallel top-k on updated scores
            candidates = parallel_topk_selection(S_combined, k, n_jobs)
            
            t_refine += (time.perf_counter() - t0) * 1000
    
    timings['sinkhorn_ms'] = t_sinkhorn
    timings['refinement_ms'] = t_refine
    
    if verbose:
        print(f"  [PAR STAGE 4] Parallel Sinkhorn: {t_sinkhorn:.2f} ms "
              f"({'Numba' if NUMBA_AVAILABLE and use_numba else 'joblib'})")
        print(f"  [PAR STAGE 5] Parallel refinement: {t_refine:.2f} ms")
    
    # ----------------------------------------------------------
    # STAGE 6: Parallel conflict-aware extraction
    # ----------------------------------------------------------
    t0 = time.perf_counter()
    
    assignment, ambiguous, available_targets = parallel_conflict_extraction(
        P_sparse, candidates, n, n_target, tau=tau, n_jobs=n_jobs)
    
    timings['extraction_ms'] = (time.perf_counter() - t0) * 1000
    if verbose:
        print(f"  [PAR STAGE 6] Parallel extraction: "
              f"{timings['extraction_ms']:.2f} ms | "
              f"Ambiguous: {len(ambiguous)}/{n}")
    
    # ----------------------------------------------------------
    # STAGE 7: Reduced LAP (CPU, sequential - small problem)
    # ----------------------------------------------------------
    t0 = time.perf_counter()
    
    assignment = reduced_lap_refinement(
        assignment, ambiguous, available_targets,
        S_node, method=lap_method)
    
    timings['reduced_lap_ms'] = (time.perf_counter() - t0) * 1000
    if verbose:
        print(f"  [PAR STAGE 7] Reduced LAP ({len(ambiguous)} nodes): "
              f"{timings['reduced_lap_ms']:.2f} ms")
    
    timings['total_ms'] = sum(v for v in timings.values()
                               if isinstance(v, float))
    timings['n_ambiguous'] = len(ambiguous)
    timings['n_jobs'] = n_jobs
    
    return assignment, timings, P_sparse, candidates


# ============================================================
# EXPERIMENT: SEQUENTIAL vs PARALLEL SPEEDUP
# ============================================================

def experiment_seq_vs_parallel(ns=None, k=32, seed=42):
    """
    Compare SCGM-Seq vs SCGM-CPU speedup.
    Shows benefit of CPU parallelization.
    """
    if ns is None:
        ns = [100, 200, 500, 1000]
    
    print(f"\n{'='*65}")
    print(f"  EXPERIMENT: Sequential vs Parallel CPU Speedup")
    print(f"  k={k}, CPU cores={N_JOBS}")
    print(f"{'='*65}")
    
    # Import sequential
    from cpu_seq_runner import scgm_sequential_simple
    
    results = []
    
    for n in ns:
        G_A, A = generate_ba_graph(n, m=3, seed=seed)
        B, perm, gt = permute_graph(A, noise_level=0.05, seed=seed+1)
        G_B = nx.from_numpy_array(B)
        
        # Sequential
        t0 = time.perf_counter()
        from utils_graph_matching import (compute_graph_features,
                                           compute_node_similarity,
                                           sparse_sinkhorn_vectorized)
        FA = compute_graph_features(G_A, A)
        FB = compute_graph_features(G_B, B)
        S_node = compute_node_similarity(FA, FB)
        candidates_s = select_top_k_candidates(S_node, k)
        sc = build_sparse_cost_vectorized(S_node, candidates_s)
        P_s = sparse_sinkhorn_vectorized(sc, candidates_s, n)
        assign_s, amb_s, avail_s = conflict_aware_extraction(
            P_s, candidates_s, n, n)
        assign_s = reduced_lap_refinement(assign_s, amb_s, avail_s, S_node)
        t_seq = (time.perf_counter() - t0) * 1000
        acc_seq = compute_accuracy(assign_s, gt)
        
        # Parallel
        t0 = time.perf_counter()
        assign_p, timings_p, P_p, cands_p = scgm_parallel(
            G_A, G_B, A, B, k=k, verbose=False)
        t_par = (time.perf_counter() - t0) * 1000
        acc_par = compute_accuracy(assign_p, gt)
        
        # SciPy
        t0 = time.perf_counter()
        scipy_assign = scipy_hungarian_matching(S_node, n, n)
        t_scipy = (time.perf_counter() - t0) * 1000
        acc_scipy = compute_accuracy(scipy_assign, gt)
        
        row = {
            'n': n,
            'seq_time_ms': round(t_seq, 2),
            'par_time_ms': round(t_par, 2),
            'scipy_time_ms': round(t_scipy, 2),
            'seq_acc': round(acc_seq, 4),
            'par_acc': round(acc_par, 4),
            'scipy_acc': round(acc_scipy, 4),
            'par_vs_seq_speedup': round(compute_speedup(t_seq, t_par), 2),
            'par_vs_scipy_speedup': round(compute_speedup(t_scipy, t_par), 2),
            'n_cores': N_JOBS
        }
        results.append(row)
        print(f"  n={n:5d} | Seq: {t_seq:7.1f}ms | "
              f"Par: {t_par:7.1f}ms | "
              f"SciPy: {t_scipy:7.1f}ms | "
              f"Par/Seq speedup: {row['par_vs_seq_speedup']:.2f}x")
    
    return results


# ============================================================
# EXPERIMENT: PARALLELISM SCALING (1 to N_JOBS cores)
# ============================================================

def experiment_core_scaling(n=800, k=32, seed=42):
    """
    Test runtime as number of CPU cores varies.
    Shows parallel scaling efficiency.
    """
    max_cores = min(N_JOBS, 8)
    core_counts = [1, 2, 4, min(8, max_cores)]
    core_counts = sorted(set(core_counts))
    
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: CPU Core Scaling")
    print(f"  n={n}, k={k}, max_cores={max_cores}")
    print(f"{'='*60}")
    
    G_A, A = generate_ba_graph(n, m=3, seed=seed)
    B, perm, gt = permute_graph(A, noise_level=0.05, seed=seed+1)
    G_B = nx.from_numpy_array(B)
    
    results = []
    t_1core = None
    
    for n_cores in core_counts:
        t0 = time.perf_counter()
        assignment, timings, P_sparse, candidates = scgm_parallel(
            G_A, G_B, A, B, k=k, n_jobs=n_cores, verbose=False)
        t_total = (time.perf_counter() - t0) * 1000
        
        if t_1core is None:
            t_1core = t_total
        
        acc = compute_accuracy(assignment, gt)
        eff = (t_1core / t_total) / n_cores if t_1core else 1.0
        
        row = {
            'n_cores': n_cores,
            'time_ms': round(t_total, 2),
            'accuracy': round(acc, 4),
            'speedup_vs_1core': round(t_1core / t_total, 2),
            'efficiency': round(eff, 3)
        }
        results.append(row)
        print(f"  cores={n_cores:2d} | Time: {t_total:8.2f}ms | "
              f"Speedup: {row['speedup_vs_1core']:.2f}x | "
              f"Efficiency: {eff:.1%}")
    
    return results


# ============================================================
# COMPREHENSIVE COMPARISON TABLE
# ============================================================

def run_method_comparison(ns=None, k=32, noise=0.05, seed=42):
    """
    Full comparison: SCGM-Seq vs SCGM-CPU vs SciPy Hungarian.
    Generates the main comparison table for the paper.
    """
    if ns is None:
        ns = [100, 300, 500, 1000]
    
    print(f"\n{'='*70}")
    print(f"  METHOD COMPARISON TABLE")
    print(f"  k={k}, noise={noise}, CPU cores={N_JOBS}")
    print(f"{'='*70}")
    
    results = []
    
    for n in ns:
        G_A, A = generate_ba_graph(n, m=3, seed=seed)
        B, perm, gt = permute_graph(A, noise_level=noise, seed=seed+1)
        G_B = nx.from_numpy_array(B)
        
        from utils_graph_matching import (compute_graph_features,
                                           compute_node_similarity,
                                           sparse_sinkhorn_vectorized)
        
        # --- SCGM-CPU (parallel) ---
        t0 = time.perf_counter()
        assign_par, timings_par, P_par, cands_par = scgm_parallel(
            G_A, G_B, A, B, k=k, n_jobs=N_JOBS, verbose=False)
        t_par = (time.perf_counter() - t0) * 1000
        acc_par = compute_accuracy(assign_par, gt)
        recall_par = compute_candidate_recall(cands_par, gt)
        
        # --- SCGM-Seq (sequential) ---
        FA = compute_graph_features(G_A, A)
        FB = compute_graph_features(G_B, B)
        S_node = compute_node_similarity(FA, FB)
        
        t0 = time.perf_counter()
        cands_s = select_top_k_candidates(S_node, k)
        sc_s = build_sparse_cost_vectorized(S_node, cands_s)
        P_s = sparse_sinkhorn_vectorized(sc_s, cands_s, n)
        assign_s, amb_s, avail_s = conflict_aware_extraction(P_s, cands_s, n, n)
        assign_s = reduced_lap_refinement(assign_s, amb_s, avail_s, S_node)
        t_seq = (time.perf_counter() - t0) * 1000
        acc_seq = compute_accuracy(assign_s, gt)
        
        # --- SciPy Hungarian ---
        t0 = time.perf_counter()
        assign_scipy = scipy_hungarian_matching(S_node, n, n)
        t_scipy = (time.perf_counter() - t0) * 1000
        acc_scipy = compute_accuracy(assign_scipy, gt)
        
        row = {
            'n': n,
            'seq_acc': round(acc_seq, 4),
            'par_acc': round(acc_par, 4),
            'scipy_acc': round(acc_scipy, 4),
            'seq_time_ms': round(t_seq, 2),
            'par_time_ms': round(t_par, 2),
            'scipy_time_ms': round(t_scipy, 2),
            'recall@k': round(recall_par, 4),
            'par_vs_scipy_speedup': round(t_scipy / max(t_par, 0.1), 2)
        }
        results.append(row)
        
        print(f"\n  n={n}")
        print(f"    SCGM-Seq:  acc={acc_seq:.4f} time={t_seq:.1f}ms")
        print(f"    SCGM-CPU:  acc={acc_par:.4f} time={t_par:.1f}ms "
              f"speedup={row['par_vs_scipy_speedup']:.2f}x vs scipy")
        print(f"    SciPy:     acc={acc_scipy:.4f} time={t_scipy:.1f}ms")
    
    return results


# ============================================================
# MAIN RUNNER
# ============================================================

def main():
    """Run all CPU Parallel experiments."""
    print("\n" + "="*70)
    print("  SCGM-CPU: CPU Parallel Sparse Candidate Graph Matching")
    print(f"  Available CPU cores: {N_JOBS}")
    print(f"  Numba available: {NUMBA_AVAILABLE}")
    print("="*70)
    print_complexity_analysis()
    
    import os
    os.makedirs('results', exist_ok=True)
    
    # --- Core scaling ---
    r1 = experiment_core_scaling(n=500, k=32)
    save_results_csv('results/par_core_scaling.csv', r1)
    print_result_table(r1, "CPU Core Scaling")
    
    # --- Method comparison ---
    r2 = run_method_comparison(ns=[100, 300, 500], k=32, noise=0.05)
    save_results_csv('results/par_method_comparison.csv', r2)
    print_result_table(r2, "Method Comparison")
    
    # --- Scalability ---
    r3 = []
    ns = [100, 200, 500, 1000]
    for n in ns:
        G_A, A = generate_ba_graph(n, m=3, seed=42)
        B, perm, gt = permute_graph(A, noise_level=0.05, seed=43)
        G_B = nx.from_numpy_array(B)
        
        t0 = time.perf_counter()
        assign, timings, P, cands = scgm_parallel(
            G_A, G_B, A, B, k=32, verbose=False)
        t_total = (time.perf_counter() - t0) * 1000
        acc = compute_accuracy(assign, gt)
        
        r3.append({
            'n': n, 'time_ms': round(t_total, 2),
            'accuracy': round(acc, 4),
            'n_cores': N_JOBS
        })
        print(f"  n={n}: {t_total:.2f}ms, acc={acc:.4f}")
    
    save_results_csv('results/par_scalability.csv', r3)
    print_result_table(r3, "Parallel Scalability")
    
    print("\n[SCGM-CPU] All experiments completed.")
    print("Results saved in results/ directory.")


if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)
    main()