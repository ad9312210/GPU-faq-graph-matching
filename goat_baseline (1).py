# -*- coding: utf-8 -*-
# ================================================================
#   GOAT BASELINE IMPLEMENTATION
#
#   GOAT: Graph Optimal Alignment Tool
#   Reference: Pentagon Fabrizio et al. (typical graph matching baseline)
#   
#   Uses Frank-Wolfe method on quadratic assignment relaxation.
# ================================================================

import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def goat_match(A, B, max_iter=30, tol=1e-6, verbose=False):
    """
    GOAT-style graph matching via Frank-Wolfe method.
    
    Solves relaxed QAP: min_P ||AP - PB||_F^2
    where P is doubly stochastic matrix.
    
    Args:
        A, B: adjacency matrices (n x n)
        max_iter: Frank-Wolfe iterations
        tol: convergence tolerance
    
    Returns:
        matching: (n,) array where matching[i] = j means A_i ↔ B_j
    """
    n = A.shape[0]
    
    # Initialize: uniform doubly stochastic
    P = np.ones((n, n), dtype=np.float32) / n
    
    prev_obj = float('inf')
    
    for iteration in range(max_iter):
        # Gradient of ||AP - PB||_F^2 with respect to P
        # ∇ = 2 * A^T * (AP - PB) - 2 * (AP - PB) * B^T
        AP = A @ P
        PB = P @ B
        gradient = 2 * (A.T @ (AP - PB) - (AP - PB) @ B.T)
        
        # Linear minimization: find best permutation matrix Q
        # minimize <gradient, Q> subject to Q being a permutation
        # This is a linear assignment problem
        row_ind, col_ind = linear_sum_assignment(gradient)
        Q = np.zeros((n, n), dtype=np.float32)
        Q[row_ind, col_ind] = 1
        
        # Line search: alpha ∈ [0, 1]
        # Minimize ||A(P + alpha(Q-P)) - (P + alpha(Q-P))B||^2
        D = Q - P
        AD = A @ D
        DB = D @ B
        
        # Numerator: -<gradient, D>
        # Denominator: 2 * ||AD - DB||^2
        num = -np.sum(gradient * D)
        denom = 2 * np.sum((AD - DB) ** 2)
        
        if denom > 1e-10:
            alpha = np.clip(num / denom, 0, 1)
        else:
            alpha = 0.5
        
        # Update
        P_new = P + alpha * D
        
        # Compute new objective
        AP_new = A @ P_new
        PB_new = P_new @ B
        curr_obj = float(np.linalg.norm(AP_new - PB_new, 'fro') ** 2)
        
        if verbose and iteration % 5 == 0:
            print(f"    GOAT iter {iteration}: obj={curr_obj:.4f}, alpha={alpha:.4f}")
        
        # Check convergence
        if abs(prev_obj - curr_obj) < tol:
            break
        
        prev_obj = curr_obj
        P = P_new
    
    # Discretize: project P to permutation matrix via Hungarian
    row_ind, col_ind = linear_sum_assignment(-P)  # negate to maximize
    matching = np.full(n, -1, dtype=np.int32)
    matching[row_ind] = col_ind
    
    return matching


def goat_match_with_timing(A, B, max_iter=30):
    """GOAT with timing information."""
    t0 = time.perf_counter()
    matching = goat_match(A, B, max_iter=max_iter)
    elapsed = time.perf_counter() - t0
    
    n = A.shape[0]
    return {
        'matching': matching,
        'timings': {'total': elapsed},
        'memory_mb': (n * n * 4 * 3) / 1e6,  # P, gradient, Q
        'method': 'GOAT (Frank-Wolfe)',
    }


# ================================================================
# DEMO
# ================================================================
if __name__ == "__main__":
    import networkx as nx
    
    print("=" * 70)
    print("  GOAT BASELINE TEST")
    print("=" * 70)
    
    # Test on karate club
    G = nx.karate_club_graph()
    A = nx.to_numpy_array(G).astype(np.float32)
    n = A.shape[0]
    
    # Create noisy pair
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    B = A[perm][:, perm].copy()
    gt = np.argsort(perm).astype(np.int32)
    
    print(f"\n  Test graph: Karate Club (n={n})")
    
    # Run GOAT
    print(f"\n  Running GOAT...")
    result = goat_match_with_timing(A, B, max_iter=30)
    acc = np.mean(result['matching'] == gt)
    
    print(f"  Time:     {result['timings']['total']*1000:.2f} ms")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Memory:   {result['memory_mb']:.4f} MB")
    
    # Compare QAP objective
    P = np.zeros((n, n), dtype=np.float32)
    valid = result['matching'] >= 0
    P[np.arange(n)[valid], result['matching'][valid]] = 1
    qap = float(np.linalg.norm(A @ P - P @ B, 'fro') ** 2)
    print(f"  QAP obj:  {qap:.2f}")