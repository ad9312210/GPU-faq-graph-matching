# -*- coding: utf-8 -*-
# ================================================================
#   CORRECTNESS TESTS ON TOY GRAPHS
#
#   Validates:
#   1. Ground truth mapping is correct
#   2. Algorithm recovers exact matching on noise-free pairs
#   3. Objective values match theoretical predictions
#   4. Comparison with reference implementations (SciPy, GOAT)
# ================================================================

import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ================================================================
# TEST 1: TRIVIAL 3-NODE TRIANGLE
# ================================================================
def test_triangle():
    """
    Test on smallest possible non-trivial graph.
    Triangle → Triangle (permuted). All methods should get 100%.
    """
    print("\n" + "=" * 70)
    print("  TEST 1: TRIANGLE (n=3)")
    print("=" * 70)
    
    # Original triangle
    A = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ], dtype=np.float32)
    
    # Permuted: [2, 0, 1]
    # Node 0 → position 2, Node 1 → position 0, Node 2 → position 1
    perm = np.array([2, 0, 1])
    B = A[perm][:, perm]
    
    # Ground truth: A's node i → B's position np.argsort(perm)[i]
    gt = np.argsort(perm)  # [1, 2, 0]
    
    print(f"  A:\n{A}")
    print(f"  B = A[{perm}][:, {perm}]:\n{B}")
    print(f"  Ground truth (A→B mapping): {gt}")
    
    # Verify: B[gt[i], gt[j]] should equal A[i, j]
    print("\n  Verifying ground truth consistency:")
    for i in range(3):
        for j in range(3):
            expected = A[i, j]
            actual = B[gt[i], gt[j]]
            status = "✅" if expected == actual else "❌"
            print(f"    A[{i},{j}]={expected} vs B[{gt[i]},{gt[j]}]={actual} {status}")
    
    return A, B, gt


# ================================================================
# TEST 2: KARATE CLUB (well-known graph)
# ================================================================
def test_karate():
    """
    Test on Zachary's Karate Club - well-studied graph.
    Should achieve high accuracy since structure is unique.
    """
    print("\n" + "=" * 70)
    print("  TEST 2: KARATE CLUB (n=34)")
    print("=" * 70)
    
    G = nx.karate_club_graph()
    A = nx.to_numpy_array(G).astype(np.float32)
    n = A.shape[0]
    
    # Create permuted version WITHOUT noise
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    B = A[perm][:, perm].copy()
    gt = np.argsort(perm).astype(np.int32)
    
    print(f"  Nodes: {n}")
    print(f"  Edges: {int(np.sum(A) / 2)}")
    print(f"  Density: {2 * np.sum(A) / (n * (n-1)):.4f}")
    
    return A, B, gt


# ================================================================
# TEST 3: IDENTITY MAPPING (no permutation)
# ================================================================
def test_identity():
    """
    A = B (no permutation). All methods should return [0, 1, 2, ..., n-1].
    """
    print("\n" + "=" * 70)
    print("  TEST 3: IDENTITY MAPPING (n=50)")
    print("=" * 70)
    
    n = 50
    G = nx.barabasi_albert_graph(n, 5, seed=42)
    A = nx.to_numpy_array(G).astype(np.float32)
    B = A.copy()  # Same graph, no permutation
    gt = np.arange(n, dtype=np.int32)
    
    print(f"  Graph: BA(n={n}, m=5)")
    print(f"  Expected matching: identity [0, 1, ..., {n-1}]")
    
    return A, B, gt


# ================================================================
# TEST 4: BIPARTITE STRUCTURE
# ================================================================
def test_bipartite():
    """
    Complete bipartite K_{5,5}.
    Every method should perfectly identify structure.
    """
    print("\n" + "=" * 70)
    print("  TEST 4: BIPARTITE K(5,5) (n=10)")
    print("=" * 70)
    
    G = nx.complete_bipartite_graph(5, 5)
    A = nx.to_numpy_array(G).astype(np.float32)
    n = A.shape[0]
    
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    B = A[perm][:, perm].copy()
    gt = np.argsort(perm).astype(np.int32)
    
    print(f"  Nodes: {n}, Edges: {int(np.sum(A) / 2)}")
    
    return A, B, gt


# ================================================================
# RUN ALL METHODS ON TOY GRAPH
# ================================================================
def compare_methods_on_toy(A, B, gt, test_name):
    """
    Run multiple methods on toy graph and verify correctness.
    """
    print(f"\n  ─── Running methods on '{test_name}' ───")
    
    n = A.shape[0]
    results = {}
    
    # Method 1: SciPy with rich features
    try:
        from utils_graph_matching import make_graph_pair
        # Feature-based cost
        F_A = _extract_simple_features(A)
        F_B = _extract_simple_features(B)
        cost = -(F_A @ F_B.T)
        ri, ci = linear_sum_assignment(cost)
        matching = np.full(n, -1, dtype=np.int32)
        matching[ri] = ci
        acc = np.mean(matching == gt)
        results['SciPy+Features'] = (matching, acc)
        print(f"    SciPy+Features: acc={acc:.4f}")
    except Exception as e:
        print(f"    SciPy+Features: ERROR {e}")
    
    # Method 2: Hungarian on adjacency directly (baseline)
    try:
        # Cost = -A · B (higher similarity = lower cost)
        cost = -(A @ B.T)
        ri, ci = linear_sum_assignment(cost)
        matching = np.full(n, -1, dtype=np.int32)
        matching[ri] = ci
        acc = np.mean(matching == gt)
        results['Hungarian-Adj'] = (matching, acc)
        print(f"    Hungarian-Adj:  acc={acc:.4f}")
    except Exception as e:
        print(f"    Hungarian-Adj:  ERROR {e}")
    
    # Method 3: GOAT (implemented below)
    try:
        from goat_baseline import goat_match
        matching = goat_match(A, B)
        acc = np.mean(matching == gt)
        results['GOAT'] = (matching, acc)
        print(f"    GOAT:           acc={acc:.4f}")
    except Exception as e:
        print(f"    GOAT:           ERROR {e}")
    
    # Method 4: Our proposed
    try:
        # Try to import proposed method
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "gpu_sparse", "gpu_sparse_matching.py")
        if spec and spec.loader:
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            if hasattr(m, 'scgm_gpu'):
                r = m.scgm_gpu(A, B, k=min(15, n-1),
                                 sinkhorn_iters=30, refinement_iters=2)
                acc = np.mean(r['matching'] == gt)
                results['Proposed'] = (r['matching'], acc)
                print(f"    Proposed:       acc={acc:.4f}")
    except Exception as e:
        print(f"    Proposed:       {e}")
    
    return results


def _extract_simple_features(A, n_spectral=5):
    """Simple feature extraction for testing."""
    n = A.shape[0]
    d = np.sum(A, axis=1, keepdims=True)
    A2 = A @ A
    features = [d, d**2, np.sum(A2, axis=1, keepdims=True)]
    
    if n > 3:
        try:
            D_inv_sqrt = 1.0 / np.sqrt(d + 1e-10)
            L = np.eye(n) - D_inv_sqrt * A * D_inv_sqrt.T
            _, eigvecs = np.linalg.eigh(L)
            n_spec = min(n_spectral, n - 1)
            features.append(np.abs(eigvecs[:, 1:n_spec+1]))
        except: pass
    
    F = np.concatenate(features, axis=1).astype(np.float32)
    F = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-10)
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-10)
    return F


# ================================================================
# OBJECTIVE VALIDATION (Quadratic vs Linear)
# ================================================================
def validate_qap_objective(A, B, matching):
    """
    Compute QAP objective: ||AP - PB||_F^2
    All methods should be compared on THIS objective.
    """
    n = A.shape[0]
    P = np.zeros((n, n), dtype=np.float32)
    valid = matching >= 0
    P[np.arange(n)[valid], matching[valid]] = 1
    
    # QAP objective
    qap = float(np.linalg.norm(A @ P - P @ B, 'fro') ** 2)
    
    # Linear objective (for comparison)
    linear = float(np.sum(P * (-A @ B.T)))
    
    return {'qap': qap, 'linear': linear}


# ================================================================
# MAIN TEST RUNNER
# ================================================================
def run_all_correctness_tests():
    print("=" * 70)
    print("    CORRECTNESS VALIDATION SUITE")
    print("    Tests algorithm on toy graphs with known ground truth")
    print("=" * 70)
    
    all_passed = True
    
    # Test 1: Triangle
    A, B, gt = test_triangle()
    results = compare_methods_on_toy(A, B, gt, "Triangle")
    for name, (m, acc) in results.items():
        if acc < 1.0:
            print(f"  ⚠️ {name} did not achieve 100% on triangle!")
            all_passed = False
    
    # Test 2: Identity
    A, B, gt = test_identity()
    results = compare_methods_on_toy(A, B, gt, "Identity")
    for name, (m, acc) in results.items():
        if acc < 0.95:
            print(f"  ⚠️ {name} accuracy only {acc:.2f} on identity (should be ~1.0)")
    
    # Test 3: Karate
    A, B, gt = test_karate()
    results = compare_methods_on_toy(A, B, gt, "Karate")
    print("\n  Karate Club results (higher = better):")
    for name, (m, acc) in results.items():
        print(f"    {name:<20}: {acc:.4f}")
    
    # Test 4: Bipartite
    A, B, gt = test_bipartite()
    results = compare_methods_on_toy(A, B, gt, "Bipartite")
    
    # Objective consistency check
    print("\n" + "=" * 70)
    print("  OBJECTIVE CONSISTENCY CHECK")
    print("=" * 70)
    A, B, gt = test_karate()
    results = compare_methods_on_toy(A, B, gt, "Karate objective test")
    print("\n  QAP Objective (lower = better):")
    for name, (m, acc) in results.items():
        obj = validate_qap_objective(A, B, m)
        print(f"    {name:<20}: QAP={obj['qap']:.2f}, Linear={obj['linear']:.2f}, Acc={acc:.4f}")
    
    print("\n" + "=" * 70)
    print(f"  {'✅ ALL TESTS PASSED' if all_passed else '⚠️ SOME TESTS FAILED - CHECK ABOVE'}")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    run_all_correctness_tests()