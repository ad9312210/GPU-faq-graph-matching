# -*- coding: utf-8 -*-
"""
MASTER RUN ALL EXPERIMENTS - Final Working Version
Uses correct signatures for all modules
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import time
import os
import sys
import json
import importlib.util
import networkx as nx
from datetime import datetime
from pathlib import Path
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    'k': 15, 'sinkhorn_iter': 20, 'sinkhorn_temp': 0.1,
    'tau': 0.5, 'n_refinement_iter': 2,
    'noise': 0.05, 'seed': 42,
    'goat_max_iter': 15, 'goat_size_limit': 500,
}

RESULTS_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_module(name, path):
    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(full_path): return None
    try:
        spec = importlib.util.spec_from_file_location(name, full_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"  ⚠️  {path}: {str(e)[:80]}")
        return None


print("=" * 90)
print("    🚀 MASTER RUN ALL EXPERIMENTS")
print("=" * 90)
print(f"\n  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

from utils_graph_matching import (
    accuracy, edge_correctness, graph_matching_cost,
    generate_ba, generate_er, generate_ws, generate_sbm,
    load_karate, load_les_mis, load_florentine, load_davis,
)
print("  ✅ utils_graph_matching")

seq_module = load_module("seq", "01_cpu_sequential_sparse_matching.py")
par_module = load_module("par", "02_cpu_parallel_sparse_matching.py")
gpu_module = load_module("gpu", "03_gpu_sparse_matching.py")
cuda_hungarian_module = load_module("cuda_hun", "05_cuda_hungarian.py")
goat_module = load_module("goat", "goat_baseline.py")
sparse_module = load_module("sparse", "true_sparse_scgm.py")
print(f"\n  All modules loaded\n")


def _parse_result(result, A, elapsed):
    """Parse various result formats into standard dict."""
    matching = None
    timings = {'total': elapsed}
    
    if isinstance(result, tuple):
        matching = result[0]
        if len(result) > 1 and isinstance(result[1], dict):
            timings = result[1]
    elif isinstance(result, dict):
        for key in ['matching', 'assignment', 'perm', 'pred']:
            if key in result:
                matching = result[key]
                break
        if 'timings' in result:
            timings = result['timings']
    elif isinstance(result, (np.ndarray, list)):
        matching = result
    
    if matching is None:
        return None
    
    matching = np.asarray(matching, dtype=np.int32)
    if 'total' not in timings:
        timings['total'] = elapsed
    
    return {
        'matching': matching,
        'timings': timings,
        'memory_mb': (A.shape[0] * CONFIG['k'] * 4) / 1e6,
    }


def run_seq(A, B):
    """CPU Sequential: (G_A, G_B, A, B, kwargs)"""
    if not seq_module or not hasattr(seq_module, 'scgm_sequential'):
        return None
    try:
        G_A = nx.from_numpy_array(A)
        G_B = nx.from_numpy_array(B)
        t0 = time.perf_counter()
        result = seq_module.scgm_sequential(
            G_A, G_B, A, B,
            k=CONFIG['k'], sinkhorn_iter=CONFIG['sinkhorn_iter'],
            sinkhorn_temp=CONFIG['sinkhorn_temp'], tau=CONFIG['tau'],
            n_refinement_iter=CONFIG['n_refinement_iter'], verbose=False)
        elapsed = time.perf_counter() - t0
        r = _parse_result(result, A, elapsed)
        if r: r['method'] = 'CPU Sequential'
        return r
    except Exception as e:
        print(f"    seq error: {str(e)[:80]}")
        return None


def run_par(A, B):
    """CPU Parallel: (G_A, G_B, A, B, kwargs)"""
    if not par_module or not hasattr(par_module, 'scgm_parallel'):
        return None
    try:
        G_A = nx.from_numpy_array(A)
        G_B = nx.from_numpy_array(B)
        t0 = time.perf_counter()
        result = par_module.scgm_parallel(
            G_A, G_B, A, B,
            k=CONFIG['k'], sinkhorn_iter=CONFIG['sinkhorn_iter'],
            sinkhorn_temp=CONFIG['sinkhorn_temp'], tau=CONFIG['tau'],
            n_refinement_iter=CONFIG['n_refinement_iter'],
            n_jobs=-1, use_numba=False, verbose=False)
        elapsed = time.perf_counter() - t0
        r = _parse_result(result, A, elapsed)
        if r: r['method'] = 'CPU Parallel'
        return r
    except Exception as e:
        print(f"    par error: {str(e)[:80]}")
        return None


def run_gpu(A, B):
    """GPU: (G_A, G_B, A_cpu, B_cpu, kwargs)"""
    if not gpu_module or not hasattr(gpu_module, 'scgm_gpu'):
        return None
    try:
        G_A = nx.from_numpy_array(A)
        G_B = nx.from_numpy_array(B)
        t0 = time.perf_counter()
        result = gpu_module.scgm_gpu(
            G_A, G_B, A, B,
            k=CONFIG['k'], sinkhorn_iter=CONFIG['sinkhorn_iter'],
            sinkhorn_temp=CONFIG['sinkhorn_temp'], tau=CONFIG['tau'],
            n_refinement_iter=CONFIG['n_refinement_iter'],
            use_custom_kernels=True, verbose=False)
        elapsed = time.perf_counter() - t0
        r = _parse_result(result, A, elapsed)
        if r: r['method'] = 'GPU'
        return r
    except Exception as e:
        print(f"    gpu error: {str(e)[:80]}")
        return None


def run_scipy(A, B):
    n = A.shape[0]
    t0 = time.perf_counter()
    d_a = np.sum(A, axis=1, keepdims=True)
    d_b = np.sum(B, axis=1, keepdims=True)
    F_A = np.concatenate([d_a, d_a**2, np.sum(A@A, axis=1, keepdims=True)], axis=1)
    F_A = F_A / (np.linalg.norm(F_A, axis=1, keepdims=True) + 1e-10)
    F_B = np.concatenate([d_b, d_b**2, np.sum(B@B, axis=1, keepdims=True)], axis=1)
    F_B = F_B / (np.linalg.norm(F_B, axis=1, keepdims=True) + 1e-10)
    cost = -(F_A @ F_B.T)
    ri, ci = linear_sum_assignment(cost)
    matching = np.full(n, -1, dtype=np.int32); matching[ri] = ci
    return {'matching': matching,
            'timings': {'total': time.perf_counter() - t0},
            'memory_mb': (n * n * 8) / 1e6, 'method': 'SciPy'}


def run_goat(A, B):
    if not goat_module or A.shape[0] > CONFIG['goat_size_limit']: return None
    try:
        return goat_module.goat_match_with_timing(
            A, B, max_iter=CONFIG['goat_max_iter'])
    except: return None


def run_sparse(A, B):
    if not sparse_module: return None
    try:
        import scipy.sparse as sp
        A_clean = np.nan_to_num(A.astype(np.float32), nan=0.0)
        B_clean = np.nan_to_num(B.astype(np.float32), nan=0.0)
        A_csr = sp.csr_matrix(A_clean)
        B_csr = sp.csr_matrix(B_clean)
        return sparse_module.true_sparse_scgm(
            A_csr, B_csr, k=CONFIG['k'],
            sinkhorn_iters=CONFIG['sinkhorn_iter'],
            refinement_iters=CONFIG['n_refinement_iter'])
    except Exception as e:
        print(f"    sparse error: {str(e)[:50]}")
        return None


def run_gpu_hungarian(A, B):
    if not cuda_hungarian_module or not getattr(cuda_hungarian_module, 'GPU_AVAILABLE', False):
        return None
    try:
        return cuda_hungarian_module.gpu_hungarian(A, B)
    except: return None


def warmup():
    print("Warming up implementations...")
    A_w, B_w, _ = generate_ba(50, m=5, noise=0.02, seed=42)
    for name, fn in [('SciPy', run_scipy), ('CPU Seq', run_seq),
                       ('CPU Par', run_par), ('GPU', run_gpu),
                       ('GOAT', run_goat), ('True Sparse', run_sparse),
                       ('GPU Hungarian', run_gpu_hungarian)]:
        try:
            r = fn(A_w, B_w)
            print(f"  {'✅' if r else '⚠️'} {name} {'ready' if r else 'not available'}")
        except Exception as e:
            print(f"  ⚠️  {name}: {str(e)[:50]}")
    print()


def exp_scalability():
    print("\n" + "=" * 100)
    print("    📊 EXPERIMENT 1: SCALABILITY (BA Scale-Free)")
    print("=" * 100)
    
    sizes = [100, 200, 500, 1000, 2000]
    print(f"\n  {'n':>5} | {'SciPy':>10} | {'GOAT':>11} | {'CPU Seq':>11} | "
          f"{'CPU Par':>11} | {'GPU':>11} | {'Sparse':>11} | {'Best':>7}")
    print("  " + "-" * 100)
    
    results = []
    for n in sizes:
        A, B, gt = generate_ba(n, m=5, noise=0.02, seed=CONFIG['seed'])
        row = {'n': n}
        for key, fn in [('scipy', run_scipy), ('goat', run_goat),
                          ('seq', run_seq), ('par', run_par),
                          ('gpu', run_gpu), ('sparse', run_sparse)]:
            try:
                r = fn(A, B)
                if r:
                    row[f't_{key}'] = r['timings'].get('total', 0) * 1000
                    row[f'acc_{key}'] = accuracy(r['matching'], gt)
            except: pass
        
        best = max([row.get(f'acc_{k}', 0) for k in ['scipy', 'goat', 'seq', 'par', 'gpu', 'sparse']])
        def fmt(k):
            t = row.get(f't_{k}')
            return f"{t:>9.1f}ms" if t else "      N/A"
        print(f"  {n:>5} | {fmt('scipy'):>10} | {fmt('goat'):>11} | {fmt('seq'):>11} | "
              f"{fmt('par'):>11} | {fmt('gpu'):>11} | {fmt('sparse'):>11} | {best:>7.3f}")
        results.append(row)
    return results


def exp_graph_types():
    print("\n" + "=" * 100)
    print("    🎨 EXPERIMENT 2: GRAPH TYPES (n=500)")
    print("=" * 100)
    
    n = 500
    graphs = [('BA', lambda: generate_ba(n, m=5)),
              ('WS', lambda: generate_ws(n, k=6, p=0.3)),
              ('ER', lambda: generate_er(n, p=0.1)),
              ('SBM', lambda: generate_sbm(n, n_blocks=4))]
    
    print(f"\n  {'Type':<6} | {'SciPy':>10} | {'CPU Seq':>10} | {'GPU':>10} | "
          f"{'Sparse':>10} | {'Best Acc':>9}")
    print("  " + "-" * 75)
    
    results = []
    for name, gen in graphs:
        try:
            A, B, gt = gen()
            row = {'type': name}
            for key, fn in [('scipy', run_scipy), ('seq', run_seq),
                              ('gpu', run_gpu), ('sparse', run_sparse)]:
                try:
                    r = fn(A, B)
                    if r:
                        row[f't_{key}'] = r['timings'].get('total', 0) * 1000
                        row[f'acc_{key}'] = accuracy(r['matching'], gt)
                except: pass
            
            best = max([row.get(f'acc_{k}', 0) for k in ['scipy', 'seq', 'gpu', 'sparse']])
            def fmt(k):
                t = row.get(f't_{k}')
                return f"{t:>8.1f}ms" if t else "      N/A"
            print(f"  {name:<6} | {fmt('scipy'):>10} | {fmt('seq'):>10} | "
                  f"{fmt('gpu'):>10} | {fmt('sparse'):>10} | {best:>9.3f}")
            results.append(row)
        except Exception as e:
            print(f"  {name}: {e}")
    return results


def exp_real_datasets():
    print("\n" + "=" * 100)
    print("    🌐 EXPERIMENT 3: REAL DATASETS")
    print("=" * 100)
    
    datasets = [('Karate', load_karate), ('LesMis', load_les_mis),
                ('Florentine', load_florentine), ('Davis', load_davis)]
    
    print(f"\n  {'Dataset':<12} | {'n':>4} | {'SciPy':>9} | {'CPU Seq':>9} | "
          f"{'GPU':>9} | {'GOAT':>9} | {'Sparse':>9} | {'Best':>7}")
    print("  " + "-" * 85)
    
    results = []
    for name, load_fn in datasets:
        try:
            A, B, gt = load_fn(noise=CONFIG['noise'])
            n = A.shape[0]
            row = {'name': name, 'n': n}
            for key, fn in [('scipy', run_scipy), ('seq', run_seq),
                              ('gpu', run_gpu), ('goat', run_goat),
                              ('sparse', run_sparse)]:
                try:
                    r = fn(A, B)
                    if r:
                        row[f't_{key}'] = r['timings'].get('total', 0) * 1000
                        row[f'acc_{key}'] = accuracy(r['matching'], gt)
                except: pass
            
            best = max([row.get(f'acc_{k}', 0) for k in ['scipy', 'seq', 'gpu', 'goat', 'sparse']])
            def fmt(k):
                t = row.get(f't_{k}')
                return f"{t:>7.1f}ms" if t else "     N/A"
            print(f"  {name:<12} | {n:>4} | {fmt('scipy'):>9} | {fmt('seq'):>9} | "
                  f"{fmt('gpu'):>9} | {fmt('goat'):>9} | {fmt('sparse'):>9} | {best:>7.3f}")
            results.append(row)
        except Exception as e:
            print(f"  {name}: ERROR {str(e)[:50]}")
    return results


def exp_hungarian():
    print("\n" + "=" * 100)
    print("    ⚔️ EXPERIMENT 4: HUNGARIAN VARIANTS")
    print("=" * 100)
    
    sizes = [200, 500, 1000, 2000]
    print(f"\n  {'n':>5} | {'SciPy':>10} | {'GPU Hun':>10} | {'GPU Auction':>11} | {'GPU Greedy':>11}")
    print("  " + "-" * 60)
    
    results = []
    for n in sizes:
        try:
            A, B, gt = generate_ba(n, m=5)
            row = {'n': n}
            r = run_scipy(A, B); row['scipy'] = r['timings']['total'] * 1000
            
            if cuda_hungarian_module and getattr(cuda_hungarian_module, 'GPU_AVAILABLE', False):
                for method in ['gpu_hungarian', 'gpu_auction', 'gpu_parallel_greedy']:
                    if hasattr(cuda_hungarian_module, method):
                        try:
                            fn = getattr(cuda_hungarian_module, method)
                            r = fn(A, B) if method != 'gpu_parallel_greedy' else fn(A, B, top_k=30)
                            row[method] = r['timings']['total'] * 1000
                        except: pass
            
            def f(k):
                v = row.get(k); return f"{v:>8.1f}ms" if v else "N/A"
            print(f"  {n:>5} | {f('scipy'):>10} | {f('gpu_hungarian'):>10} | "
                  f"{f('gpu_auction'):>11} | {f('gpu_parallel_greedy'):>11}")
            results.append(row)
        except: pass
    return results


def exp_memory():
    print("\n" + "=" * 100)
    print("    💾 EXPERIMENT 5: MEMORY SCALING")
    print("=" * 100)
    
    print(f"\n  {'n':>7} | {'Dense':>14} | {'Sparse':>16} | {'Reduction':>12}")
    print("  " + "-" * 60)
    results = []
    for n in [100, 500, 1000, 2000, 5000, 10000, 50000, 100000]:
        dense = (n * n * 4) / (1024 * 1024)
        sparse = (n * CONFIG['k'] * 4) / (1024 * 1024)
        red = dense / sparse if sparse > 0 else 0
        print(f"  {n:>7,} | {dense:>12.2f}MB | {sparse:>14.4f}MB | {red:>10.1f}× less")
        results.append({'n': n, 'dense_mb': dense, 'sparse_mb': sparse, 'reduction': red})
    return results


def exp_latency():
    print("\n" + "=" * 100)
    print("    ⚡ EXPERIMENT 6: LATENCY (P50/P95/P99)")
    print("=" * 100)
    
    print(f"\n  {'n':>5} | {'Mean':>8} | {'P50':>8} | {'P95':>8} | {'P99':>8} | Status")
    print("  " + "-" * 65)
    results = []
    for n in [200, 500, 1000, 2000]:
        try:
            latencies = []
            for trial in range(10):
                A, B, _ = generate_ba(n, m=5, seed=trial)
                t0 = time.perf_counter()
                run_sparse(A, B)
                latencies.append((time.perf_counter() - t0) * 1000)
            mean = np.mean(latencies); p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95); p99 = np.percentile(latencies, 99)
            status = "✅ REAL-TIME" if p95 < 100 else "⚡ INTERACTIVE" if p95 < 500 else "⏳ BATCH"
            print(f"  {n:>5} | {mean:>6.1f}ms | {p50:>6.1f}ms | {p95:>6.1f}ms | {p99:>6.1f}ms | {status}")
            results.append({'n': n, 'mean': mean, 'p50': p50, 'p95': p95, 'p99': p99,
                            'realtime': bool(p95 < 100)})
        except: pass
    return results


def save_results(all_results):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    def convert(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        return obj
    def convert_dict(d):
        if isinstance(d, dict): return {str(k): convert_dict(v) for k, v in d.items()}
        if isinstance(d, (list, tuple)): return [convert_dict(x) for x in d]
        return convert(d)
    try:
        json_path = RESULTS_DIR / f"results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(convert_dict(all_results), f, indent=2, default=str)
        print(f"\n  💾 Saved: {json_path.name}")
    except Exception as e:
        print(f"\n  ⚠️ Save failed: {e}")


def print_summary(all_results):
    print("\n\n" + "=" * 100)
    print("    🏆 FINAL SUMMARY")
    print("=" * 100)
    
    sparse_vs_scipy = []
    sparse_vs_goat = []
    sparse_vs_gpu = []
    accs_scipy = []; accs_sparse = []; accs_goat = []; accs_gpu = []
    
    for exp_data in all_results.values():
        if isinstance(exp_data, list):
            for r in exp_data:
                if isinstance(r, dict):
                    if r.get('t_scipy') and r.get('t_sparse'):
                        sparse_vs_scipy.append(r['t_scipy'] / r['t_sparse'])
                    if r.get('t_goat') and r.get('t_sparse'):
                        sparse_vs_goat.append(r['t_goat'] / r['t_sparse'])
                    if r.get('t_gpu') and r.get('t_sparse'):
                        sparse_vs_gpu.append(r['t_gpu'] / r['t_sparse'])
                    if r.get('acc_scipy') is not None: accs_scipy.append(r['acc_scipy'])
                    if r.get('acc_sparse') is not None: accs_sparse.append(r['acc_sparse'])
                    if r.get('acc_goat') is not None: accs_goat.append(r['acc_goat'])
                    if r.get('acc_gpu') is not None: accs_gpu.append(r['acc_gpu'])
    
    print(f"""
   📊 KEY RESULTS:
   
   🏆 Sparse SCGM vs SciPy:
      Tests: {len(sparse_vs_scipy)}
      Avg speedup: {np.mean(sparse_vs_scipy) if sparse_vs_scipy else 0:.2f}×
      Accuracy: SciPy={np.mean(accs_scipy):.3f}, Sparse={np.mean(accs_sparse):.3f}
      Improvement: {(np.mean(accs_sparse) - np.mean(accs_scipy))*100:+.1f}%
   
   🏆 Sparse SCGM vs GOAT:
      Tests: {len(sparse_vs_goat)}
      Avg speedup: {np.mean(sparse_vs_goat) if sparse_vs_goat else 0:.2f}× faster
      Max speedup: {np.max(sparse_vs_goat) if sparse_vs_goat else 0:.2f}×
   
   💾 Memory: Up to 6666× less than dense at n=100K
""")


def main():
    warmup()
    all_results = {}
    experiments = [
        ('exp1', "Scalability", exp_scalability),
        ('exp2', "Graph Types", exp_graph_types),
        ('exp3', "Real Datasets", exp_real_datasets),
        ('exp4', "Hungarian Variants", exp_hungarian),
        ('exp5', "Memory", exp_memory),
        ('exp6', "Latency", exp_latency),
    ]
    for exp_id, exp_name, exp_fn in experiments:
        try:
            all_results[exp_id] = exp_fn()
        except KeyboardInterrupt:
            print(f"\n{exp_name} INTERRUPTED"); break
        except Exception as e:
            print(f"\n{exp_name} FAILED: {e}")
            all_results[exp_id] = None
    
    print_summary(all_results)
    save_results(all_results)
    print("\n🎯 ALL EXPERIMENTS COMPLETE!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback; traceback.print_exc()