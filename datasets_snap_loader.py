# -*- coding: utf-8 -*-
# ================================================================
#   SNAP + GRAPH500 DATASETS LOADER
#
#   Datasets for performance and scalability evaluation:
#   - SNAP real-world graphs (G1-G12)
#   - Graph500 synthetic benchmarks (G13-G15)
#
#   Range: 5K nodes (G1) to 65M nodes (G12)
#   Auto-downloads and caches locally
# ================================================================

import numpy as np
import networkx as nx
import os
import gzip
import urllib.request
import time
from typing import Tuple, Dict, Optional
from pathlib import Path


# ================================================================
# DATASET REGISTRY (as per your specification)
# ================================================================
DATASET_REGISTRY = {
    'G1': {
        'name': 'ca-GrQc',
        'nodes': 5_242,
        'edges': 14_496,
        'avg_degree': 5.53,
        'nature': 'undirected',
        'purpose': 'Initial correctness and implementation validation',
        'url': 'https://snap.stanford.edu/data/ca-GrQc.txt.gz',
        'category': 'collaboration',
    },
    'G2': {
        'name': 'email-Enron',
        'nodes': 36_692,
        'edges': 183_831,
        'avg_degree': 10.02,
        'nature': 'undirected',
        'purpose': 'Small-scale real-world graph alignment',
        'url': 'https://snap.stanford.edu/data/email-Enron.txt.gz',
        'category': 'communication',
    },
    'G3': {
        'name': 'com-DBLP',
        'nodes': 317_080,
        'edges': 1_049_866,
        'avg_degree': 6.62,
        'nature': 'undirected',
        'purpose': 'Medium-scale graph-matching evaluation',
        'url': 'https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz',
        'category': 'collaboration',
    },
    'G4': {
        'name': 'com-Amazon',
        'nodes': 334_863,
        'edges': 925_872,
        'avg_degree': 5.53,
        'nature': 'undirected',
        'purpose': 'Sparse community-structure matching',
        'url': 'https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz',
        'category': 'product',
    },
    'G5': {
        'name': 'web-Google',
        'nodes': 875_713,
        'edges': 5_105_039,
        'avg_degree': 5.83,
        'nature': 'directed',
        'purpose': 'Directed-graph matching experiment',
        'url': 'https://snap.stanford.edu/data/web-Google.txt.gz',
        'category': 'web',
    },
    'G6': {
        'name': 'com-Youtube',
        'nodes': 1_134_890,
        'edges': 2_987_624,
        'avg_degree': 5.27,
        'nature': 'undirected',
        'purpose': 'Initial million-node scalability test',
        'url': 'https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz',
        'category': 'social',
    },
    'G7': {
        'name': 'roadNet-CA',
        'nodes': 1_965_206,
        'edges': 2_766_607,
        'avg_degree': 2.82,
        'nature': 'undirected',
        'purpose': 'Very sparse and low-degree structural graph',
        'url': 'https://snap.stanford.edu/data/roadNet-CA.txt.gz',
        'category': 'infrastructure',
    },
    'G8': {
        'name': 'soc-Pokec',
        'nodes': 1_632_803,
        'edges': 30_622_564,
        'avg_degree': 18.75,
        'nature': 'directed',
        'purpose': 'Dense directed social-network evaluation',
        'url': 'https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz',
        'category': 'social',
    },
    'G9': {
        'name': 'as-Skitter',
        'nodes': 1_696_415,
        'edges': 11_095_298,
        'avg_degree': 13.08,
        'nature': 'undirected',
        'purpose': 'Irregular degree-distribution evaluation',
        'url': 'https://snap.stanford.edu/data/as-skitter.txt.gz',
        'category': 'internet',
    },
    'G10': {
        'name': 'com-LiveJournal',
        'nodes': 3_997_962,
        'edges': 34_681_189,
        'avg_degree': 17.35,
        'nature': 'undirected',
        'purpose': 'Multi-million-node graph alignment',
        'url': 'https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz',
        'category': 'social',
    },
    'G11': {
        'name': 'com-Orkut',
        'nodes': 3_072_441,
        'edges': 117_185_083,
        'avg_degree': 76.28,
        'nature': 'undirected',
        'purpose': 'High-density and high-throughput GPU test',
        'url': 'https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz',
        'category': 'social',
    },
    'G12': {
        'name': 'com-Friendster',
        'nodes': 65_608_366,
        'edges': 1_806_067_135,
        'avg_degree': 55.06,
        'nature': 'undirected',
        'purpose': 'Billion-edge final scalability and stress test',
        'url': 'https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz',
        'category': 'social',
        'warning': 'HUGE: ~30GB download, requires >100GB RAM to fully load',
    },
    'G13': {
        'name': 'Graph500 Scale 20',
        'nodes': 1_048_576,
        'edges': 16_777_216,
        'avg_degree': 32.00,
        'nature': 'undirected',
        'purpose': 'Controlled million-node synthetic benchmark',
        'url': None,  # Generated synthetically
        'category': 'synthetic',
        'scale': 20,
    },
    'G14': {
        'name': 'Graph500 Scale 24',
        'nodes': 16_777_216,
        'edges': 268_435_456,
        'avg_degree': 32.00,
        'nature': 'undirected',
        'purpose': 'Large-scale single- and multi-GPU evaluation',
        'url': None,
        'category': 'synthetic',
        'scale': 24,
    },
    'G15': {
        'name': 'Graph500 Scale 26',
        'nodes': 67_108_864,
        'edges': 1_073_741_824,
        'avg_degree': 32.00,
        'nature': 'undirected',
        'purpose': 'Billion-edge synthetic stress and scalability test',
        'url': None,
        'category': 'synthetic',
        'scale': 26,
    },
}


# ================================================================
# DIRECTORY MANAGEMENT
# ================================================================
DATASET_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "datasets"
DATASET_DIR.mkdir(parents=True, exist_ok=True)


def get_dataset_info(gid: str) -> Dict:
    """Get dataset metadata by GID."""
    if gid not in DATASET_REGISTRY:
        raise ValueError(f"Unknown GID: {gid}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[gid]


def list_datasets(max_size: Optional[int] = None) -> None:
    """Print all available datasets."""
    print("\n" + "=" * 110)
    print("    📊 AVAILABLE DATASETS")
    print("=" * 110)
    print(f"  {'GID':>3} | {'Name':<25} | {'Nodes':>15} | {'Edges':>17} | "
          f"{'Deg':>6} | {'Type':<10} | Purpose")
    print("  " + "-" * 108)
    
    for gid, info in DATASET_REGISTRY.items():
        if max_size and info['nodes'] > max_size:
            continue
        
        nodes_str = f"{info['nodes']:>15,}"
        edges_str = f"{info['edges']:>17,}"
        
        print(f"  {gid:>3} | {info['name']:<25} | {nodes_str} | {edges_str} | "
              f"{info['avg_degree']:>6.2f} | {info['nature']:<10} | {info['purpose'][:40]}")


# ================================================================
# DOWNLOAD SNAP DATASET
# ================================================================
def download_snap_dataset(gid: str, verbose: bool = True) -> Optional[Path]:
    """Download SNAP dataset if not already cached."""
    info = get_dataset_info(gid)
    if info.get('url') is None:
        if verbose:
            print(f"  ⚠️ {gid} ({info['name']}) is synthetic, no download needed")
        return None
    
    url = info['url']
    filename = url.split('/')[-1]
    filepath = DATASET_DIR / filename
    
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        if verbose:
            print(f"  ✅ {gid} already cached: {filepath.name} ({size_mb:.1f} MB)")
        return filepath
    
    # Warn about large downloads
    if 'warning' in info:
        print(f"  ⚠️ WARNING: {info['warning']}")
    
    if verbose:
        print(f"  📥 Downloading {gid} ({info['name']})...")
        print(f"     URL: {url}")
    
    try:
        # Download with progress indication
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                mb_done = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                if verbose and block_num % 100 == 0:
                    print(f"     Progress: {percent:.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)",
                          end='\r')
        
        urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
        
        size_mb = filepath.stat().st_size / (1024 * 1024)
        if verbose:
            print(f"\n  ✅ Downloaded: {filepath.name} ({size_mb:.1f} MB)")
        return filepath
    except Exception as e:
        if verbose:
            print(f"\n  ❌ Download failed: {e}")
        return None


# ================================================================
# LOAD SNAP GRAPH (edge list to adjacency matrix)
# ================================================================
def load_snap_graph(gid: str, max_nodes: Optional[int] = None,
                     verbose: bool = True) -> Optional[np.ndarray]:
    """
    Load SNAP dataset as adjacency matrix.
    
    Args:
        gid: Dataset ID (G1-G12)
        max_nodes: If graph is larger, subsample via BFS
        verbose: Print progress
    
    Returns:
        Adjacency matrix (np.ndarray) or None if failed
    """
    info = get_dataset_info(gid)
    filepath = download_snap_dataset(gid, verbose)
    if filepath is None:
        return None
    
    # Determine if directed
    is_directed = (info['nature'] == 'directed')
    
    if verbose:
        print(f"  📊 Loading {info['name']} ({info['nodes']:,} nodes)...")
    
    t_start = time.perf_counter()
    
    # Build graph from edge list
    G = nx.DiGraph() if is_directed else nx.Graph()
    
    open_fn = gzip.open if str(filepath).endswith('.gz') else open
    with open_fn(filepath, 'rt') as f:
        n_edges = 0
        for line_num, line in enumerate(f):
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    u, v = int(parts[0]), int(parts[1])
                    G.add_edge(u, v)
                    n_edges += 1
                    if verbose and n_edges % 100_000 == 0:
                        print(f"     Loaded {n_edges:,} edges...", end='\r')
                except ValueError:
                    continue
    
    load_time = time.perf_counter() - t_start
    if verbose:
        print(f"\n  ✅ Graph loaded in {load_time:.1f}s: "
              f"{G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # Get largest connected component
    if is_directed:
        largest_cc = max(nx.strongly_connected_components(G), key=len)
    else:
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
        else:
            largest_cc = list(G.nodes())
    
    G = G.subgraph(largest_cc).copy()
    if verbose:
        print(f"  📉 Largest component: {G.number_of_nodes():,} nodes, "
              f"{G.number_of_edges():,} edges")
    
    # Subsample if too large for memory
    if max_nodes and G.number_of_nodes() > max_nodes:
        if verbose:
            print(f"  ✂️ Subsampling to {max_nodes:,} nodes via BFS...")
        G = _bfs_sample(G, max_nodes)
    
    # Relabel to consecutive integers
    G = nx.convert_node_labels_to_integers(G)
    
    # Convert to adjacency matrix
    n = G.number_of_nodes()
    if verbose:
        mem_mb = (n * n * 4) / (1024 * 1024)
        print(f"  💾 Memory needed: {mem_mb:.1f} MB (dense {n}x{n})")
        if mem_mb > 4096:
            print(f"  ⚠️ WARNING: Dense matrix requires {mem_mb/1024:.1f} GB!")
            print(f"     Consider using sparse format for graphs this large.")
    
    A = nx.to_numpy_array(G, dtype=np.float32)
    
    if verbose:
        print(f"  ✅ Adjacency matrix: {A.shape}, {A.nbytes/(1024*1024):.1f} MB")
    
    return A


def _bfs_sample(G: nx.Graph, max_nodes: int) -> nx.Graph:
    """BFS sampling from highest-degree node."""
    # Start from highest degree node
    degrees = dict(G.degree())
    if not degrees:
        return G
    
    center = max(degrees, key=degrees.get)
    
    nodes = set([center])
    queue = [center]
    
    while len(nodes) < max_nodes and queue:
        node = queue.pop(0)
        for neighbor in G.neighbors(node):
            if len(nodes) >= max_nodes:
                break
            if neighbor not in nodes:
                nodes.add(neighbor)
                queue.append(neighbor)
    
    return G.subgraph(nodes).copy()


# ================================================================
# GENERATE GRAPH500 (Kronecker/RMAT graphs)
# ================================================================
def generate_graph500(scale: int, edge_factor: int = 16,
                       seed: int = 42, verbose: bool = True) -> np.ndarray:
    """
    Generate Graph500-style RMAT graph.
    
    Args:
        scale: log2(nodes), e.g., scale=20 → 1M nodes
        edge_factor: edges per node (default 16)
        seed: random seed
    
    Returns:
        Adjacency matrix
    """
    n_nodes = 2 ** scale
    n_edges = n_nodes * edge_factor
    
    if verbose:
        print(f"  🔨 Generating Graph500 Scale {scale}: {n_nodes:,} nodes, "
              f"{n_edges:,} edges")
    
    if n_nodes > 10_000_000:
        print(f"  ⚠️ WARNING: {n_nodes:,} nodes is very large!")
        print(f"     Dense matrix would need {(n_nodes*n_nodes*4)/(1024**3):.1f} GB")
        print(f"     Returning None - use sparse implementation instead")
        return None
    
    t_start = time.perf_counter()
    
    # RMAT parameters (standard Graph500 values)
    a, b, c = 0.57, 0.19, 0.19  # d = 1 - a - b - c = 0.05
    
    rng = np.random.default_rng(seed)
    
    # Generate edges using RMAT recursive matrix
    edges = set()
    max_edges = min(n_edges, n_nodes * (n_nodes - 1) // 2)
    
    while len(edges) < max_edges:
        # Generate batch of edges
        batch_size = min(10_000, max_edges - len(edges))
        for _ in range(batch_size):
            u, v = 0, 0
            for level in range(scale):
                r = rng.random()
                if r < a:
                    pass  # (u, v) stays same
                elif r < a + b:
                    v |= (1 << level)
                elif r < a + b + c:
                    u |= (1 << level)
                else:
                    u |= (1 << level)
                    v |= (1 << level)
            
            if u != v:
                edges.add((min(u, v), max(u, v)))
        
        if verbose and len(edges) % 100_000 == 0:
            print(f"     Generated {len(edges):,} edges...", end='\r')
    
    gen_time = time.perf_counter() - t_start
    if verbose:
        print(f"\n  ✅ Generated in {gen_time:.1f}s")
    
    # Build adjacency matrix
    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
    
    return A


# ================================================================
# LOAD DATASET (Universal loader for G1-G15)
# ================================================================
def load_dataset(gid: str, max_nodes: Optional[int] = None,
                  verbose: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Universal dataset loader for any GID.
    
    Args:
        gid: Dataset ID (G1-G15)
        max_nodes: Subsample if larger
        verbose: Print progress
    
    Returns:
        (adjacency_matrix, info_dict)
    """
    info = get_dataset_info(gid)
    
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  Loading {gid}: {info['name']}")
        print(f"  Purpose: {info['purpose']}")
        print(f"{'=' * 70}")
    
    if info['category'] == 'synthetic':
        # Graph500 synthetic
        scale = info['scale']
        A = generate_graph500(scale, verbose=verbose)
    else:
        # SNAP dataset
        A = load_snap_graph(gid, max_nodes, verbose=verbose)
    
    return A, info


# ================================================================
# CREATE MATCHING PAIR
# ================================================================
def make_matching_pair(A: np.ndarray, noise: float = 0.05,
                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create noisy permuted graph pair with ground truth.
    
    Args:
        A: Original adjacency matrix
        noise: Fraction of edges to flip
        seed: Random seed
    
    Returns:
        (A, B, ground_truth) where gt[i] = j means A's node i ↔ B's node j
    """
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    
    perm = rng.permutation(n)
    B = A[perm][:, perm].copy()
    
    if noise > 0:
        n_edges = int(np.sum(B) / 2)
        n_flips = int(noise * n_edges)
        for _ in range(n_flips):
            i, j = rng.integers(0, n, 2)
            if i != j:
                B[i, j] = 1 - B[i, j]
                B[j, i] = B[i, j]
    
    # Ground truth = inverse permutation
    gt = np.argsort(perm).astype(np.int32)
    
    return A.astype(np.float32), B.astype(np.float32), gt


# ================================================================
# MEMORY SAFETY CHECK
# ================================================================
def check_memory_safety(gid: str, max_gb: float = 4.0) -> bool:
    """Check if dataset can fit in memory."""
    info = get_dataset_info(gid)
    n = info['nodes']
    mem_gb = (n * n * 4) / (1024 ** 3)
    
    if mem_gb > max_gb:
        print(f"  ⚠️ {gid} needs {mem_gb:.1f} GB (dense) but limit is {max_gb:.1f} GB")
        print(f"     Suggest max_nodes = {int(np.sqrt(max_gb * 1024**3 / 4)):,}")
        return False
    return True


# ================================================================
# BATCH DOWNLOAD (all SNAP datasets)
# ================================================================
def download_all_snap(verbose: bool = True) -> Dict[str, bool]:
    """Download all SNAP datasets (G1-G12)."""
    print("\n" + "=" * 70)
    print("    📥 BATCH DOWNLOAD: All SNAP Datasets")
    print("=" * 70)
    
    results = {}
    for gid in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6',
                 'G7', 'G8', 'G9', 'G10', 'G11', 'G12']:
        info = get_dataset_info(gid)
        print(f"\n  {gid} ({info['name']}):")
        
        if gid == 'G12' and verbose:
            print("     ⚠️ SKIPPING G12 by default (30GB download)")
            print("     To download, call: download_snap_dataset('G12')")
            results[gid] = False
            continue
        
        filepath = download_snap_dataset(gid, verbose=False)
        results[gid] = filepath is not None
        if filepath:
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"     ✅ Cached ({size_mb:.1f} MB)")
        else:
            print(f"     ❌ Failed")
    
    return results


# ================================================================
# DEMO
# ================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("    SNAP + GRAPH500 DATASET LOADER")
    print("=" * 70)
    
    # List all available datasets
    list_datasets(max_size=10_000_000)
    
    print("\n" + "=" * 70)
    print("    TEST: Loading small datasets")
    print("=" * 70)
    
    # Test loading small datasets that fit in memory
    test_gids = ['G1', 'G2']  # Small datasets
    
    for gid in test_gids:
        try:
            A, info = load_dataset(gid, max_nodes=2000, verbose=True)
            if A is None: continue
            
            # Create matching pair
            print(f"\n  Creating matching pair with 5% noise...")
            A_pair, B_pair, gt = make_matching_pair(A, noise=0.05, seed=42)
            
            print(f"  ✅ Pair created:")
            print(f"     Shape:    {A_pair.shape}")
            print(f"     Edges:    {int(np.sum(A_pair) / 2):,}")
            print(f"     Density:  {2 * np.sum(A_pair) / (A_pair.shape[0]**2):.4f}")
            print(f"     GT sample: {gt[:5].tolist()}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("    Memory safety checks")
    print("=" * 70)
    
    for gid in ['G1', 'G3', 'G6', 'G10', 'G12']:
        check_memory_safety(gid, max_gb=4.0)
    
    print("\n" + "=" * 70)
    print("    Available commands:")
    print("=" * 70)
    print("""
   from datasets_snap_loader import (
       load_dataset,           # Load any GID
       make_matching_pair,     # Create A, B, gt
       list_datasets,          # Show all datasets
       download_all_snap,      # Batch download G1-G11
       check_memory_safety,    # Check if fits in RAM
   )
   
   # Examples:
   A, info = load_dataset('G1', max_nodes=5000)
   A_pair, B_pair, gt = make_matching_pair(A, noise=0.05)
   
   list_datasets()             # See all 15 datasets
   download_all_snap()         # Cache all locally
""")