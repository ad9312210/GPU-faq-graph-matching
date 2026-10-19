"""Tolerance-aware end-to-end scoring validation; not an identical-score selection oracle."""
import argparse
import csv
import json
import math
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from topk_cli import (add_topk_arguments, topk_configuration, native_topk_arguments,
                      verify_topk_metadata, GPUUnavailable)


def load_graph(path):
    with path.open(encoding="utf-8") as stream:
        n, declared_edges = map(int, stream.readline().split())
        edges = [tuple(map(int, line.split()[:2])) for line in stream if line.strip()]
    if len(edges) != declared_edges:
        raise ValueError(f"{path}: header says {declared_edges} edges, read {len(edges)}")
    adjacency = [set() for _ in range(n)]
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n) or u == v:
            raise ValueError(f"{path}: invalid edge {(u, v)}")
        if v in adjacency[u]:
            raise ValueError(f"{path}: duplicate edge {(u, v)}")
        adjacency[u].add(v)
        adjacency[v].add(u)
    return adjacency


def reference_features(adjacency):
    """Independent transparent reference; returns raw six-column features."""
    n = len(adjacency)
    degree = np.asarray([len(neighbors) for neighbors in adjacency], dtype=np.float64)
    triangles = np.zeros(n, dtype=np.float64)
    for u in range(n):
        for w in adjacency[u]:
            if w <= u:
                continue
            for z in adjacency[w]:
                if z > w and z in adjacency[u]:
                    triangles[u] += 1
                    triangles[w] += 1
                    triangles[z] += 1
    features = np.zeros((n, 6), dtype=np.float64)
    features[:, 0] = degree
    features[:, 4] = triangles
    features[:, 5] = degree * degree
    for v, neighbors in enumerate(adjacency):
        d = len(neighbors)
        if d >= 2:
            features[v, 1] = triangles[v] / (d * (d - 1) / 2)
        if d:
            neighbor_degrees = degree[list(neighbors)]
            features[v, 2] = neighbor_degrees.mean()
            if d > 1:
                features[v, 3] = neighbor_degrees.std(ddof=1)
    return features


def joint_normalize(source, target):
    # Match the native method's shared graph-pair population z-score.
    combined = np.concatenate((source, target), axis=0)
    mean = combined.mean(axis=0)
    std = combined.std(axis=0)
    std[std < 1e-12] = 1.0
    return (source - mean) / std, (target - mean) / std


def analyze_dense_row(saved, dense_scores, k, score_tolerance, tie_tolerance):
    """Compare native score/order invariants exactly and reference scoring tolerantly."""
    dense_scores = np.asarray(dense_scores, dtype=float)
    if not 1 <= k <= len(dense_scores) or not np.isfinite(dense_scores).all():
        raise ValueError('invalid dense scores or K')
    errors = []
    ids = [int(target) for target, _ in saved]
    valid_ids = all(0 <= target < len(dense_scores) for target in ids)
    finite = all(math.isfinite(score) for _, score in saved)
    unique = len(ids) == len(set(ids))
    if len(saved) != k: errors.append('wrong_candidate_count')
    if not unique: errors.append('duplicate_targets')
    if not valid_ids: errors.append('target_out_of_range')
    if not finite: errors.append('nonfinite_native_score')
    ordered = finite and all(
        saved[i-1][1] > saved[i][1] or
        (saved[i-1][1] == saved[i][1] and saved[i-1][0] < saved[i][0])
        for i in range(1, len(saved)))
    if not ordered: errors.append('native_order_violation')
    order = np.lexsort((np.arange(len(dense_scores)), -dense_scores))
    cutoff = float(dense_scores[order[k-1]])
    selected = set(ids) if valid_ids else set()
    outside = [target for target in ids if 0 <= target < len(dense_scores)
               and dense_scores[target] < cutoff - tie_tolerance]
    missing = [int(target) for target in np.flatnonzero(dense_scores > cutoff + tie_tolerance)
               if int(target) not in selected]
    if outside: errors.append('selected_below_reference_cutoff')
    if missing: errors.append('omitted_strictly_better_reference_candidate')
    score_error = (max((abs(float(score)-float(dense_scores[target])) for target, score in saved), default=0.0)
                   if valid_ids and finite else None)
    if score_error is not None and score_error > score_tolerance:
        errors.append('score_error_exceeds_tolerance')
    return {'k': k, 'native_saved_candidate_count': len(saved),
            'saved_candidate_count_matches_k': len(saved) == k,
            'unique_targets': unique, 'native_order_valid': ordered,
            'dense_cutoff_similarity': cutoff,
            'dense_cutoff_tie_count': int(np.count_nonzero(np.abs(dense_scores-cutoff) <= tie_tolerance)),
            'native_dense_topk_overlap': len(selected & set(map(int, order[:k]))),
            'max_native_vs_dense_similarity_error': score_error,
            'native_candidates_below_dense_cutoff': outside,
            'omitted_strictly_better_reference_candidates': missing,
            'errors': errors, 'pass': not errors}


def read_native_candidates(path, selected, n, m):
    rows = {source: [] for source in selected}
    with path.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            source, target, score = int(row['source']), int(row['target']), float(row['similarity'])
            if not 0 <= source < n or not 0 <= target < m or not math.isfinite(score):
                raise ValueError('invalid candidate endpoint or score in export')
            if source in rows:
                if 'rank' in row and int(row['rank']) != len(rows[source]):
                    raise ValueError('noncontiguous candidate ranks')
                rows[source].append((target, score))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--executable', type=Path, default=Path('build-l40s/scgm_cuda'))
    parser.add_argument('--mode', choices=('cpu','gpu'), default='cpu')
    parser.add_argument('--k', type=int, default=64)
    parser.add_argument('--rows', type=int, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tie-tolerance', type=float, default=2e-5)
    parser.add_argument('--score-tolerance', type=float, default=2e-5)
    add_topk_arguments(parser)
    args = parser.parse_args()
    config = topk_configuration(args, parser)
    if any(not math.isfinite(value) or value < 0 for value in (args.tie_tolerance,args.score_tolerance)):
        parser.error('tolerances must be finite and nonnegative')
    source, target = load_graph(args.source), load_graph(args.target)
    n, m = len(source), len(target)
    if n == 0 or not 1 <= args.k <= m: parser.error('require nonempty source and 1 <= K <= targets')
    selected = sorted(set(args.rows if args.rows is not None else [i for i in (0,1,17,n//2,n-1) if 0 <= i < n]))
    if any(i < 0 or i >= n for i in selected): parser.error('selected row out of range')
    executable = args.executable.resolve()
    if not executable.is_file(): parser.error(f'executable missing: {executable}')
    x, y = joint_normalize(reference_features(source), reference_features(target))
    report = {'source_graph':str(args.source), 'target_graph':str(args.target), 'mode':args.mode,
              'k':args.k, 'selected_rows':selected,
              'validation_scope':'end_to_end_float_scoring_with_tolerances',
              'selection_oracle_note':'Exact identical-score selection is checked by test_gpu_topk.',
              'score_tolerance':args.score_tolerance,'cutoff_tolerance':args.tie_tolerance}
    code = 0
    try:
        with tempfile.TemporaryDirectory(prefix='scgm_dense_ref_') as folder:
            root = Path(folder); candidates = root/'candidates.csv'; timing = root/'timing.json'
            command = [str(executable),'--source',str(args.source.resolve()),'--target',str(args.target.resolve()),
                       '--k',str(args.k),f'--{args.mode}','--candidates',str(candidates),
                       '--output',str(root/'assignment.csv'),'--timings',str(timing)]
            command += native_topk_arguments(config)
            completed = subprocess.run(command,capture_output=True,text=True)
            report.update(native_exit_code=completed.returncode, command=command)
            if completed.returncode not in (0,2) or not timing.is_file() or not candidates.is_file():
                raise RuntimeError(f'native candidate run failed: {completed.returncode}: {completed.stderr}')
            native = json.loads(timing.read_text())
            verify_topk_metadata(native, config, args.mode, args.k)
            report['gpu_topk'] = native.get('gpu_topk')
            rows = read_native_candidates(candidates,selected,n,m)
        results = []
        norms = np.linalg.norm(y,axis=1)
        for source_id in selected:
            denominators = norms*np.linalg.norm(x[source_id]); dots = y@x[source_id]
            scores = np.clip(np.divide(dots,denominators,out=np.zeros_like(dots),where=denominators>=1e-12),-1,1)
            results.append({'source':source_id,**analyze_dense_row(rows[source_id],scores,args.k,
                           args.score_tolerance,args.tie_tolerance)})
        report.update(rows=results,failed_rows=[row['source'] for row in results if not row['pass']],
                      all_selected_rows_pass=all(row['pass'] for row in results))
        report['status'] = 'passed' if report['all_selected_rows_pass'] else 'failed'
        code = 0 if report['all_selected_rows_pass'] else 1
    except GPUUnavailable as error:
        report.update(status='skipped',reason=str(error),all_selected_rows_pass=False); code=77
    except (ValueError,RuntimeError,OSError) as error:
        report.update(status='failed',reason=str(error),all_selected_rows_pass=False); code=1
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
