"""Paired exhaustive vs CPU structural-index comparisons on one saved graph pair.

Candidate exports/quality and isolated RSS profiles are outside timing trials.
Each trial runs both backends sequentially in alternating order. No support repair
is performed: infeasible supports are reported, with unavailable final quality.
Repeat on independent saved pairs/seeds; timing repeats are not graph replication.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
from itertools import groupby
from pathlib import Path

import benchmark_native as native
from evaluate_assignment import read_graph, read_mapping, evaluate
from index_cli import add_index_arguments, index_configuration, configuration_id


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stats(values):
    values = [float(v) for v in values if v is not None]
    return None if not values else {'median': statistics.median(values), 'min': min(values), 'max': max(values), 'count': len(values)}


def ratio(a, b):
    return a / b if a is not None and b is not None and b > 0 else None


def candidate_rows(path):
    with path.open(newline='') as stream:
        for source, group in groupby(csv.DictReader(stream), lambda row: int(row['source'])):
            row = [(int(x['target']), float(x['similarity'])) for x in group]
            yield source, row


def compare_support(left, right, truth, n, k, tolerance):
    left_iter, right_iter = iter(candidate_rows(left)), iter(candidate_rows(right))
    overlap = left_hits = right_hits = below_cutoff = lost = gained = 0
    for source in range(n):
        a = next(left_iter, None)
        b = next(right_iter, None)
        if a is None or b is None or a[0] != source or b[0] != source:
            raise ValueError('missing/out-of-order candidate row')
        aa, bb = dict(a[1]), dict(b[1])
        if len(a[1]) != k or len(b[1]) != k or len(aa) != k or len(bb) != k:
            raise ValueError('candidate count or uniqueness error')
        overlap += len(aa.keys() & bb.keys())
        # Same CPU scoring: candidates strictly below the exhaustive cutoff
        # indicate a ranking loss; equal-cutoff substitutions can be ties.
        cutoff = min(aa.values())
        below_cutoff += sum(score < cutoff - tolerance for score in bb.values())
        ah, bh = truth[source] in aa, truth[source] in bb
        left_hits += ah
        right_hits += bh
        lost += ah and not bh
        gained += bh and not ah
    if next(left_iter, None) is not None or next(right_iter, None) is not None:
        raise ValueError('unexpected extra candidate rows')
    return {'mean_exhaustive_topk_overlap': overlap / (n*k),
            'indexed_fraction_strictly_below_exhaustive_cutoff': below_cutoff / (n*k),
            'cutoff_score_tolerance': tolerance,
            'exhaustive_candidate_recall': left_hits/n,
            'indexed_candidate_recall': right_hits/n,
            'candidate_recall_delta_indexed_minus_exhaustive': (right_hits-left_hits)/n,
            'truth_targets_lost_from_exhaustive_topk': lost,
            'truth_targets_gained_over_exhaustive_topk': gained,
            'note': 'Top-K/support loss metrics, not pre-Top-K retrieval recall; cutoff ties may change IDs.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, default=Path('build-l40s/scgm_cuda'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--ground-truth', type=Path, required=True)
    parser.add_argument('--ks', type=int, nargs='+', default=[64])
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cutoff-tolerance', type=float, default=2e-6)
    parser.add_argument('--skip-memory-profile', action='store_true', help='Skip separate GNU time host RSS runs')
    add_index_arguments(parser)
    parser.set_defaults(candidate_backend='structural-index', mode='cpu')
    args = parser.parse_args()
    if args.candidate_backend != 'structural-index':
        parser.error('this runner always compares exhaustive against structural-index')
    config = index_configuration(args, parser)
    for path in (args.executable, args.source, args.target, args.ground_truth):
        if not path.is_file(): parser.error(f'file not found: {path}')
    if args.trials < 1 or args.warmup < 0:
        parser.error('trials must be positive and warmup nonnegative')
    if not math.isfinite(args.cutoff_tolerance) or args.cutoff_tolerance < 0:
        parser.error('cutoff tolerance must be finite and nonnegative')
    source_graph, target_graph = read_graph(args.source), read_graph(args.target)
    n, m = source_graph[0], target_graph[0]
    truth = read_mapping(args.ground_truth, 'source_vertex', 'target_vertex')
    if n == 0 or n != m or set(truth) != set(range(n)) or set(truth.values()) != set(range(m)):
        parser.error('requires a nonempty equal-size pair and complete one-to-one ground truth')
    if any(k < 1 or k > m for k in args.ks) or len(set(args.ks)) != len(args.ks):
        parser.error('K values must be unique and in [1, target count]')
    if not args.skip_memory_profile and not Path('/usr/bin/time').is_file():
        parser.error('GNU /usr/bin/time is required; alternatively use --skip-memory-profile')
    provenance = {'mode': 'cpu', 'structural_index': config, 'ks': args.ks,
                  'source': str(args.source.resolve()), 'target': str(args.target.resolve()),
                  'ground_truth': str(args.ground_truth.resolve()),
                  'source_sha256': digest(args.source), 'target_sha256': digest(args.target),
                  'ground_truth_sha256': digest(args.ground_truth),
                  'executable_sha256': digest(args.executable),
                  'warmup': args.warmup, 'trials': args.trials,
                  'cutoff_tolerance': args.cutoff_tolerance, 'memory_profile': not args.skip_memory_profile,
                  'support_repair': {'enabled': False}}
    directory = args.output / ('comparison_' + configuration_id(provenance))
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'configuration.json').write_text(json.dumps(provenance, indent=2)+'\n')
    revision = subprocess.run(['git','rev-parse','HEAD'], capture_output=True, text=True)
    dirty = subprocess.run(['git','status','--porcelain'], capture_output=True, text=True)
    summary = {'configuration': provenance, 'environment': {'python': platform.python_version(),
               'platform': platform.platform(), 'git_revision': revision.stdout.strip() if revision.returncode == 0 else None,
               'git_worktree_dirty': bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
               'runner_sha256': digest(Path(__file__))}, 'comparisons': [],
               'limitations': ['CPU indexed backend only.',
                 'No support repair; infeasible cases skip assignment/Sinkhorn and cannot establish complete-pipeline speedups.',
                 'RSS is a separate isolated process high-water mark including native execution without candidate CSV export; not stage attribution.',
                 'No GPU memory comparison is applicable to CPU runs.',
                 'Repeated timings use the same saved pair, not independent graph samples.']}
    backends = [('exhaustive', None), ('structural-index', config)]
    for k in args.ks:
        kd = directory / f'K{k:03d}'
        runs = {}; quality = {}; artifacts = {}; profiles = {}
        for backend, policy in backends:
            bd = kd/backend; bd.mkdir(parents=True, exist_ok=True)
            artifacts[backend] = bd
            export = native.run_once(args.executable, args.source, args.target, k, 'cpu', bd, 0,
                                     write_candidates=True, candidate_backend=backend, index_config=policy)
            assignment = read_mapping(bd/'assignment_cpu_0.csv', 'source', 'target')
            quality[backend] = evaluate(source_graph, target_graph, assignment, truth)
            quality[backend].update({'maximum_matching_size': export['maximum_matching_size'],
                                    'support_feasible': export['maximum_matching_size'] == n,
                                    'native_assignment_status': export['assignment_status'],
                                    'index_diagnostics': export.get('index_diagnostics'),
                                    'csr_memory_bytes': export['csr_memory_bytes']})
            runs[backend] = []
            for i in range(args.warmup):
                native.run_once(args.executable, args.source, args.target, k, 'cpu', bd, f'warmup_{i}',
                                write_candidates=False, candidate_backend=backend, index_config=policy)
        for trial in range(args.trials):
            order = backends if trial % 2 == 0 else list(reversed(backends))
            for backend, policy in order:
                record = native.run_once(args.executable, args.source, args.target, k, 'cpu',
                                         artifacts[backend], f'trial_{trial}', write_candidates=False,
                                         candidate_backend=backend, index_config=policy)
                if record['maximum_matching_size'] != quality[backend]['maximum_matching_size']:
                    raise RuntimeError('candidate-support feasibility changed across timing trials')
                runs[backend].append(record)
        for backend, policy in backends:
            profiles[backend] = None
            if not args.skip_memory_profile:
                bd = artifacts[backend]
                profile = native.run_once(args.executable, args.source, args.target, k, 'cpu', bd, 'rss',
                                          write_candidates=False, candidate_backend=backend, index_config=policy,
                                          profile_host_memory=True)
                profiles[backend] = profile['peak_host_rss_bytes']
        overlap = compare_support(artifacts['exhaustive']/'candidates_cpu_0.csv',
                                  artifacts['structural-index']/'candidates_cpu_0.csv', truth,n,k,args.cutoff_tolerance)
        for backend, key in [('exhaustive', 'exhaustive_candidate_recall'),
                             ('structural-index', 'indexed_candidate_recall')]:
            quality[backend]['candidate_recall_at_k'] = overlap[key]
            quality[backend]['candidate_recall_note'] = 'Membership in the saved backend Top-K support.'
        measurements = {}
        for backend, _ in backends:
            records = runs[backend]
            measurements[backend] = {'wall_ms': stats([r['wall_clock_ms'] for r in records]),
              'candidate_with_index_setup_ms': stats([r['candidate_generation_ms']+r.get('index_build_ms',0) for r in records]),
              'host_peak_rss_bytes_isolated_profile': profiles[backend],
              'quality': quality[backend], 'trials': records}
        a, b = measurements['exhaustive'], measurements['structural-index']
        complete = all(quality[name]['support_feasible'] and quality[name]['one_to_one'] for name,_ in backends)
        ia = quality['structural-index']['index_diagnostics']
        result = {'k': k, 'backends': measurements, 'candidate_comparison': overlap,
                  'exhaustive_scored_pairs': n*m, 'indexed_scored_pairs': ia['actual_scored_pairs'],
                  'scored_pair_reduction_fraction': 1-ia['actual_scored_pairs']/(n*m),
                  'candidate_with_setup_speedup': ratio(a['candidate_with_index_setup_ms']['median'],b['candidate_with_index_setup_ms']['median']),
                  'observed_wall_ratio_exhaustive_over_indexed': ratio(a['wall_ms']['median'],b['wall_ms']['median']),
                  'complete_pipeline_speedup': ratio(a['wall_ms']['median'],b['wall_ms']['median']) if complete else None,
                  'host_peak_rss_ratio_indexed_over_exhaustive': ratio(profiles['structural-index'],profiles['exhaustive']),
                  'maximum_matching_size_delta_indexed_minus_exhaustive': quality['structural-index']['maximum_matching_size']-quality['exhaustive']['maximum_matching_size'],
                  'edge_preservation_delta_indexed_minus_exhaustive': (quality['structural-index']['edge_preservation']-quality['exhaustive']['edge_preservation']) if complete else None,
                  'final_accuracy_delta_indexed_minus_exhaustive': (quality['structural-index']['mapping_accuracy']-quality['exhaustive']['mapping_accuracy']) if complete else None}
        summary['comparisons'].append(result)
        (directory/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        print(f"K={k}: scored-pair reduction={result['scored_pair_reduction_fraction']:.1%}; "
              f"recall delta={overlap['candidate_recall_delta_indexed_minus_exhaustive']:.4f}; "
              f"both complete={complete}")
    print(f'Results: {directory / "comparison.json"}')


if __name__ == '__main__':
    main()
