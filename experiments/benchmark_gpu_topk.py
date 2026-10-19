"""Benchmark identical-score selectors, then optionally compare complete GPU pipelines.

Run this on the intended GPU device. Reports never label serial fallback as a
parallel speedup or infeasible assignment as a completed matching speedup.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from benchmark_native import run_once
from index_cli import configuration_id, native_index_arguments
from topk_cli import GPUUnavailable


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): result.update(block)
    return result.hexdigest()


def stats(values):
    values = [float(value) for value in values]
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('missing/nonfinite/negative timing values')
    return {'median':statistics.median(values),'min':min(values),'max':max(values),'n':len(values)}


def ratio(a,b):
    return a/b if a is not None and b is not None and a >= 0 and b > 0 else None


def micro_comparisons(cases):
    groups = {}
    for case in cases:
        if case.get('kind') != 'case' or case.get('oracle_exact') is not True or not case.get('trials'):
            raise ValueError('native benchmark case lacks exact oracle evidence/raw trials')
        key = (case['pool_size'],case['k'],case['distribution'],case['batch_requested'],case['scratch_budget_bytes'])
        policy = case['requested']
        if policy not in ('serial_heap','parallel') or policy in groups.setdefault(key,{}):
            raise ValueError('unknown or duplicate microbenchmark selector')
        for trial in case['trials']:
            if trial['scratch_peak_bytes'] > case['scratch_budget_bytes']:
                raise ValueError('native benchmark exceeded scratch budget')
            expected_impl='streaming_parallel_merge' if trial['used']=='parallel' else 'streaming_serial_heap'
            if trial.get('implementation')!=expected_impl: raise ValueError('native implementation diagnostic mismatch')
            if trial['used'] not in ('serial_heap','parallel') or trial['fallback'] not in (
                    'none','unsupported_k','scratch_budget_too_small','device_allocation_failure'):
                raise ValueError('unknown native effective selector/fallback')
        groups[key][policy] = case
    comparisons=[]
    for key, group in sorted(groups.items()):
        if set(group) != {'serial_heap','parallel'}: raise ValueError('incomplete serial/parallel pair')
        serial,parallel = group['serial_heap'],group['parallel']
        if serial['input_fingerprint_fnv64'] != parallel['input_fingerprint_fnv64']:
            raise ValueError('serial/parallel inputs differ')
        eligible = key[0] > 0 and all(t['used']=='serial_heap' and t['fallback']=='none' for t in serial['trials']) and all(
            t['used']=='parallel' and t['fallback']=='none' and t['batches']>0 for t in parallel['trials'])
        summary={policy:{field:stats([t[field] for t in case['trials']])
                        for field in ('wall_ms','selection_ms','transfer_ms')}
                 for policy,case in group.items()}
        comparisons.append({'pool_size':key[0],'k':key[1],'distribution':key[2],
            'batch_requested':key[3],'scratch_budget_bytes':key[4],
            'input_fingerprint_fnv64':serial['input_fingerprint_fnv64'],
            'exact_oracle_passed':True,'actual_parallel_execution':eligible,'timings':summary,
            'observed_wall_ratio':ratio(summary['serial_heap']['wall_ms']['median'],summary['parallel']['wall_ms']['median']),
            'parallel_selection_speedup':ratio(summary['serial_heap']['selection_ms']['median'],summary['parallel']['selection_ms']['median']) if eligible else None,
            'parallel_wall_speedup':ratio(summary['serial_heap']['wall_ms']['median'],summary['parallel']['wall_ms']['median']) if eligible else None,
            'scratch_peak_bytes':max(t['scratch_peak_bytes'] for case in group.values() for t in case['trials']),
            'effective_fallbacks':sorted(set(t['fallback'] for t in parallel['trials']))})
    return comparisons


def graph(path):
    with Path(path).open(encoding='utf-8') as stream:
        n,e=map(int,stream.readline().split()); edges=[tuple(map(int,line.split())) for line in stream if line.strip()]
    if len(edges)!=e: raise ValueError('graph header edge count mismatch')
    adjacency=[set() for _ in range(n)]
    for a,b in edges:
        if not 0<=a<n or not 0<=b<n or a==b or b in adjacency[a]: raise ValueError('invalid graph edge')
        adjacency[a].add(b); adjacency[b].add(a)
    return adjacency


def candidate_rows(path,n,m,k):
    rows=[[] for _ in range(n)]
    with Path(path).open(newline='',encoding='utf-8') as stream:
        for item in csv.DictReader(stream):
            source,target,score=int(item['source']),int(item['target']),float(item['similarity'])
            if not 0<=source<n or not 0<=target<m or not math.isfinite(score): raise ValueError('invalid candidate export')
            if int(item['rank'])!=len(rows[source]): raise ValueError('invalid candidate ranks')
            rows[source].append((target,score))
    for row in rows:
        if len(row)!=k or len({target for target,_ in row})!=k: raise ValueError('candidate count/uniqueness mismatch')
        if any(not (row[i-1][1]>row[i][1] or (row[i-1][1]==row[i][1] and row[i-1][0]<row[i][0])) for i in range(1,k)):
            raise ValueError('candidate export violates deterministic order')
    return rows


def quality(rows,assignment_path,source,target,truth):
    n,m=len(source),len(target); mapping={}; costs=[]
    with Path(assignment_path).open(newline='',encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            a,b=int(row['source']),int(row['target'])
            if a in mapping: raise ValueError('duplicate assignment source')
            mapping[a]=b
            if b>=0:
                cost=float(row['cost'])
                if not math.isfinite(cost): raise ValueError('nonfinite assignment cost')
                costs.append(cost)
    valid=set(mapping)==set(range(n)) and all(0<=b<m for b in mapping.values()) and len(set(mapping.values()))==n
    edges=sum(len(neighbors) for neighbors in source)//2
    preserved=sum(mapping[b] in target[mapping[a]] for a in range(n) for b in source[a] if b>a) if valid else None
    return {'one_to_one':valid,'assignment_cost':sum(costs) if valid else None,
            'candidate_recall_at_k':sum(truth[a] in {b for b,_ in rows[a]} for a in range(n))/n if truth else None,
            'final_accuracy':sum(mapping[a]==truth[a] for a in range(n))/n if truth and valid else None,
            'edge_preservation':preserved/edges if valid and edges else None}


def pipeline_gate(serial,parallel,comparison):
    """Return eligibility; observed time ratios alone do not establish a speedup."""
    effective=bool(serial['trials'] and parallel['trials']) and all(
        run['gpu_topk']['used']=='serial_heap' and run['gpu_topk']['fallback']=='none' for run in serial['trials']) and all(
        run['gpu_topk']['used']=='parallel' and run['gpu_topk']['fallback']=='none' for run in parallel['trials'])
    complete=serial['quality']['one_to_one'] and parallel['quality']['one_to_one'] and all(
        run.get('assignment_feasible') and run.get('assignment_one_to_one') for result in (serial,parallel) for run in result['trials'])
    invariant=all(serial['quality'].get(key)==parallel['quality'].get(key)
                  for key in ('candidate_recall_at_k','final_accuracy','edge_preservation','assignment_cost'))
    return bool(effective and complete and invariant and comparison['ordered_target_ids_equal'] and comparison['scores_within_tolerance'])


def run_pipeline(args,directory):
    source,target=graph(args.source),graph(args.target); n,m=len(source),len(target)
    if not n or any(k>m for k in args.ks): raise ValueError('pipeline requires nonempty sources and K <= targets')
    truth=None
    if args.ground_truth:
        truth={}
        with args.ground_truth.open(newline='',encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                a,b=int(row['source_vertex']),int(row['target_vertex'])
                if a in truth or not 0<=b<m: raise ValueError('invalid truth mapping')
                truth[a]=b
        if set(truth)!=set(range(n)): raise ValueError('truth must cover every source')
    policies=['streaming-serial','parallel']+(['serial'] if args.include_legacy else [])
    results=[]
    index_config={'degree_bin_width':args.index_degree_width,'neighbor_bin_width':args.index_neighbor_width,
                  'minimum_radius':args.index_min_radius,'maximum_radius':args.index_max_radius,
                  'pool_multiplier':args.index_pool_multiplier,'fallback':'exhaustive'}
    for backend in args.backends:
        for k in args.ks:
            variants={}; exports={}
            for policy in policies:
                folder=directory/backend/f'K{k}'/policy; folder.mkdir(parents=True,exist_ok=True)
                config={'policy':policy,'scratch_mib':64 if policy=='serial' and backend=='exhaustive' else args.scratch_mib,
                        'batch_candidates':65536 if policy=='serial' and backend=='exhaustive' else args.pipeline_batch}
                config_index=index_config if backend=='structural-index' else None
                record=run_once(args.executable.resolve(),args.source.resolve(),args.target.resolve(),k,'gpu',folder,0,
                    candidate_backend=backend,index_config=config_index,topk_config=config)
                rows=candidate_rows(folder/'candidates_gpu_0.csv',n,m,k)
                exports[policy]=rows
                variants[policy]={'quality':quality(rows,folder/'assignment_gpu_0.csv',source,target,truth),
                    'export_diagnostics':record,'trials':[],'folder':str(folder),'config':config,'index_config':config_index}
                for warmup in range(args.warmup):
                    run_once(args.executable.resolve(),args.source.resolve(),args.target.resolve(),k,'gpu',folder,f'warmup{warmup}',
                        write_candidates=False,candidate_backend=backend,index_config=config_index,topk_config=config)
            for trial in range(args.trials):
                order=policies if trial%2==0 else list(reversed(policies))
                for policy in order:
                    variant=variants[policy]
                    variant['trials'].append(run_once(args.executable.resolve(),args.source.resolve(),args.target.resolve(),k,'gpu',
                        Path(variant['folder']),f'trial{trial}',write_candidates=False,candidate_backend=backend,
                        index_config=variant['index_config'],topk_config=variant['config']))
            for variant in variants.values():
                variant['wall_ms']=stats([run['wall_clock_ms'] for run in variant['trials']])
                variant['candidate_stage_wall_ms']=stats([run['gpu_topk']['candidate_stage_wall_ms'] for run in variant['trials']])
                variant['selection_ms']=stats([run['gpu_topk']['selection_ms'] for run in variant['trials']])
            baseline,parallel=variants['streaming-serial'],variants['parallel']
            left,right=exports['streaming-serial'],exports['parallel']
            equal=all([target for target,_ in a]==[target for target,_ in b] for a,b in zip(left,right))
            score_error=max((abs(a[i][1]-b[i][1]) for a,b in zip(left,right) for i in range(k)),default=0) if equal else None
            comparison={'ordered_target_ids_equal':equal,'max_shared_rank_score_error':score_error,
                        'scores_within_tolerance':score_error is not None and score_error<=args.score_tolerance}
            eligible=pipeline_gate(baseline,parallel,comparison)
            results.append({'backend':backend,'k':k,'variants':variants,'candidate_comparison':comparison,
                'ground_truth_available':truth is not None,'quality_export_run_separate_from_timing_trials':True,
                'supported_pipeline_comparison':eligible,
                'observed_wall_ratio':ratio(baseline['wall_ms']['median'],parallel['wall_ms']['median']),
                'parallel_pipeline_speedup':ratio(baseline['wall_ms']['median'],parallel['wall_ms']['median']) if eligible else None,
                'note':'Fixed saved pair; repeated trials are not independent graph samples. Legacy mode is context only.'})
    return results


def findings(report):
    lines=['# Parallel Top-K findings','',f"Status: {report['status']}",'',
           'Only configurations with exact microbenchmark oracle checks and actual parallel execution receive parallel speedup fields.',
           'Ratios above one favor parallel; these are observed medians, not a universal or statistical-significance claim.','',
           '| Pool | K | Distribution | Batch maximum | Selection ratio | Wall ratio | Actual parallel |',
           '|---:|---:|---|---:|---:|---:|---|']
    def number(value): return 'unavailable' if value is None else f'{value:.3f}'
    for row in report.get('micro_comparisons',[]):
        lines.append(f"| {row['pool_size']} | {row['k']} | {row['distribution']} | {row['batch_requested']} | "
                     f"{number(row['parallel_selection_speedup'])} | {number(row['parallel_wall_speedup'])} | {row['actual_parallel_execution']} |")
    lines += ['', 'Pipeline comparisons use streaming-serial versus parallel with the same scoring producer.',
              'A complete-pipeline speedup field also requires feasible one-to-one assignments and unchanged measured candidate/quality evidence.','']
    for row in report.get('pipeline_comparisons',[]):
        lines.append(f"- {row['backend']}, K={row['k']}: supported={row['supported_pipeline_comparison']}; "
                     f"pipeline median ratio={number(row['parallel_pipeline_speedup'])}; truth supplied={row['ground_truth_available']}.")
    if not report.get('pipeline_comparisons'): lines.append('Pipeline impact and matching quality were not measured in this run.')
    if report.get('reason'): lines += ['',report['reason']]
    lines += ['', 'See benchmark.json for raw trials, fallback reasons, exact scratch peaks, hardware, configuration and executable/input hashes.']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-executable',type=Path,default=Path('build-l40s/benchmark_gpu_topk_streaming'))
    parser.add_argument('--executable',type=Path,default=Path('build-l40s/scgm_cuda'))
    parser.add_argument('--pool-sizes',type=int,nargs='+',default=[257,4097,65537])
    parser.add_argument('--ks',type=int,nargs='+',default=[1,64,256,512])
    parser.add_argument('--batch-sizes',type=int,nargs='+',default=[256,4096,65536])
    parser.add_argument('--distributions',nargs='+',choices=('random','ties','negative','zeros'),default=['random','ties'])
    parser.add_argument('--scratch-mib',type=int,default=64)
    parser.add_argument('--warmup',type=int,default=2); parser.add_argument('--trials',type=int,default=7)
    parser.add_argument('--seed',type=int,default=173); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--findings-output',type=Path)
    parser.add_argument('--source',type=Path); parser.add_argument('--target',type=Path); parser.add_argument('--ground-truth',type=Path)
    parser.add_argument('--backends',nargs='+',choices=('exhaustive','structural-index'),default=['exhaustive','structural-index'])
    parser.add_argument('--pipeline-batch',type=int,default=65536); parser.add_argument('--include-legacy',action='store_true')
    parser.add_argument('--score-tolerance',type=float,default=2e-5)
    parser.add_argument('--index-degree-width',type=float,default=.5); parser.add_argument('--index-neighbor-width',type=float,default=.5)
    parser.add_argument('--index-min-radius',type=int,default=1); parser.add_argument('--index-max-radius',type=int,default=3)
    parser.add_argument('--index-pool-multiplier',type=int,default=4)
    args=parser.parse_args()
    if not 0<=args.warmup<=2147483647 or not 1<=args.trials<=2147483647 or not 0<=args.seed<=2147483647 or not 1<=args.scratch_mib<=2147483647 or not 1<=args.pipeline_batch<=2147483647:
        parser.error('invalid warmup/trials/seed/budget/batch configuration')
    if any(n<0 or n>2147483647 for n in args.pool_sizes) or any(k<1 or k>2147483647 for k in args.ks+args.batch_sizes):
        parser.error('pool/K/batch values outside supported range')
    if any(len(items)!=len(set(items)) for items in (args.pool_sizes,args.ks,args.batch_sizes,args.distributions,args.backends)):
        parser.error('duplicate benchmark configurations')
    if bool(args.source)!=bool(args.target) or (args.ground_truth and not args.source): parser.error('supply both source/target for pipeline tests')
    if not math.isfinite(args.score_tolerance) or args.score_tolerance<0: parser.error('invalid score tolerance')
    if not (0<=args.index_min_radius<=args.index_max_radius<=2147483647) or not 1<=args.index_pool_multiplier<=2147483647:
        parser.error('invalid index radii or pool multiplier')
    if any(not math.isfinite(width) or width<=0 for width in (args.index_degree_width,args.index_neighbor_width)):
        parser.error('index widths must be finite and positive')
    paths=[args.benchmark_executable]+([args.executable,args.source,args.target] if args.source else [])+([args.ground_truth] if args.ground_truth else [])
    for path in paths:
        if not path.is_file(): parser.error(f'missing file: {path}')
    requested={key:value for key,value in vars(args).items() if key not in ('output','findings_output')}
    requested={key:str(value) if isinstance(value,Path) else value for key,value in requested.items()}
    requested['file_sha256']={str(path.resolve()):digest(path) for path in paths}
    requested['analysis_sha256']={name:digest(Path(__file__).with_name(name)) for name in
        ('benchmark_gpu_topk.py','benchmark_native.py','topk_cli.py','index_cli.py')}
    directory=args.output/('config_'+configuration_id(requested)); directory.mkdir(parents=True,exist_ok=True)
    report={'status':'pending','configuration':requested,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'host':platform.node(),'platform':platform.platform(),'python':platform.python_version(),
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'micro_comparisons':[],'pipeline_comparisons':[]}
    cache=args.benchmark_executable.resolve().parent/'CMakeCache.txt'
    if cache.is_file(): report['cmake_build_configuration']=[line for line in cache.read_text().splitlines() if line.startswith(
        ('CMAKE_BUILD_TYPE:','CMAKE_CUDA_ARCHITECTURES:','CMAKE_CUDA_COMPILER:','CMAKE_CXX_COMPILER:'))]
    command=[str(args.benchmark_executable.resolve()),'--json','--sizes',','.join(map(str,args.pool_sizes)),
        '--ks',','.join(map(str,args.ks)),'--batches',','.join(map(str,args.batch_sizes)),
        '--budget-bytes',str(args.scratch_mib*1024*1024),'--warmup',str(args.warmup),'--trials',str(args.trials),
        '--seed',str(args.seed),'--distributions',','.join(args.distributions)]
    report['micro_command']=command
    code=0
    try:
        completed=subprocess.run(command,capture_output=True,text=True)
        (directory/'micro_stdout.jsonl').write_text(completed.stdout)
        (directory/'micro_stderr.txt').write_text(completed.stderr)
        objects=[json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
        if completed.returncode==77: raise GPUUnavailable('microbenchmark skipped: no usable GPU')
        if completed.returncode!=0: raise RuntimeError(f'native benchmark failed ({completed.returncode}): {completed.stderr}')
        environments=[item for item in objects if item.get('kind')=='environment']
        if len(environments)!=1: raise ValueError('native benchmark did not report one GPU environment')
        report['gpu_environment']=environments[0]
        report['micro_cases']=[item for item in objects if item.get('kind')=='case']
        expected=len(args.pool_sizes)*len(args.ks)*len(args.batch_sizes)*len(args.distributions)*2
        if len(report['micro_cases'])!=expected: raise ValueError('native benchmark case count mismatch')
        if any(len(case['trials'])!=args.trials for case in report['micro_cases']): raise ValueError('native trial count mismatch')
        report['micro_comparisons']=micro_comparisons(report['micro_cases'])
        if args.source: report['pipeline_comparisons']=run_pipeline(args,directory/'pipeline')
        report['status']='measured'
    except GPUUnavailable as error:
        report.update(status='skipped',reason=str(error)); code=77
    except (ValueError,RuntimeError,OSError) as error:
        report.update(status='failed',reason=str(error)); code=1
    (directory/'benchmark.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    findings_path=args.findings_output or directory/'parallel_topk_findings.md'
    findings_path.parent.mkdir(parents=True,exist_ok=True); findings_path.write_text(findings(report))
    print(f"{report['status']}: {directory/'benchmark.json'}\nFindings: {findings_path}")
    return code


if __name__=='__main__':
    raise SystemExit(main())
