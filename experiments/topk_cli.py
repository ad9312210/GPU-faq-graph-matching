"""Shared selector policy, native argument forwarding, and diagnostics checks."""
import math


class GPUUnavailable(RuntimeError):
    """Requested GPU execution fell back to CPU; do not report GPU evidence."""


def add_topk_arguments(parser):
    parser.add_argument('--gpu-selector', choices=('serial', 'streaming-serial', 'parallel'), default=None)
    parser.add_argument('--gpu-topk-scratch-mib', type=int, default=None)
    parser.add_argument('--gpu-topk-batch-candidates', type=int, default=None)


def topk_configuration(args, parser):
    explicit = any(getattr(args, name, None) is not None for name in
                   ('gpu_selector', 'gpu_topk_scratch_mib', 'gpu_topk_batch_candidates'))
    if args.mode != 'gpu':
        if explicit:
            parser.error('GPU selector options require --mode gpu')
        return None
    policy = args.gpu_selector or 'serial'
    budget = 64 if args.gpu_topk_scratch_mib is None else args.gpu_topk_scratch_mib
    batch = 65536 if args.gpu_topk_batch_candidates is None else args.gpu_topk_batch_candidates
    if not 1 <= budget <= 2147483647 or not 1 <= batch <= 2147483647:
        parser.error('GPU scratch MiB and batch candidates must be in [1, INT_MAX]')
    explicit_limits = args.gpu_topk_scratch_mib is not None or args.gpu_topk_batch_candidates is not None
    if policy == 'serial' and getattr(args, 'candidate_backend', 'exhaustive') == 'exhaustive' and explicit_limits:
        parser.error('legacy serial has no scratch cap; choose streaming-serial or parallel')
    return {'policy': policy, 'scratch_mib': budget, 'batch_candidates': batch}


def native_topk_arguments(config, backend='exhaustive'):
    if config is None:
        return []
    command = ['--gpu-selector', config['policy']]
    if config['policy'] != 'serial' or backend == 'structural-index':
        command += ['--gpu-topk-scratch-mib', str(config['scratch_mib']),
                    '--gpu-topk-batch-candidates', str(config['batch_candidates'])]
    return command


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f'invalid native selector integer: {name}')
    return value


def verify_topk_metadata(native, config, mode, k):
    actual_mode = native.get('effective_mode')
    if mode == 'gpu' and actual_mode == 'cpu':
        raise GPUUnavailable('native GPU request executed on CPU; GPU result is skipped')
    if actual_mode != mode:
        raise RuntimeError('native effective mode does not match request')
    data = native.get('gpu_topk')
    if mode == 'cpu':
        if data is not None:
            raise RuntimeError('CPU execution unexpectedly reports a GPU selector')
        return None
    if native.get('k') != k:
        raise RuntimeError('native effective K differs from requested K')
    config = config or {'policy': 'serial', 'scratch_mib': 64, 'batch_candidates': 65536}
    if not isinstance(data, dict) or data.get('schema_version') != 1:
        raise RuntimeError('missing/version-mismatched selector diagnostics; rebuild the native executable')
    policy = config['policy']
    requested = 'parallel' if policy == 'parallel' else 'serial_heap'
    if data.get('requested_policy') != policy or data.get('requested') != requested:
        raise RuntimeError('native requested selection policy mismatch')
    if data.get('used') not in ('parallel', 'serial_heap'):
        raise RuntimeError('unknown effective selector')
    if data.get('chunk_size') != 256 or data.get('parallel_k_min') != 1 or data.get('parallel_k_max') != 256:
        raise RuntimeError('native chunk/K contract mismatch')
    bounded = native.get('candidate_backend') == 'structural-index' or policy != 'serial'
    if data.get('budget_enforced') is not bounded:
        raise RuntimeError('incorrect scratch-budget scope diagnostic')
    implementation = ('legacy_tiled_heap' if not bounded else
                      'streaming_parallel_merge' if data['used'] == 'parallel' else 'streaming_serial_heap')
    if data.get('implementation') != implementation:
        raise RuntimeError('effective implementation does not match selector/budget scope')
    reason = data.get('fallback')
    if reason not in ('none', 'unsupported_k', 'device_allocation_failure', 'scratch_budget_too_small'):
        raise RuntimeError('unknown fallback reason')
    if requested == 'serial_heap' and (data['used'] != 'serial_heap' or reason != 'none'):
        raise RuntimeError('serial request incorrectly reports parallel/fallback')
    if requested == 'parallel':
        expected_reason = ('unsupported_k' if k > 256 else
                           'scratch_budget_too_small' if config['batch_candidates'] < 256 else None)
        if expected_reason and (reason != expected_reason or data['used'] != 'serial_heap'):
            raise RuntimeError('required serial fallback is not recorded')
        if not expected_reason and not ((data['used'] == 'parallel' and reason == 'none') or
                 (data['used'] == 'serial_heap' and reason == 'device_allocation_failure')):
            raise RuntimeError('unexpected effective implementation/fallback')
    budget = _integer(data.get('scratch_budget_bytes'), 'scratch_budget_bytes', 1)
    if budget != config['scratch_mib'] * 1024 * 1024:
        raise RuntimeError('native scratch budget differs from request')
    if data.get('requested_batch_candidates') != config['batch_candidates']:
        raise RuntimeError('native requested batch maximum differs from request')
    capacity = _integer(data.get('batch_capacity'), 'batch_capacity')
    batches = _integer(data.get('batches'), 'batches')
    _integer(data.get('shared_bytes_per_block'), 'shared_bytes_per_block')
    if bounded:
        peak = _integer(data.get('scratch_peak_bytes'), 'scratch_peak_bytes')
        if peak > budget or capacity > config['batch_candidates']:
            raise RuntimeError('native scratch or effective batch exceeds configured limit')
        if batches and not capacity:
            raise RuntimeError('nonempty execution reports zero batch capacity')
    elif data.get('scratch_peak_bytes') is not None:
        raise RuntimeError('legacy unbudgeted scratch must be marked unavailable')
    for key in ('selection_ms', 'candidate_stage_wall_ms'):
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise RuntimeError(f'invalid selector timing: {key}')
    return data
