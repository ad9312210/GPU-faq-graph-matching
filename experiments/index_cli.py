"""Shared structural-index CLI contract for native experiment runners."""
import hashlib
import json
import math


def add_index_arguments(parser):
    parser.add_argument('--candidate-backend', choices=('exhaustive', 'structural-index'), default='exhaustive')
    for flag in ('degree-width', 'neighbor-width'):
        parser.add_argument('--index-' + flag, type=float, default=None)
    for flag in ('min-radius', 'max-radius', 'pool-multiplier'):
        parser.add_argument('--index-' + flag, type=int, default=None)
    parser.add_argument('--index-fallback', choices=('exhaustive',), default=None)


def index_configuration(args, parser):
    values = dict(degree_bin_width=args.index_degree_width,
                  neighbor_bin_width=args.index_neighbor_width,
                  minimum_radius=args.index_min_radius,
                  maximum_radius=args.index_max_radius,
                  pool_multiplier=args.index_pool_multiplier,
                  fallback=args.index_fallback)
    if args.candidate_backend == 'exhaustive':
        if any(value is not None for value in values.values()):
            parser.error('index options require --candidate-backend structural-index')
        return None
    defaults = dict(degree_bin_width=.5, neighbor_bin_width=.5,
                    minimum_radius=1, maximum_radius=3, pool_multiplier=4,
                    fallback='exhaustive')
    config = {key: defaults[key] if value is None else value for key, value in values.items()}
    if any(not math.isfinite(config[key]) or config[key] <= 0
           for key in ('degree_bin_width', 'neighbor_bin_width')):
        parser.error('index widths must be finite and positive')
    if not 0 <= config['minimum_radius'] <= config['maximum_radius'] <= 2147483647:
        parser.error('require 0 <= min radius <= max radius <= INT_MAX')
    if not 1 <= config['pool_multiplier'] <= 2147483647:
        parser.error('pool multiplier must be in [1, INT_MAX]')
    return config


def native_index_arguments(backend, config):
    command = ['--candidate-backend', backend]
    if config is not None:
        for flag, key in [('degree-width','degree_bin_width'), ('neighbor-width','neighbor_bin_width'),
                          ('min-radius','minimum_radius'), ('max-radius','maximum_radius'),
                          ('pool-multiplier','pool_multiplier'), ('fallback','fallback')]:
            command += ['--index-' + flag, str(config[key])]
    return command


def verify_index_metadata(native, backend, config, mode):
    if native.get('candidate_backend') != backend or native.get('effective_mode') != mode:
        raise RuntimeError('native effective backend/mode does not match requested configuration')
    actual = native.get('structural_index')
    if config is None:
        if actual is not None:
            raise RuntimeError('unexpected structural-index metadata')
    elif not isinstance(actual, dict) or any(actual.get(k) != v for k,v in config.items()):
        raise RuntimeError('native structural-index configuration mismatch')


def configuration_id(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()[:16]
