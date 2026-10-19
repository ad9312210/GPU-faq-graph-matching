"""Check both CLI candidate backends, budget reporting, and batch boundaries."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    args = parser.parse_args()
    executable = args.executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    n, k = 513, 64
    with tempfile.TemporaryDirectory(prefix='gpu_topk_cli_') as folder:
        root = Path(folder)
        graph = root / 'empty_edges.txt'
        graph.write_text(f'{n} 0\n', encoding='utf-8')
        for backend in ('exhaustive', 'structural-index'):
            candidates = root / f'{backend}.csv'
            timings = root / f'{backend}.json'
            command = [str(executable), '--source', str(graph), '--target', str(graph),
                       '--gpu', '--k', str(k), '--candidate-backend', backend,
                       '--gpu-selector', 'parallel', '--gpu-topk-scratch-mib', '1',
                       '--gpu-topk-batch-candidates', '256', '--candidates', str(candidates),
                       '--timings', str(timings)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
            if timings.is_file():
                availability = json.loads(timings.read_text(encoding='utf-8'))
                if availability.get('effective_mode') == 'cpu':
                    print('SKIPPED: native GPU request fell back to CPU')
                    return 77
            # Tied scores retain targets 0..63. Global support is infeasible;
            # exit 2 is expected after candidate/timing export, not a selector failure.
            if completed.returncode != 2:
                raise AssertionError(f'{backend}: expected infeasible-support exit 2, got '
                                     f'{completed.returncode}\n{completed.stdout}\n{completed.stderr}')
            metadata = json.loads(timings.read_text(encoding='utf-8'))
            assert metadata['effective_mode'] == 'gpu'
            assert metadata['candidate_backend'] == backend
            selection = metadata['gpu_topk']
            assert selection['requested'] == selection['used'] == 'parallel'
            assert selection['fallback'] == 'none'
            assert selection['budget_enforced'] is True
            assert selection['scratch_peak_bytes'] <= selection['scratch_budget_bytes'] == 1024 * 1024
            assert selection['batch_capacity'] == 256
            assert selection['batches'] == n * 3
            with candidates.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == n * k
            for source in range(n):
                selected = rows[source * k:(source + 1) * k]
                assert all(int(row['source']) == source for row in selected)
                assert [int(row['target']) for row in selected] == list(range(k))
                assert all(float(row['similarity']) == 0.0 for row in selected)
            print(f'{backend}: PASS (multi-batch candidates and budget metadata)')


if __name__ == '__main__':
    raise SystemExit(main())
