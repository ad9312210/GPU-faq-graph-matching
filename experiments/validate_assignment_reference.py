"""Compare the native assignment driver with SciPy on identical float32 costs.

Small dense reference matrices are intentional: infinity marks forbidden edges,
while stored zero/negative edges remain valid. No graph features or truth labels
are used. Exit 0 means all cases passed; exit 1 means at least one case failed.
"""
import argparse
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment


def serialize(rows, cols, edges):
    lines = [f"{rows} {cols} {len(edges)}"]
    lines.extend(f"{i} {j} {float(c):.9g}" for i, j, c in edges)
    return "\n".join(lines) + "\n"


def reference(rows, cols, edges):
    if rows > cols:
        return {"feasible": False, "total_cost": None}
    if rows == 0:
        return {"feasible": True, "total_cost": 0.0}
    matrix = np.full((rows, cols), np.inf, dtype=np.float64)
    for i, j, value in edges:
        matrix[i, j] = float(value)
    try:
        rr, cc = linear_sum_assignment(matrix)
    except ValueError as error:
        if "infeasible" not in str(error).lower():
            raise
        return {"feasible": False, "total_cost": None}
    return {"feasible": True, "total_cost": float(matrix[rr, cc].sum())}


def check_result(result, expected, rows, cols, edges):
    if type(result.get("feasible")) is not bool:
        raise ValueError("missing or nonboolean feasibility")
    if result["feasible"] != expected["feasible"]:
        raise ValueError("native/reference feasibility disagreement")
    forward, reverse = result.get("row_to_col"), result.get("col_to_row")
    if not isinstance(forward, list) or len(forward) != rows:
        raise ValueError("invalid forward mapping length")
    if not isinstance(reverse, list) or len(reverse) != cols:
        raise ValueError("invalid reverse mapping length")
    if not result["feasible"]:
        if result.get("total_cost") is not None:
            raise ValueError("infeasible result must have null cost")
        return
    if any(type(j) is not int or not 0 <= j < cols for j in forward):
        raise ValueError("invalid target index")
    if len(set(forward)) != rows:
        raise ValueError("duplicate assigned targets")
    if any(type(i) is not int for i in reverse):
        raise ValueError("noninteger reverse mapping")
    allowed = {(i, j): float(c) for i, j, c in edges}
    total = 0.0
    for i, j in enumerate(forward):
        if (i, j) not in allowed:
            raise ValueError("selected edge absent from support")
        if reverse[j] != i:
            raise ValueError("forward/reverse mappings disagree")
        total += allowed[i, j]
    used = set(forward)
    if any(reverse[j] != -1 for j in range(cols) if j not in used):
        raise ValueError("unused target reverse mapping is not -1")
    reported = result.get("total_cost")
    if type(reported) not in (float, int) or not math.isfinite(reported):
        raise ValueError("nonfinite or missing total cost")
    if not math.isclose(total, reported, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("reported cost disagrees with selected edges")
    if not math.isclose(total, expected["total_cost"], rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("native cost disagrees with SciPy optimum")


def fixed_cases():
    return [
        ("negative", 2, 3, [(0, 0, -3), (0, 1, 2), (1, 1, -2), (1, 2, 4)]),
        ("ties", 2, 2, [(i, j, 1) for i in range(2) for j in range(2)]),
        ("zero", 2, 2, [(0, 0, 0), (1, 1, 0)]),
        ("empty_row", 2, 2, [(0, 0, 1)]),
        ("too_many_sources", 3, 2, [(i, j, i+j) for i in range(3) for j in range(2)]),
        ("hall_deficiency", 4, 4, [(i, j, 1) for i in range(3) for j in range(2)] + [(3, 2, 1), (3, 3, 1)]),
        ("rectangular_competition", 2, 4, [(0, 0, 1), (0, 1, 2), (0, 2, 8), (1, 0, 1), (1, 1, 10), (1, 3, 9)]),
        ("empty", 0, 3, []),
    ]


def random_case(seed, index, max_size):
    rng = np.random.default_rng(seed)
    rows = int(rng.integers(1, max_size + 1))
    guaranteed = index % 2 == 0
    cols = int(rng.integers(rows if guaranteed else 1, max_size + 1))
    density = (0.15, 0.35, 0.75, 1.0)[(index // 2) % 4]
    support = rng.random((rows, cols)) < density
    if guaranteed:
        support[np.arange(rows), rng.permutation(cols)[:rows]] = True
    # Both integer tie-heavy and noninteger float32 inputs.
    if (index // 8) % 2 == 0:
        values = rng.integers(-5, 6, size=(rows, cols)).astype(np.float32)
    else:
        values = rng.uniform(-5, 5, size=(rows, cols)).astype(np.float32)
    edges = [(int(i), int(j), float(values[i, j])) for i, j in zip(*np.nonzero(support))]
    rng.shuffle(edges)  # Also exercise the driver's sorting.
    return f"random_{index}_seed_{seed}", rows, cols, edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("build-l40s/assignment_solver_driver"))
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--cases", type=int, default=200, help="Random cases, plus eight explicit cases")
    parser.add_argument("--max-size", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--output", type=Path, required=True, help="Summary JSON; failing fixtures saved beside it")
    args = parser.parse_args()
    if args.seed < 0 or args.cases < 0 or args.max_size < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("seed/cases must be nonnegative; max-size and timeout must be positive")
    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"executable not found: {executable}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = fixed_cases() + [random_case(args.seed + i, i, args.max_size) for i in range(args.cases)]
    failures = []
    feasible = 0
    corpus = hashlib.sha256()
    for name, rows, cols, edges in cases:
        # Fixed cases also follow the native float32 storage contract.
        edges = [(i, j, float(np.float32(c))) for i, j, c in edges]
        problem = serialize(rows, cols, edges)
        corpus.update(problem.encode())
        expected = reference(rows, cols, edges)
        feasible += int(expected["feasible"])
        stdout = stderr = ""
        try:
            process = subprocess.run([str(executable)], input=problem, text=True,
                                     capture_output=True, timeout=args.timeout)
            stdout, stderr = process.stdout, process.stderr
            if process.returncode:
                raise ValueError(f"driver exited with status {process.returncode}")
            result = json.loads(stdout)
            if not isinstance(result, dict):
                raise ValueError("driver did not return a JSON object")
            check_result(result, expected, rows, cols, edges)
        except (ValueError, OSError, subprocess.TimeoutExpired) as error:
            failure_dir = args.output.parent / (args.output.stem + "_failures")
            failure_dir.mkdir(exist_ok=True)
            fixture = failure_dir / (name + ".txt")
            fixture.write_text(problem)
            detail = {"case": name, "error": str(error), "reference": expected,
                      "stdout": stdout, "stderr": stderr, "fixture": str(fixture)}
            fixture.with_suffix(".json").write_text(json.dumps(detail, indent=2) + "\n")
            failures.append(detail)
            print(f"FAIL {name}: {error}")
    summary = {
        "reference": "scipy.optimize.linear_sum_assignment; infinity for forbidden edges",
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "executable": str(executable), "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "seed": args.seed, "random_cases": args.cases, "explicit_cases": 8, "max_size": args.max_size,
        "corpus_sha256": corpus.hexdigest(), "total": len(cases),
        "passed": len(cases)-len(failures), "failed": len(failures),
        "reference_feasible": feasible, "reference_infeasible": len(cases)-feasible,
        "failures": failures,
    }
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Passed {summary['passed']}/{summary['total']}; failures: {summary['failed']}")
    print(f"Summary: {args.output}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
