"""Compare two saved Top-K rankings independently of ground truth."""
import argparse
import csv
import json
from pathlib import Path


def load(path, n):
    rows = [[] for _ in range(n)]
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows[int(row["source"])].append(
                (int(row["rank"]), int(row["target"]), float(row["similarity"]))
            )
    for row in rows:
        row.sort()
    if not rows or any(not row for row in rows):
        raise ValueError(f"incomplete candidate ranking: {path}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--left", type=Path, required=True)
    ap.add_argument("--right", type=Path, required=True)
    ap.add_argument("--vertices", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    left, right = load(args.left, args.vertices), load(args.right, args.vertices)
    if len(left[0]) != len(right[0]) or any(len(a) != len(b) for a, b in zip(left, right)):
        raise ValueError("rankings must use equal K for every source")
    n, k = args.vertices, len(left[0])
    same_sets = 0
    overlap = 0
    rank_target_equal = 0
    score_errors = []
    for a, b in zip(left, right):
        aset, bset = {x[1] for x in a}, {x[1] for x in b}
        same_sets += aset == bset
        overlap += len(aset & bset)
        for x, y in zip(a, b):
            rank_target_equal += x[1] == y[1]
            score_errors.append(abs(x[2] - y[2]))
    result = {
        "vertices": n,
        "k": k,
        "rows_with_identical_candidate_sets": same_sets,
        "candidate_set_overlap_fraction": overlap / (n * k),
        "rank_positions_with_same_target_fraction": rank_target_equal / (n * k),
        "mean_absolute_rankwise_similarity_difference": sum(score_errors) / len(score_errors),
        "max_absolute_rankwise_similarity_difference": max(score_errors),
        "left": str(args.left),
        "right": str(args.right),
        "ground_truth_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
