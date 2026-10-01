"""Generate fixed-seed permutation pairs at requested target-noise percentages."""

import argparse
import json
from pathlib import Path

from generate_paired_graphs import generate_pair


SEEDS = (11, 22, 33)


def generate_variants(dataset_dir, output_dir, dataset, noise_levels):
    source_graph = dataset_dir / "G.graph"
    if not source_graph.is_file():
        raise FileNotFoundError(f"cleaned graph does not exist: {source_graph}")

    for noise_percent in noise_levels:
        if not 0 <= noise_percent <= 100:
            raise ValueError("noise percentages must be between 0 and 100")
        for seed in SEEDS:
            pair_dir = output_dir / f"seed_{seed}" / f"noise_{noise_percent:02d}"
            generate_pair(
                pair_dir,
                num_vertices=_vertex_count(source_graph),
                edge_probability=0.5,  # ignored when a source graph is supplied
                seed=seed,
                edge_noise=noise_percent / 100.0,
                source_graph=source_graph,
            )
            manifest_path = pair_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update({
                "dataset": dataset,
                "pair_seed": seed,
                "noise_percent": noise_percent,
                "noise_policy": (
                    "After permutation, remove floor(noise_percent * target_edges) "
                    "uniformly sampled edges, then insert the same number of edges "
                    "uniformly sampled from nonedges of the remaining target graph."
                ),
                "ground_truth_unchanged_by_noise": True,
                "matcher_inputs": ["source.graph", "target.graph"],
                "evaluation_only": ["ground_truth.csv"],
            })
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _vertex_count(path):
    with path.open(encoding="utf-8") as graph_file:
        return int(graph_file.readline().split()[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="Prepared dataset directory containing G.graph")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True, help="Run label, for example G03")
    parser.add_argument("--noise-levels", type=int, nargs="+", required=True,
                        help="Integer percentages, for example 0 10")
    args = parser.parse_args()
    generate_variants(args.dataset_dir, args.output_dir, args.dataset, args.noise_levels)
    print(f"Generated noise variants under {args.output_dir}")


if __name__ == "__main__":
    main()
