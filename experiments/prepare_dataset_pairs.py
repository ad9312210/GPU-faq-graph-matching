"""Clean a SNAP edge list and create exact, reproducible permutation pairs.

The output contains a single deterministic internal-ID mapping and one graph
pair for each of the fixed seeds 11, 22, and 33. Ground truth is for evaluation
only; matchers should receive only source.graph and target.graph.
"""

import argparse
import csv
import gzip
import json
from pathlib import Path

from generate_paired_graphs import generate_pair, write_graph


SEEDS = (11, 22, 33)


def open_input(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def load_clean_graph(path):
    """Return sorted original IDs and unique undirected edges in internal IDs."""
    edges = set()
    vertices = set()
    with open_input(path) as graph_file:
        for line_number, raw_line in enumerate(graph_file, 1):
            line = raw_line.strip()
            if not line or line.startswith(("#", "%")):
                continue
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"{path}:{line_number}: expected two vertex IDs")
            u, v = int(fields[0]), int(fields[1])
            if u == v:
                continue
            edges.add(tuple(sorted((u, v))))
            vertices.update((u, v))

    if not edges:
        raise ValueError(f"{path}: no usable edges found")

    original_ids = sorted(vertices)
    to_internal = {original_id: internal_id for internal_id, original_id in enumerate(original_ids)}
    internal_edges = {
        tuple(sorted((to_internal[u], to_internal[v]))) for u, v in edges
    }
    return original_ids, internal_edges


def prepare(input_path, output_dir, label):
    original_ids, edges = load_clean_graph(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = output_dir / "original_id_mapping.csv"
    with mapping_path.open("w", newline="", encoding="utf-8") as mapping_file:
        writer = csv.writer(mapping_file)
        writer.writerow(("original_file_id", "internal_id"))
        writer.writerows((original_id, internal_id) for internal_id, original_id in enumerate(original_ids))

    source_path = output_dir / "G.graph"
    write_graph(source_path, len(original_ids), edges)

    for seed in SEEDS:
        pair_dir = output_dir / f"seed_{seed}"
        generate_pair(
            pair_dir,
            len(original_ids),
            edge_probability=0.5,  # ignored when --source-graph is supplied
            seed=seed,
            edge_noise=0.0,
            source_graph=source_path,
        )
        pair_manifest_path = pair_dir / "manifest.json"
        manifest = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
        manifest.update({
            "dataset": label,
            "input_file": str(input_path),
            "input_graph": "../G.graph",
            "original_id_mapping": "../original_id_mapping.csv",
            "seed": seed,
            "edge_noise": 0.0,
            "pair_definition": "target_edges = {(truth[u], truth[v]) for (u, v) in source_edges}",
            "matcher_inputs": ["source.graph", "target.graph"],
            "evaluation_only": ["ground_truth.csv", "../original_id_mapping.csv"],
        })
        pair_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    summary = {
        "dataset": label,
        "input_file": str(input_path),
        "num_vertices": len(original_ids),
        "num_undirected_edges": len(edges),
        "cleaning": "discard self-loops; deduplicate and canonicalize undirected edges",
        "internal_id_order": "ascending original file ID",
        "seeds": list(SEEDS),
        "original_id_mapping": mapping_path.name,
        "cleaned_graph": source_path.name,
        "pair_directories": [f"seed_{seed}" for seed in SEEDS],
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {label}: {len(original_ids)} vertices, {len(edges)} undirected edges")
    print(f"Pairs: {', '.join(f'seed_{seed}' for seed in SEEDS)}")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="SNAP edge list, optionally gzip-compressed")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True, help="Dataset label, for example G01")
    args = parser.parse_args()
    prepare(args.input, args.output_dir, args.label)


if __name__ == "__main__":
    main()
