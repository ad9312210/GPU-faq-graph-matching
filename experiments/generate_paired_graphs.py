'''
generate_paired_graphs.py

Creates a source graph.
Applies a known vertex permutation to create the target graph.
Optionally adds/removes edges as controlled noise.
Writes: 
    - source.graph
    - target.graph
    - ground_truth.txt
    - manifest.json
'''
"""Generate a reproducible graph pair for native SCGM experiments."""

import argparse
import csv
import json
import random
from pathlib import Path


def make_edges(num_vertices, edge_probability, rng):
    edges = set()
    for source in range(num_vertices):
        for target in range(source + 1, num_vertices):
            if rng.random() < edge_probability:
                edges.add((source, target))
    if not edges:
        raise ValueError("edge_probability produced a graph with no edges")
    return edges


def write_graph(path, num_vertices, edges):
    with path.open("w", encoding="utf-8") as graph_file:
        graph_file.write(f"{num_vertices} {len(edges)}\n")
        for source, target in sorted(edges):
            graph_file.write(f"{source} {target}\n")


def read_native_graph(path):
    with path.open(encoding="utf-8") as graph_file:
        header = graph_file.readline().split()
        if len(header) != 2:
            raise ValueError(f"{path}: first line must contain vertex and edge counts")
        num_vertices, declared_edges = map(int, header)
        edges = set()
        for line_number, raw_line in enumerate(graph_file, 2):
            fields = raw_line.split()
            if not fields:
                continue
            if len(fields) < 2:
                raise ValueError(f"{path}:{line_number}: invalid edge")
            source, target = map(int, fields[:2])
            if source == target:
                raise ValueError(f"{path}:{line_number}: self-loop is not allowed")
            edges.add(tuple(sorted((source, target))))
    if len(edges) != declared_edges:
        raise ValueError(f"{path}: header edge count does not match file")
    return num_vertices, edges


def generate_pair(output_dir, num_vertices, edge_probability, seed, edge_noise,
                  source_graph=None):
    if not 0.0 < edge_probability < 1.0:
        raise ValueError("edge_probability must be between 0 and 1")
    if not 0.0 <= edge_noise <= 1.0:
        raise ValueError("edge_noise must be between 0 and 1")

    rng = random.Random(seed)
    if source_graph is None:
        source_edges = make_edges(num_vertices, edge_probability, rng)
    else:
        input_vertices, source_edges = read_native_graph(source_graph)
        if input_vertices != num_vertices:
            raise ValueError("--num-vertices does not match --source-graph")
    permutation = list(range(num_vertices))
    rng.shuffle(permutation)
    target_edges = {
        tuple(sorted((permutation[source], permutation[target])))
        for source, target in source_edges
    }

    noise_count = int(len(target_edges) * edge_noise)
    if noise_count:
        removed_edges = rng.sample(sorted(target_edges), noise_count)
        target_edges.difference_update(removed_edges)
        # Rejection-sample uniformly from unordered vertex pairs. This avoids
        # materializing O(n^2) possible pairs for sparse real-world graphs.
        inserted = 0
        while inserted < noise_count:
            source = rng.randrange(num_vertices)
            target = rng.randrange(num_vertices)
            if source == target:
                continue
            edge = tuple(sorted((source, target)))
            if edge in target_edges:
                continue
            target_edges.add(edge)
            inserted += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    write_graph(output_dir / "source.graph", num_vertices, source_edges)
    write_graph(output_dir / "target.graph", num_vertices, target_edges)
    with (output_dir / "ground_truth.csv").open("w", newline="", encoding="utf-8") as truth_file:
        writer = csv.writer(truth_file)
        writer.writerow(("source_vertex", "target_vertex"))
        writer.writerows(enumerate(permutation))

    manifest = {
        "seed": seed,
        "num_vertices": num_vertices,
        "source_edges": len(source_edges),
        "target_edges": len(target_edges),
        "edge_probability": edge_probability,
        "edge_noise": edge_noise,
        "mapping_direction": "source_vertex_to_target_vertex",
        "files": ["source.graph", "target.graph", "ground_truth.csv"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-vertices", type=int, default=100)
    parser.add_argument("--edge-probability", type=float, default=0.08)
    parser.add_argument("--edge-noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--source-graph", type=Path)
    args = parser.parse_args()
    if args.num_vertices < 2:
        parser.error("--num-vertices must be at least 2")
    generate_pair(
        args.output_dir,
        args.num_vertices,
        args.edge_probability,
        args.seed,
        args.edge_noise,
        args.source_graph,
    )
    print(f"Generated paired graph files in {args.output_dir}")


if __name__ == "__main__":
    main()
