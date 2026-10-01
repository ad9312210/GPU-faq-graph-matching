"""Convert a SNAP-style edge list into the native undirected graph format."""

import argparse
import gzip
from pathlib import Path


def open_input(path):
	if path.suffix == ".gz":
		return gzip.open(path, "rt", encoding="utf-8")
	return path.open(encoding="utf-8")


def convert_dataset(input_path, output_path):
	edges = set()
	vertices = set()
	with open_input(input_path) as input_file:
		for line_number, raw_line in enumerate(input_file, 1):
			line = raw_line.strip()
			if not line or line.startswith(("#", "%")):
				continue
			fields = line.split()
			if len(fields) < 2:
				raise ValueError(f"{input_path}:{line_number}: expected two vertex IDs")
			source, target = int(fields[0]), int(fields[1])
			if source == target:
				continue
			edge = tuple(sorted((source, target)))
			edges.add(edge)
			vertices.update(edge)

	if not vertices:
		raise ValueError(f"{input_path}: no usable edges found")

	# SNAP IDs are not guaranteed to be contiguous; native Graph requires IDs
	# in [0, num_vertices), so remap them deterministically.
	remap = {vertex: index for index, vertex in enumerate(sorted(vertices))}
	native_edges = sorted((remap[source], remap[target]) for source, target in edges)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as output_file:
		output_file.write(f"{len(remap)} {len(native_edges)}\n")
		for source, target in native_edges:
			output_file.write(f"{source} {target}\n")

	print(f"Converted {input_path} -> {output_path}")
	print(f"Vertices: {len(remap)}")
	print(f"Undirected edges: {len(native_edges)}")
	print("Self-loops discarded: yes")
	print("Direction discarded: yes")


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input", type=Path, required=True)
	parser.add_argument("--output", type=Path, required=True)
	args = parser.parse_args()
	convert_dataset(args.input, args.output)


if __name__ == "__main__":
	main()