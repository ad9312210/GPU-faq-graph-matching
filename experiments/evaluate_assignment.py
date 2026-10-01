"""Evaluate a native SCGM assignment against generated ground truth."""

import argparse
import csv
import json
from pathlib import Path


def read_graph(path):
	edges = set()
	with path.open(encoding="utf-8") as graph_file:
		header = None
		for line_number, raw_line in enumerate(graph_file, 1):
			line = raw_line.strip()
			if not line or line.startswith(("#", "%")):
				continue
			fields = line.split()
			if header is None:
				if len(fields) != 2:
					raise ValueError(f"{path}:{line_number}: invalid graph header")
				header = (int(fields[0]), int(fields[1]))
				continue
			if len(fields) < 2:
				raise ValueError(f"{path}:{line_number}: invalid edge")
			source, target = int(fields[0]), int(fields[1])
			if source == target:
				raise ValueError(f"{path}:{line_number}: self-loop is not allowed")
			edges.add(tuple(sorted((source, target))))
	if header is None:
		raise ValueError(f"{path}: missing graph header")
	num_vertices, declared_edges = header
	if len(edges) != declared_edges:
		raise ValueError(
			f"{path}: header declares {declared_edges} edges, found {len(edges)}"
		)
	return num_vertices, edges


def read_mapping(path, source_column, target_column):
	mapping = {}
	with path.open(newline="", encoding="utf-8") as mapping_file:
		for row in csv.DictReader(mapping_file):
			source = int(row[source_column])
			target = int(row[target_column])
			if source in mapping:
				raise ValueError(f"{path}: duplicate source vertex {source}")
			mapping[source] = target
	return mapping


def read_candidates(path):
	candidates = {}
	with path.open(newline="", encoding="utf-8") as candidate_file:
		for row in csv.DictReader(candidate_file):
			source = int(row["source"])
			target = int(row["target"])
			candidates.setdefault(source, set()).add(target)
	return candidates


def evaluate(source_graph, target_graph, assignment, ground_truth, candidates=None):
	num_vertices, source_edges = source_graph
	target_vertices, target_edges = target_graph
	if num_vertices != target_vertices:
		raise ValueError("source and target graphs have different vertex counts")

	predicted_targets = list(assignment.values())
	in_range = all(0 <= target < num_vertices for target in predicted_targets)
	one_to_one = (
		len(assignment) == num_vertices
		and set(assignment) == set(range(num_vertices))
		and in_range
		and len(set(predicted_targets)) == num_vertices
	)
	correct = sum(
		1 for source in range(num_vertices)
		if assignment.get(source) == ground_truth.get(source)
	)
	mapped_edges = {
		tuple(sorted((assignment[source], assignment[target])))
		for source, target in source_edges
		if source in assignment and target in assignment and in_range
	}
	preserved_edges = len(mapped_edges & target_edges)
	valid_assignment = one_to_one and len(assignment) == num_vertices
	if candidates is None:
		candidate_recall = None
		candidate_recall_note = "Unavailable: pass --candidates from scgm_cuda."
	else:
		found = sum(
			1 for source in range(num_vertices)
			if ground_truth.get(source) in candidates.get(source, set())
		)
		candidate_recall = found / num_vertices
		candidate_recall_note = "Recall of ground-truth targets in exported Top-K lists."
	return {
		"num_vertices": num_vertices,
		"source_edges": len(source_edges),
		"target_edges": len(target_edges),
		"assigned_vertices": len(assignment),
		"mapping_accuracy": correct / num_vertices if valid_assignment else None,
		"correct_vertices": correct if valid_assignment else None,
		"partial_correct_vertices": correct,
		"one_to_one": one_to_one,
		"targets_in_range": in_range,
		"edge_preservation": (preserved_edges / len(source_edges)
			if valid_assignment and source_edges else (1.0 if valid_assignment else None)),
		"preserved_edges": preserved_edges if valid_assignment else None,
		"candidate_recall_at_k": candidate_recall,
		"candidate_recall_note": candidate_recall_note,
	}


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--source-graph", type=Path, required=True)
	parser.add_argument("--target-graph", type=Path, required=True)
	parser.add_argument("--assignment", type=Path, required=True)
	parser.add_argument("--ground-truth", type=Path, required=True)
	parser.add_argument("--candidates", type=Path)
	parser.add_argument("--output", type=Path)
	args = parser.parse_args()

	source_graph = read_graph(args.source_graph)
	target_graph = read_graph(args.target_graph)
	assignment = read_mapping(args.assignment, "source", "target")
	ground_truth = read_mapping(args.ground_truth, "source_vertex", "target_vertex")
	candidates = read_candidates(args.candidates) if args.candidates else None
	metrics = evaluate(source_graph, target_graph, assignment, ground_truth, candidates)

	formatted = json.dumps(metrics, indent=2)
	print(formatted)
	if args.output:
		args.output.write_text(formatted + "\n", encoding="utf-8")


if __name__ == "__main__":
	main()