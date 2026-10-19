"""Evaluate a native SCGM assignment against generated ground truth."""

import argparse
import csv
import json
from pathlib import Path
try:
	from .quality_metrics import load_support, mapping_metrics, support_metrics, fingerprint
except ImportError:
	from quality_metrics import load_support, mapping_metrics, support_metrics, fingerprint


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
	if num_vertices < 0 or declared_edges < 0 or any(
		u < 0 or v >= num_vertices for u, v in edges):
		raise ValueError(f"{path}: invalid dimensions or edge endpoints")
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
	# Common metrics validate truth and support rectangular injections.
	base = mapping_metrics(source_edges, target_edges, assignment, ground_truth,
		num_vertices, target_vertices)
	found = None if candidates is None else sum(
		ground_truth[s] in candidates.get(s, set()) for s in range(num_vertices))
	base["candidate_recall_at_k"] = None if found is None else (
		found / num_vertices if num_vertices else 1.0)
	base["candidate_recall_note"] = "Recall over supplied support; repaired rows may have unequal lengths."
	return base

def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--source-graph", type=Path, required=True)
	parser.add_argument("--target-graph", type=Path, required=True)
	parser.add_argument("--assignment", type=Path, required=True)
	parser.add_argument("--ground-truth", type=Path, required=True)
	parser.add_argument("--candidates", type=Path)
	parser.add_argument("--output", type=Path)
	parser.add_argument("--timings", type=Path)
	parser.add_argument("--variant", default="baseline")
	args = parser.parse_args()

	source_graph = read_graph(args.source_graph)
	target_graph = read_graph(args.target_graph)
	assignment = read_mapping(args.assignment, "source", "target")
	ground_truth = read_mapping(args.ground_truth, "source_vertex", "target_vertex")
	candidates = read_candidates(args.candidates) if args.candidates else None
	metrics = evaluate(source_graph, target_graph, assignment, ground_truth, candidates)
	if args.candidates:
		support = load_support(args.candidates, source_graph[0], target_graph[0])
		metrics.update(mapping_metrics(source_graph[1], target_graph[1], assignment,
			ground_truth, source_graph[0], target_graph[0], support))
		metrics["support"] = support_metrics(support, target_graph[0])
	else:
		metrics["support"] = None
	metrics["variant"] = args.variant
	metrics["native_timings"] = json.loads(args.timings.read_text()) if args.timings else None
	metrics["inputs"] = {key: str(getattr(args, key)) if getattr(args, key) else None
		for key in ("source_graph", "target_graph", "assignment", "ground_truth", "candidates", "timings")}
	metrics["input_fingerprints"] = {key: fingerprint(getattr(args, key))
		for key in ("source_graph", "target_graph", "assignment", "ground_truth", "candidates")
		if getattr(args, key)}
	metrics["interpretation"] = "Descriptor optimality does not establish correct correspondence."

	formatted = json.dumps(metrics, indent=2, allow_nan=False)
	print(formatted)
	if args.output:
		args.output.parent.mkdir(parents=True, exist_ok=True)
		args.output.write_text(formatted + "\n", encoding="utf-8")


if __name__ == "__main__":
	main()
