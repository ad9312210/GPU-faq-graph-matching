"""Independent checks for evaluation metrics and label-free refinement."""
import csv
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from quality_metrics import mapping_metrics, support_metrics
from refine_assignment_by_edge_agreement import refine, preserved_edges
from analyze_feature_candidates import audit


def exhaustive_matching(rows):
    def visit(s, used):
        if s == len(rows):
            return 0
        best = visit(s + 1, used)
        for t in rows[s]:
            if t not in used:
                best = max(best, 1 + visit(s + 1, used | {t}))
        return best
    return visit(0, set())


class QualityTests(unittest.TestCase):
    def test_matching_against_exhaustive(self):
        rng = random.Random(20261019)
        for _ in range(100):
            n, m = rng.randrange(6), rng.randrange(6)
            rows = [{t: 0.0 for t in range(m) if rng.randrange(2)} for _ in range(n)]
            result = support_metrics(rows, m)
            self.assertEqual(result["maximum_matching_size"], exhaustive_matching(rows))
            self.assertEqual(result["support_feasible"], exhaustive_matching(rows) == n)

    def test_metrics_are_separate(self):
        edges = {(0, 1), (1, 2)}
        # A path reversal preserves every edge but disagrees with two labels.
        result = mapping_metrics(edges, edges, {0: 2, 1: 1, 2: 0},
            {0: 0, 1: 1, 2: 2}, 3, 3, [{t: 1.0 for t in range(3)} for _ in range(3)])
        self.assertTrue(result["restricted_assignment_feasible"])
        self.assertEqual(result["descriptor_assignment_cost"], 0.0)
        self.assertEqual(result["edge_preservation"], 1.0)
        self.assertAlmostEqual(result["mapping_accuracy"], 1 / 3)
        incomplete = mapping_metrics(edges, edges, {0: -1, 1: 1, 2: 2},
            {0: 0, 1: 1, 2: 2}, 3, 3)
        self.assertFalse(incomplete["one_to_one"])
        self.assertIsNone(incomplete["mapping_accuracy"])

    def test_refinement_without_truth(self):
        adjacency = [{1, 4}, {0, 2, 3}, {1, 4}, {1}, {0, 2}]
        support = [set(range(5)) for _ in range(5)]
        initial = [1, 0, 2, 3, 4]
        result, info = refine(adjacency, adjacency, support, initial, 2000, 42, 60)
        again, _ = refine(adjacency, adjacency, support, initial, 2000, 42, 60)
        self.assertEqual(result, again)
        self.assertEqual(sorted(result), list(range(5)))
        self.assertEqual(initial, [1, 0, 2, 3, 4])
        self.assertGreaterEqual(preserved_edges(adjacency, adjacency, result),
                                preserved_edges(adjacency, adjacency, initial))
        self.assertEqual(info["attempted_swaps"], 2000)
        self.assertEqual(info["stop_reason"], "attempt_limit")
        with self.assertRaises(ValueError):
            refine(adjacency, adjacency, [set() for _ in range(5)], initial, 10, 42, 60)

    def test_empty_and_single_vertex(self):
        self.assertEqual(refine([], [], [], [], 10, 1, 60)[0], [])
        self.assertEqual(refine([set()], [set()], [{0}], [0], 10, 1, 60)[0], [0])

    def test_empty_candidate_row_counts_as_omission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = root / "graph.txt"
            graph.write_text("3 0\n")
            truth = root / "truth.csv"
            truth.write_text("source_vertex,target_vertex\n0,0\n1,1\n2,2\n")
            candidates = root / "candidates.csv"
            candidates.write_text("source,rank,target,similarity\n0,0,0,0\n1,0,0,0\n")
            result = audit(graph, graph, candidates, truth, root / "audit.json", 1, 1e-6,
                           print_result=False)
            self.assertEqual(result["omitted_true_target_count"], 2)
            self.assertAlmostEqual(result["candidate_recall"], 1 / 3)
            self.assertEqual(result["support"]["maximum_matching_size"], 1)
            self.assertEqual(result["support"]["empty_source_rows"], 1)
            retained = audit(graph, graph, candidates, truth, root / "retained.json", 1, 1e-6,
                             print_result=False, support_kind="retained-support")
            self.assertIsNone(retained["sources_with_ties_at_saved_k_boundary"])


if __name__ == "__main__":
    unittest.main()
