"""Tests for candidate comparison semantics; no GPU or datasets required."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from benchmark_structural_index import compare_support, ratio


class ComparisonMetricsTests(unittest.TestCase):
    def write_candidates(self, path, rows):
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['source', 'rank', 'target', 'similarity'])
            writer.writerows(rows)

    def test_tie_substitution_is_distinct_from_strict_score_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            left, right = Path(temporary)/'left.csv', Path(temporary)/'right.csv'
            self.write_candidates(left, [(0,0,0,1), (1,0,1,.9)])
            self.write_candidates(right, [(0,0,1,1), (1,0,0,.5)])
            result = compare_support(left,right,{0:0,1:1},2,1,1e-6)
            self.assertEqual(result['mean_exhaustive_topk_overlap'],0)
            self.assertEqual(result['indexed_fraction_strictly_below_exhaustive_cutoff'],.5)
            self.assertEqual(result['candidate_recall_delta_indexed_minus_exhaustive'],-1)
            self.assertEqual(result['truth_targets_lost_from_exhaustive_topk'],2)

    def test_duplicate_candidates_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'bad.csv'
            self.write_candidates(path,[(0,0,0,1),(0,1,0,1)])
            with self.assertRaises(ValueError):
                compare_support(path,path,{0:0},1,2,1e-6)

    def test_unknown_or_zero_denominator_does_not_become_speedup(self):
        self.assertIsNone(ratio(None,1))
        self.assertIsNone(ratio(2,0))
        self.assertEqual(ratio(6,3),2)


if __name__ == '__main__':
    unittest.main()
