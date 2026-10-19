"""CPU-only checks for strict validation, selector provenance, and claim gates."""
import argparse
import contextlib
import io
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from topk_cli import (add_topk_arguments,topk_configuration,native_topk_arguments,
                      verify_topk_metadata,GPUUnavailable)
from validate_topk_dense_reference import analyze_dense_row
from benchmark_gpu_topk import micro_comparisons,pipeline_gate


def metadata(policy='parallel',k=64,backend='exhaustive'):
    used='serial_heap' if policy!='parallel' or k>256 else 'parallel'
    bounded=backend=='structural-index' or policy!='serial'
    return {'effective_mode':'gpu','candidate_backend':backend,'k':k,'gpu_topk':{
        'schema_version':1,'requested_policy':policy,'requested':'parallel' if policy=='parallel' else 'serial_heap',
        'used':used,'implementation':'legacy_tiled_heap' if not bounded else
            'streaming_parallel_merge' if used=='parallel' else 'streaming_serial_heap',
        'fallback':'unsupported_k' if policy=='parallel' and k>256 else 'none',
        'chunk_size':256,'parallel_k_min':1,'parallel_k_max':256,'budget_enforced':bounded,
        'scratch_budget_bytes':64*1024*1024,'requested_batch_candidates':65536,
        'scratch_peak_bytes':1052676 if bounded else None,'batch_capacity':65536 if bounded else 256,
        'batches':2 if bounded else 0,'shared_bytes_per_block':2056 if used=='parallel' else 0,
        'selection_ms':1.0,'candidate_stage_wall_ms':2.0}}


class SelectorDiagnostics(unittest.TestCase):
    def test_modes_and_recorded_fallback(self):
        for policy in ('serial','streaming-serial','parallel'):
            for backend in ('exhaustive','structural-index'):
                for k in (64,512):
                    config={'policy':policy,'scratch_mib':64,'batch_candidates':65536}
                    verify_topk_metadata(metadata(policy,k,backend),config,'gpu',k)

    def test_cpu_fallback_is_skip(self):
        with self.assertRaises(GPUUnavailable): verify_topk_metadata({'effective_mode':'cpu'},None,'gpu',64)

    def test_missing_or_false_provenance_is_rejected(self):
        for field,value in [('implementation','streaming_parallel_merge_typo'),('used','serial_heap'),
            ('chunk_size',128),('scratch_peak_bytes',64*1024*1024+1),('selection_ms',float('nan')),
            ('schema_version',0),('requested_policy','serial'),('scratch_budget_bytes',True)]:
            native=metadata(); native['gpu_topk'][field]=value
            with self.subTest(field=field),self.assertRaises(RuntimeError): verify_topk_metadata(native,
                {'policy':'parallel','scratch_mib':64,'batch_candidates':65536},'gpu',64)

    def test_streaming_serial_arguments_and_cpu_rejection(self):
        parser=argparse.ArgumentParser(); parser.add_argument('--mode',default='gpu'); add_topk_arguments(parser)
        args=parser.parse_args(['--gpu-selector','streaming-serial','--gpu-topk-batch-candidates','256'])
        config=topk_configuration(args,parser)
        self.assertIn('streaming-serial',native_topk_arguments(config))
        self.assertIn('256',native_topk_arguments(config))
        args.mode='cpu'
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit): topk_configuration(args,parser)


class DenseReferenceValidation(unittest.TestCase):
    def check(self,saved,scores,k=1): return analyze_dense_row(saved,scores,k,1e-5,1e-5)
    def test_exact_tie_order(self):
        self.assertTrue(self.check([(0,1),(1,1)],[1,1],2)['pass'])
        self.assertFalse(self.check([(1,1),(0,1)],[1,1],2)['pass'])
    def test_count_duplicate_range_and_nonfinite_fail(self):
        for saved in ([],[(0,1),(0,1)],[(2,1)],[(0,float('nan'))]):
            self.assertFalse(self.check(saved,[1,0])['pass'])
    def test_score_error_is_not_hidden_by_valid_ids(self):
        result=self.check([(0,.5)],[1,0])
        self.assertFalse(result['pass']); self.assertIn('score_error_exceeds_tolerance',result['errors'])
    def test_only_near_cutoff_scoring_substitution_is_accepted(self):
        self.assertTrue(self.check([(1,1.0)],[1.0,1.0-1e-6])['pass'])
        self.assertFalse(self.check([(1,.5)],[1.0,.5])['pass'])
    def test_omitted_clear_winner_is_rejected(self):
        result=self.check([(1,.5),(2,.5)],[1,.5,.5],k=2)
        self.assertIn('omitted_strictly_better_reference_candidate',result['errors'])


class ClaimGates(unittest.TestCase):
    def case(self,requested,used=None,fallback='none'):
        return {'kind':'case','pool_size':257,'k':64,'distribution':'ties','batch_requested':256,
                'scratch_budget_bytes':8196,'input_fingerprint_fnv64':'abc','requested':requested,'oracle_exact':True,
                'trials':[{'used':used or requested,'fallback':fallback,'batches':2,'scratch_peak_bytes':8196,
                           'implementation':'streaming_parallel_merge' if (used or requested)=='parallel' else 'streaming_serial_heap',
                           'wall_ms':2 if requested=='serial_heap' else 1,
                           'selection_ms':1 if requested=='serial_heap' else .5,'transfer_ms':.1}]}
    def test_real_parallel_has_ratio_and_fallback_does_not(self):
        serial=self.case('serial_heap')
        real=micro_comparisons([serial,self.case('parallel')])[0]
        self.assertEqual(real['parallel_selection_speedup'],2)
        fallback=micro_comparisons([serial,self.case('parallel','serial_heap','device_allocation_failure')])[0]
        self.assertIsNone(fallback['parallel_selection_speedup'])
        self.assertIsNone(fallback['parallel_wall_speedup'])
    def test_different_inputs_or_budget_overrun_rejected(self):
        serial=self.case('serial_heap'); other=self.case('parallel'); other['input_fingerprint_fnv64']='different'
        with self.assertRaises(ValueError): micro_comparisons([serial,other])
        other=self.case('parallel'); other['trials'][0]['scratch_peak_bytes']=9000
        with self.assertRaises(ValueError): micro_comparisons([serial,other])
    def test_infeasible_pipeline_is_not_a_complete_speedup(self):
        def variant(used): return {'quality':{'one_to_one':True,'candidate_recall_at_k':1,'final_accuracy':1,'edge_preservation':1},
            'trials':[{'gpu_topk':{'used':used,'fallback':'none'},'assignment_feasible':True,'assignment_one_to_one':True}]}
        serial,parallel=variant('serial_heap'),variant('parallel')
        comparison={'ordered_target_ids_equal':True,'scores_within_tolerance':True}
        self.assertTrue(pipeline_gate(serial,parallel,comparison))
        parallel['trials'][0]['assignment_feasible']=False
        self.assertFalse(pipeline_gate(serial,parallel,comparison))
        parallel=variant('parallel'); parallel['quality']['final_accuracy']=.9
        self.assertFalse(pipeline_gate(serial,parallel,comparison))


if __name__=='__main__': unittest.main()
