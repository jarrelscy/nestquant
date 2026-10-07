"""CPU regression tests of actual serving function ASTs, without CUDA/vLLM imports."""
import ast
import collections
import functools
import os
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
S=types.SimpleNamespace

def functions(path,*names):
    tree=ast.parse((ROOT/path).read_text())
    return [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]

def execute(nodes,ns):
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'serving-under-test','exec'),ns)
    return ns

class UptimeTests(unittest.TestCase):
    def test_firstn_tracks_prefill_and_cleanup(self):
        path='sm120/serve/nq_pfblock.py'
        install=functions(path,'install_dec')[0]
        wrapper=next(n for n in install.body if isinstance(n,ast.FunctionDef) and n.name=='execute_model')
        for prompt in (4,1024,300000,900000):
            waits=[]
            ns=dict(os=os,FNKNOB='/nonexistent/nq-test-firstn',_FN=dict(m=-1,v=64,after=.04,pl={}),
                    _G=dict(fresh=False),PI=S(new=False),MX=8,orig=lambda *a,**k:None)
            execute(functions(path,'_firstn_skip')+[wrapper],ns)
            def so(n,new=(),nc=(),finished=()):
                return S(total_num_scheduled_tokens=n,scheduled_new_reqs=list(new),finished_req_ids=list(finished),
                         scheduled_cached_reqs=S(req_ids=['r'] if nc else [],num_computed_tokens=list(nc)))
            with patch.dict(sys.modules,nq_vllm=S(RT=S(PFB=S(dec_wait=waits.append,late=None)))):
                f=ns['execute_model']
                f(None,so(prompt,[S(req_id='r',prompt_token_ids=range(prompt))]))
                f(None,so(4,nc=[prompt]));self.assertIsNone(waits[-1])
                f(None,so(4,nc=[prompt+63]));self.assertIsNone(waits[-1])
                f(None,so(4,nc=[prompt+64]));self.assertEqual(waits[-1],.04)
                f(None,so(0,finished=['r']));self.assertNotIn('r',ns['_FN']['pl'])
                f(None,so(prompt,[S(req_id='dummy',prompt_token_ids=range(prompt))]),dummy_run=True)
                self.assertNotIn('dummy',ns['_FN']['pl'])

    def test_hit_counter_wrap(self):
        ns=execute(functions('sm120/serve/nq_vllm.py','_hit_delta'),{})
        prev=np.array([2147483646,-2,10],dtype=np.int64)
        cur=np.array([-2147483646,3,18],dtype=np.int64)
        np.testing.assert_array_equal(ns['_hit_delta'](cur,prev),[4,5,8])

    def test_health_check_catches_background_error_even_on_graph_replay(self):
        path='sm120/serve/nq_vllm.py'
        rt=S(err=None,SR=None)
        calls=[]
        class Worker:
            def execute_model(self,*a,**k):
                calls.append(1)
                if k.get('fail_during'):rt.err=ValueError('background failure')
                return 'graph-result'
        mods={'vllm':types.ModuleType('vllm'),'vllm.v1':types.ModuleType('vllm.v1'),
              'vllm.v1.worker':types.ModuleType('vllm.v1.worker'),
              'vllm.v1.worker.gpu_worker':S(Worker=Worker)}
        ns=execute(functions(path,'_check_health','_hook_sched'),dict(functools=functools))
        with patch.dict(sys.modules,mods):ns['_hook_sched'](rt)
        w=Worker();self.assertEqual(w.execute_model(None),'graph-result')
        with self.assertRaisesRegex(RuntimeError,'background'):w.execute_model(None,fail_during=True)
        n=len(calls)
        with self.assertRaisesRegex(RuntimeError,'background'):w.execute_model(None)
        self.assertEqual(len(calls),n)

    def test_graph_capture_failure_releases_gate(self):
        path='sm120/serve/nq_vllm.py'
        class CV:
            allowed=False
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def wait_for(self,*a,**k):return self.allowed
            def notify_all(self):pass
        class Graph:
            fail=False
            def capture_begin(self):
                if self.fail:raise ValueError('capture failed')
            def capture_end(self):pass
        rt=S(cv=CV(),ncap=0,in_iter=False,X=S(ops={}))
        ns=execute(functions(path,'_gate_captures'),dict(torch=S(cuda=S(CUDAGraph=Graph))))
        ns['_gate_captures'](rt);g=Graph()
        with self.assertRaises(TimeoutError):g.capture_begin()
        self.assertEqual(rt.ncap,0)
        rt.cv.allowed=True;g.fail=True
        with self.assertRaises(ValueError):g.capture_begin()
        self.assertEqual(rt.ncap,0)
        g.fail=False;g.capture_begin();self.assertEqual(rt.ncap,1)
        g.capture_end();self.assertEqual(rt.ncap,0)

    def test_lookahead_pool_detaches_before_reuse(self):
        tree=ast.parse((ROOT/'sm120/serve/nq_lookahead.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LA')
        service=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='service')
        pre=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='pre')
        ns=execute([service,pre],dict(MODE='chunk',stats=lambda *a:None,MEAS_N=('a','b','c','d'),
            torch=S(linalg=S(vector_norm=lambda *a,**k:S(square=lambda:None)),float32=None),NE=256))
        # Exhausted pool must drop optional work before touching pending buffers or issuing CUDA work.
        s=S(lead=True,layers=[3,4],delta={},meas=False,free=collections.deque())
        ns['pre'](s,4,None,S(),S(float=lambda:None),None)
        # Consumer operates on a detached host snapshot, even if a freed buffer is immediately reused.
        backing=np.ones((10,256));backing[4]=4
        buf=S(numpy=lambda:backing)
        class ReusePool(collections.deque):
            def append(self,b):backing.fill(0);super().append(b)
        s=S(q=collections.deque([(3,1,None,buf,S(query=lambda:True),'chunk',0,1,False)]),
            free=ReusePool(),cs=collections.defaultdict(lambda:collections.defaultdict(float)),plan={},layers=[3,4])
        self.assertEqual(ns['service'](s,S(li={3:0}),None,None),1)
        self.assertAlmostEqual(s.cs[1]['served_a'],1.0)
        self.assertEqual(len(s.free),1)

if __name__=='__main__':unittest.main()
