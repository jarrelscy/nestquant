# nq-prefill: imported at startup by every Python process of the NQ serve container (PYTHONPATH=/nq/sm120/serve).
# Unless NQ_PREFILL_BORROW=0 (default 1) installs the EngineCore-side prefill-borrow patches (nq_pb_engine.py)
# as post-import hooks. Any failure leaves the process untouched.
import os
if os.environ.get('NQ_PREFILL_BORROW','1')=='1':
    try:
        import nq_pb_engine as _nq_pb_engine;_nq_pb_engine.install()
    except Exception as _e:
        import sys;print(f'nq_pb sitecustomize: {_e!r}',file=sys.stderr)
elif os.environ.get('NQ_LMPF','0')=='1':      # NQ_LMPF without prefill-borrow: own scheduler hook (else chained by nq_pb_engine)
    try:
        import nq_lmpf_engine as _nq_lmpf_engine;_nq_lmpf_engine.install()
    except Exception as _e:
        import sys;print(f'nq_lmpf sitecustomize: {_e!r}',file=sys.stderr)
if os.environ.get('NQ_DBG_FUSE_M') or os.environ.get('NQ_DBG_LIN_ROWSPLIT') or os.environ.get('NQ_DBG_MLA_BMM_ROWSPLIT') or os.environ.get('NQ_DBG_NO_BF16_RED') or os.environ.get('NQ_DBG_LMHEAD_FP32') or os.environ.get('NQ_DBG_FP8_OPROJ_ONLY') or os.environ.get('NQ_DBG_FP8_TARGETS'):     # nq-kld debug numerics switches (nq_dbg_numerics.py)
    try:
        import nq_dbg_numerics as _nq_dbg;_nq_dbg.install()
    except Exception as _e:
        import sys;print(f'nq_dbg sitecustomize: {_e!r}',file=sys.stderr)
if os.environ.get('NQ_DEFER_START','0')=='1':
    # spark (64 GB/GPU): allocate the slot pool after the whole load (main model + MTP drafter), so the drafter's load-time
    # staging and the slot pool are never resident together. The pool is added to the weights figure vLLM budgets KV with.
    def _nq_defer_patch(m):
        W=m.Worker;f0=W.determine_available_memory
        def determine_available_memory(self,*a,**k):
            import sys,torch
            rt=getattr(sys.modules.get('nq_vllm'),'RT',None)
            if rt is not None and rt.lay and not rt.started:
                torch.cuda.empty_cache();a0=torch.cuda.memory_allocated();rt.start()
                self.model_runner.model_memory_usage+=torch.cuda.memory_allocated()-a0
            return f0(self,*a,**k)
        W.determine_available_memory=determine_available_memory
    try:
        import sys as _sys
        class _NQDeferFinder:
            def find_spec(self,name,path=None,target=None):
                if name!='vllm.v1.worker.gpu_worker':return None
                for f in _sys.meta_path:
                    if f is self or not hasattr(f,'find_spec'):continue
                    spec=f.find_spec(name,path,target)
                    if spec is not None:break
                else:return None
                ex0=spec.loader.exec_module
                def exec_module(m,_ex0=ex0):_ex0(m);_nq_defer_patch(m)
                spec.loader.exec_module=exec_module
                return spec
        if 'vllm.v1.worker.gpu_worker' in _sys.modules:_nq_defer_patch(_sys.modules['vllm.v1.worker.gpu_worker'])
        else:_sys.meta_path.insert(0,_NQDeferFinder())
    except Exception as _e:
        import sys;print(f'nq_defer sitecustomize: {_e!r}',file=sys.stderr)
