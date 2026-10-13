"""Experimental eager-runtime publication at completed-layer boundaries.

No budget, ranking, pointer-release or stream changes. The original layer and
routing snapshot enqueue before mailbox kernels on the same current stream.
Only layers already executed in this pass qualify. Executor remains solely
responsible for freeing slots after observing device acknowledgement.
"""
import functools

def flush_completed(runtime):
    visited=runtime._publication_visited
    sequences=runtime._publication_sequences
    pending={}
    for (layer,expert),(kind,seq) in runtime.executor.wait_apply.items():
        if layer in visited and sequences.get((layer,expert))!=seq:
            pending.setdefault(layer,[]).append((expert,seq))
    for layer in sorted(pending):
        runtime.layers[layer][1].apply()
        for expert,seq in pending[layer]:sequences[layer,expert]=seq
    return len(pending)

def patch_runtime(module):
    cls=module.Runtime
    if getattr(cls,'_post_layer_publication_patched',False):return
    begin,forward,finish=cls.begin,cls.forward,cls.finish
    @functools.wraps(begin)
    def wrapped_begin(self,*args,**kwargs):
        self._publication_visited=set()
        self._publication_sequences={}
        return begin(self,*args,**kwargs)
    @functools.wraps(forward)
    def wrapped_forward(self,layer,*args,**kwargs):
        result=forward(self,layer,*args,**kwargs)
        if not self.dummy:
            if not hasattr(self,'_publication_visited'):
                raise RuntimeError('Publication boundary missing begin')
            self._publication_visited.add(layer)
            # Completion observation and original scheduling; never release a
            # slot here, never synchronize the GPU or modify wanted masks.
            self.poll()
            flush_completed(self)
        return result
    @functools.wraps(finish)
    def wrapped_finish(self,*args,**kwargs):
        result=finish(self,*args,**kwargs)
        if not self.dummy:
            # Original finish owns and synchronizes routing snapshots. Observe
            # loads completed during that wait before predictor inference.
            self.poll()
            flush_completed(self)
        return result
    cls.begin,cls.forward,cls.finish=wrapped_begin,wrapped_forward,wrapped_finish
    cls._post_layer_publication_patched=True
