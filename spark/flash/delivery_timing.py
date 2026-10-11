"""Exclusive host-phase accounting; no CUDA sync or change of scheduling."""
import time,collections,functools
class PhaseMeter:
 def __init__(self,clock=time.monotonic):self.clock=clock;self.last=clock();self.stack=[];self.total=collections.defaultdict(float);self.calls=collections.Counter()
 def tick(self):
  now=self.clock();self.total[self.stack[-1] if self.stack else 'outside_hooks']+=now-self.last;self.last=now;return now
 def snapshot(self):self.tick();return dict(self.total)
 def delta(self,before):
  after=self.snapshot();return {k:v-before.get(k,0.) for k,v in after.items()}
 def wrap(self,obj,method,label):
  fn=getattr(obj,method)
  @functools.wraps(fn)
  def timed(*args,**kwargs):
   self.tick();self.stack.append(label);self.calls[label]+=1
   try:return fn(*args,**kwargs)
   finally:self.tick();self.stack.pop()
  setattr(obj,method,timed)
 def stats(self):return dict(exclusive_seconds=self.snapshot(),calls=dict(self.calls))
