"""Single-rank floating-pool state machine for Flash's published jT policy.

The executor owns CUDA completion and slot reuse. A requested pool is not treated
as landed: callbacks alone advance state. Superseded in-flight loads finish before
being demoted, avoiding concurrent writes to one expert's mailbox.
"""
import numpy as np


class Pool:
    def __init__(self, layers, ne, wanted):
        self.layers=list(layers); self.li={l:i for i,l in enumerate(layers)}
        self.state=np.zeros((len(layers),ne),np.int8)
        self.wanted=np.asarray(wanted,bool).copy()
        if self.wanted.shape != self.state.shape:
            raise ValueError('Wrong desired-pool shape')
        self.read_errors=0

    def operations(self):
        down=np.argwhere((self.state==2)&~self.wanted)
        up=np.argwhere((self.state==0)&self.wanted)
        self.state[tuple(down.T)]=3
        self.state[tuple(up.T)]=1
        keys=lambda a:[(self.layers[int(i)],int(e)) for i,e in a]
        return keys(up),keys(down)

    def landed(self,L,E):
        i=self.li[L]
        if self.state[i,E]!=1:raise RuntimeError(f'Unexpected landing {L}/{E}')
        self.state[i,E]=2

    def released(self,L,E):
        i=self.li[L]
        if self.state[i,E]!=3:raise RuntimeError(f'Unexpected release {L}/{E}')
        self.state[i,E]=0

    def failed(self,L,E,read_error=False):
        self.state[self.li[L],E]=0
        if read_error:
            self.read_errors+=1
            raise IOError(f'Failed Flash residual read L{L}/E{E}')

    def cancelled(self,L,E):self.failed(L,E)
