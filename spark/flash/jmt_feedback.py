"""Bounded, request-local online horizon/window tuning using delivered routing.

Rewards are observational, not counterfactual estimates. Keep the slot budget,
mixture and pacing fixed. Discard settling rows after a switch. Never learn from
uncommitted drafts, prefill, previous requests or benchmark task scores.
"""
import math

class FeedbackTuner:
    ARMS=((0,3),(-1,3),(1,3),(0,6),(0,1))
    def __init__(self, window=128, settle=32):
        if window<1 or settle<0:raise ValueError('Invalid feedback window')
        self.window=window;self.settle=settle;self.reset()
    def reset(self):
        self.arm=0;self.remaining=self.settle
        self.n=0;self.mass=0.;self.hot=0.;self.late=0.;self.seconds=0.
        self.results=[dict(visits=0,reward=0.,salience=None,tps=None) for _ in self.ARMS]
        self.reference_tps=None;self.rounds=0;self.switches=0;self.last=None
    def choice(self,base):
        offset,width=self.ARMS[self.arm]
        return max(1,min(16,int(base)+offset)),width
    def observe(self,committed,total,hot,late,seconds):
        if not isinstance(committed,int) or committed<0:raise ValueError('Invalid committed count')
        vals=(total,hot,late,seconds)
        if any(not math.isfinite(v) or v<0 for v in vals):raise ValueError('Invalid feedback')
        if hot>total+1e-6 or late>total+1e-6:raise ValueError('Invalid coverage')
        if not committed:return False
        if self.remaining>0:
            self.remaining=max(0,self.remaining-committed)
            return False
        if not total or not seconds:return False
        self.n+=committed;self.mass+=total;self.hot+=hot;self.late+=late;self.seconds+=seconds
        if self.n<self.window:return False
        coverage=self.hot/self.mass;late_fraction=self.late/self.mass;tps=self.n/self.seconds
        if self.arm==0:
            self.reference_tps=tps if self.reference_tps is None else .7*self.reference_tps+.3*tps
        acceptable=self.reference_tps is None or tps>=.9*self.reference_tps
        reward=coverage-.05*late_fraction if acceptable else -1.
        r=self.results[self.arm];r['visits']+=1
        r['reward']=reward if r['visits']==1 else .7*r['reward']+.3*reward
        r.update(salience=coverage,tps=tps)
        self.last=dict(arm=self.arm,first_offset=self.ARMS[self.arm][0],width=self.ARMS[self.arm][1],
                       salience=coverage,late_salience=late_fraction,tps=tps,speed_floor_pass=acceptable,reward=reward,rows=self.n)
        self.rounds+=1
        untried=[i for i,x in enumerate(self.results) if x['visits']==0]
        if untried:next_arm=untried[0]
        # Recheck baseline periodically, since task difficulty and token speed drift.
        elif self.rounds%5==0:next_arm=0
        else:next_arm=max(range(len(self.ARMS)),key=lambda i:self.results[i]['reward']+.015*math.sqrt(math.log(self.rounds+1)/self.results[i]['visits']))
        if next_arm!=self.arm:self.switches+=1
        self.arm=next_arm;self.remaining=self.settle
        self.n=0;self.mass=0.;self.hot=0.;self.late=0.;self.seconds=0.
        return True
    def stats(self):
        return dict(arm=self.arm,offset=self.ARMS[self.arm][0],width=self.ARMS[self.arm][1],
                    measurement_rows=self.n,settling_rows_left=self.remaining,rounds=self.rounds,switches=self.switches,
                    reference_tps=self.reference_tps,last_window=self.last,arms=self.results)
