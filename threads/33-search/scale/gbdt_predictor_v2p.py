"""GBDTPredictorV2 + prefill warm-start (T33j scale).  The shipped predictor ignores prefill chunks (ntok > 16), so
every request's decode starts cold (floating_default, empty EMAs): heldout decode first 256 / 1024 tokens lose ~3.5 /
~1.1 pt all-slot sal-hot vs a warm state.  prefill() seeds the state from the prompt's per-expert sums only:

  p.prefill(counts_sum[len(layers), NE], sal_sum[len(layers), NE], ntok, reset=True)
      counts_sum = routed hits per expert summed over the prefill's computed tokens, sal_sum = sum of w^2 * xn over the
      same slots (same definition as step()'s sal), ntok = number of computed prefill tokens.
      reset=True: clear the predictor state first (new session / cold request), then feed 16 synthetic blocks of the
      prompt's mean per-block counts / salience (= 256 tokens of stationary history).
      reset=False: session carry-over; append min(16, ceil(ntok/16)) synthetic mean blocks to the existing state.
      A refresh follows (sync: new S applied now; next_refresh: queued as a normal refresh).
      Returns True if a new score matrix was applied.
Serve cost: two [75, 256] float sums accumulated over the prefill (the routing and w^2*xn are already computed), then
16 block updates (+ one GBDT predict) once per request, ~2 ms.  Offline definition: reqstart.py arm 'warmmean';
parity: parity_prefill.py."""
import math
import sys
import os

import numpy as np

sys.path.insert(0, "/home/coder/git/nestquant/streaming")
from gbdt_predictor import G  # noqa: E402
from gbdt_predictor_v2 import GBDTPredictorV2  # noqa: E402


class GBDTPredictorV2P(GBDTPredictorV2):
    def _reset_state(self):
        NL, NE = self.NL, self.NE
        for k in range(len(self.E)):
            self.E[k] = np.zeros((NL, NE), np.float32)
        self.Et = np.zeros((NL, NE), np.float32); self.Ea = np.zeros_like(self.Et); self.wt = self.wa = 0.0
        self.last = np.full((NL, NE), -10 ** 6, np.int64)
        self.nblk = 0
        self.bc = np.zeros((NL, NE), np.float32); self.bca = np.zeros((NL, NE), np.float32); self.btok = 0; self.bans = 0
        self.seg = 0
        z = lambda: np.zeros((NL, NE), np.float64)  # noqa: E731
        self.Es = [z() for _ in self.sag]; self.Ec = [z() for _ in self.sag]
        self.bs = z(); self.s16 = z()

    def prefill(self, counts_sum, sal_sum, ntok, reset=True, nblk=None):
        if ntok <= 0:
            return False
        if reset:
            self._reset_state()
        nb = nblk if nblk is not None else (16 if reset else int(min(16, math.ceil(ntok / G))))
        c = np.asarray(counts_sum, np.float64) / ntok * G
        s = np.asarray(sal_sum, np.float64) / ntok * G
        # a partially filled decode block is dropped (a new request starts on a block boundary)
        for _ in range(nb):
            self.bc = c.astype(np.float32)
            if self.seg == 1:
                self.bca = self.bc.copy(); self.bans = G
            else:
                self.bca = np.zeros_like(self.bc); self.bans = 0
            self.bs = s.copy(); self.btok = G
            self._close_block()
        return self._refresh()
