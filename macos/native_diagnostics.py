import json
from pathlib import Path
import time


class NativeDiagnostics:
    def __init__(self, path: Path, synchronize_layers=False):
        self.stream = path.open('x')
        self.synchronize_layers = synchronize_layers
        self.phase = 'startup'
        self.decode_step = None
        self.record('diagnostics_started', synchronize_layers=synchronize_layers,
                    limitation='Host wall intervals can contain GPU waits. Diagnostic timing is not ordinary throughput.')

    def record(self, event, **values):
        row = dict(event=event, phase=self.phase, decode_step=self.decode_step,
                   monotonic_seconds=time.perf_counter(), **values)
        self.stream.write(json.dumps(row, allow_nan=False) + '\n')
        self.stream.flush()

    def set_phase(self, phase):
        if phase not in ('startup', 'prefill', 'decode', 'complete'):
            raise ValueError('Unexpected native diagnostic phase')
        self.phase = phase
        self.record('phase_changed')

    def begin_step(self, step):
        if self.phase != 'decode' or type(step) is not int or step < 0:
            raise ValueError('Expected a nonnegative decode step')
        if self.synchronize_layers and step >= 128:
            raise ValueError('Synchronized diagnostics allow at most 128 decode steps')
        self.decode_step = step

    def close(self):
        self.stream.close()
