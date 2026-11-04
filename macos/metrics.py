import math
import time

import mlx.core as mx
import torch

from resources import process_usage, system_pressure



def snapshot():
    usage = process_usage()
    return dict(
        disk_read_bytes=usage.diskio_bytesread,
        disk_written_bytes=usage.diskio_byteswritten,
        physical_footprint_bytes=usage.phys_footprint,
        lifetime_peak_physical_footprint_bytes=usage.lifetime_max_phys_footprint,
        mlx_active_bytes=mx.get_active_memory(), mlx_peak_bytes=mx.get_peak_memory(),
        mlx_cached_bytes=mx.get_cache_memory(),
        mps_driver_allocated_bytes=torch.mps.driver_allocated_memory() if torch.backends.mps.is_available() else 0,
        mps_allocated_bytes=torch.mps.current_allocated_memory() if torch.backends.mps.is_available() else 0,
    )


class MemoryGuard:
    def __init__(self, limit_gib=84):
        if not math.isfinite(limit_gib) or limit_gib <= 0:
            raise ValueError('Process memory limit must be positive and finite')
        self.limit_bytes = min(int(limit_gib * 1024**3), int(mx.device_info()['memory_size'] * 0.7))
        self.last_check = 0.0
        self.check(force=True)

    def check(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_check < 1:
            return
        self.last_check = now
        footprint = process_usage().phys_footprint
        if footprint > self.limit_bytes:
            raise MemoryError(f'Process footprint {footprint / 1024**3:.2f} GiB exceeds '
                              f'{self.limit_bytes / 1024**3:.2f} GiB guard')
        if system_pressure() != 1:
            raise MemoryError('macOS reports memory pressure; stop inference and reduce memory use')
