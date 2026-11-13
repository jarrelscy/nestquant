from concurrent.futures import ThreadPoolExecutor


class BoundedReader:
    def __init__(self, read, budget_bytes, max_pending=2, max_workers=1):
        if budget_bytes < 1 or max_pending < 1 or max_workers < 1:
            raise ValueError('Read-ahead limits must be positive')
        self.read = read
        self.budget_bytes = budget_bytes
        self.max_pending = max_pending
        self.pending = {}
        self.reserved_bytes = 0
        self.peak_reserved_bytes = 0
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='nestquant-ssd')

    def submit(self, key, payload_bytes):
        if self.executor is None:
            raise RuntimeError('Read-ahead worker is closed')
        # Reserve the shard copies, merged arrays, and installation copy together.
        reservation = 3 * payload_bytes + 1024**2
        if reservation > self.budget_bytes:
            raise ValueError('One expert exceeds the read-ahead staging budget')
        if key in self.pending or len(self.pending) >= self.max_pending:
            return False
        if self.reserved_bytes + reservation > self.budget_bytes:
            return False
        future = self.executor.submit(self.read, *key)
        self.pending[key] = future, reservation
        self.reserved_bytes += reservation
        self.peak_reserved_bytes = max(self.peak_reserved_bytes, self.reserved_bytes)
        return True

    def poll(self, accept):
        for key, (future, reservation) in list(self.pending.items()):
            if not future.done():
                continue
            try:
                if not future.cancelled():
                    value = future.result()
                    try:
                        accept(key, value)
                    finally:
                        del value
            finally:
                del self.pending[key]
                self.reserved_bytes -= reservation

    def consume(self, key, accept, on_error=None, on_cancel=None):
        entry = self.pending.get(key)
        if entry is None:
            return False
        future, reservation = entry
        try:
            if future.cancel():
                if on_cancel is not None:
                    on_cancel(key)
                return False
            try:
                value = future.result()
            except Exception as failure:
                if on_error is None:
                    raise
                on_error(key, failure)
                return False
            accept(value)
            return True
        finally:
            del self.pending[key]
            self.reserved_bytes -= reservation

    def retire(self, predicate, accept, on_error=None, on_cancel=None):
        for key, (future, reservation) in list(self.pending.items()):
            if not predicate(key) or (not future.done() and not future.cancel()):
                continue
            try:
                if future.cancelled():
                    if on_cancel is not None:
                        on_cancel(key)
                else:
                    try:
                        value = future.result()
                    except Exception as failure:
                        if on_error is None:
                            raise
                        on_error(key, failure)
                    else:
                        accept(key, value)
            finally:
                del self.pending[key]
                self.reserved_bytes -= reservation

    def discard(self, accept=None):
        error = None
        for future, _ in self.pending.values():
            future.cancel()
            if not future.cancelled():
                try:
                    result = future.result()
                    if accept is not None:
                        accept(result)
                    del result
                except BaseException as failure:
                    error = error or failure
        self.pending.clear()
        self.reserved_bytes = 0
        if error is not None:
            raise error

    def close(self):
        if self.executor is not None:
            executor, self.executor = self.executor, None
            try:
                self.discard()
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
