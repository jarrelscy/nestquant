from concurrent.futures import ThreadPoolExecutor
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from expert_major_base import read_into
from probe_required_reads import read_batches, read_row


class RequiredReadTest(unittest.TestCase):
    def fixture(self, handle, count=8):
        payloads = [bytes((expert + position) % 251 for position in range(256)) for expert in range(count)]
        handle.write(b''.join(payloads))
        handle.flush()
        rows = [dict(fd=handle.fileno(), offset=expert * 256, bytes=256,
                     first=payload[:64], last=payload[-64:]) for expert, payload in enumerate(payloads)]
        return rows, payloads

    def test_row_preserves_all_bytes_and_rejects_truncation_or_wrong_address(self):
        with tempfile.TemporaryFile() as handle:
            rows, payloads = self.fixture(handle)
            buffer = bytearray(512)
            count, seconds = read_row(rows[3], buffer)
            self.assertEqual(count, 256)
            self.assertGreaterEqual(seconds, 0)
            self.assertEqual(buffer[:256], payloads[3])
            self.assertEqual(buffer[256:], bytes(256))
            with self.assertRaisesRegex(RuntimeError, 'wrong expert row'):
                read_row(dict(rows[3], offset=4 * 256), buffer)
            with self.assertRaises(EOFError):
                read_row(dict(rows[3], offset=8 * 256), buffer)

    def test_concurrent_batches_keep_buffer_ownership_and_count_identical_bytes(self):
        with tempfile.TemporaryFile() as handle:
            rows, _ = self.fixture(handle)
            for width in (1, 2, 4):
                buffers = [bytearray(256) for _ in range(4)]
                active, peaks = set(), []
                lock = threading.Lock()
                barrier = threading.Barrier(width, timeout=2)

                def concurrent_read(fd, views, offset):
                    identity = id(views[0].obj)
                    with lock:
                        self.assertNotIn(identity, active)
                        active.add(identity)
                        peaks.append(len(active))
                    try:
                        barrier.wait()
                        return read_into(fd, views, offset)
                    finally:
                        with lock:
                            active.remove(identity)

                check = Mock()
                with self.subTest(width=width), patch('probe_required_reads.read_into', side_effect=concurrent_read), \
                        ThreadPoolExecutor(max_workers=width) as executor:
                    result = read_batches(executor, rows, width, buffers, check)
                self.assertEqual(result['bytes'], 8 * 256)
                self.assertEqual(len(result['read_seconds']), 8)
                self.assertEqual(check.call_count, 2 * (8 // width))
                self.assertEqual(max(peaks), width)
                self.assertEqual(active, set())

    def test_executor_context_joins_other_reads_after_a_failure(self):
        started, release, finished = threading.Event(), threading.Event(), threading.Event()
        failures = []

        def reader(row, buffer):
            if row == 0:
                raise EOFError('read failed')
            started.set()
            if not release.wait(2):
                raise TimeoutError('Test did not release the reader')
            return 256, 0.01

        def run():
            try:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    read_batches(executor, [0, 1], 2, [bytearray(256), bytearray(256)], lambda: None)
            except Exception as failure:
                failures.append(failure)
            finally:
                finished.set()

        with patch('probe_required_reads.read_row', side_effect=reader):
            worker = threading.Thread(target=run)
            worker.start()
            try:
                self.assertTrue(started.wait(2))
                self.assertFalse(finished.is_set())
            finally:
                release.set()
                worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(failures), 1)
            self.assertIsInstance(failures[0], EOFError)


if __name__ == '__main__':
    unittest.main()
