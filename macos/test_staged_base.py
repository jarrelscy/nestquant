from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import mlx.core as mx

from expert_major_base import convert_layer
from native_assets import REVISION
from packed_expert_bank import BaseReadStaging, PackedExpertBank
from test_expert_major_base import source_fixture


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fields = source_fixture(root)
        archive = root / 'archive'
        archive.mkdir()
        convert_layer(root, archive, 3, lambda: None)
        staging = BaseReadStaging()
        banks = []
        try:
            for _ in range(2):
                banks.append(PackedExpertBank(archive / 'L3.safetensors', 3, 32, REVISION, staging=staging))
            yield root, fields, staging, banks
        finally:
            for bank in banks:
                bank.close()


class StagedBaseTest(unittest.TestCase):
    def test_shared_staging_preserves_gpu_rows_across_reuse_and_eviction(self):
        with fixture() as (_, fields, staging, banks):
            self.assertEqual(len(staging.storage), 16 * 1024**2)
            for first in (0, 80, 160, 240, 16, 0):
                selected = tuple(range(first, first + 8))
                outputs = []
                for bank, experts in zip(banks, (selected, tuple(reversed(selected))), strict=True):
                    route = mx.array(bank.select(experts), dtype=mx.uint32)
                    actual = bank.planes['gate'][0]['suh2'][route]
                    bank.retain(actual)
                    outputs.append((actual, mx.array(fields['gate.suh2'][list(experts)])))
                staging.storage[:9] = b'\xff' * 9
                for actual, expected in outputs:
                    self.assertTrue(mx.array_equal(actual, expected).item())
                for bank in banks:
                    for expert in selected:
                        stored = bank.slots[expert]
                        for name, (target, _, _) in bank.host.items():
                            self.assertEqual(target[stored].tobytes(), fields[name][expert].tobytes())
            for bank in banks:
                self.assertGreater(bank.stats['evictions'], 0)
                self.assertGreater(bank.stats['read_copy_seconds'], 0)
                self.assertEqual(bank.stats['read_bytes'], bank.stats['misses'] * 9)

    def test_read_and_install_failures_release_staging_without_publishing_a_slot(self):
        with fixture() as (root, _, staging, banks):
            bank = banks[0]
            original = os.preadv
            calls = []

            def truncate(fd, views, offset):
                calls.append(offset)
                if len(calls) > 1:
                    return 0
                return original(fd, [views[0][:3]], offset)

            with patch('expert_major_base.os.preadv', side_effect=truncate):
                with self.assertRaises(EOFError):
                    bank.select((17,))
            self.assertFalse(staging.lock.locked())
            self.assertEqual(bank.slots, {})
            self.assertEqual(len(bank.free_slots), 32)
            self.assertEqual(bank.stats['read_bytes'], 0)
            with patch.object(bank, '_install_payload', side_effect=RuntimeError('copy failed')):
                with self.assertRaisesRegex(RuntimeError, 'copy failed'):
                    bank.select((17,))
            self.assertFalse(staging.lock.locked())
            self.assertEqual(bank.slots, {})
            self.assertEqual(len(bank.free_slots), 32)
            self.assertEqual(bank.select((17,)), (0,))
            bank.retain(mx.array(0))
            with self.assertRaisesRegex(ValueError, 'expert-major'):
                PackedExpertBank(root / 'L3.safetensors', 3, 32, REVISION, staging=staging)

    def test_second_bank_cannot_reuse_staging_until_the_first_copy_finishes(self):
        with fixture() as (_, fields, staging, banks):
            first, second = banks
            entered, release, attempted, finished = (threading.Event() for _ in range(4))
            install = first._install_payload

            def pause_copy(slot, payload):
                entered.set()
                if not release.wait(2):
                    raise TimeoutError('The test did not release the first copy')
                install(slot, payload)

            def second_fill():
                attempted.set()
                second._fill(0, 81)
                finished.set()

            with patch.object(first, '_install_payload', side_effect=pause_copy), ThreadPoolExecutor(2) as executor:
                one = executor.submit(first._fill, 0, 17)
                try:
                    self.assertTrue(entered.wait(2))
                    two = executor.submit(second_fill)
                    self.assertTrue(attempted.wait(2))
                    self.assertFalse(finished.wait(.05))
                finally:
                    release.set()
                one.result()
                two.result()
            self.assertFalse(staging.lock.locked())
            for bank, expert in ((first, 17), (second, 81)):
                for name, (target, _, _) in bank.host.items():
                    self.assertEqual(target[0].tobytes(), fields[name][expert].tobytes())


if __name__ == '__main__':
    unittest.main()
