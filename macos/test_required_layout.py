from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from expert_major_base import convert_layer
from native_assets import REVISION
from packed_expert_bank import PackedExpertBank, _shared_plane
from probe_required_layout import make_trace, measure_layer, VARIANTS
from test_expert_major_base import source_fixture


class RequiredLayoutTest(unittest.TestCase):
    def test_trace_is_deterministic_and_each_layer_has_disjoint_rows(self):
        trace = make_trace(range(3, 45))
        self.assertEqual(trace, make_trace(range(3, 45)))
        self.assertEqual(len(trace), 42)
        for experts in trace.values():
            self.assertEqual(len(experts), 32)
            self.assertEqual(len(set(experts)), 32)
            self.assertTrue(all(0 <= expert < 288 for expert in experts))

    def test_real_bank_variants_preserve_bytes_and_release_mapping_after_success_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fields = source_fixture(root)
            archive = root / 'archive'
            archive.mkdir()
            convert_layer(root, archive, 3, lambda: None)
            bank = PackedExpertBank(archive / 'L3.safetensors', 3, 32, REVISION)
            mappings = []

            def capture(*args):
                result = _shared_plane(*args)
                mappings.append(result[0])
                return result

            def reject_memory():
                raise MemoryError('guard')

            try:
                experts = make_trace([3])[3]
                with patch('probe_required_layout._shared_plane', side_effect=capture):
                    result = measure_layer(bank, experts, VARIANTS[::-1], lambda: None)
                    self.assertTrue(mappings[-1].closed)
                    self.assertEqual(result['validated_rows'], 8)
                    self.assertEqual(result['logical_buffer_bytes'], 50 * 9)
                    self.assertEqual([item['variant'] for item in result['measurements']], list(VARIANTS[::-1]))
                    for item in result['measurements']:
                        self.assertEqual(item['bytes'], 8 * 9)
                        self.assertEqual(len(item['read_seconds']), 8)
                        self.assertEqual(len(item['copy_seconds']), 8 if item['variant'] == 'staged' else 0)
                    for slot in range(16):
                        expert = experts[slot if slot < 8 else slot + 16]
                        for name, (target, _, _) in bank.host.items():
                            self.assertEqual(target[slot].tobytes(), fields[name][expert].tobytes())
                    with self.assertRaisesRegex(MemoryError, 'guard'):
                        measure_layer(bank, experts, VARIANTS, reject_memory)
                    self.assertTrue(mappings[-1].closed)
                    with patch('probe_required_layout.read_into', side_effect=EOFError('truncated')):
                        with self.assertRaisesRegex(EOFError, 'truncated'):
                            measure_layer(bank, experts, VARIANTS, lambda: None)
                    self.assertTrue(mappings[-1].closed)
                with self.assertRaises(ValueError):
                    measure_layer(bank, experts, (*VARIANTS, 'private'), lambda: None)
            finally:
                bank.close()

    def test_staged_corruption_fails_the_post_measurement_byte_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_fixture(root)
            archive = root / 'archive'
            archive.mkdir()
            convert_layer(root, archive, 3, lambda: None)
            bank = PackedExpertBank(archive / 'L3.safetensors', 3, 32, REVISION)
            install = bank._install_payload

            def corrupt(slot, payload):
                install(slot, payload)
                bank.host['gate.base'][0][slot, 0] ^= 1

            try:
                with patch.object(bank, '_install_payload', side_effect=corrupt):
                    with self.assertRaisesRegex(RuntimeError, 'differs from the source'):
                        measure_layer(bank, make_trace([3])[3], VARIANTS, lambda: None)
            finally:
                bank.close()


if __name__ == '__main__':
    unittest.main()
