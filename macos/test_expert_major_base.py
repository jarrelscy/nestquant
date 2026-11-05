import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from expert_major_base import FORMAT, convert_layer, plane_layouts, read_into
from mapped_weights import DTYPES, read_layouts
from native_assets import REVISION, fingerprint
from packed_expert_bank import PackedExpertBank


def source_fixture(directory):
    fields = {'gate.base': (np.arange(288 * 5) % 251).astype(np.uint8).reshape(288, 5),
              'gate.suh2': np.arange(288 * 2, dtype=np.float16).reshape(288, 2)}
    header = {'__metadata__': dict(format='nestquant-resident-base-v1', revision=REVISION,
                                   layer='3', geometry=json.dumps({'gate': {}}))}
    payload = bytearray()
    for name, array in fields.items():
        raw = array.tobytes()
        header[name] = dict(dtype='U8' if array.dtype == np.uint8 else 'F16',
                            shape=list(array.shape), data_offsets=[len(payload), len(payload) + len(raw)])
        payload.extend(raw)
    encoded = json.dumps(header).encode()
    file = directory / 'L3.safetensors'
    file.write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)
    (directory / 'L3.json').write_text(json.dumps(dict(revision=REVISION, state=fingerprint(file),
                                                     sha256=hashlib.sha256(file.read_bytes()).hexdigest())))
    return fields


class ExpertMajorTest(unittest.TestCase):
    def test_short_reads_cross_buffer_boundaries(self):
        with tempfile.TemporaryFile() as handle:
            handle.write(b'--abcdefgh')
            handle.flush()
            buffers = [bytearray(2), bytearray(5), bytearray(1)]

            def short_read(fd, views, offset):
                data = os.pread(fd, min(3, sum(view.nbytes for view in views)), offset)
                position = 0
                for view in views:
                    count = min(view.nbytes, len(data) - position)
                    view[:count] = data[position:position + count]
                    position += count
                    if position == len(data):
                        break
                return len(data)

            with patch('expert_major_base.os.preadv', side_effect=short_read) as read:
                self.assertEqual(read_into(handle.fileno(), buffers, 2), 8)
            self.assertEqual(b''.join(buffers), b'abcdefgh')
            self.assertEqual(read.call_count, 3)
            with self.assertRaises(EOFError):
                read_into(handle.fileno(), [bytearray(20)], 0)

    def test_invalid_buffers_are_rejected(self):
        for buffers in ([], [bytearray()], [b'readonly']):
            with self.subTest(buffers=buffers), self.assertRaises(ValueError):
                read_into(-1, buffers, 0)

    def test_malformed_layouts_fail_before_bank_allocation(self):
        layout = {'experts': (0, 288 * 9, DTYPES['U8'], (288, 9))}
        original = [dict(name='gate.base', dtype='U8', shape=[5], offset=0),
                    dict(name='gate.suh2', dtype='F16', shape=[2], offset=5)]
        parsed, stride = plane_layouts(layout, dict(format=FORMAT, plane_layouts=json.dumps(original)))
        self.assertEqual(stride, 9)
        self.assertEqual(set(parsed), {'gate.base', 'gate.suh2'})
        cases = [original[:1], original + [original[0]], [None],
                 [original[0], dict(original[1], offset=4)],
                 [dict(original[0], offset=False), original[1]],
                 [original[0], dict(original[1], dtype='I64')],
                 [original[0], dict(original[1], shape=[3])]]
        for descriptions in cases:
            with self.subTest(descriptions=descriptions), self.assertRaises(ValueError):
                plane_layouts(layout, dict(format=FORMAT, plane_layouts=json.dumps(descriptions)))

    def test_conversion_preserves_every_row_and_required_fill_uses_one_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / 'source', root / 'output'
            source.mkdir()
            output.mkdir()
            fields = source_fixture(source)
            checks = []
            result = convert_layer(source, output, 3, lambda: checks.append(True))
            file = output / 'L3.safetensors'
            self.assertEqual(result['row_bytes'], 9)
            self.assertEqual(len(checks), 290)
            proof = json.loads((output / 'L3.json').read_text())
            self.assertEqual(proof['sha256'], hashlib.sha256(file.read_bytes()).hexdigest())
            with file.open('rb') as handle:
                layouts, metadata, offset = read_layouts(handle, file.stat().st_size)
                self.assertEqual(metadata['format'], FORMAT)
                planes, stride = plane_layouts(layouts, metadata)
                for expert in range(288):
                    actual = os.pread(handle.fileno(), stride, offset + expert * stride)
                    self.assertEqual(actual, b''.join(array[expert].tobytes() for array in fields.values()))
                bank = PackedExpertBank.__new__(PackedExpertBank)
                bank.expert_row_bytes, bank.data_offset, bank.fd = stride, offset, handle
                bank.host = {name: (np.zeros((2, end - start), dtype=np.uint8), offset + start, end - start)
                             for name, (start, end, _, _) in planes.items()}
                bank.stats = dict(read_bytes=0, read_seconds=0.0)
                original = os.preadv
                with patch('expert_major_base.os.preadv', wraps=original) as read:
                    bank._fill(0, 137)
                self.assertEqual(read.call_count, 1)
                self.assertEqual(bank.stats['read_bytes'], stride)
                for name, (target, _, _) in bank.host.items():
                    self.assertEqual(target[0].tobytes(), fields[name][137].tobytes())
                payload, count, elapsed = bank.read_payload(42)
                later, _, _ = bank.read_payload(43)
                self.assertEqual(count, stride)
                self.assertGreaterEqual(elapsed, 0)
                self.assertIsNot(payload['gate.base'].obj, later['gate.base'].obj)
                bank._install_payload(1, payload)
                for name, (target, _, _) in bank.host.items():
                    self.assertEqual(target[1].tobytes(), fields[name][42].tobytes())

    def test_disk_reserve_and_existing_targets_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_fixture(root)
            output = root / 'output'
            output.mkdir()
            with patch('expert_major_base.shutil.disk_usage', return_value=SimpleNamespace(free=100 * 1024**3)):
                with self.assertRaisesRegex(RuntimeError, 'disk reserve'):
                    convert_layer(root, output, 3, lambda: None)
            self.assertEqual(list(output.iterdir()), [])
            target = output / 'L3.safetensors'
            target.write_bytes(b'existing')
            with self.assertRaises(FileExistsError):
                convert_layer(root, output, 3, lambda: None)
            self.assertEqual(target.read_bytes(), b'existing')

    def test_source_change_cannot_publish_an_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_fixture(root)
            output = root / 'output'
            output.mkdir()
            file = root / 'L3.safetensors'
            calls = 0

            def check():
                nonlocal calls
                calls += 1
                if calls == 5:
                    state = file.stat()
                    os.utime(file, ns=(state.st_atime_ns, state.st_mtime_ns + 1000000))

            with self.assertRaisesRegex(RuntimeError, 'changed during preparation'):
                convert_layer(root, output, 3, check)
            self.assertFalse((output / 'L3.safetensors').exists())
            self.assertFalse((output / 'L3.json').exists())


if __name__ == '__main__':
    unittest.main()
