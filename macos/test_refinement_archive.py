"""CPU-only tests for the expert-major refinement archive (no MLX operations, no GPU).

Synthetic tests cover the format (layout, alignment, padding rule, index validation, round trip through every
read path). When the converted archive and checkpoint exist (NQ_REFINEMENT_ARCHIVE / NQ_TEST_MODEL or the
default paths), sampled experts are compared byte for byte with RefinementReader after PackedRefinementBank's
own write().
"""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import refinement_archive as ra

MODEL = Path(os.environ.get('NQ_TEST_MODEL', '~/Models/GLM/GLM-5.3-Flash-NestQuant-1.5-4bit')).expanduser()
BASE = Path(os.environ.get('NQ_TEST_BASE', '~/Models/GLM/resident-banks-v1')).expanduser()
ARCHIVE = Path(os.environ.get('NQ_REFINEMENT_ARCHIVE', '~/Models/GLM/refinements-expert-major-v1')).expanduser()
META = dict(k=64, n=32, tk=2, tn=1, shard_axis='n', base_K=1.5, res_rule=dict(kind='uniform', K=2.5))
PROJ_META = dict(gate=META, up=META, down=dict(META, k=32, n=64, res_rule=dict(kind='uniform', K=2.8125)))
RANKS = dict(gate=2, up=2, down=4)


def bank_write(host, slot, payload):
    """Run PackedRefinementBank.write (host-memory only) on NumPy planes."""
    from packed_expert_bank import PackedRefinementBank
    PackedRefinementBank.write(SimpleNamespace(host=host), slot, payload)


def host_planes(fields, rows):
    host = {}
    for item in fields:
        host.setdefault(item['projection'], {})[item['field']] = np.zeros(
            (rows, *item['shape']), dtype=ra.DTYPES[item['dtype']])
    return host


def synthetic_payload(rng, fields, ranks):
    payload = {p: {} for p in ra.PROJECTIONS}
    for item in fields:
        dtype = ra.DTYPES[item['dtype']]
        shape = list(item['shape'])
        if item['field'] == 'lrU4':
            shape[0] = ranks[item['projection']]
            if not shape[0]:
                continue
        if dtype.kind == 'f':
            value = rng.standard_normal(shape).astype(dtype)
        else:
            value = rng.integers(1, 2**31 - 1 if dtype.itemsize == 4 else 255, size=shape).astype(dtype)
        if item['field'] == 'word':
            value |= 1  # nonzero denominators
        payload[item['projection']][item['field']] = value
    return payload


def build_archive(directory, layers, seed=0):
    """Write a small archive with random payloads; returns {(layer, expert): payload}."""
    rng = np.random.default_rng(seed)
    fields, record_bytes, stride = ra.layer_fields(PROJ_META, RANKS)
    payloads, entries = {}, {}
    for layer in layers:
        path = Path(directory) / f'L{layer}.bin'
        image = np.zeros(stride, dtype=np.uint8)
        with path.open('wb') as handle:
            for expert in range(ra.EXPERTS):
                ranks = dict(RANKS, down=expert % 5)  # 0..4 rows, padded to the bank rank of 4
                payload = synthetic_payload(rng, fields, ranks)
                payloads[layer, expert] = payload
                image.fill(0)
                ra.slot_image(payload, fields, image)
                handle.write(image.tobytes())
        entries[str(layer)] = dict(file=path.name, sha256='0' * 64, stride=stride, record_bytes=record_bytes,
                                   ranks=RANKS, fields=fields, state=ra.fingerprint(path), source={},
                                   file_bytes=path.stat().st_size)
    index = dict(format=ra.FORMAT, layout_version=ra.LAYOUT_VERSION, revision=ra.REVISION, alignment=ra.ALIGNMENT,
                 experts=ra.EXPERTS, layers=entries)
    (Path(directory) / 'index.json').write_text(json.dumps(index))
    return payloads, fields


class FormatTests(unittest.TestCase):
    def test_layout_alignment(self):
        fields, record_bytes, stride = ra.layer_fields(PROJ_META, RANKS)
        self.assertEqual(stride % ra.ALIGNMENT, 0)
        self.assertGreaterEqual(stride, record_bytes)
        self.assertTrue(all(item['offset'] % ra.FIELD_ALIGNMENT == 0 for item in fields))
        self.assertEqual([(f['projection'], f['field']) for f in fields][:5],
                         [('gate', f) for f in ra.FIELDS])
        ra.validate_fields(fields, record_bytes, stride)
        without = ra.layer_fields(PROJ_META, dict(RANKS, gate=0))[0]
        self.assertNotIn(('gate', 'lrU4'), {(f['projection'], f['field']) for f in without})

    def test_validate_rejects_bad_tables(self):
        fields, record_bytes, stride = ra.layer_fields(PROJ_META, RANKS)
        overlapping = [dict(item) for item in fields]
        overlapping[1]['offset'] = 0
        unaligned = [dict(item) for item in fields]
        unaligned[-1]['offset'] += 2
        for bad, size, step in ((overlapping, record_bytes, stride), (unaligned, record_bytes, stride),
                                (fields, record_bytes, stride + 1), (fields[:-1], record_bytes, stride),
                                ([dict(fields[0], dtype='f8')] + fields[1:], record_bytes, stride)):
            with self.assertRaises(ValueError):
                ra.validate_fields(bad, size, step)

    def test_slot_image_matches_bank_write(self):
        rng = np.random.default_rng(1)
        fields, record_bytes, stride = ra.layer_fields(PROJ_META, RANKS)
        for down_rank in range(5):
            payload = synthetic_payload(rng, fields, dict(RANKS, down=down_rank))
            image = np.full(record_bytes, 0xAB, dtype=np.uint8)  # stale bytes must be overwritten
            ra.slot_image(payload, fields, image)
            host = host_planes(fields, 1)
            for planes in host.values():
                for plane in planes.values():
                    plane.view(np.uint8).fill(0xCD)
            bank_write(host, 0, payload)
            for item in fields:
                expected = host[item['projection']][item['field']][0]
                actual = image[item['offset']:item['offset'] + item['bytes']].view(expected.dtype).reshape(expected.shape)
                np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
        with self.assertRaises(ValueError):
            bad = synthetic_payload(rng, fields, RANKS)
            bad['gate']['word'] = bad['gate']['word'][:-1]
            ra.slot_image(bad, fields, np.zeros(record_bytes, np.uint8))


class RoundTripTests(unittest.TestCase):
    layers = (3, 4)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.payloads, self.fields = build_archive(self.directory.name, self.layers)

    def tearDown(self):
        self.directory.cleanup()

    def reader(self, **kwargs):
        return ra.ExpertMajorRefinementReader(self.directory.name, layers=self.layers, **kwargs)

    def test_read_matches_bank_install_bytes(self):
        for cache, readahead in ((False, True), (True, False)):
            reader = self.reader(cache=cache, readahead=readahead)
            try:
                for layer, expert in ((3, 0), (3, 4), (4, 287), (4, 3)):
                    payload, count, seconds = reader.read(layer, expert)
                    self.assertEqual(count, reader.layers[layer]['record_bytes'])
                    self.assertGreaterEqual(seconds, 0)
                    host = host_planes(self.fields, 3)
                    bank_write(host, 0, self.payloads[layer, expert])
                    bank_write(host, 1, payload)
                    self.assertEqual(reader.read_into_slot(layer, expert, SimpleNamespace(host=host), 2)[0], count)
                    for planes in host.values():
                        for plane in planes.values():
                            np.testing.assert_array_equal(plane[1], plane[0])
                            np.testing.assert_array_equal(plane[2], plane[0])
            finally:
                reader.close()

    def test_views_share_one_pooled_buffer(self):
        reader = self.reader()
        try:
            payload, _, _ = reader.read(3, 5)
            bases = {id(value.base) for planes in payload.values() for value in planes.values()}
            self.assertEqual(len(bases), 1)
            del payload
            self.assertEqual(len(reader.pool.free), 1)  # returned once the views are gone
            reader.read(3, 6)
            self.assertEqual(reader.pool.allocated, 1)
        finally:
            reader.close()

    def test_prefill_staging_row(self):
        reader = self.reader()
        rows, max_rank = 2, 4
        host = host_planes(self.fields, rows)
        for name in ra.PROJECTIONS:  # staging always has MAX_RANK low-rank rows
            host[name]['lrU4'] = np.full((rows, max_rank, PROJ_META[name]['n']), 7, dtype=np.float16)
        try:
            count, _ = reader.read_into(4, 9, SimpleNamespace(host=host), 1)
            self.assertEqual(count, reader.layers[4]['record_bytes'])
            expected = host_planes(self.fields, 1)
            bank_write(expected, 0, self.payloads[4, 9])
            for name, planes in expected.items():
                for field, plane in planes.items():
                    got = host[name][field][1]
                    if field == 'lrU4':
                        np.testing.assert_array_equal(got[:plane.shape[1]], plane[0])
                        self.assertFalse(got[plane.shape[1]:].any())
                    else:
                        np.testing.assert_array_equal(got, plane[0])
        finally:
            reader.close()

    def test_index_validation(self):
        path = Path(self.directory.name) / 'index.json'
        original = json.loads(path.read_text())
        cases = [dict(original, revision='0' * 40), dict(original, layout_version=2),
                 dict(original, alignment=4096)]
        broken = json.loads(json.dumps(original))
        broken['layers']['3']['stride'] += 16384
        cases.append(broken)
        for case in cases:
            path.write_text(json.dumps(case))
            with self.assertRaises((ValueError, RuntimeError)):
                self.reader()
        path.write_text(json.dumps(original))
        with self.assertRaises(ValueError):  # missing layer
            ra.ExpertMajorRefinementReader(self.directory.name, layers=(3, 4, 5))
        os.utime(Path(self.directory.name) / 'L3.bin', ns=(1, 1))  # changed after conversion
        with self.assertRaises(RuntimeError):
            self.reader()
        self.reader(check=False).close()

    def test_zero_denominator_rejected(self):
        directory = Path(self.directory.name)
        fields = self.fields
        word = next(f for f in fields if (f['projection'], f['field']) == ('up', 'word'))
        stride = json.loads((directory / 'index.json').read_text())['layers']['3']['stride']
        with (directory / 'L3.bin').open('r+b') as handle:
            handle.seek(2 * stride + word['offset'])
            handle.write(b'\0' * 4)
        reader = self.reader(check=False)
        try:
            with self.assertRaises(ValueError):
                reader.read(3, 2)
            reader.read(3, 1)
        finally:
            reader.close()

    def test_environment_options(self):
        os.environ['NQ_REFINEMENT_CACHE'] = '1'
        os.environ['NQ_REFINEMENT_READAHEAD'] = '0'
        try:
            reader = self.reader()
            self.assertTrue(reader.cache)
            self.assertFalse(reader.readahead)
            reader.close()
        finally:
            del os.environ['NQ_REFINEMENT_CACHE'], os.environ['NQ_REFINEMENT_READAHEAD']
        reader = self.reader()
        self.assertEqual((reader.cache, reader.readahead), (False, True))
        reader.close()
        saved = os.environ.pop('NQ_REFINEMENT_ARCHIVE', None)
        default = ra.DEFAULT_ARCHIVE
        try:
            # No variable and no archive at the default path: shard reader.
            ra.DEFAULT_ARCHIVE = Path(tempfile.gettempdir()) / 'no-such-refinement-archive'
            self.assertIsNone(ra.reader_from_env(None))
            # Explicitly disabled even when an archive exists.
            ra.DEFAULT_ARCHIVE = default
            os.environ['NQ_REFINEMENT_ARCHIVE'] = '0'
            self.assertIsNone(ra.reader_from_env(None))
        finally:
            ra.DEFAULT_ARCHIVE = default
            os.environ.pop('NQ_REFINEMENT_ARCHIVE', None)
            if saved is not None:
                os.environ['NQ_REFINEMENT_ARCHIVE'] = saved


class DirectInstallTests(unittest.TestCase):
    def test_deferred_read_fills_reserved_slot_and_flips_later(self):
        from unittest.mock import Mock, patch
        from native_refinements import NativeHotRefinements

        hot = NativeHotRefinements.__new__(NativeHotRefinements)
        hot.worker = Mock()
        hot.predictor = SimpleNamespace(reset=Mock(), budgets={layer: 2 for layer in range(3, 45)},
                                        defaults=[], fixed_ids=[])
        with patch('native_refinements.PredictorMetrics'):
            hot.reset()
        hot.background_installs, hot.install_threads, hot._copies, hot._copy_pool = True, 2, [], None
        bank = Mock(slots={}, staging={})
        bank.reserve.return_value = 1
        hot.banks = {3: bank}
        hot.wanted[0, 7] = True
        hot.reader = Mock()
        hot.reader.read_into_slot.return_value = (1000, 0.5)
        try:
            hot._stage((3, 7), NativeHotRefinements._deferred(3, 7))
            hot.flip(wait=True)
        finally:
            hot._copy_pool.shutdown()
        hot.reader.read_into_slot.assert_called_once_with(3, 7, bank, 1)
        bank.write.assert_not_called()
        bank.activate.assert_called_once_with(7, 1)
        self.assertEqual((hot.reads['decode_bytes'], hot.reads['installed']), (1000, 1))


@unittest.skipUnless((ARCHIVE / 'index.json').exists() and (MODEL / 'COMPLETE').exists(),
                     'converted archive or checkpoint not present')
class CheckpointEqualityTests(unittest.TestCase):
    """Installed bytes from the archive equal those from RefinementReader for sampled experts."""

    def test_sampled_experts_install_identically(self):
        from experts import RefinementReader
        from prefill_stream import MAX_RANK, ShardedRefinementReader

        old, new = RefinementReader(MODEL), ra.ExpertMajorRefinementReader(ARCHIVE, root=MODEL)
        sharded = ShardedRefinementReader(MODEL)
        rng = np.random.default_rng(int(os.environ.get('NQ_TEST_SEED', '0')))
        keys = [(3, 0), (44, 287)] + [(int(l), int(e)) for l, e in zip(rng.integers(3, 45, 30), rng.integers(0, 288, 30))]
        try:
            for layer, expert in keys:
                entry = new.layers[layer]
                self.assertEqual(entry['ranks'], ra.ranks_from_base(BASE, layer))
                host = host_planes(entry['fields'], 3)
                bank_write(host, 0, old.read(layer, expert)[0])
                bank_write(host, 1, new.read(layer, expert)[0])
                new.read_into_slot(layer, expert, SimpleNamespace(host=host), 2)
                staging = {p: {f: np.zeros((2, *plane.shape[1:]), plane.dtype) for f, plane in planes.items()}
                           for p, planes in host.items()}
                for p in ra.PROJECTIONS:
                    staging[p]['lrU4'] = np.zeros((2, MAX_RANK, staging[p]['svh4'].shape[1]), np.float16)
                sharded.read_into(layer, expert, SimpleNamespace(host=staging), 0)
                new.read_into(layer, expert, SimpleNamespace(host=staging), 1)
                for p, planes in host.items():
                    for f, plane in planes.items():
                        for slot in (1, 2):
                            self.assertTrue(np.array_equal(plane[slot].view(np.uint8), plane[0].view(np.uint8)),
                                            f'L{layer} E{expert} {p}.{f} slot {slot}')
                    for f in staging[p]:
                        self.assertTrue(np.array_equal(staging[p][f][1].view(np.uint8), staging[p][f][0].view(np.uint8)),
                                        f'L{layer} E{expert} prefill {p}.{f}')
        finally:
            old.close()
            new.close()
            sharded.close()


if __name__ == '__main__':
    unittest.main()
