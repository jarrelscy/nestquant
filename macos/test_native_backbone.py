import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_vlm.models.glm5_next import language
from mlx_vlm.models.glm5_next.config import TextConfig

from native_assets import REVISION, fingerprint
from native_backbone import load_prepared


class NativeBackboneTest(unittest.TestCase):
    def fixture(self, root, bits, embedding_bits, legacy=False):
        source = json.dumps(dict(text_config=dict(num_hidden_layers=1, vocab_size=16, hidden_size=128,
                                                 intermediate_size=128, linear_head_dim=64, linear_num_heads=2))).encode()
        (root / 'config.json').write_bytes(source)
        model = language.LanguageModel(TextConfig.from_dict(json.loads(source)['text_config'])).eval()
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight = module.weight.astype(mx.bfloat16)
        if embedding_bits:
            model.model.embed_tokens = model.model.embed_tokens.to_quantized(group_size=64, bits=embedding_bits)
        model.lm_head = model.lm_head.to_quantized(group_size=64, bits=bits)
        nn.quantize(model.layers[0], group_size=64, bits=bits,
                    class_predicate=lambda _, module: isinstance(module, nn.Linear) and module.weight.dtype != mx.float32)
        directory = root / 'backbone'
        directory.mkdir()
        files = {}
        modules = dict(embed=model.model.embed_tokens, norm=model.model.norm, head=model.lm_head, L0=model.layers[0])
        for name, module in modules.items():
            params = dict(tree_flatten(module.parameters()))
            mx.eval(params)
            file = directory / f'{name}.safetensors'
            mx.save_safetensors(file, params)
            files[name] = dict(state=fingerprint(file), tensor_bytes=sum(array.nbytes for array in params.values()))
        manifest = dict(revision=REVISION, config_sha256=hashlib.sha256(source).hexdigest(), files=files)
        if not legacy:
            manifest['quantization'] = dict(bits=bits, embedding_bits=embedding_bits)
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        return model, directory

    def test_prepared_parameters_and_outputs_roundtrip_for_legacy_and_compact_archives(self):
        for bits, embedding_bits, legacy in ((8, 0, True), (8, 0, False), (4, 0, False), (4, 8, False)):
            with self.subTest(bits=bits, embedding_bits=embedding_bits, legacy=legacy), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                expected, archive = self.fixture(root, bits, embedding_bits, legacy)
                event = Mock()
                actual = load_prepared(root, archive, event)
                self.assertEqual([call.args[0] for call in event.call_args_list], ['embed', 'norm', 'head', 'L0'])
                a, b = dict(tree_flatten(actual.parameters())), dict(tree_flatten(expected.parameters()))
                self.assertEqual(a.keys(), b.keys())
                for name in a:
                    self.assertTrue(mx.array_equal(a[name], b[name]).item(), name)
                for model in (actual, expected):
                    mx.eval(model.parameters())
                ids = mx.array([[0, 3, 15]])
                first = actual.lm_head(actual.model.norm(actual.model.embed_tokens(ids)))
                second = expected.lm_head(expected.model.norm(expected.model.embed_tokens(ids)))
                self.assertTrue(mx.array_equal(first, second).item())

    def test_metadata_rejects_unsupported_precision_before_model_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, archive = self.fixture(root, 8, 0)
            path = archive / 'manifest.json'
            manifest = json.loads(path.read_text())
            for quantization in (None, {}, {'bits': 6, 'embedding_bits': 0}, {'bits': 4, 'embedding_bits': 4},
                                 {'bits': 4.0, 'embedding_bits': 8}, {'bits': 8, 'embedding_bits': False}):
                manifest['quantization'] = quantization
                path.write_text(json.dumps(manifest))
                with self.subTest(quantization=quantization), patch('native_backbone.language.LanguageModel') as model, \
                        self.assertRaises(ValueError):
                    load_prepared(root, archive, Mock())
                model.assert_not_called()

    def test_embedding_metadata_must_match_stored_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, archive = self.fixture(root, 4, 8)
            path = archive / 'manifest.json'
            manifest = json.loads(path.read_text())
            manifest['quantization']['embedding_bits'] = 0
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'embedding tensors disagree'):
                load_prepared(root, archive, Mock())


if __name__ == '__main__':
    unittest.main()
