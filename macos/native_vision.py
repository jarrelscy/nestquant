"""Image input for the native runtime: GLM-5.3-Flash vision tower, image loading and prompt expansion.

The text model uses plain 1-D positions, so an image is a run of image-token positions whose input embeddings are
replaced by the vision tower's features. In prompts the image tokens carry negative pseudo ids derived from the
image's hash and the position inside the image; prefix reuse then matches a cached image only when it is the same
image. `real_ids` maps them back to the image token id before they reach the model, and FeatureEmbedding supplies
the features while the prefill embeds the prompt in order.
"""

import base64
from contextlib import contextmanager
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import urllib.request

import mlx.core as mx
import mlx.nn as nn
import numpy as np

MAX_IMAGE_BYTES = 20 * 2**20
MAX_IMAGES = 16
_INDEX_BITS = 24          # position inside an image (GLM caps an image at 8,000 tokens)


class ImageError(ValueError):
    pass


def load_image_bytes(source):
    """Bytes of an image given as a base64 data URL or an http(s) URL."""
    if not isinstance(source, str) or not source:
        raise ImageError('Image parts need a URL')
    if source.startswith('data:'):
        header, _, data = source.partition(',')
        if ';base64' not in header:
            raise ImageError('Image data URLs must be base64 encoded')
        try:
            raw = base64.b64decode(data, validate=False)
        except ValueError as error:
            raise ImageError('Invalid base64 image data') from error
    elif source.startswith(('http://', 'https://')):
        request = urllib.request.Request(source, headers={'User-Agent': 'nestquant-server'})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read(MAX_IMAGE_BYTES + 1)
        except OSError as error:
            raise ImageError(f'Could not fetch image: {error}') from error
    else:
        raise ImageError('Images must be data: or http(s):// URLs')
    if len(raw) > MAX_IMAGE_BYTES:
        raise ImageError(f'Images are limited to {MAX_IMAGE_BYTES // 2**20} MiB')
    return raw


def image_key(raw):
    return int(hashlib.sha256(raw).hexdigest()[:15], 16)


def pseudo_id(key, index):
    return -(1 + (key << _INDEX_BITS) + index)


def split_pseudo(token):
    value = -token - 1
    return value >> _INDEX_BITS, value & ((1 << _INDEX_BITS) - 1)


def expand_prompt(ids, image_token_id, images):
    """Replace the k-th image marker in `ids` with images[k] = (key, count) pseudo ids."""
    out, k = [], 0
    for token in ids:
        if token == image_token_id:
            if k >= len(images):
                raise ImageError('The prompt has more image markers than images')
            key, count = images[k]
            out.extend(pseudo_id(key, index) for index in range(count))
            k += 1
        else:
            out.append(token)
    if k != len(images):
        raise ImageError('The number of images does not match the image markers')
    return out


def real_ids(ids, image_token_id):
    return [image_token_id if token < 0 else token for token in ids]


class FeatureEmbedding(nn.Module):
    """Token embedding that substitutes supplied feature rows at image-token positions, in order of appearance.
    Without supplied rows it is the wrapped embedding."""

    def __init__(self, base, image_token_id):
        super().__init__()
        self.base = base
        self.image_token_id = image_token_id
        self._supply = SimpleNamespace(rows=None, cursor=0)

    def __call__(self, ids):
        out = self.base(ids)
        supply = self._supply
        if supply.rows is None:
            return out
        mask = ids == self.image_token_id
        count = int(mask.sum().item())
        if not count:
            return out
        if supply.cursor + count > supply.rows.shape[0]:
            raise RuntimeError('More image tokens than supplied image features')
        rows = supply.rows[supply.cursor:supply.cursor + count]
        supply.cursor += count
        flat = mask.reshape(-1)
        index = mx.where(flat, mx.cumsum(flat.astype(mx.int32)) - 1, 0)
        gathered = rows[index].reshape(*ids.shape, rows.shape[-1]).astype(out.dtype)
        return mx.where(mask[..., None], gathered, out)

    def as_linear(self, x):
        return self.base.as_linear(x)

    @contextmanager
    def supplying(self, rows):
        supply = self._supply
        supply.rows, supply.cursor = rows, 0
        try:
            yield
            if rows is not None and supply.cursor != rows.shape[0]:
                raise RuntimeError(f'Prefill used {supply.cursor} of {rows.shape[0]} image feature rows')
        finally:
            supply.rows, supply.cursor = None, 0


class VisionInputs:
    """The vision tower, the image processor and a small cache of computed image features."""

    def __init__(self, model_path, cache_entries=MAX_IMAGES):
        from mlx_vlm.models.glm5_next.config import VisionConfig
        from mlx_vlm.models.glm5_next.processing import Glm5NextImageProcessor
        from mlx_vlm.models.glm5_next.vision import VisionModel

        model_path = Path(model_path)
        config = json.loads((model_path / 'config.json').read_text())
        self.image_token_id = int(config['image_token_id'])
        self.tower = VisionModel(VisionConfig.from_dict(config['vision_config']))
        weights = mx.load(str(model_path / 'vision_tower.safetensors'))
        prefix = 'model.visual.'
        weights = self.tower.sanitize({key[len(prefix):]: value for key, value in weights.items()
                                       if key.startswith(prefix)})
        self.tower.load_weights(list(weights.items()), strict=True)
        mx.eval(self.tower.parameters())
        settings = json.loads((model_path / 'processor_config.json').read_text())['image_processor']
        settings.pop('image_processor_type', None)
        self.processor = Glm5NextImageProcessor(**settings)
        self.merge = int(settings.get('merge_size', 2))
        self.cache_entries = cache_entries
        self.features = OrderedDict()   # key -> mx.array (tokens, hidden)
        self.pending = {}               # key -> (pixel values, grid) awaiting the tower
        self.counts = {}                # key -> image tokens
        self.lock = threading.Lock()    # prepare runs on request threads, features on the model thread

    def prepare(self, raw_images):
        """Preprocess images on the host. Returns [(key, image tokens)] in order."""
        from PIL import Image

        if len(raw_images) > MAX_IMAGES:
            raise ImageError(f'At most {MAX_IMAGES} images per request')
        out = []
        for raw in raw_images:
            key = image_key(raw)
            with self.lock:
                known = key in self.counts
            if not known:
                try:
                    image = Image.open(io.BytesIO(raw))
                    image.load()
                except Exception as error:     # PIL raises several types for bad data
                    raise ImageError(f'Could not decode image: {type(error).__name__}') from error
                processed = self.processor([image.convert('RGB')])
                grid = np.asarray(processed['image_grid_thw'])
                with self.lock:
                    self.pending[key] = (np.asarray(processed['pixel_values']), grid)
                    self.counts[key] = int(np.prod(grid[0])) // self.merge ** 2
                    # Requests rejected after preprocessing leave pending entries; keep only the newest.
                    while len(self.pending) > 2 * MAX_IMAGES:
                        old = next(iter(self.pending))
                        del self.pending[old]
                        if old not in self.features:
                            self.counts.pop(old, None)
            with self.lock:
                out.append((key, self.counts[key]))
        return out

    def _features(self, key):
        with self.lock:
            return self._features_locked(key)

    def _features_locked(self, key):
        if key in self.features:
            self.features.move_to_end(key)
            return self.features[key]
        pixels, grid = self.pending.pop(key)
        dtype = self.tower.patch_embed.proj.weight.dtype
        features = self.tower(mx.array(pixels).astype(dtype), mx.array(grid))
        mx.eval(features)
        if features.shape[0] != self.counts[key]:
            raise RuntimeError(f'Vision tower gave {features.shape[0]} rows for {self.counts[key]} image tokens')
        self.features[key] = features
        while len(self.features) > self.cache_entries:
            old, _ = self.features.popitem(last=False)
            self.counts.pop(old, None)
        return features

    def rows_for(self, ids):
        """Feature rows for the pseudo ids in `ids`, in order (None if there are none)."""
        wanted = [split_pseudo(token) for token in ids if token < 0]
        if not wanted:
            return None
        rows, start = [], 0
        # Consecutive positions of one image become one slice.
        while start < len(wanted):
            key, first = wanted[start]
            end = start + 1
            while end < len(wanted) and wanted[end] == (key, first + end - start):
                end += 1
            rows.append(self._features(key)[first:first + end - start])
            start = end
        return mx.concatenate(rows, axis=0) if len(rows) > 1 else rows[0]
