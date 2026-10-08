"""CPU format regressions; no CUDA context, compilation, or checkpoint required."""
import importlib.util
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_cpu_moe():
    stub = types.ModuleType('build')
    stub.get = lambda: None
    spec = importlib.util.spec_from_file_location('flash_cpu_moe', ROOT/'sm120/moe.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'build': stub}):
        spec.loader.exec_module(module)
    return module


class FlashFormat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_cpu_moe()

    def test_patterns(self):
        self.assertEqual(self.m.RKP[self.m.RK_OF[1.5]], (1, 0xAAAA))
        self.assertEqual(self.m.RKP[self.m.RK_OF[2.8125]], (2, 0xFBDE))
        self.assertEqual(self.m.rbits(5), 96)
        self.assertEqual(self.m.rbits(10), 180)

    def test_packed_layout_against_bit_oracle(self):
        # Includes full-GLM base/residual codes to guard existing formats.
        gen = torch.Generator().manual_seed(771)
        for code in (0, 1, 2, 5, 9, 10):
            bits = self.m.rbits(code)
            words = torch.randint(0, 2**32, (64, (bits+31)//32), generator=gen)
            if bits % 32:
                words[:, -1] &= (1 << (bits % 32))-1
            packed = self.m.pack_words(words, bits)
            self.assertTrue(torch.equal(self.m.unpack_words(packed, 64, bits), words))
            # Independently assemble each sub-array from little-endian integers.
            raw = bytearray()
            off = 0
            for width in (128, 64, 32, 16):
                count = (bits-off)//width
                for record in words.tolist():
                    value = sum(w << (32*i) for i, w in enumerate(record))
                    if count:
                        raw.extend(((value >> off) & ((1 << (count*width))-1)).to_bytes(count*width//8, 'little'))
                off += count*width
            tail = bits-off
            if tail:
                value = sum(((int(row[-1]) >> (off % 32)) & ((1 << tail)-1)) << (i*tail) for i,row in enumerate(words))
                raw.extend(value.to_bytes(64*tail//8, 'little'))
                raw.extend(b'\0'*4)
            self.assertEqual(packed.view(torch.uint8).numpy().tobytes(), bytes(raw))

    def test_actual_cpp_ring_extension(self):
        src = (ROOT/'sm120/nqmoe.cu').read_text()
        start = src.index('template <int BITS, int NW>\n')
        end = src.index('// LV:', start)
        function = src[start:end].replace('__device__', '').replace('__forceinline__', 'inline')
        checks = []
        for bits in (96,112,128,160,164,180):
            nw = (bits+31)//32
            checks.append(f'''{{ uint32_t w[8]={{0x12345678,0xAABBCCDD,0x87654321,0xDEADBEEF,0xA5A5A5A5,0x87654321}};
              uint32_t before[8]; for(int i=0;i<8;i++) before[i]=w[i];
              uint32_t nb=0xF19D83AB; ext_words<{bits},{nw}>(w,nb);
              for(int i=0;i<{bits+32};i++) {{ unsigned expected=i<{bits} ? (before[i/32]>>(i%32))&1 : (nb>>((i-{bits})%32))&1;
                if(((w[i/32]>>(i%32))&1)!=expected) return 1; }} }}''')
        program = '#include <cstdint>\n'+function+'\nint main(){'+''.join(checks)+'return 0;}'
        with tempfile.TemporaryDirectory() as d:
            source = pathlib.Path(d)/'test.cc'; source.write_text(program)
            subprocess.run(['g++','-std=c++17','-O2',str(source),'-o',d+'/test'], check=True)
            subprocess.run([d+'/test'], check=True)
        self.assertNotIn('ext_words<BKB<RC>::BITS, 4>', src)


if __name__ == '__main__':
    unittest.main()
