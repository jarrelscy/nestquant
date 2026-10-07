"""CPU-only compatibility and finite fallback regressions; does not load CUDA modules."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parents[1]

class Compatibility(unittest.TestCase):
    def test_loader_metadata(self):
        tree=ast.parse((ROOT/'sm120/moe.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MoELayer')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='validate')
        env={'had_dn':lambda e:e.had_dn,'base_code':lambda p:p.bk,'torch':NS(float16='f16')}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'validate','exec'),env)
        layer=NS(H=6144,I=512,M=NS(rk_codes=lambda:[0x1c9,0x1c9]))
        def ex():return NS(H=6144,I=512,had_dn=512,rg=2,rd=1,lr=NS(dtype='f16'),gu=NS(bk=0,rk=0),dn=NS(bk=0,rk=7))
        env['validate'](layer,ex(),4)
        for key,val in [('H',4096),('had_dn',256),('rg',5),('rd',-1),('lr',None),('gu.bk',1),('gu.bk',-1),('dn.rk',2),('dn.rk',32)]:
            x=ex(); obj=x
            if '.' in key:part,key=key.split('.');obj=getattr(x,part)
            setattr(obj,key,val)
            with self.subTest(key=key,val=val),self.assertRaises(ValueError):env['validate'](layer,x,4)
    def test_safe_fallback(self):
        import torch
        tree=ast.parse((ROOT/'sm120/serve/nq_dbg_numerics.py').read_text())
        outer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='patch_fp8o')
        fn=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='fp8_w8a16_linear')
        fn.decorator_list=[]
        env={'torch':torch,'FUSED':False}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'safe_fallback','exec'),env)
        # Both old half partial overflow and old BF16->half input/output narrowing.
        for value in (1280.,8_000_000.):
            x=torch.full((1,16),value,dtype=torch.bfloat16)
            w=torch.full((1,16),448.,dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8)
            scale=torch.tensor([.001])
            out=env['fp8_w8a16_linear'](x,w,scale,64,6)
            expected=(x.float().sum(1,keepdim=True)*448.*scale).to(x.dtype)
            self.assertTrue(torch.isfinite(out).all());torch.testing.assert_close(out,expected)
if __name__=='__main__':unittest.main()
