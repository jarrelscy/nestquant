"""P4 record layout (one record per (layer, expert) per TP rank, the bytes of one level-4 slot).
Segments, in order, each 256 B aligned: gu.p4 | gu.d4 | dn.p4 | dn.d4 | lr4 (U4_g | U4_u | U4_d, fp16, rank <= RMAX).
All records of a layer have the same size (rounded up to ALIGN = 4 KiB), so record i of layer L sits at
layer_off[L] + i * rec_bytes[L] and one O_DIRECT read fills a slot. Slot size = max rec_bytes over layers (a parameter).
  layout(ex_any, H, I)       segment offsets/lengths for a layer (from any expert of that layer: p4/d4 sizes are fixed by K)
  pack(ex, lay) -> bytes     record of one kernel expert (nqload.kernel_expert)
  row(ex, lay, slot_ptr)     level-4 table row with the P4 pointers pointing into a slot that holds the record"""
import torch
SEG_ALIGN=256;ALIGN=4096;RMAX=4
SEGS=('gu.p4','gu.d4','dn.p4','dn.d4','lr4')
up=lambda n,a:(n+a-1)//a*a

def layout(ex,H,I):
    n={'gu.p4':ex.gu.p4.numel()*4,'gu.d4':ex.gu.d4.numel()*4,'dn.p4':ex.dn.p4.numel()*4,'dn.d4':ex.dn.d4.numel()*4,
       'lr4':RMAX*(2*I+H)*2}                               # U4_g, U4_u [r,I] + U4_d [r,H] at the largest rank
    seg={};o=0
    for k in SEGS:seg[k]=(o,n[k]);o=up(o+n[k],SEG_ALIGN)
    return dict(seg=seg,rec_bytes=up(o,ALIGN))

def _tensors(ex):
    lr4=ex.lr4 if ex.lr4 is not None else torch.zeros(0,dtype=torch.float16)
    return {'gu.p4':ex.gu.p4,'gu.d4':ex.gu.d4,'dn.p4':ex.dn.p4,'dn.d4':ex.dn.d4,'lr4':lr4}

def pack(ex,lay):
    b=bytearray(lay['rec_bytes'])
    for k,t in _tensors(ex).items():
        o,n=lay['seg'][k];raw=t.contiguous().cpu().view(torch.uint8).numpy().tobytes()
        assert len(raw)<=n,(k,len(raw),n);b[o:o+len(raw)]=raw
    return bytes(b)

def row(ex,lay,slot_ptr,entry):
    """entry = moe.entry; the resident planes (base, var, signs, lr) stay where they are."""
    r=entry(ex,4);s=lay['seg']
    r[2],r[3],r[6],r[7]=(slot_ptr+s[k][0] for k in ('gu.p4','gu.d4','dn.p4','dn.d4'))
    if ex.rg+ex.rd:r[15]=slot_ptr+s['lr4'][0]
    if hasattr(ex,'sc'):r[9]=ex.sc[4].data_ptr()           # level-4 scale vectors (resident)
    return r
