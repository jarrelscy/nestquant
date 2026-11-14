"""regs.py TAG [SETTINGS_JSON]: GPR count (metadata field 0.0, calibrated: N live floats -> N + 3) and code size of every
window kernel expert_mlp_fused launches for decode-1/verify-2/verify-8 base and mixed, under decoder SETTINGS."""
import os
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from desc import sections
from inner import walk

ARCHIVE = os.path.join(HERE, 'archive')  # swiftc -O archive.swift -o archive
PY = os.path.expanduser('~/src/nestquant/.venv-macos/bin/python')
tag = sys.argv[1]
settings = sys.argv[2] if len(sys.argv) > 2 else '{}'
work = os.path.join('/tmp', 'nq_regs_' + tag)
os.makedirs(work, exist_ok=True)
log = os.path.join(work, 'log.txt')
with open(log, 'w') as f:
    subprocess.run([PY, os.path.join(HERE, 'capture.py'), work, settings], stdout=f, stderr=subprocess.STDOUT, check=True)
subprocess.run([PY, os.path.join(HERE, 'split.py'), log, work], check=True)


def gprs(path):
    md = {seg: body for seg, sect, size, body in walk(sections(path)['__compute'])}
    buf = md['__GPU_METADATA']
    root = struct.unpack('<I', buf[:4])[0]
    vt = root - struct.unpack('<i', buf[root:root + 4])[0]
    f0 = struct.unpack('<H', buf[vt + 4:vt + 6])[0]
    sub = root + f0 + struct.unpack('<I', buf[root + f0:root + f0 + 4])[0]
    vt2 = sub - struct.unpack('<i', buf[sub:sub + 4])[0]
    g0 = struct.unpack('<H', buf[vt2 + 4:vt2 + 6])[0]
    return struct.unpack('<I', buf[sub + g0:sub + g0 + 4])[0], len(md['__TEXT'])


for name in sorted(os.listdir(work)):
    if not name.endswith('.metal'):
        continue
    src = os.path.join(work, name)
    binary = src[:-6] + '.bin'
    subprocess.run([ARCHIVE, src, binary], check=True, stdout=subprocess.DEVNULL)
    regs, text = gprs(binary)
    params = name.split('__')[1][:-6].split('_')   # LEVEL TK TN BANKED MIXED PACKED CHUNKS MAXR ...
    proj = 'gate/up' if '_pair' in name else 'down'
    kind = name.split('nomask')[1].split('__')[0]
    print(f'{tag:14s} {proj:8s} MAXR={params[7]} MIXED={params[4]} {kind:22s} gprs={regs:4d} code={text:6d}')
