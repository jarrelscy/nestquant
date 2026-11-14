"""desc.py ARCHIVE...: print __compute size and the __descriptor section as 32-bit words for each archive."""
import struct
import sys


def sections(path):
    data = open(path, 'rb').read()
    n = struct.unpack('>I', data[4:8])[0]
    for i in range(n):
        cpu, sub, o, size, align = struct.unpack('>iiIII', data[8 + 20 * i:28 + 20 * i])
        if cpu != 16777235:
            continue
        m = data[o:o + size]
        ncmds = struct.unpack('<I', m[16:20])[0]
        p, out = 32, {}
        for c in range(ncmds):
            cmd, csize = struct.unpack('<II', m[p:p + 8])
            if cmd == 0x19:
                nsects = struct.unpack('<I', m[p + 64:p + 68])[0]
                q = p + 72
                for s in range(nsects):
                    sect = m[q:q + 16].rstrip(b'\0').decode()
                    addr, ssize, soff = struct.unpack('<QQI', m[q + 32:q + 52])
                    out[sect] = m[soff:soff + ssize]
                    q += 80
            p += csize
        return out


for path in (sys.argv[1:] if __name__ == "__main__" else []):
    s = sections(path)
    words = struct.unpack(f'<{len(s["__descriptor"]) // 4}I', s['__descriptor'])
    print(path, 'code', len(s['__compute']), ' '.join(f'{w:x}' for w in words[-36:]))

if __name__ == "__main__" and len(sys.argv) == 3:
    a, b = (sections(p) for p in sys.argv[1:])
    for name in a:
        if name in ('__compute', '__metallib'):
            continue
        x, y = a[name], b[name]
        print(name, len(x), len(y))
        for i in range(0, min(len(x), len(y)), 4):
            if x[i:i + 4] != y[i:i + 4]:
                print('  ', i, x[i:i + 4].hex(), y[i:i + 4].hex())
    x, y = a['__compute'], b['__compute']
    print('compute head', x[:128].hex(), '\n            ', y[:128].hex())
