"""inner.py ARCHIVE...: parse the Mach-O nested in __compute; print sections and small section bytes, plus symbols."""
import struct
import sys

from desc import sections


def walk(m):
    ncmds = struct.unpack('<I', m[16:20])[0]
    p, out = 32, []
    for c in range(ncmds):
        cmd, csize = struct.unpack('<II', m[p:p + 8])
        if cmd == 0x19:
            nsects = struct.unpack('<I', m[p + 64:p + 68])[0]
            q = p + 72
            for s in range(nsects):
                sect = m[q:q + 16].rstrip(b'\0').decode()
                seg = m[q + 16:q + 32].rstrip(b'\0').decode()
                addr, ssize, soff = struct.unpack('<QQI', m[q + 32:q + 52])
                out.append((seg, sect, ssize, m[soff:soff + ssize]))
                q += 80
        elif cmd == 0x2:
            symoff, nsyms, stroff, strsize = struct.unpack('<IIII', m[p + 8:p + 24])
            syms = []
            for i in range(nsyms):
                strx, ntype, nsect, ndesc, value = struct.unpack('<IBBHQ', m[symoff + 16 * i:symoff + 16 * i + 16])
                name = m[stroff + strx:m.index(b'\0', stroff + strx)].decode()
                syms.append((name, nsect, value))
            out.append(('SYMTAB', '', len(syms), syms))
        else:
            out.append(('cmd', hex(cmd), csize, m[p:p + csize]))
        p += csize
    return out


for path in (sys.argv[1:] if __name__ == "__main__" else []):
    print('==', path)
    for seg, sect, size, body in walk(sections(path)['__compute']):
        if seg == 'SYMTAB':
            print('  symbols', [(n[:40], s, hex(v)) for n, s, v in body])
        elif seg == 'cmd':
            print('  cmd', sect, size, body[8:].hex())
        else:
            print('  ', seg, sect, size, body.hex() if size <= 256 else '')
