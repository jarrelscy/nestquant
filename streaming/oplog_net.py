"""Leader/follower op log across hosts (NQ_OPLOG=dist; 2x DGX Spark = TP2 over two nodes). Same records and the same
reader semantics as oplog.OpLog, carried over TCP instead of a /dev/shm file (a follower on another node cannot see the
leader's /dev/shm). Rank 0 listens on NQ_OPLOG_ADDR (host:port, default $MASTER_ADDR or $VLLM_HOST_IP : 29611); every
follower connects and sends its rank; the leader's constructor waits until all TP-1 followers are connected
(NQ_OPLOG_WAIT_S, 600), so no record is ever sent before a follower can read it.
  NetOpLog(addr, writer, rank, nfollow)   writer: put(ups, downs) / put_reclaim(ep) -> sendall to every follower
                                          reader: get(gate) -> [(ups, downs)] complete records received since last get
  publish(d)    io stats of this rank (dict): follower -> leader as one JSON line; leader keeps its own copy
  peer_stats()  leader: {rank: last published dict} (the Runtime.io_all() API, no /dev/shm files)
  put_mark(kind, i, v)   leader: in-band marker (nq_pfblock: kind 0 = prefill plan of layer index i for chunk v, kind 1 =
                decode step v, i = 0). The follower's get() stores it in marks[kind][i] = (v, gen, off after the marker), so
                (log.gen, log.off) >= (marks[..][1], marks[..][2]) holds once it has replayed every op sent before the marker,
                the same test nq_pfblock makes against the /dev/shm marker files on one host.
gen is always 0 and off counts stream bytes (writer: sent, reader: consumed), so positions compare like OpLog's.
The log never rotates (a stream has no file to grow). A broken connection raises in put / get: the streaming thread
stops and every expert keeps its current valid row, as for any other host-loop error."""
import os,json,time,socket,select,threading,numpy as np
from oplog import MAGIC,XB
PORT=29611;NMARK=256

def addr():
    a=os.environ.get('NQ_OPLOG_ADDR')
    if a:h,_,p=a.rpartition(':');return h,int(p)
    h=os.environ.get('MASTER_ADDR') or os.environ.get('VLLM_HOST_IP') or '127.0.0.1'
    return h,PORT

def _nodelay(c):
    c.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1);return c

class NetOpLog:
    def __init__(s,ad,writer,rank,nfollow,wait_s=None):
        s.writer=writer;s.rank=rank;s.buf=bytearray();s.stats_in={};s.sbuf={};s.lock=threading.Lock();s.mine=None
        s.gen=0;s.off=0;s.marks={0:np.zeros((NMARK,3),np.int64),1:np.zeros((1,3),np.int64)}
        wait_s=float(os.environ.get('NQ_OPLOG_WAIT_S','600')) if wait_s is None else wait_s
        if writer:
            ls=socket.socket(socket.AF_INET,socket.SOCK_STREAM);ls.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            ls.bind(('0.0.0.0',ad[1]));ls.listen(max(nfollow,1));ls.settimeout(wait_s);s.peers={}
            t=time.time()
            while len(s.peers)<nfollow:
                try:c,_=ls.accept()
                except socket.timeout:raise RuntimeError(f'NestQuant oplog: {len(s.peers)}/{nfollow} followers connected to :{ad[1]} after {wait_s:.0f}s')
                c.settimeout(wait_s);r=int.from_bytes(_recvn(c,4),'little');c.settimeout(None);s.peers[r]=_nodelay(c);s.sbuf[r]=bytearray()
            ls.close();s.t_conn=time.time()-t
        else:
            t=time.time()
            while True:
                try:c=socket.create_connection(ad,timeout=5);break
                except OSError:
                    if time.time()-t>wait_s:raise RuntimeError(f'NestQuant oplog: rank {rank} could not reach the leader at {ad[0]}:{ad[1]}')
                    time.sleep(0.5)
            c.settimeout(None);_nodelay(c).sendall(int(rank).to_bytes(4,'little'));s.sock=c;c.setblocking(False);s.t_conn=time.time()-t
    # ---- writer ----
    def _w(s,a):
        b=a.tobytes()
        with s.lock:                                      # the streaming thread and the decode-step waiter both write
            for c in s.peers.values():c.sendall(b)
            s.off+=len(b)
    def put_mark(s,kind,i,v):s._w(np.array([MAGIC,-3,int(kind),int(i),int(v)],np.int32))
    def put_reclaim(s,ep):s._w(np.array([MAGIC,-2,int(ep)],np.int32))
    def put(s,ups,downs):
        if not ups and not downs:return
        s._w(np.array([MAGIC,len(ups),len(downs)]+[x for p in list(ups)+list(downs) for x in p],np.int32))
    # ---- reader ----
    def _recv(s):
        while True:
            try:b=s.sock.recv(1<<20)
            except (BlockingIOError,InterruptedError):return
            if not b:raise RuntimeError('NestQuant oplog: leader closed the connection')
            s.buf+=b
            if len(b)<1<<20:return
    def get(s,gate=None):
        s._recv();out=[]
        a=np.frombuffer(bytes(s.buf[:len(s.buf)//4*4]),np.int32);i=0
        while i+3<=len(a):
            assert a[i]==MAGIC,f'oplog stream: bad record at int {i}'
            nu,nd=int(a[i+1]),int(a[i+2])
            if nu==-2:                                    # nq-prefill reclaim marker
                if gate is not None and gate.x_done<nd:break
                i+=3;continue
            if nu==-3:                                    # nq_pfblock marker
                if i+5>len(a):break
                s.marks[int(a[i+2])][int(a[i+3])]=(int(a[i+4]),s.gen,s.off+4*(i+5));i+=5;continue
            j=i+3+2*(nu+nd)
            if j>len(a):break                             # record not complete yet
            if gate is not None and nu and int(a[i+3:i+3+2*nu:2].max())>=XB and gate.x_have<int(a[i+3:i+3+2*nu:2].max())//XB:break
            p=a[i+3:j].reshape(-1,2).tolist();out.append(([tuple(x) for x in p[:nu]],[tuple(x) for x in p[nu:]]));i=j
        del s.buf[:4*i];s.off+=4*i
        return out
    # ---- io stats ----
    def publish(s,d):
        if s.writer:s.mine=d;return
        b=(json.dumps(d)+'\n').encode();s.sock.setblocking(True)
        try:s.sock.sendall(b)
        finally:s.sock.setblocking(False)
    def peer_stats(s):
        if not s.writer:return {}
        out={}
        if s.mine is not None:out[s.rank]=s.mine
        rd,_,_=select.select(list(s.peers.values()),[],[],0)
        for r,c in s.peers.items():
            if c in rd:
                b=c.recv(1<<20)
                if b:s.sbuf[r]+=b
        for r,b in s.sbuf.items():
            if b'\n' in b:
                ln=bytes(b).rsplit(b'\n',2);last=ln[-2] if len(ln)>=2 else b'';s.sbuf[r]=bytearray(ln[-1])
                try:s.stats_in[r]=json.loads(last)
                except ValueError:pass
        out.update(s.stats_in);return out
    def close(s):
        for c in (list(s.peers.values()) if s.writer else [s.sock]):
            try:c.close()
            except OSError:pass

def _recvn(c,n):
    b=b''
    while len(b)<n:
        x=c.recv(n-len(b))
        if not x:raise RuntimeError('NestQuant oplog: peer closed during handshake')
        b+=x
    return b
