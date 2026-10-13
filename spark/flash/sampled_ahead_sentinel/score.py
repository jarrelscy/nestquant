"""Full-vocabulary BF16 teacher KL against position-indexed Spark decode logprobs."""
import argparse,json,pathlib,struct,math
import numpy as np

def safemmap(path):
 with open(path,'rb') as f:
  n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n))
 t=h['logits'];assert t['dtype']=='F32'
 return np.memmap(path,mode='r',dtype='<f4',offset=8+n+t['data_offsets'][0],shape=tuple(t['shape']))
def lsm(x):
 x=x.astype(np.float64);m=x.max(axis=-1,keepdims=True);return x-m-np.log(np.exp(x-m).sum(axis=-1,keepdims=True))
def score(teacher,student,tokens):
 n,v=teacher.shape;assert student.shape==(n,v);assert len(tokens)==n+1
 sums=np.zeros(4);per=[]
 for a in range(0,n,8):
  b=min(a+8,n);lp=lsm(teacher[a:b]);q=np.asarray(student[a:b],dtype=np.float64)
  if not np.isfinite(q).all():raise ValueError('nonfinite student log probabilities')
  if np.max(np.abs(np.exp(q).sum(-1)-1))>1e-4:raise ValueError('student log probabilities not normalized')
  kl=(np.exp(lp)*(lp-q)).sum(-1);target=tokens[a+1:b+1];idx=np.arange(b-a)
  sums+=np.array([kl.sum(),(lp.argmax(-1)==q.argmax(-1)).sum(),-q[idx,target].sum(),-lp[idx,target].sum()]);per.extend(kl.tolist())
 return dict(positions=n,vocab=v,kld=float(sums[0]/n),top1_agreement=float(sums[1]/n),student_ppl=math.exp(sums[2]/n),teacher_ppl=math.exp(sums[3]/n),per_position_kld=per)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('teacher');p.add_argument('student');p.add_argument('tokens');p.add_argument('metadata');p.add_argument('output');a=p.parse_args()
 meta=json.load(open(a.metadata));assert meta['positions']==meta['expected_positions']==2047 and not meta['missing'] and meta['dup']==0 and meta['first_mismatch']==0
 t=safemmap(a.teacher);s=np.memmap(a.student,mode='r',dtype='<f4',shape=t.shape);r=score(t,s,np.load(a.tokens));r['capture']=meta;pathlib.Path(a.output).write_text(json.dumps(r,indent=2))
