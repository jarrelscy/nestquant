"""Fetch the exact public BF16 confirmation panel; no GPU use."""
import pathlib,json,hashlib,shutil
from huggingface_hub import hf_hub_download
P=pathlib.Path('/eval');REV='95f4fdd94bf29989db2e0d1054e4931f55edb6aa';REPO='brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
(P/'tokens').mkdir(exist_ok=True)
for row in json.loads((P/'teacher-hashes.json').read_text()):
 tag=row['tag']
 for name,want in [(f'logits/full-panel/confirmation/{tag}.safetensors',row['teacher_sha256']),(f'calibration/panel-v1/arrays/{tag}.tokens.npy',row['tokens_file_sha256'])]:
  p=pathlib.Path(hf_hub_download(REPO,name,repo_type='dataset',revision=REV,local_dir=P/'teacher'));assert sha(p)==want,name
  if name.endswith('.npy'):shutil.copyfile(p,P/'tokens'/p.name)
print('Verified four BF16 teacher files and token windows at pinned revision '+REV)
