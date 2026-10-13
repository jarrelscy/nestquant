try:
 import startup_thp
except BaseException:
 import os;os._exit(78)
import os,pathlib,hashlib,json,importlib.machinery
expected={'nq_flash_runtime': '49482ea7c50b0a1dd33f3867a6e01c0c98106f5fe650865e4f74146f880e8c24', 'jt_runtime': '8c06583c210dc1d4a44766164c6a3da46f41f83094310df39963401d2b4f88ee', 'nq_flash_plugin': '079f9d3f6c046eb8b0fd8d479df6889a21b16d4e30eab2823d50d4699df05b13'}
try:
 proof={}
 for name,h in expected.items():
  spec=importlib.machinery.PathFinder.find_spec(name,['/nq/spark/flash']);path=pathlib.Path(spec.origin);actual=hashlib.sha256(path.read_bytes()).hexdigest()
  assert str(path)==f'/nq/spark/flash/{name}.py' and actual==h,(name,str(path),actual)
  proof[name]={'origin':str(path),'sha256':actual}
 pub=importlib.machinery.PathFinder.find_spec('post_layer_publication',['/diag']);pp=pathlib.Path(pub.origin);ph=hashlib.sha256(pp.read_bytes()).hexdigest();assert str(pp)=='/diag/post_layer_publication.py' and ph=='1f5d5e732c4df4a3a48fbac7ef998499adeb2ad3029f242af41644299309091c';proof['publication']={'origin':str(pp),'sha256':ph}
 for name,want in {'startup_sentinel.py': '20d2ab578bf7adf5cad6e12cd09d0d048dbd734c957001b8f3923733e83d13b5'}.items():
  actual=hashlib.sha256(pathlib.Path("/diag",name).read_bytes()).hexdigest();assert actual==want;proof[name]={"origin":"/diag/"+name,"sha256":actual}
 for name,want in {'packet_timing.py': '1119d77107272da87b3edcf335ac25dd77f522f0d770df5ad05bfa28bec1c3ec'}.items():
  actual=hashlib.sha256(pathlib.Path("/diag",name).read_bytes()).hexdigest();assert actual==want;proof[name]={"origin":"/diag/"+name,"sha256":actual}
 for name,want in {'prewarm_mla.py': 'c173bfd9c3a1d779f73a3254af66cb27288660875ffaa3cf7488360ad3aa56b0', 'loader_memory_diagnostic.py': '55a62d1d6350b8b90867fcf2e26803d4705b3b5ee525e83f04d560dd7247ce79'}.items():
  actual=hashlib.sha256(pathlib.Path('/diag',name).read_bytes()).hexdigest();assert actual==want,(name,actual);proof[name]={'origin':'/diag/'+name,'sha256':actual}
 indexer_path=pathlib.Path("/opt/vllm/vllm/model_executor/layers/sparse_attn_indexer_kpool.py");indexer_sha=hashlib.sha256(indexer_path.read_bytes()).hexdigest();assert indexer_sha=="62d88a567877c7f22e03dc2eaf860fbaed1b355486ae5c8d088a5cc6693b9758";proof["indexer"]={"origin":str(indexer_path),"sha256":indexer_sha}
 out=pathlib.Path('/data/kld-sampled-ahead-sentinel');out.mkdir(exist_ok=True);(out/f'module-provenance-{os.getpid()}.json').write_text(json.dumps(proof))
except BaseException:os._exit(78)
import loader_memory_diagnostic
startup_thp.install(loader_memory_diagnostic)
_original_patch_runner=loader_memory_diagnostic.patch_runner
def _patch_runner(module):
    import nq_flash_runtime,jt_runtime,nq_flash_plugin
    for name,m in [('nq_flash_runtime',nq_flash_runtime),('jt_runtime',jt_runtime),('nq_flash_plugin',nq_flash_plugin)]:
        if hashlib.sha256(pathlib.Path(m.__file__).read_bytes()).hexdigest()!=expected[name]:os._exit(78)
    _original_patch_runner(module)
    import packet_timing
    packet_timing.install(nq_flash_runtime,jt_runtime,str(out))
    import nq_spark_kld
    nq_spark_kld.install_force()
loader_memory_diagnostic.patch_runner=_patch_runner
loader_memory_diagnostic.install()
