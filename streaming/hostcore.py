"""Build/load nqhost (nqhost.cpp: C++ host loop for NQ_HOSTLOOP=cpp). Plain pybind11 module, pybind11 headers from the
installed torch (no libtorch link), -ffp-contract=off for bit-exact float math. Built once into $NQ_BUILD/nqhost
(rebuilt when nqhost.cpp is newer), under a file lock so the TP workers do not race."""
import os,sys,sysconfig,subprocess,fcntl,importlib.util
D=os.path.dirname(os.path.abspath(__file__));_m=None
def mod():
    global _m
    if _m is not None:return _m
    import torch
    b=os.environ.get('NQ_BUILD','/data/Jarrel/nq-build')+'/nqhost';os.makedirs(b,exist_ok=True)
    so=f"{b}/nqhost{sysconfig.get_config_var('EXT_SUFFIX')}";src=D+'/nqhost.cpp'
    with open(b+'/.lock','w') as lk:
        fcntl.flock(lk,fcntl.LOCK_EX)
        if not os.path.exists(so) or os.path.getmtime(so)<os.path.getmtime(src):
            cmd=[os.environ.get('CXX','g++'),'-O3','-std=c++17','-shared','-fPIC','-fvisibility=hidden','-ffp-contract=off',
                 '-I',sysconfig.get_paths()['include'],'-I',os.path.dirname(torch.__file__)+'/include',src,'-o',so+'.tmp']
            subprocess.run(cmd,check=True);os.replace(so+'.tmp',so)
    sp=importlib.util.spec_from_file_location('nqhost',so);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);_m=m
    return m
