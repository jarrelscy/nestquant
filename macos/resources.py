import ctypes
import os


class RUsageInfoV4(ctypes.Structure):
    _fields_ = [('uuid', ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            'user_time', 'system_time', 'pkg_idle_wkups', 'interrupt_wkups', 'pageins',
            'wired_size', 'resident_size', 'phys_footprint', 'proc_start_abstime', 'proc_exit_abstime',
            'child_user_time', 'child_system_time', 'child_pkg_idle_wkups', 'child_interrupt_wkups',
            'child_pageins', 'child_elapsed_abstime', 'diskio_bytesread', 'diskio_byteswritten',
            'cpu_time_qos_default', 'cpu_time_qos_maintenance', 'cpu_time_qos_background',
            'cpu_time_qos_utility', 'cpu_time_qos_legacy', 'cpu_time_qos_user_initiated',
            'cpu_time_qos_user_interactive', 'billed_system_time', 'serviced_system_time',
            'logical_writes', 'lifetime_max_phys_footprint', 'instructions', 'cycles',
            'billed_energy', 'serviced_energy', 'interval_max_phys_footprint', 'runnable_time',
        )
    ]


_LIBPROC = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
_LIBPROC.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
_LIBPROC.proc_pid_rusage.restype = ctypes.c_int


def process_usage(pid=None):
    usage = RUsageInfoV4()
    if _LIBPROC.proc_pid_rusage(os.getpid() if pid is None else pid, 4, ctypes.byref(usage)) != 0:
        raise OSError(ctypes.get_errno(), 'proc_pid_rusage failed')
    return usage



_LIBPROC.sysctlbyname.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]
_LIBPROC.sysctlbyname.restype = ctypes.c_int


def system_pressure():
    value = ctypes.c_int()
    size = ctypes.c_size_t(ctypes.sizeof(value))
    if _LIBPROC.sysctlbyname(b'kern.memorystatus_vm_pressure_level', ctypes.byref(value),
                           ctypes.byref(size), None, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot read macOS memory pressure')
    return value.value
