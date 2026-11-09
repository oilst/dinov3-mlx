"""The macOS process footprint and its lifetime maximum, from proc_pid_rusage.

The footprint is the memory that macOS charges to a process. On unified memory it includes the
GPU memory of torch-mps and mlx, so one measure serves both backends (D5 of
specs/03-baseline/explore.html). The footprint does not fall after torch.mps.empty_cache().
"""

import ctypes
import ctypes.util
import os
from dataclasses import dataclass

RUSAGE_INFO_V4 = 4

# struct rusage_info_v4 of sys/resource.h: a 16-byte uuid, then 35 uint64 fields.
_V4_FIELDS = [
    "ri_user_time",
    "ri_system_time",
    "ri_pkg_idle_wkups",
    "ri_interrupt_wkups",
    "ri_pageins",
    "ri_wired_size",
    "ri_resident_size",
    "ri_phys_footprint",
    "ri_proc_start_abstime",
    "ri_proc_exit_abstime",
    "ri_child_user_time",
    "ri_child_system_time",
    "ri_child_pkg_idle_wkups",
    "ri_child_interrupt_wkups",
    "ri_child_pageins",
    "ri_child_elapsed_abstime",
    "ri_diskio_bytesread",
    "ri_diskio_byteswritten",
    "ri_cpu_time_qos_default",
    "ri_cpu_time_qos_maintenance",
    "ri_cpu_time_qos_background",
    "ri_cpu_time_qos_utility",
    "ri_cpu_time_qos_legacy",
    "ri_cpu_time_qos_user_initiated",
    "ri_cpu_time_qos_user_interactive",
    "ri_billed_system_time",
    "ri_serviced_system_time",
    "ri_logical_writes",
    "ri_lifetime_max_phys_footprint",
    "ri_instructions",
    "ri_cycles",
    "ri_billed_energy",
    "ri_serviced_energy",
    "ri_interval_max_phys_footprint",
    "ri_runnable_time",
]


class _RusageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in _V4_FIELDS]


_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_RusageInfoV4)]
_libc.proc_pid_rusage.restype = ctypes.c_int


@dataclass(frozen=True)
class Footprint:
    """The physical footprint of one process.

    Attributes:
        current_bytes: The footprint now.
        lifetime_max_bytes: The largest footprint since the process started.
    """

    current_bytes: int
    lifetime_max_bytes: int


def read_footprint(pid: int | None = None) -> Footprint:
    """Return the current physical footprint and its lifetime maximum, in bytes.

    The footprint includes the GPU memory of torch-mps and mlx on unified memory.

    Args:
        pid: The process. None means this process.

    Returns:
        The footprint of the process.

    Raises:
        OSError: If proc_pid_rusage fails.
    """
    info = _RusageInfoV4()
    if _libc.proc_pid_rusage(pid or os.getpid(), RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
    return Footprint(info.ri_phys_footprint, info.ri_lifetime_max_phys_footprint)
