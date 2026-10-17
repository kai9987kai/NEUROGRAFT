"""Lightweight runtime diagnostics without importing torch."""
from __future__ import annotations
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import sys


def memory_info():
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {"total_ram_gb": round(memory.total / 2**30, 3),
                "available_ram_gb": round(memory.available / 2**30, 3)}
    except ImportError:
        if sys.platform == "win32":
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in
                    ("total_physical", "available_physical", "total_page", "available_page",
                     "total_virtual", "available_virtual", "available_extended")]
            memory = MemoryStatus()
            memory.length = ctypes.sizeof(memory)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                return {"total_ram_gb": round(memory.total_physical / 2**30, 3),
                        "available_ram_gb": round(memory.available_physical / 2**30, 3)}
        return {"total_ram_gb": None, "available_ram_gb": None}


def doctor():
    packages = {}
    for name in ("torch", "transformers", "pyarrow", "pytest"):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        packages[name] = {"installed": importlib.util.find_spec(name) is not None, "version": version}
    return {"python": sys.version, "executable": sys.executable, "machine": platform.machine(),
            "logical_processors": os.cpu_count(), "free_disk_gb": round(shutil.disk_usage('.').free / 2**30, 2),
            "packages": packages, **memory_info(),
            "note": "Package discovery is not a model or compiled-operator test; no CUDA assumption."}
