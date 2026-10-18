"""Make the CUDA DLLs that pip installs (nvidia-cublas-cu12, nvidia-cudnn-cu12) visible to ctranslate2 on Windows.

Import this before faster_whisper. On Linux/macOS, or when the DLLs are already on PATH, it does nothing harmful.
"""
import os
import sys
from pathlib import Path


def add_cuda_dlls():
    if sys.platform != "win32":
        return
    nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not nvidia.is_dir():
        return
    for bin_dir in nvidia.glob("*/bin"):
        os.add_dll_directory(str(bin_dir))
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


add_cuda_dlls()
