from __future__ import annotations

import os
import sysconfig
from pathlib import Path

# Debemos conservar los handles durante toda la ejecución.
# Si se destruyen, Windows puede dejar de usar esas rutas para buscar DLL.
_DLL_DIRECTORY_HANDLES: list[object] = []


def _cuda_version_key(path: Path) -> tuple[int, ...]:
    """Convierte nombres como v12.8 en una tupla comparable: (12, 8)."""
    try:
        return tuple(
            int(part)
            for part in path.name.lower().removeprefix("v").split(".")
        )
    except ValueError:
        return (0,)


def _find_cuda_bin() -> Path | None:
    """Encuentra automáticamente la instalación CUDA 12 más reciente."""
    cuda_path = os.environ.get("CUDA_PATH")

    if cuda_path:
        configured_bin = Path(cuda_path) / "bin"

        if configured_bin.is_dir():
            return configured_bin

    cuda_root = Path(
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    )

    if not cuda_root.is_dir():
        return None

    installations = [
        directory
        for directory in cuda_root.glob("v12.*")
        if (directory / "bin").is_dir()
    ]

    if not installations:
        return None

    latest_installation = max(
        installations,
        key=_cuda_version_key,
    )

    return latest_installation / "bin"


def get_nvidia_dll_directories() -> list[Path]:
    """Obtiene las carpetas que contienen CUDA, cuBLAS, cuDNN y NVRTC."""
    if os.name != "nt":
        return []

    site_packages = Path(
        sysconfig.get_paths()["purelib"]
    )

    candidates = [
        site_packages / "nvidia" / "cublas" / "bin",
        site_packages / "nvidia" / "cudnn" / "bin",
        site_packages / "nvidia" / "cuda_nvrtc" / "bin",
    ]

    cuda_bin = _find_cuda_bin()

    if cuda_bin is not None:
        candidates.append(cuda_bin)

    existing_directories: list[Path] = []

    for directory in candidates:
        if directory.is_dir() and directory not in existing_directories:
            existing_directories.append(directory)

    return existing_directories


def configure_nvidia_runtime() -> list[Path]:
    """
    Registra las carpetas de DLL de NVIDIA para el proceso actual.

    No modifica permanentemente el PATH de Windows.
    """
    if os.name != "nt":
        return []

    configured_directories: list[Path] = []

    for directory in get_nvidia_dll_directories():
        directory_text = str(directory)

        current_path = os.environ.get("PATH", "")
        path_entries = current_path.split(os.pathsep)

        if directory_text not in path_entries:
            os.environ["PATH"] = (
                f"{directory_text}{os.pathsep}{current_path}"
            )

        handle = os.add_dll_directory(directory_text)
        _DLL_DIRECTORY_HANDLES.append(handle)
        configured_directories.append(directory)

    return configured_directories
