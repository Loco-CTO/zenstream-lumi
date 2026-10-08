"""Install and probe a llama-cpp-python release wheel without network access."""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import venv
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

_LLAMA_CPP_VERSION = "0.3.35"
_GPU_BACKENDS = ("cuda", "vulkan")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheelhouse",
        type=Path,
        action="append",
        help="Local wheel directories used for an isolated offline install; repeat as needed.",
    )
    parser.add_argument(
        "--require-backend",
        choices=_GPU_BACKENDS,
        action="append",
        default=[],
        help=(
            "Require this GPU backend shared library and static registry entry "
            "in the wheel (repeatable)."
        ),
    )
    parser.add_argument(
        "--verify-installed",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    arguments = parser.parse_args()
    if arguments.verify_installed:
        if arguments.wheelhouse:
            parser.error("--verify-installed cannot be combined with --wheelhouse")
    elif not arguments.wheelhouse:
        parser.error("at least one --wheelhouse is required")
    return arguments


def _python_in_venv(environment: Path) -> Path:
    relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    python = environment / relative
    if not python.is_file():
        raise RuntimeError(f"virtual environment interpreter was not created: {python}")
    return python


def _install_offline_and_verify(
    wheelhouses: Sequence[Path], required_backends: Sequence[str]
) -> None:
    resolved_wheelhouses = [path.expanduser().resolve() for path in wheelhouses]
    for wheelhouse in resolved_wheelhouses:
        if not wheelhouse.is_dir():
            raise RuntimeError(f"wheelhouse directory does not exist: {wheelhouse}")
        if not any(wheelhouse.glob("*.whl")):
            raise RuntimeError(f"wheelhouse contains no wheels: {wheelhouse}")

    with tempfile.TemporaryDirectory(prefix="lumi-llama-cpp-wheel-smoke-") as temporary:
        environment = Path(temporary) / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        python = _python_in_venv(environment)
        install = [str(python), "-m", "pip", "install", "--no-index"]
        for wheelhouse in resolved_wheelhouses:
            install.extend(("--find-links", str(wheelhouse)))
        install.append(f"llama-cpp-python=={_LLAMA_CPP_VERSION}")
        print("Installing llama-cpp-python from local wheelhouses into a clean venv")
        subprocess.run(install, check=True, timeout=600)

        verify = [str(python), str(Path(__file__).resolve()), "--verify-installed"]
        for backend in required_backends:
            verify.extend(("--require-backend", backend))
        environment_variables = os.environ.copy()
        environment_variables["PYTHONNOUSERSITE"] = "1"
        subprocess.run(
            verify,
            check=True,
            timeout=120,
            env=environment_variables,
            cwd=Path(__file__).resolve().parents[1],
        )


def _backend_libraries(directory: Path, backend: str) -> list[Path]:
    marker = f"ggml-{backend}".casefold()
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and marker in path.name.casefold()
        and (path.name.casefold().endswith(".dll") or ".so" in path.name.casefold())
    )


def _load_ggml_library(directory: Path) -> ctypes.CDLL:
    if os.name == "nt":
        candidates = [directory / "ggml.dll"]
    else:
        candidates = sorted(directory.glob("libggml.so*"))
    if not candidates:
        raise RuntimeError(f"could not find the ggml core shared library under {directory}")

    errors: list[str] = []
    for candidate in candidates:
        try:
            return ctypes.CDLL(str(candidate))
        except OSError as error:
            errors.append(f"{candidate.name}: {error}")
    raise RuntimeError("unable to load ggml core library: " + "; ".join(errors))


def _backend_registry(
    library: ctypes.CDLL,
    base_library: ctypes.CDLL,
    directory: Path | None = None,
) -> set[str]:
    try:
        load_all = library.ggml_backend_load_all_from_path
        count = library.ggml_backend_reg_count
        get = library.ggml_backend_reg_get
        name = base_library.ggml_backend_reg_name
    except AttributeError as error:
        raise RuntimeError(
            "the packaged ggml core/base libraries do not expose the upstream backend registry API"
        ) from error

    load_all.argtypes = [ctypes.c_char_p]
    load_all.restype = None
    count.argtypes = []
    count.restype = ctypes.c_size_t
    get.argtypes = [ctypes.c_size_t]
    get.restype = ctypes.c_void_p
    name.argtypes = [ctypes.c_void_p]
    name.restype = ctypes.c_char_p

    if directory is not None:
        # Load any package-local dynamic backends before llama_backend_init() so the wrapper
        # observes this same populated registry and does not search the process path.
        load_all(os.fsencode(directory))
    return_value = count()
    backends: set[str] = set()
    for index in range(return_value):
        registry = get(index)
        if not registry:
            raise RuntimeError(f"ggml backend registry returned a null entry at {index}")
        raw_name = name(registry)
        if not raw_name:
            raise RuntimeError(f"ggml backend registry returned an unnamed entry at {index}")
        backends.add(raw_name.decode("utf-8", errors="replace").casefold())
    return backends


def _nvidia_device_probe() -> tuple[bool, str]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return False, "nvidia-smi is unavailable on this runner"
    try:
        result = subprocess.run(
            [executable, "-L"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"nvidia-smi probe failed: {error}"
    devices = [
        line.strip()
        for line in result.stdout.splitlines()
        if re.match(r"\s*GPU\s+\d+\s*:", line, re.IGNORECASE)
    ]
    if result.returncode == 0 and devices:
        return True, ", ".join(devices)
    if result.returncode == 0:
        return False, "nvidia-smi reports no NVIDIA GPU devices"
    detail = result.stderr.strip().splitlines()
    suffix = f": {detail[-1][:240]}" if detail else ""
    return (
        False,
        f"nvidia-smi could not enumerate a usable device "
        f"(exit {result.returncode}){suffix}",
    )


def _vulkan_device_probe() -> tuple[bool, str]:
    executable = shutil.which("vulkaninfo") or shutil.which("vulkaninfo.exe")
    if executable is None:
        return False, "vulkaninfo is unavailable, so this runner's Vulkan device state is unknown"
    try:
        result = subprocess.run(
            [executable, "--summary"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"vulkaninfo probe failed: {error}"
    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0:
        detail = next((line.strip() for line in result.stderr.splitlines() if line.strip()), "")
        suffix = f": {detail[:240]}" if detail else ""
        return False, f"vulkaninfo could not enumerate devices (exit {result.returncode}){suffix}"

    count_match = re.search(r"Devices\s*:\s*count\s*=\s*(\d+)", output, re.IGNORECASE)
    if count_match is not None:
        device_count = int(count_match.group(1))
        return (device_count > 0, f"vulkaninfo reports {device_count} physical device(s)")
    device_names = re.findall(r"deviceName\s*=\s*([^\r\n]+)", output, re.IGNORECASE)
    if device_names:
        return True, ", ".join(name.strip() for name in device_names)
    if re.search(r"\bGPU\s*\d+\s*:", output, re.IGNORECASE):
        return True, "vulkaninfo reports a physical GPU"
    return False, "vulkaninfo reports no physical devices"


def _verify_installed(required_backends: Sequence[str]) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

    import llama_cpp

    from lumi.runtime.acceleration import choose_acceleration, detect_gpu_devices

    if getattr(llama_cpp, "__version__", None) != _LLAMA_CPP_VERSION:
        raise RuntimeError(
            f"expected llama-cpp-python {_LLAMA_CPP_VERSION}, got "
            f"{getattr(llama_cpp, '__version__', 'unknown')}"
        )
    library_directory = Path(llama_cpp.__file__).resolve().parent / "lib"
    if not library_directory.is_dir():
        raise RuntimeError(
            f"llama-cpp-python native library directory is missing: {library_directory}"
        )

    gpu_backend_libraries: dict[str, list[Path]] = {}
    for backend in required_backends:
        libraries = _backend_libraries(library_directory, backend)
        if not libraries:
            raise RuntimeError(
                f"the installed wheel is missing its required "
                f"{backend.upper()} backend library"
            )
        gpu_backend_libraries[backend] = libraries
        print(f"{backend.upper()} backend libraries: {', '.join(p.name for p in libraries)}")

    print(
        "GPU package checks: backend shared libraries and statically linked registry "
        "entries are required on every x64 runner. Device availability checks are "
        "conditional on matching hardware being detected."
    )

    ggml = _load_ggml_library(library_directory)
    from llama_cpp._ctypes_extensions import load_shared_library

    ggml_base = load_shared_library("ggml-base", library_directory)
    _backend_registry(ggml, ggml_base, library_directory)
    llama_cpp.llama_backend_init()
    backends = _backend_registry(ggml, ggml_base)
    if "cpu" not in backends:
        raise RuntimeError(
            f"the initialized backend registry has no CPU backend: {sorted(backends)}"
        )
    for backend in gpu_backend_libraries:
        if backend not in backends:
            raise RuntimeError(
                f"the wheel contains the {backend.upper()} backend library, but its "
                f"statically linked registry is missing {backend.upper()}: {sorted(backends)}"
            )
        print(f"Verified statically linked {backend.upper()} backend registry entry")
    print(f"Initialized llama.cpp backend registry: {', '.join(sorted(backends))}")
    print(f"llama_supports_gpu_offload: {llama_cpp.llama_supports_gpu_offload()}")

    devices = detect_gpu_devices(SimpleNamespace(ggml=ggml, ggml_base=ggml_base))
    print(
        "Detected Lumi GPU devices: "
        + (
            ", ".join(
                f"{device.backend}: {device.name} "
                f"({device.free_vram_bytes}/{device.total_vram_bytes} bytes free/total)"
                for device in devices
            )
            or "none"
        )
    )
    fallback = choose_acceleration("automatic", (), model_size_bytes=1, total_layers=1)
    if fallback.backend != "cpu" or fallback.offloaded_layers != 0:
        raise RuntimeError("automatic mode did not retain CPU fallback with no GPU devices")
    print(f"Verified automatic CPU fallback with no devices: {fallback.fallback_reason}")

    device_probes = {
        "cuda": _nvidia_device_probe,
        "vulkan": _vulkan_device_probe,
    }
    for backend in required_backends:
        has_device, reason = device_probes[backend]()
        if not has_device:
            print(f"SKIP {backend.upper()} device availability check: {reason}")
            continue
        if backend not in backends:
            raise RuntimeError(
                f"{backend.upper()} device was detected ({reason}), but the initialized "
                f"registry does not contain {backend.upper()}: {sorted(backends)}"
            )
        print(f"Verified {backend.upper()} registry on detected device: {reason}")
        if backend == "cuda" and not any(device.backend == "cuda" for device in devices):
            raise RuntimeError("CUDA hardware is present, but Lumi detected no CUDA device")

    print("Installed wheel import and backend package/registry smoke checks passed")


def main() -> None:
    arguments = _arguments()
    if arguments.verify_installed:
        _verify_installed(arguments.require_backend)
    else:
        _install_offline_and_verify(arguments.wheelhouse, arguments.require_backend)


if __name__ == "__main__":
    main()
