"""Device selection and conservative GGUF layer offload policy."""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from typing import Literal

AccelerationMode = Literal["automatic", "cpu_only", "gpu_preferred"]
SUPPORTED_ACCELERATION_MODES = frozenset({"automatic", "cpu_only", "gpu_preferred"})
SUPPORTED_GPU_BACKENDS = frozenset({"cuda", "vulkan", "hip", "metal"})


@dataclass(frozen=True, slots=True)
class GpuDevice:
    """A device reported by an in-process llama.cpp backend."""

    backend: str
    name: str
    free_vram_bytes: int
    total_vram_bytes: int
    native_device_name: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in SUPPORTED_GPU_BACKENDS:
            raise ValueError("Unsupported GPU backend")
        if not self.name or len(self.name) > 160:
            raise ValueError("GPU device name is invalid")
        if self.native_device_name is not None and (
            not isinstance(self.native_device_name, str)
            or not self.native_device_name.strip()
            or len(self.native_device_name) > 160
        ):
            raise ValueError("Native GPU device name is invalid")
        if (
            isinstance(self.free_vram_bytes, bool)
            or isinstance(self.total_vram_bytes, bool)
            or not isinstance(self.free_vram_bytes, int)
            or not isinstance(self.total_vram_bytes, int)
            or self.free_vram_bytes < 0
            or self.total_vram_bytes < 0
        ):
            raise ValueError("GPU memory values must be nonnegative byte counts")


@dataclass(frozen=True, slots=True)
class AccelerationChoice:
    """The safe backend and layer count to try for one GGUF model."""

    backend: str
    device: str
    offloaded_layers: int
    total_layers: int
    device_memory_free_bytes: int
    device_memory_total_bytes: int
    fallback_reason: str | None = None
    native_device_name: str | None = None

    @property
    def uses_gpu(self) -> bool:
        return self.offloaded_layers > 0


def choose_acceleration(
    mode: AccelerationMode,
    devices: tuple[GpuDevice, ...] | list[GpuDevice],
    *,
    model_size_bytes: int,
    total_layers: int,
    memory_reserve_bytes: int = 1_073_741_824,
    max_layers: int | None = None,
) -> AccelerationChoice:
    """Select a preferred backend and budget partial offload from available VRAM.

    CUDA is preferred first, then HIP for AMD and Vulkan as the cross-vendor
    fallback. A fixed reserve plus 15% of reported device memory is kept free for
    context/KV buffers, command buffers, and other processes.
    """

    if mode not in SUPPORTED_ACCELERATION_MODES:
        raise ValueError("Unsupported acceleration mode")
    for label, value in (
        ("model_size_bytes", model_size_bytes),
        ("total_layers", total_layers),
        ("memory_reserve_bytes", memory_reserve_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a nonnegative integer")
    if max_layers is not None and (
        isinstance(max_layers, bool) or not isinstance(max_layers, int) or max_layers < 0
    ):
        raise ValueError("max_layers must be a nonnegative integer")

    if mode == "cpu_only":
        return _cpu_choice(total_layers, None)

    candidates = _preferred_devices(devices)
    if not candidates:
        return _cpu_choice(
            total_layers,
            "No supported GPU backend or device with a native identifier is available.",
        )
    if total_layers < 1 or model_size_bytes < 1:
        return _cpu_choice(total_layers, "The model layer layout could not be measured safely.")

    estimated_bytes_per_layer = max(1, math.ceil(model_size_bytes / total_layers * 1.15))
    best_candidate: tuple[GpuDevice, int] | None = None
    for selected in candidates:
        reported_memory = min(selected.free_vram_bytes, selected.total_vram_bytes)
        if reported_memory <= 0:
            continue

        # Use a reserve proportional to larger cards while retaining at least 1 GiB.
        reserve = max(memory_reserve_bytes, math.ceil(selected.total_vram_bytes * 0.15))
        offload_budget = max(0, reported_memory - reserve)
        safe_layers = offload_budget // estimated_bytes_per_layer
        if max_layers is not None:
            safe_layers = min(safe_layers, max_layers)
        safe_layers = min(total_layers, safe_layers)
        if safe_layers > 0:
            return AccelerationChoice(
                backend=selected.backend,
                device=selected.name,
                offloaded_layers=int(safe_layers),
                total_layers=total_layers,
                device_memory_free_bytes=selected.free_vram_bytes,
                device_memory_total_bytes=selected.total_vram_bytes,
                native_device_name=selected.native_device_name or selected.name,
            )
        if best_candidate is None or selected.free_vram_bytes > best_candidate[0].free_vram_bytes:
            best_candidate = (selected, int(safe_layers))

    return _cpu_choice(
        total_layers,
        "Available GPU memory is below the safe offload threshold.",
        best_candidate[0] if best_candidate is not None else candidates[0],
    )


def detect_gpu_devices(binding: object) -> tuple[GpuDevice, ...]:
    """Read GPU devices and memory from llama.cpp's in-process ggml backend API.

    This intentionally treats a missing symbol or failing backend probe as no
    accelerated device. CPU-only llama-cpp-python builds therefore remain fully
    usable without loading a second runtime or starting a helper process.
    """

    core_library = getattr(binding, "ggml", None)
    base_library = getattr(binding, "ggml_base", None)
    library = getattr(binding, "_lib", None)
    if library is None:
        nested = getattr(binding, "llama_cpp", None)
        library = getattr(nested, "_lib", None)
    libraries = tuple(
        candidate
        for candidate in (core_library, base_library, library)
        if candidate is not None
    )
    if not libraries:
        return ()

    def native_function(name: str) -> object:
        for candidate in libraries:
            try:
                return getattr(candidate, name)
            except AttributeError:
                continue
        raise AttributeError(name)

    try:
        count_fn = native_function("ggml_backend_dev_count")
        get_fn = native_function("ggml_backend_dev_get")
        device_type_fn = native_function("ggml_backend_dev_type")
        name_fn = native_function("ggml_backend_dev_name")
        description_fn = native_function("ggml_backend_dev_description")
        memory_fn = native_function("ggml_backend_dev_memory")
        backend_fn = native_function("ggml_backend_dev_backend_reg")
        backend_name_fn = native_function("ggml_backend_reg_name")
        count_fn.restype = ctypes.c_size_t
        count_fn.argtypes = []
        get_fn.restype = ctypes.c_void_p
        get_fn.argtypes = [ctypes.c_size_t]
        device_type_fn.restype = ctypes.c_int
        device_type_fn.argtypes = [ctypes.c_void_p]
        name_fn.restype = ctypes.c_char_p
        name_fn.argtypes = [ctypes.c_void_p]
        description_fn.restype = ctypes.c_char_p
        description_fn.argtypes = [ctypes.c_void_p]
        memory_fn.restype = None
        memory_fn.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        backend_fn.restype = ctypes.c_void_p
        backend_fn.argtypes = [ctypes.c_void_p]
        backend_name_fn.restype = ctypes.c_char_p
        backend_name_fn.argtypes = [ctypes.c_void_p]
        device_count = min(int(count_fn()), 64)
    except (AttributeError, OSError, TypeError, ValueError):
        return ()

    devices: list[GpuDevice] = []
    for index in range(device_count):
        try:
            pointer = get_fn(index)
            # ggml-backend.h defines CPU=0, GPU=1, IGPU=2. Both GPU classes can
            # execute layers; IGPU devices with shared memory usually report 0/0
            # and will fail the conservative VRAM budget below.
            if not pointer or int(device_type_fn(pointer)) not in {1, 2}:
                continue
            backend_pointer = backend_fn(pointer)
            backend_name = _decode_backend_string(backend_name_fn(backend_pointer))
            name = _decode_backend_string(name_fn(pointer))
            description = _decode_backend_string(description_fn(pointer))
            backend = _normalize_backend(backend_name, name, description)
            if backend is None:
                continue
            free_memory = ctypes.c_size_t()
            total_memory = ctypes.c_size_t()
            memory_fn(pointer, ctypes.byref(free_memory), ctypes.byref(total_memory))
            devices.append(
                GpuDevice(
                    backend=backend,
                    name=(description or name)[:160],
                    free_vram_bytes=int(free_memory.value),
                    total_vram_bytes=int(total_memory.value),
                    native_device_name=name,
                )
            )
        except (OSError, TypeError, ValueError):
            continue
    return tuple(devices)


def _decode_backend_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _normalize_backend(backend: str, name: str, description: str) -> str | None:
    identity = f"{backend} {name} {description}".casefold()
    if "cuda" in identity:
        return "cuda"
    if "hip" in identity or "rocm" in identity:
        return "hip"
    if "vulkan" in identity:
        return "vulkan"
    if "metal" in identity or "apple" in identity:
        return "metal"
    return None


def _preferred_devices(devices: tuple[GpuDevice, ...] | list[GpuDevice]) -> list[GpuDevice]:
    def preference(device: GpuDevice) -> tuple[int, int]:
        name = device.name.casefold()
        if device.backend == "cuda" or "nvidia" in name or "geforce" in name:
            tier = 0 if device.backend == "cuda" else 2
        elif device.backend == "hip" or "radeon" in name or "amd" in name:
            tier = 1 if device.backend == "hip" else 3
        elif device.backend == "metal":
            tier = 0
        else:
            tier = 4 if device.backend == "vulkan" else 5
        return tier, -device.free_vram_bytes

    # The display name is descriptive only (for example, "NVIDIA GeForce").
    # llama.cpp requires its exact backend identifier (such as "CUDA0") to pin
    # model placement, so devices without that identifier cannot be selected.
    return sorted(
        (device for device in devices if device.native_device_name), key=preference
    )


def _cpu_choice(
    total_layers: int,
    fallback_reason: str | None,
    device: GpuDevice | None = None,
) -> AccelerationChoice:
    return AccelerationChoice(
        backend="cpu",
        device="CPU",
        offloaded_layers=0,
        total_layers=total_layers,
        device_memory_free_bytes=device.free_vram_bytes if device else 0,
        device_memory_total_bytes=device.total_vram_bytes if device else 0,
        fallback_reason=fallback_reason,
        native_device_name="CPU",
    )
