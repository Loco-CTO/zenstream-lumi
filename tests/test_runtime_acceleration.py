from __future__ import annotations

import unittest
from types import SimpleNamespace

from lumi.runtime.acceleration import GpuDevice, choose_acceleration, detect_gpu_devices


class _NativeFunction:
    def __init__(self, callback):
        self.callback = callback
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        return self.callback(*args)


class _NativeLibrary:
    def __init__(self) -> None:
        device_types = {1: 0, 2: 1, 3: 2, 4: 1}
        names = {1: b"CPU", 2: b"NVIDIA GeForce", 3: b"AMD Radeon", 4: b"Unknown accelerator"}
        backend_names = {101: b"CPU", 102: b"CUDA", 103: b"Vulkan", 104: b"OpenCL"}
        memory = {
            2: (1_500_000_000, 4_000_000_000),
            3: (3_000_000_000, 8_000_000_000),
            4: (1_000_000_000, 2_000_000_000),
        }

        self.ggml_backend_dev_count = _NativeFunction(lambda: 4)
        self.ggml_backend_dev_get = _NativeFunction(lambda index: index + 1)
        self.ggml_backend_dev_type = _NativeFunction(lambda pointer: device_types[pointer])
        self.ggml_backend_dev_name = _NativeFunction(lambda pointer: names[pointer])
        self.ggml_backend_dev_description = _NativeFunction(lambda pointer: names[pointer])
        self.ggml_backend_dev_backend_reg = _NativeFunction(lambda pointer: pointer + 100)
        self.ggml_backend_reg_name = _NativeFunction(lambda pointer: backend_names[pointer])

        def fill_memory(pointer, free_pointer, total_pointer):
            free_pointer._obj.value, total_pointer._obj.value = memory[pointer]

        self.ggml_backend_dev_memory = _NativeFunction(fill_memory)


class RuntimeAccelerationTests(unittest.TestCase):
    def test_detection_includes_cuda_and_vulkan_gpu_devices_but_skips_cpu_and_unknown(self) -> None:
        devices = detect_gpu_devices(type("Binding", (), {"_lib": _NativeLibrary()})())

        self.assertEqual([device.backend for device in devices], ["cuda", "vulkan"])
        self.assertEqual(devices[0].name, "NVIDIA GeForce")
        self.assertEqual(devices[0].free_vram_bytes, 1_500_000_000)
        self.assertEqual(devices[1].total_vram_bytes, 8_000_000_000)

    def test_detection_supports_binding_split_across_ggml_and_ggml_base(self) -> None:
        native = _NativeLibrary()
        core = SimpleNamespace(
            ggml_backend_dev_count=native.ggml_backend_dev_count,
            ggml_backend_dev_get=native.ggml_backend_dev_get,
            ggml_backend_dev_type=native.ggml_backend_dev_type,
            ggml_backend_dev_memory=native.ggml_backend_dev_memory,
        )
        base = SimpleNamespace(
            ggml_backend_dev_name=native.ggml_backend_dev_name,
            ggml_backend_dev_description=native.ggml_backend_dev_description,
            ggml_backend_dev_backend_reg=native.ggml_backend_dev_backend_reg,
            ggml_backend_reg_name=native.ggml_backend_reg_name,
        )

        devices = detect_gpu_devices(SimpleNamespace(ggml=core, ggml_base=base))

        self.assertEqual([device.backend for device in devices], ["cuda", "vulkan"])

    def test_automatic_prefers_cuda_and_offloads_only_the_safe_vram_budget(self) -> None:
        devices = (
            GpuDevice("vulkan", "AMD Radeon", 7_000_000_000, 8_000_000_000),
            GpuDevice("cuda", "NVIDIA GeForce", 1_500_000_000, 4_000_000_000),
        )

        choice = choose_acceleration(
            "automatic", devices, model_size_bytes=1_400_000_000, total_layers=28
        )

        self.assertEqual(choice.backend, "cuda")
        self.assertGreater(choice.offloaded_layers, 0)
        self.assertLess(choice.offloaded_layers, 28)

    def test_vulkan_is_selected_when_cuda_is_unavailable(self) -> None:
        choice = choose_acceleration(
            "gpu_preferred",
            (GpuDevice("vulkan", "AMD Radeon", 5_000_000_000, 8_000_000_000),),
            model_size_bytes=1_400_000_000,
            total_layers=28,
        )

        self.assertEqual(choice.backend, "vulkan")
        self.assertGreater(choice.offloaded_layers, 0)

    def test_insufficient_vram_selects_cpu(self) -> None:
        choice = choose_acceleration(
            "automatic",
            (GpuDevice("vulkan", "AMD Radeon", 900_000_000, 4_000_000_000),),
            model_size_bytes=1_400_000_000,
            total_layers=28,
        )

        self.assertEqual(choice.backend, "cpu")
        self.assertEqual(choice.offloaded_layers, 0)
        self.assertIsNotNone(choice.fallback_reason)


if __name__ == "__main__":
    unittest.main()
