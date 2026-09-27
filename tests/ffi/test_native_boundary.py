"""Enforce the current native/host and S2.D operation boundaries."""

from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]


class NativeBoundary(unittest.TestCase):
    def test_executor_has_no_host_message_surface(self):
        for relative in (
            "src/Cuda/JuicerCudaExecutor.h",
            "src/Cuda/JuicerCudaExecutor.cpp",
            "native/juicer_cuda_prepared.h",
            "native/juicer_cuda_prepared.cpp",
            "native/juicer_cuda_api.cpp",
        ):
            with self.subTest(path=relative):
                source = (ROOT / relative).read_text(encoding="utf-8")
                self.assertNotRegex(source, r"\b(?:DirFailureMessage|sendMessage)\b|OFX::")

    def test_production_still_uses_direct_cpp_execution(self):
        source = (ROOT / "src/mainProcessing.cpp").read_text(encoding="utf-8")
        self.assertNotRegex(source, r"\bfj_cuda_render\s*\(")
        self.assertRegex(source, r"JuicerCuda::execute_direct\s*\(")
        self.assertRegex(source, r"JuicerCuda::execute_print\s*\(")
        self.assertIn('"FilmJuicerDeferredCudaFailure"', source)

    def test_only_accepted_operations_are_defined(self):
        definitions = []
        for path in (ROOT / "native").glob("*.cpp"):
            definitions.extend(re.findall(r"^FjStatus (fj_cuda_\w+)\([^;]*?\)\s*\{", path.read_text(encoding="utf-8"), re.MULTILINE))
        self.assertCountEqual(definitions, ["fj_cuda_create", "fj_cuda_inspect", "fj_cuda_render"])

    def test_native_code_does_not_reset_host_contexts(self):
        paths = list((ROOT / "native").glob("*.cpp"))
        paths.extend((ROOT / "src/Cuda").rglob("*.cpp"))
        paths.extend((ROOT / "src/Cuda").rglob("*.cu"))
        paths.append(ROOT / "src/ProcessRoot.cpp")
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertNotRegex(path.read_text(encoding="utf-8"), r"\b(?:cudaDeviceReset|cuDevicePrimaryCtxReset|cuCtxReset)\s*\(")


if __name__ == "__main__":
    unittest.main()
