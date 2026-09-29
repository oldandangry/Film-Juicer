"""Enforce the current native/host and production operation boundaries."""

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

    def test_production_uses_one_admitted_c_record_render_body(self):
        source = (ROOT / "src/mainProcessing.cpp").read_text(encoding="utf-8")
        self.assertNotRegex(source, r"JuicerCuda::execute_(?:direct|print)\s*\(")
        self.assertNotIn("cudaStreamSynchronize", source)
        self.assertEqual(source.count("JuicerCuda::project_and_render("), 1)
        self.assertIn("FJ_RENDER_DEFERRED_DIR_ERROR", source)
        self.assertNotIn('find("component=dir")', source)
        self.assertIn('"FilmJuicerDeferredCudaFailure"', source)
        projection = (ROOT / "src/CudaRenderProjection.cpp").read_text(encoding="utf-8")
        self.assertEqual(projection.count("call.render("), 1)
        self.assertNotRegex(projection, r'#include\s+".*(?:tests/|prepared_boundary|prepared_descriptors\.h)')
        boundary = (ROOT / "native/juicer_cuda_api.cpp").read_text(encoding="utf-8")
        self.assertEqual(boundary.count("JuicerCuda::execute_prepared_host_data("), 1)
        self.assertEqual(boundary.count("FjRenderOutcome JuicerCuda::NativeCall::render("), 1)
        self.assertIn("call.render(context, frame, submission, prepared", boundary)

    def test_only_accepted_operations_are_defined(self):
        definitions = []
        for path in (ROOT / "native").glob("*.cpp"):
            definitions.extend(re.findall(r"^(?:FjStatus|FjRenderOutcome) (fj_cuda_\w+)\([^;]*?\)\s*\{", path.read_text(encoding="utf-8"), re.MULTILINE))
        self.assertCountEqual(definitions, ["fj_cuda_create", "fj_cuda_inspect", "fj_cuda_render", "fj_cuda_retire_instance", "fj_cuda_shutdown", "fj_cuda_destroy"])

    def test_resource_destruction_does_not_reenter_root(self):
        source = (ROOT / "src/Cuda/JuicerCudaResources.cpp").read_text(encoding="utf-8")
        self.assertNotIn("ProcessRoot.h", source)
        self.assertNotRegex(source, r"JuicerProcess::|retire_idle_context")
        source = (ROOT / "native/juicer_cuda_api.cpp").read_text(encoding="utf-8")
        self.assertNotIn("SF_TEMP_BRIDGE_release_cuda_owner", source)
        self.assertNotIn("shutdown_if_initialized", source)

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
