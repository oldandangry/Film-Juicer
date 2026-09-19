from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
TOOL = ROOT / "tests/diffusion/reference"
BUNDLE = ROOT / ".tmp/diffusion/cuda-convolution-fixtures"
MANIFEST = BUNDLE / "manifest.json"
HASHES = BUNDLE / "bundle.sha256.json"

CANDIDATES = (
    ("1024x1024", 1024, 1024, 128, 773, 771),
    ("2048x2048", 2048, 2048, 512, 1027, 1031),
    ("3072x3072", 3072, 3072, 1024, 2051, 2053),
    ("4096x4096", 4096, 4096, 1024, 2051, 2053),
    ("4608x4608", 4608, 4608, 2159, 4320, 4322),
    ("5120x5120", 5120, 5120, 2159, 4320, 4322),
    ("6144x6144", 6144, 6144, 2159, 4320, 4322),
    ("8192x6144", 8192, 6144, 2159, 4320, 4322),
    ("6144x8192", 6144, 8192, 2159, 4320, 4322),
    ("8192x8192", 8192, 8192, 2159, 4320, 4322),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CudaConvolutionFixtureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.hashes = json.loads(HASHES.read_text(encoding="utf-8"))

    def test_closed_identity_and_reference_authority(self) -> None:
        self.assertEqual(
            self.data["schema"], "film-juicer.diffusion-cuda-convolution.v1"
        )
        self.assertEqual(self.data["semantic_channel_order"], ["R", "G", "B"])
        self.assertEqual(
            self.data["family_order"],
            ["glimmerglass", "black_pro_mist", "pro_mist", "cinebloom"],
        )
        self.assertEqual(
            self.data["advanced_classes"],
            [
                "default",
                "proportional",
                "all_zero_nondefault_sizes",
                "runtime_outside_widget",
            ],
        )
        source = ROOT / self.data["reference_source"]["path"]
        self.assertEqual(
            self.data["reference_source"]["sha256"],
            "640006bef2fc89d4861f69add2a5c388658b0dde1fc694ac75d0e7a9c2b13d88",
        )
        self.assertEqual(_sha256(source), self.data["reference_source"]["sha256"])
        self.assertTrue(
            Path(self.data["reference_source"]["import_origin"]).resolve().is_relative_to(
                (ROOT / "external/spektrafilm").resolve()
            )
        )

    def test_stage_family_control_and_extent_coverage(self) -> None:
        stages = {
            stage["label"]
            for case in self.data["cases"]
            for stage in case["stages"]
        }
        self.assertEqual(stages, {"camera", "enlarger"})
        self.assertTrue(
            any(
                [stage["label"] for stage in case["stages"]]
                == ["camera", "enlarger"]
                for case in self.data["cases"]
            )
        )
        self.assertEqual(
            {stage["params"]["filter_family"] for case in self.data["cases"] for stage in case["stages"]},
            set(self.data["family_order"]),
        )
        self.assertEqual(
            {case["advanced_class"] for case in self.data["cases"]},
            set(self.data["advanced_classes"]),
        )
        extents = {(case["height"], case["width"]) for case in self.data["cases"]}
        self.assertTrue(any(height % 2 == 0 and width % 2 == 0 for height, width in extents))
        self.assertTrue(any(height % 2 == 1 and width % 2 == 1 for height, width in extents))
        self.assertTrue(any(height > width for height, width in extents))
        self.assertTrue(any(width > height for height, width in extents))

    def test_binary_arrays_are_finite_little_endian_float32_in_range(self) -> None:
        import numpy as np

        records = {row["id"]: row for row in self.data["arrays"]}
        for case in self.data["cases"]:
            for role in ("normalized_input", "normalized_output", "hdr_input", "hdr_output"):
                row = records[case[role]]
                self.assertEqual(row["dtype"], "<f4")
                path = BUNDLE / row["path"]
                values = np.memmap(path, dtype="<f4", mode="r", shape=tuple(row["shape"]))
                self.assertTrue(np.isfinite(values).all(), row["id"])
                if role == "normalized_input":
                    self.assertGreaterEqual(float(values.min()), 0.0)
                    self.assertLessEqual(float(values.max()), 1.0)
                if role == "hdr_input":
                    self.assertGreaterEqual(float(values.min()), 0.0)
                    self.assertLessEqual(float(values.max()), 16.0)
            for stage in case["stages"]:
                psf = records[stage["psf"]]
                self.assertEqual(psf["dtype"], "<f4")
                self.assertEqual(psf["shape"][2], 3)

    def test_every_candidate_has_exact_domain_and_seam_coverage(self) -> None:
        rows = {row["id"]: row for row in self.data["candidates"]}
        self.assertEqual(set(rows), {row[0] for row in CANDIDATES})
        for candidate_id, width, height, radius, frame_h, frame_w in CANDIDATES:
            row = rows[candidate_id]
            self.assertEqual(
                (row["width"], row["height"], row["radius"], row["frame_height"], row["frame_width"]),
                (width, height, radius, frame_h, frame_w),
            )
            valid_w = width - 2 * radius
            valid_h = height - 2 * radius
            self.assertEqual((row["valid_tile_width"], row["valid_tile_height"]), (valid_w, valid_h))
            self.assertGreater(math.ceil(frame_w / valid_w), 1)
            self.assertGreater(math.ceil(frame_h / valid_h), 1)
            self.assertNotEqual(frame_w % valid_w, 0)
            self.assertNotEqual(frame_h % valid_h, 0)
            self.assertEqual(row["seams"], {"horizontal": True, "vertical": True, "partial_bottom_right": True})

    def test_manifest_generator_and_every_binary_are_hashed(self) -> None:
        self.assertEqual(
            self.hashes["schema"],
            "film-juicer.diffusion-cuda-convolution-hashes.v1",
        )
        expected = {
            "manifest.json",
            "tests/diffusion/reference/generate_cuda_convolution_cases.py",
        }
        expected.update(row["path"] for row in self.data["arrays"])
        rows = {row["path"]: row["sha256"] for row in self.hashes["files"]}
        self.assertEqual(set(rows), expected)
        for relative, digest in rows.items():
            path = ROOT / relative if relative.startswith("tools/") else BUNDLE / relative
            self.assertEqual(_sha256(path), digest, relative)


if __name__ == "__main__":
    unittest.main()
