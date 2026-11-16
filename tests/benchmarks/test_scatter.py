"""Offline membership and integrity checks for scatter precision reports."""

import copy
import unittest

import scatter


class ScatterReportTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "rows": [
                {
                    "id": "row",
                    "pixel_size_um": 1.0,
                    "parameters": {"scatter_amount": 1.0, "halation_amount": 0.0},
                    "input": {"sha256": "a" * 64},
                    "post_reflection": {"sha256": "b" * 64},
                }
            ]
        }
        self.accuracy = {
            "schema": "scatter-halation-precision-accuracy.v1",
            "existing_gate_pass": True,
            "records": [
                {
                    "id": "row",
                    "stage": "post_reflection",
                    "variant": variant,
                    "padding": padding,
                    "channels": [{"count": 1}] * 3,
                    "input_sha256": "c" * 64,
                    "output_sha256": "d" * 64,
                    "reference_input_sha256": "a" * 64,
                    "reference_output_sha256": "b" * 64,
                    "fixture_gate_pass": True,
                    "existing_ordinary_equal": True,
                }
                for variant in scatter.VARIANTS
                for padding in (0, 7)
            ],
        }
        self.timing = {
            "schema": "scatter-halation-precision-timing.v1",
            "mode": "precision-timing",
            "width": 1920,
            "height": 1080,
            "warmupCount": 10,
            "precision_sample_count": 30,
            "admittedBytes": 1,
            "sourceChecksum": "e" * 64,
            "descriptorRecipeHash": 1,
            "host_factor_derivation": "included",
            "gpu_event_interval": "includes gaps",
            "precision_rows": [
                {
                    "pair": pair,
                    "order": position,
                    "variant": variant,
                    "stage_samples_ms": [1.0] * 30,
                    "host_samples_ms": [2.0] * 30,
                    "cold_stage_ms": 1.0,
                    "cold_host_ms": 2.0,
                    "stage_p50_ms": 1.0,
                    "stage_p95_ms": 1.0,
                    "host_p50_ms": 2.0,
                    "host_p95_ms": 2.0,
                    "registers": {"horizontal": 1, "vertical_accumulation": 1, "vertical_core": 1},
                }
                for pair in range(3)
                for position, variant in enumerate(
                    scatter.VARIANTS if pair % 2 == 0 else tuple(reversed(scatter.VARIANTS))
                )
            ],
        }

    def test_variant_gate_failure_is_characterization(self):
        self.accuracy["records"][2]["fixture_gate_pass"] = False
        self.assertEqual(scatter.validate_accuracy(self.accuracy, self.manifest)["fp32_direct"], 1)

    def test_accuracy_rejects_missing_and_tampered_inputs(self):
        for mutate in (
            lambda r: r["records"].pop(),
            lambda r: r["records"][0].update(reference_input_sha256="f" * 64),
            lambda r: r["records"][0].update(reference_output_sha256="f" * 64),
            lambda r: r["records"][0].update(input_sha256="f" * 64),
            lambda r: r["records"][1].update(output_sha256="f" * 64),
        ):
            report = copy.deepcopy(self.accuracy)
            mutate(report)
            with self.assertRaises(ValueError):
                scatter.validate_accuracy(report, self.manifest)

    def test_timing_rejects_missing_and_wrong_order(self):
        scatter.validate_timing(self.timing, 1920, 1080)
        for mutate in (
            lambda r: r["precision_rows"].pop(),
            lambda r: r["precision_rows"][0].update(order=2),
            lambda r: r["precision_rows"][0].update(stage_p95_ms=3.0),
            lambda r: r["precision_rows"][0].update(host_p50_ms=3.0),
            lambda r: r["precision_rows"][0].update(cold_host_ms=float("nan")),
            lambda r: r["precision_rows"][0].update(registers={}),
            lambda r: r.update(sourceChecksum="bad"),
        ):
            report = copy.deepcopy(self.timing)
            mutate(report)
            with self.assertRaises(ValueError):
                scatter.validate_timing(report, 1920, 1080)


if __name__ == "__main__":
    unittest.main()
