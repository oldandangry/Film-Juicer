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

    @staticmethod
    def highlight_accuracy():
        rows = []
        for variant in scatter.HIGHLIGHT_VARIANTS:
            for padding in (0, 7):
                flags = {
                    "status_pass": True,
                    "maximum_pass": True,
                    "repeat_pass": True,
                    "layout_pass": True,
                    "exact_pass": True,
                    "finite_pass": True,
                    "monotonic_pass": True,
                    "tolerance_pass": True,
                }
                rows.append(
                    {
                        "id": "case",
                        "variant": variant,
                        "padding": padding,
                        "pass": True,
                        **flags,
                        "tolerance_failures": 0,
                        "samples": 8,
                        "max_absolute": 0.0,
                        "max_relative": 0.0,
                        "rms": 0.0,
                        "max_ulp": 0,
                        "worst_sample": None,
                    }
                )
        return {
            "schema": "highlight-precision-accuracy.v1",
            "rows": rows,
            "production_pass": True,
            "experimental_pass": True,
        }

    @staticmethod
    def highlight_timing():
        pairs = []
        for pair_index in range(3):
            order = "reverse" if pair_index == 1 else "forward"
            variants = []
            ordered_variants = (
                scatter.HIGHLIGHT_VARIANTS if order == "forward" else tuple(reversed(scatter.HIGHLIGHT_VARIANTS))
            )
            for variant in ordered_variants:
                stats = {"raw_ms": [float(index + 1) for index in range(30)], "p50_ms": 15.0, "p95_ms": 29.0}
                variants.append(
                    {
                        "variant": variant,
                        "first_sample": {"cuda_ms": 1.0, "host_ms": 2.0, "wait_ms": 1.0},
                        "cuda_stage": stats.copy(),
                        "host_launch_readback_wait": stats.copy(),
                        "wait": stats.copy(),
                        "finite_pass": True,
                        "baseline_pass": True,
                        "baseline_failures": 0,
                        "max_absolute": 0.0,
                        "registers": {"partial": 1, "final": 1, "apply": 1},
                    }
                )
            pairs.append({"pair": pair_index, "order": order, "variants": variants})
        return {
            "schema": "highlight-precision-timing.v1",
            "width": 1920,
            "height": 1080,
            "density": "sparse",
            "warmups": 10,
            "samples": 30,
            "source_algorithm": "fixed test pattern",
            "descriptor": {"recipe_hash": 1, "protection": 2.944, "sharpness": 10.0, "peak_gain": 16.0},
            "device": "test GPU",
            "driver_version": 1,
            "runtime_version": 1,
            "workspace_bytes": 64,
            "production_pass": True,
            "pairs": pairs,
        }

    def test_highlight_run_schema_and_preset(self):
        for preset in ("linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release"):
            self.assertTrue(
                scatter.highlight_run_kind(
                    {"schema": "highlight-production-run.v1", "status": "passed", "preset": preset}
                )
            )
        self.assertFalse(
            scatter.highlight_run_kind(
                {"schema": "highlight-precision-run.v1", "status": "completed", "preset": "linux-release"}
            )
        )
        for result in (
            {"schema": "highlight-production-run.v1", "status": "completed", "preset": "linux-release"},
            {"schema": "highlight-production-run.v1", "status": "passed", "preset": "other"},
            {"schema": "highlight-precision-run.v1", "status": "completed", "preset": "linux-debug"},
            {"schema": "unknown", "status": "passed", "preset": "linux-release"},
        ):
            with self.assertRaises(ValueError):
                scatter.highlight_run_kind(result)

    def test_production_accuracy_membership_and_failure_gates(self):
        report = self.highlight_accuracy()
        report.update(schema="highlight-production-accuracy.v1", rows=report["rows"][:2])
        report.pop("experimental_pass")
        for row in report["rows"]:
            row.update(
                finite_samples=8, nonfinite_count=0, first_failing_sample=None, metrics_scope="finite_samples_only"
            )
        summary = scatter.validate_highlight_accuracy(report, ["case"], production=True)
        self.assertEqual(summary["variant_pass"], {"production": True})
        for mutate in (
            lambda r: r["rows"].pop(),
            lambda r: r["rows"].append(copy.deepcopy(r["rows"][0])),
            lambda r: r["rows"][0].update(variant="fp32_stable"),
            lambda r: r["rows"][0].update(tolerance_failures=1),
            lambda r: r["rows"][0].update(nonfinite_count=1),
            lambda r: r["rows"][0].update(finite_samples=9),
            lambda r: r.update(production_pass=False),
            lambda r: r.update(schema="highlight-precision-accuracy.v1"),
        ):
            broken = copy.deepcopy(report)
            mutate(broken)
            with self.assertRaises(ValueError):
                scatter.validate_highlight_accuracy(broken, ["case"], production=True)
        failed = copy.deepcopy(report)
        failed["rows"][0].update(
            {
                "pass": False,
                "finite_pass": False,
                "nonfinite_count": 1,
                "first_failing_sample": {"actual_bits": 2143289344},
            }
        )
        failed["production_pass"] = False
        with self.assertRaisesRegex(ValueError, "production accuracy"):
            scatter.validate_highlight_accuracy(failed, ["case"], production=True)

    def test_highlight_timing_rejects_missing_pair_and_variant(self):
        for mutate in (
            lambda report: report["pairs"].pop(),
            lambda report: report["pairs"][0]["variants"].pop(),
        ):
            report = self.highlight_timing()
            mutate(report)
            with self.assertRaises(ValueError):
                scatter.validate_highlight_timing(report, 1920, 1080, "sparse")

    def test_highlight_timing_rejects_tampered_percentile(self):
        report = self.highlight_timing()
        report["pairs"][0]["variants"][0]["cuda_stage"]["p95_ms"] = 30.0
        with self.assertRaisesRegex(ValueError, "percentile"):
            scatter.validate_highlight_timing(report, 1920, 1080, "sparse")

    def test_highlight_timing_failure_cannot_be_qualified_by_accuracy_pass(self):
        accuracy = self.highlight_accuracy()
        accuracy_summary = scatter.validate_highlight_accuracy(accuracy, ["case"])
        self.assertTrue(accuracy_summary["variant_pass"]["fp32_direct"])

        timing = self.highlight_timing()
        direct = next(row for row in timing["pairs"][0]["variants"] if row["variant"] == "fp32_direct")
        direct.update(baseline_pass=False, baseline_failures=1)
        timing_summary = scatter.validate_highlight_timing(timing, 1920, 1080, "sparse")
        combined = scatter.combine_highlight_passes(accuracy_summary, [timing_summary])
        self.assertFalse(combined["variant_pass"]["fp32_direct"])
        self.assertFalse(combined["experimental_pass"])
        self.assertTrue(combined["variant_pass"]["fp32_stable"])

    def test_highlight_timing_rejects_contradictory_baseline_flags(self):
        report = self.highlight_timing()
        report["pairs"][0]["variants"][0]["baseline_failures"] = 1
        with self.assertRaisesRegex(ValueError, "disagrees"):
            scatter.validate_highlight_timing(report, 1920, 1080, "sparse")

    def test_highlight_failed_production_baseline_is_fatal(self):
        accuracy = self.highlight_accuracy()
        row = next(row for row in accuracy["rows"] if row["variant"] == "production")
        row["pass"] = row["finite_pass"] = False
        accuracy["production_pass"] = False
        with self.assertRaisesRegex(ValueError, "baseline production"):
            scatter.validate_highlight_accuracy(accuracy, ["case"])

        timing = self.highlight_timing()
        timing["production_pass"] = False
        with self.assertRaisesRegex(ValueError, "production baseline"):
            scatter.validate_highlight_timing(timing, 1920, 1080, "sparse")


if __name__ == "__main__":
    unittest.main()
