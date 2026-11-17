"""Known cases for the maintained G3 offline decision arithmetic."""

import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import grain
import grain_g3 as g3
import grain_g3_analysis as analysis


def invocation(gpu_ms):
    return {
        "samples": [
            {"phase": phase, "index": index, "time": index, "gpu_ms": gpu_ms}
            for phase, count in analysis.PHASE_COUNTS.items()
            for index in range(count)
        ]
    }


def pairs(a_values, b_values):
    return [
        {"order": order, "a": invocation(a), "b": invocation(b)}
        for order, a, b in zip(analysis.PAIR_ORDER, a_values, b_values, strict=True)
    ]


def row(baseline=(100, 100, 100), candidate=(100, 100, 100)):
    result = {"id": "row", "disposition": "qualified", "mandatory": True, "baseline_pairs": pairs((100,) * 3, baseline)}
    if candidate is not None:
        result["candidate_pairs"] = pairs((100,) * 3, candidate)
    return result


class GrainG3AnalysisTests(unittest.TestCase):
    def test_quantiles_use_linear_interpolation(self):
        self.assertEqual(analysis.linear_quantile([4, 1, 3, 2], 0.5), 2.5)
        self.assertEqual(analysis.linear_quantile([0, 10], 0.9), 9)
        for values, q in (([], 0.5), ([1, math.inf], 0.5), ([1], math.nan), ([1], -0.1)):
            with self.assertRaises(ValueError):
                analysis.linear_quantile(values, q)

    def test_pair_orientation_uses_named_a_and_b_not_execution_order(self):
        result = analysis.analyze_row(row(baseline=(102, 98, 101), candidate=(100, 100, 100)))
        decision = result["decision"]
        self.assertAlmostEqual(decision["baseline_ratios"]["0.5"][0], 1.02)
        self.assertAlmostEqual(decision["baseline_ratios"]["0.5"][1], 0.98)
        self.assertEqual(decision["disposition"], "within_budget")

    def test_equal_values_with_distinct_identities_are_valid_and_raw_retained(self):
        source = row()
        result = analysis.analyze_row(source)
        self.assertIs(result["baseline_pairs"], source["baseline_pairs"])
        self.assertEqual(result["decision"]["disposition"], "within_budget")

    def test_complete_sample_membership_rejects_duplicate_or_missing_identity(self):
        for mutate in (
            lambda samples: samples.append(copy.deepcopy(samples[0])),
            lambda samples: samples.pop(),
        ):
            source = row()
            mutate(source["baseline_pairs"][0]["a"]["samples"])
            self.assertEqual(analysis.analyze_row(source)["decision"]["disposition"], "invalid")

    def test_nonfinite_and_nonpositive_decision_times_are_invalid(self):
        for value in (0, -1, math.inf, math.nan):
            source = row()
            source["baseline_pairs"][0]["a"]["samples"][-1]["gpu_ms"] = value
            self.assertEqual(analysis.analyze_row(source)["decision"]["disposition"], "invalid")

    def test_order_and_distinct_invocation_requirements(self):
        source = row()
        source["baseline_pairs"][1]["order"] = "ab"
        self.assertEqual(analysis.analyze_row(source)["decision"]["disposition"], "invalid")
        source = row()
        source["baseline_pairs"][0]["b"] = source["baseline_pairs"][0]["a"]
        self.assertEqual(analysis.analyze_row(source)["decision"]["disposition"], "invalid")

    def test_repeatability_gate_exposes_allowance_and_omits_candidate(self):
        result = analysis.analyze_row(row(baseline=(110, 100, 100), candidate=None))["decision"]
        self.assertEqual(result["disposition"], "unresolved")
        self.assertFalse(result["candidate_admitted"])
        self.assertAlmostEqual(result["allowance_factors"]["0.5"], 1.1)

    def test_candidate_block_is_rejected_when_repeatability_gate_fails(self):
        self.assertEqual(analysis.analyze_row(row(baseline=(110, 100, 100)))["decision"]["disposition"], "invalid")

    def test_budget_boundary_is_inclusive_and_over_budget_is_strict(self):
        at_boundary = analysis.analyze_row(row(candidate=(105, 105, 105)))["decision"]
        above_boundary = analysis.analyze_row(row(candidate=(105.001, 105.001, 105.001)))["decision"]
        self.assertEqual(at_boundary["disposition"], "within_budget")
        self.assertEqual(above_boundary["disposition"], "over_budget")
        self.assertLessEqual(at_boundary["upper_envelopes"]["0.5"], analysis.BUDGET)
        self.assertGreater(above_boundary["lower_envelopes"]["0.5"], analysis.BUDGET)

    def test_unresolved_row_and_all_inventory_dispositions_are_preserved(self):
        unresolved = analysis.analyze_row(row(candidate=(105.0, 110.0, 100.0)))
        self.assertEqual(unresolved["decision"]["disposition"], "unresolved")
        rows = [
            row(),
            {"id": "unmatched", "disposition": "unmatched"},
            {"id": "unrun", "disposition": "unrun", "mandatory": True},
            {"id": "deferred", "disposition": "deferred"},
        ]
        report = analysis.summarize_campaign(rows)
        self.assertEqual(report["disposition_counts"], {"within_budget": 1, "unmatched": 1, "unrun": 1, "deferred": 1})
        self.assertEqual(report["mandatory_selection"], "not_established")

    def test_invalid_and_duplicate_manifest_rows_are_reported_or_rejected(self):
        self.assertEqual(
            analysis.analyze_row({"id": "bad", "disposition": "qualified"})["decision"]["disposition"], "invalid"
        )
        with self.assertRaises(ValueError):
            analysis.summarize_campaign([{"id": "duplicate"}, {"id": "duplicate"}])


class StageReceiptTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.rows = [
            dict(id=preset, preset=preset, workload="portra", role="decision", time_mode="unchanged", offset=0.0)
            for preset in g3.PRESETS
        ]
        carrier = {"file": "input", "metadata": "metadata"}
        for key, hash_key in (("file", "sha256"), ("metadata", "metadata_sha256")):
            path = self.root / carrier[key]
            path.write_text(key, encoding="utf-8")
            carrier[hash_key] = grain.execution.digest(path)
        self.manifest = {
            "schema": "grain-g3-workloads-v1",
            "budget": g3.BUDGET,
            "pair_orders": g3.ORDERS,
            "workloads": {"portra": {"carrier": "portra", "current": {}, "film": {"bank": "bank", "tables": "tables"}}},
            "carriers": {"portra": carrier},
            "g2_inputs": {},
            "coverage": {},
            "rows": self.rows,
        }
        self.manifest_path = self.root / "manifest.json"
        self.manifest_hash = self.write(self.manifest_path, self.manifest)
        expansion = patch.object(g3, "expand_rows", return_value=self.rows)
        expansion.start()
        self.addCleanup(expansion.stop)

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        grain.execution.write_json(path, value)
        return grain.execution.digest(path)

    def receipt(self, row, *, quad=False, legacy_arms=False):
        preset = row["preset"]
        directory = self.root / preset
        path = directory / "result.json"
        settings = {"film_variant": "quad", "film_block": "32x8"} if quad else {}
        build = f"out/build/{preset}/tests"
        binary = f"{build}/bin/JuicerGrainPrototype" + (".exe" if preset.startswith("windows") else "")
        success = {"returncode": 0, "drained": True, "process_status": "completed"}
        check = {
            "schema": "grain-g3-native-v1",
            "mode": "g3-check",
            "status": "passed",
            "preset": preset,
            "manifest_sha256": self.manifest_hash,
            "binary_sha256": ("a" if preset == "linux-release" else "b") * 64,
            "commands": [success],
            "checks": {str(i): {} for i in range(26)},
            **settings,
        }
        result = {
            **check,
            "mode": "g3",
            "manifest": "manifest.json",
            "binary": binary,
            "commands": [],
            "check_result": f"{preset}/check.json",
            "check_sha256": self.write(directory / "check.json", check),
            "rows": {},
            "controls": {},
        }
        if quad:
            request = g3.stage_request(self.root, self.manifest, row, "film", "verify") | settings
            observations = [
                {
                    "identity": i,
                    "order": order,
                    "time": i // 2 - 10 + i % 2 * 0.5,
                    "bit_mismatches": 0,
                    "nonfinite": 0,
                    "first_mismatch": None,
                    "replay_hash_mismatch": False,
                    "changed": [1, 1, 1],
                }
                for order, identities in (("forward", range(80)), ("reverse", range(79, -1, -1)))
                for i in identities
            ]
            raw = {
                "request": request,
                "observations": observations,
                "frame_identities": 80,
                "bit_mismatches": 0,
                "nonfinite": 0,
                "inactive_frames": 0,
                "replay_hash_mismatches": 0,
            }
            agreement = {
                **check,
                "schema": "grain-optimization-v1",
                "mode": "g3-verify-opt",
                "variant": "quad",
                "block": "32x8",
                "observations": {
                    row["id"]: {
                        "report": "verification.json",
                        "variant": "quad",
                        "block": "32x8",
                        "sha256": self.write(directory / "verification.json", raw),
                    }
                },
            }
            result.update(
                identities_verified=True,
                agreement_result=f"{preset}/agreement.json",
                agreement_sha256=self.write(directory / "agreement.json", agreement),
            )
        measured = dict(row)
        for block in ("baseline", "candidate"):
            measured[f"{block}_pairs"] = []
            for index, order in enumerate(g3.ORDERS, 1):
                pair = {"order": order.lower()}
                for arm in order.lower():
                    identity = f"{block}-{index}-{arm}"
                    engine = "film" if block == "candidate" and arm == "b" else "current"
                    name = f"{row['id']}-{identity}"
                    request = g3.stage_request(self.root, self.manifest, row, engine, identity)
                    if engine == "film" and not legacy_arms:
                        request.update(settings)
                    raw = {
                        **invocation(10),
                        "request": request,
                        "changed": [1, 1, 1],
                        "replay_bit_mismatches": 0,
                        **dict.fromkeys(
                            (
                                "cold_setup_ms",
                                "cold_first_completed_ms",
                                "input_read_ms",
                                "bank_read_ms",
                                "table_read_ms",
                                "bank_upload_ms",
                                "carrier_allocation_upload_ms",
                                "input_bytes",
                                "carrier_bytes",
                                "host_input_bytes",
                                "device_bank_bytes",
                                "film_table_bytes",
                                "current_workspace_bytes",
                            ),
                            0,
                        ),
                    }
                    for sample in raw["samples"]:
                        sample.update(
                            time=0,
                            resolved={"frame_index": 0, "fraction": 0},
                            host_prepare_ms=0,
                            host_submit_ms=0,
                            host_wait_ms=0,
                            reset_ms=0,
                        )
                    if quad and engine == "film" and not legacy_arms:
                        raw.update(
                            device_quad_bytes=53477376,
                            device_sat_bytes=53477376,
                            quad_host_bytes=53477376,
                            quad_pack_source_and_staging_bytes=66951372,
                            quad_pack_ms=10,
                            quad_upload_ms=5,
                        )
                    report = f"{name}/stage.json"
                    pair[arm] = {
                        **raw,
                        "invocation": identity,
                        "engine": engine,
                        "report": report,
                        "sha256": self.write(directory / report, raw),
                    }
                    request_path = directory / name / "request.json"
                    self.write(request_path, request)
                    result["commands"].append(
                        {
                            **success,
                            "name": name,
                            "command": [
                                str(self.root / binary),
                                "stage",
                                str(self.root / build / "Resources"),
                                str(request_path),
                            ],
                        }
                    )
                measured[f"{block}_pairs"].append(pair)
        measured = analysis.analyze_row(measured)
        report = f"{row['id']}.json"
        result["rows"][row["id"]] = {
            "report": report,
            "sha256": self.write(directory / report, measured),
            "disposition": measured["decision"]["disposition"],
        }
        self.write(path, result)
        return path, result

    def read(self, *paths):
        output = self.root / "output.json"
        output.unlink(missing_ok=True)
        g3.offline_report(self.manifest_path, paths, [], output, root=self.root)
        return g3.read_json(output)

    def test_legacy_reference_and_failed_parent(self):
        path, parent = self.receipt(self.rows[0])
        report = self.read(path)
        reference = report["rows"][0]["stage"]["candidate_pairs"][0]["b"]
        self.assertEqual(reference["film_variant"], "reference")
        self.assertNotIn("device_quad_bytes", reference)
        self.assertEqual(report["counts"], {"within_budget": 1, "unrun": 1})
        parent.update(status="failed", error="input drift")
        self.write(path, parent)
        report = self.read(path)
        self.assertEqual(report["counts"], {"invalid": 1, "unrun": 1})
        self.assertEqual(report["rows"][0]["diagnostic_disposition"], "within_budget")
        self.assertNotIn("stage", report["rows"][0])
        self.assertFalse(report["stage_results"][0]["admitted"])
        self.assertEqual(report["stage_results"][0]["error"], "input drift")

    def test_positive_quad_and_incompatible_presets(self):
        path, _ = self.receipt(self.rows[0], quad=True)
        report = self.read(path)
        arm = report["rows"][0]["stage"]["candidate_pairs"][0]["b"]
        self.assertEqual(arm["film_variant"], "quad")
        self.assertEqual(arm["device_bank_bytes"], 0)
        self.assertEqual(arm["device_sat_bytes"], 53477376)
        self.assertEqual(arm["quad_pack_source_and_staging_bytes"], 66951372)
        self.assertEqual(arm["quad_upload_ms"], 5)
        other, _ = self.receipt(self.rows[1])
        with self.assertRaisesRegex(ValueError, "incompatible Film selections"):
            self.read(path, other)

    def test_quad_parent_cannot_relabel_reference_arms(self):
        path, _ = self.receipt(self.rows[0], quad=True, legacy_arms=True)
        with self.assertRaisesRegex(ValueError, "Film arm selection differs"):
            self.read(path)

    def test_missing_or_mismatched_agreement(self):
        path, parent = self.receipt(self.rows[0], quad=True)
        agree = self.root / parent.pop("agreement_result")
        self.write(path, parent)
        with self.assertRaisesRegex(ValueError, "agreement missing"):
            self.read(path)
        agreement = g3.read_json(agree)
        agreement["commands"][0]["process_status"] = "incomplete"
        parent.update(
            agreement_result=agree.relative_to(self.root).as_posix(), agreement_sha256=self.write(agree, agreement)
        )
        self.write(path, parent)
        with self.assertRaisesRegex(ValueError, "agreement did not complete"):
            self.read(path)
        agreement["commands"][0]["process_status"] = "completed"
        agreement["binary_sha256"] = "c" * 64
        parent.update(
            agreement_result=agree.relative_to(self.root).as_posix(), agreement_sha256=self.write(agree, agreement)
        )
        self.write(path, parent)
        with self.assertRaisesRegex(ValueError, "agreement receipt does not match"):
            self.read(path)

    def test_reused_arms_and_reordered_processes(self):
        path, parent = self.receipt(self.rows[0])
        member = parent["rows"][self.rows[0]["id"]]
        row_path = path.parent / member["report"]
        measured = g3.read_json(row_path)
        original = copy.deepcopy(measured)
        measured["baseline_pairs"][0]["b"] = copy.deepcopy(measured["baseline_pairs"][0]["a"])
        member["sha256"] = self.write(row_path, measured)
        self.write(path, parent)
        with self.assertRaisesRegex(ValueError, "wrong invocation path"):
            self.read(path)
        member["sha256"] = self.write(row_path, original)
        parent["commands"][0], parent["commands"][1] = parent["commands"][1], parent["commands"][0]
        self.write(path, parent)
        with self.assertRaisesRegex(ValueError, "reordered"):
            self.read(path)


if __name__ == "__main__":
    unittest.main()
