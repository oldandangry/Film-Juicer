"""Regression checks for optimization evidence admission and arm identity."""

import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import execution
import grain
import grain_g3 as g3
import grain_optimization as optimization


class GrainOptimizationTests(unittest.TestCase):
    def test_quad_copy_requires_complete_exact_word_representation(self):
        with tempfile.TemporaryDirectory() as temporary:
            bank = Path(temporary).resolve()
            path = bank / "copy.json"
            report = {
                "bank": str(bank),
                "quad_count": 3342336,
                "checked_words": 13369344,
                "bit_mismatches": 0,
                "device_quad_bytes": 53477376,
            }
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(grain.quad_copy_report(path, bank), report)
            for key in ("quad_count", "checked_words", "bit_mismatches", "device_quad_bytes"):
                changed = report | {key: report[key] + 1}
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    grain.quad_copy_report(path, bank)
            path.write_text(json.dumps(report | {"bank": str(bank / "other")}), encoding="utf-8")
            with self.assertRaises(ValueError):
                grain.quad_copy_report(path, bank)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        workloads = {}
        for name, (_, routes) in g3.QUALIFIED.items():
            for route in routes:
                workloads[f"{name}-{g3.ROUTES[route]}"] = {
                    "current": {"stock": name, "route": route, "width": 8, "height": 8},
                    "film": {"bank": "bank", "tables": "tables"},
                    "carrier": "carrier",
                }
        self.manifest = {"workloads": workloads, "carriers": {"carrier": {"file": "input.f32"}}}
        self.manifest["rows"] = [
            {
                "id": f"row-{index}-{offset}",
                "preset": "linux-release",
                "workload": name,
                "role": "decision",
                "time_mode": "unchanged",
                "offset": offset,
            }
            for index, name in enumerate(workloads)
            for offset in (0.0, 0.5)
        ]

    def member(self, row, invocation, variant, time):
        request = g3.stage_request(self.root, self.manifest, row, "film", invocation)
        request.update(film_variant=variant, film_block="32x4")
        directory = self.root / f"{row['id']}-{invocation}"
        directory.mkdir(exist_ok=True)
        path = directory / "stage.json"
        raw = {
            "request": request,
            "changed": [1, 1, 1],
            "replay_bit_mismatches": 0,
            "samples": [
                {
                    "phase": "warmup" if index < 10 else "measurement",
                    "index": index if index < 10 else index - 10,
                    "time": row["offset"],
                    "gpu_ms": time,
                    "resolved": {"frame_index": 0, "fraction": row["offset"]},
                }
                for index in range(40)
            ],
        }
        path.write_text(json.dumps(raw), encoding="utf-8")
        return {
            "report": path.relative_to(self.root).as_posix(),
            "sha256": execution.digest(path),
            "variant": variant,
            "block": "32x4",
        }

    def screen(self):
        result = {
            "schema": "grain-optimization-v1",
            "mode": "g3-screen",
            "status": "passed",
            "preset": "linux-release",
            "variant": "scalar",
            "block": "32x4",
            "screen_pairs": [],
            "observations": {},
            "commands": [],
        }
        gold = optimization.selected_row(self.manifest, "linux-release", "gold-uhd-36-default-direct", 0.5)
        for index, order in enumerate(g3.ORDERS, 1):
            result["screen_pairs"].append(
                {
                    "order": order.lower(),
                    "a": self.member(gold, f"pair-{index}-a", "reference", 20),
                    "b": self.member(gold, f"pair-{index}-b", "scalar", 25),
                }
            )
            for arm in order.lower():
                member = result["screen_pairs"][-1][arm]
                self.command(result, f"{gold['id']}-pair-{index}-{arm}", member)
        for name, offset in optimization.SCREEN_CONTROLS:
            row = optimization.selected_row(self.manifest, "linux-release", name, offset)
            result["observations"][row["id"]] = {
                "reference": self.member(row, "control-reference", "reference", 20),
                "optimized": self.member(row, "control-optimized", "scalar", 25),
            }
            for arm in ("reference", "optimized"):
                self.command(result, f"{row['id']}-control-{arm}", result["observations"][row["id"]][arm])
        return result

    def command(self, result, name, member):
        result["commands"].append(
            {
                "name": name,
                "command": [
                    "native",
                    "stage",
                    "resources",
                    str(self.root / Path(member["report"]).parent / "request.json"),
                ],
                "returncode": 0,
                "drained": True,
            }
        )

    def test_request_paths_match_across_native_platforms_and_reject_escape(self):
        windows = {
            "input": r"C:\Dev\Codex\out\carrier.f32",
            "bank": r"C:\Dev\Codex\out\bank",
            "tables": r"C:\Dev\Codex\out\tables",
            "invocation": "verify",
        }
        linux = {
            key: value.replace("C:\\Dev\\Codex", "/mnt/c/Dev/Codex").replace("\\", "/")
            for key, value in windows.items()
        }
        self.assertTrue(optimization.requests_match(Path("C:/Dev/Codex"), windows, linux))
        self.assertTrue(optimization.requests_match(Path("/mnt/c/Dev/Codex"), linux, windows))
        wrong = {**linux, "invocation": "different"}
        self.assertFalse(optimization.requests_match(Path("C:/Dev/Codex"), windows, wrong))
        with self.assertRaisesRegex(ValueError, "escapes"):
            optimization.repository_path(Path("C:/Dev/Codex"), "C:/Dev/Codex/out/../carrier.f32")
        with self.assertRaisesRegex(ValueError, "outside"):
            optimization.repository_path(Path("C:/Dev/Codex"), "C:/Dev/Codex-other/carrier.f32")

    def test_screen_rejects_reordered_process_records(self):
        result = self.screen()
        result["commands"][2:4] = reversed(result["commands"][2:4])
        with self.assertRaisesRegex(ValueError, "process order"):
            optimization.summarize_screen(self.root, result, self.root, self.manifest)

    def test_screen_telemetry_is_bound_to_each_arm(self):
        result = self.screen()
        member = result["screen_pairs"][0]["a"]
        name = Path(member["report"]).parent.name
        member["device_observations"] = {}
        for phase in ("before", "after"):
            for suffix in ("", "-activity"):
                path = self.root / f"{name}-{phase}{suffix}.log"
                path.write_text("device observation")
                member["device_observations"][path.name] = execution.digest(path)
        optimization.summarize_screen(self.root, result, self.root, self.manifest)
        path.write_text("changed observation")
        with self.assertRaisesRegex(ValueError, "telemetry changed"):
            optimization.summarize_screen(self.root, result, self.root, self.manifest)

    def test_new_screen_cannot_omit_per_invocation_telemetry(self):
        result = self.screen()
        result["device_observation_scope"] = "per-invocation"
        with self.assertRaisesRegex(ValueError, "omitted per-invocation"):
            optimization.summarize_screen(self.root, result, self.root, self.manifest)

    def test_screen_ratios_use_distinct_named_arms_and_reject_relabelled_processes(self):
        result = self.screen()
        summary = optimization.summarize_screen(self.root, result, self.root, self.manifest)
        self.assertEqual([pair["p90"] for pair in summary["gold_half_pair_ratios"]], [1.25] * 3)
        result["screen_pairs"][0]["a"], result["screen_pairs"][2]["a"] = (
            result["screen_pairs"][2]["a"],
            result["screen_pairs"][0]["a"],
        )
        with self.assertRaisesRegex(ValueError, "request differs"):
            optimization.summarize_screen(self.root, result, self.root, self.manifest)

    def test_screen_rejects_wrong_workload_even_with_complete_control_count(self):
        result = self.screen()
        key = next(iter(result["observations"]))
        result["observations"]["wrong-row"] = result["observations"].pop(key)
        with self.assertRaisesRegex(ValueError, "membership differs"):
            optimization.summarize_screen(self.root, result, self.root, self.manifest)

    def test_screen_uses_declared_scalar_baseline_and_rejects_old_reference_arm(self):
        result = self.screen()
        result.update(baseline_variant="scalar", baseline_block="32x8")
        members = [pair["a"] for pair in result["screen_pairs"]]
        members.extend(arms["reference"] for arms in result["observations"].values())
        for member in members:
            path = self.root / member["report"]
            raw = json.loads(path.read_text())
            raw["request"].update(film_variant="scalar", film_block="32x8")
            path.write_text(json.dumps(raw))
            member.update(variant="scalar", block="32x8", sha256=execution.digest(path))
        summary = optimization.summarize_screen(self.root, result, self.root, self.manifest)
        self.assertEqual([pair["median"] for pair in summary["gold_half_pair_ratios"]], [1.25] * 3)
        members[0]["variant"] = "reference"
        with self.assertRaisesRegex(ValueError, "variant identity"):
            optimization.summarize_screen(self.root, result, self.root, self.manifest)

    def test_profile_command_preserves_legacy_sections_and_records_sat_settings(self):
        command = ["binary", "stage", "resources", "request.json"]
        legacy = optimization.profiler_command("ncu", self.root, command)
        self.assertNotIn("--import-source", legacy)
        sat = optimization.profiler_command("ncu", self.root, command, optimization.SAT_SECTIONS)
        self.assertIn("MemoryWorkloadAnalysis_Tables", sat)
        self.assertEqual(sat[sat.index("--import-source") + 1], "on")
        self.assertEqual(optimization.profile_sections(list(optimization.SECTIONS)), optimization.SECTIONS)
        with self.assertRaisesRegex(ValueError, "unsupported profiler"):
            optimization.profiler_command("ncu", self.root, command, ("unknown",))

    def test_pc_reader_uses_numeric_instance_kind_and_rejects_invalid_values(self):
        class Metric:
            ValueKind_FLOAT, ValueKind_DOUBLE, ValueKind_UINT32, ValueKind_UINT64 = 3, 4, 5, 6
            values = [0, 7, 19]

            def has_correlation_ids(self):
                return True

            def correlation_ids(self):
                return self

            def num_instances(self):
                return len(self.values)

            def has_value(self, kind=1):
                return kind in (1, 6)

            def kind(self, index):
                return self.ValueKind_UINT64

            def as_double(self, index):
                return self.values[index]

            def as_uint64(self, index):
                return 100 + index

        metric = Metric()
        self.assertFalse(metric.has_value(2))
        self.assertEqual(optimization.pc_values(metric), [(101, 7), (102, 19)])
        metric.values = [float("nan")]
        with self.assertRaisesRegex(ValueError, "invalid SAT PC"):
            optimization.pc_values(metric)

    def test_screen_scalar_baseline_requires_same_candidate_agreement(self):
        result = {"baseline_variant": "scalar", "baseline_block": "32x8"}
        with self.assertRaisesRegex(ValueError, "baseline lacks"):
            optimization.validate_screen_baseline(result, {"variant": "scalar", "block": "32x8"})
        optimization.validate_screen_baseline(result, {"baseline_variant": "scalar", "baseline_block": "32x8"})

    def test_sat_source_readback_rejects_stale_imported_text(self):
        metric = SimpleNamespace(has_value=lambda: True, as_double=lambda: 1)
        action = SimpleNamespace(
            metric_by_name=lambda name: metric,
            source_files=lambda: {"C:/repo/tests/grain/film_reference.cu": "stale\r\nsource\r\n"},
        )
        scope = SimpleNamespace(num_actions=lambda: 3, action_by_idx=lambda index: action)
        report = SimpleNamespace(num_ranges=lambda: 1, range_by_idx=lambda index: scope)
        reader = SimpleNamespace(load_report=lambda path: report)
        with self.assertRaisesRegex(ValueError, "imported sampler source differs"):
            optimization.read_sat_profile(
                self.root / "report.ncu-rep", reader, optimization.source_text_digest("current\nsource\n")
            )
        self.assertEqual(
            optimization.source_text_digest("stale\r\nsource\r\n"), optimization.source_text_digest("stale\nsource\n")
        )

    def test_combined_sat_ratio_requires_each_counter_to_be_valid(self):
        source = {"C:/repo/tests/grain/film_reference.cu": "source\n"}
        for operation, command, global_count, local_count, wavefronts, expected in (
            ("ld", "read", 10, -1, 20, None),
            ("st", "write", 10, -1, 20, None),
            ("st", "write", -1, 10, 20, None),
            ("st", "write", 10, 0, -1, None),
            ("ld", "read", 0, 0, 20, None),
            ("ld", "read", 10, 0, 20, 2),
            ("st", "write", 0, 10, 0, 0),
        ):
            with self.subTest(operation=operation, counts=(global_count, local_count, wavefronts)):
                counts = {
                    f"l1tex__t_requests_pipe_lsu_mem_global_op_{operation}.sum": global_count,
                    f"l1tex__t_requests_pipe_lsu_mem_local_op_{operation}.sum": local_count,
                    f"l1tex__data_pipe_lsu_wavefronts_mem_lg_cmd_{command}.sum": wavefronts,
                }
                action = SimpleNamespace(
                    metric_by_name=lambda name: SimpleNamespace(
                        has_value=lambda: True, as_double=lambda: counts.get(name, 1)
                    ),
                    source_files=lambda: source,
                    source_info=lambda pc: SimpleNamespace(file_name=lambda: next(iter(source)), line=lambda: 1),
                    sass_by_pc=lambda pc: "LDG",
                )
                scope = SimpleNamespace(num_actions=lambda: 3, action_by_idx=lambda index: action)
                report = SimpleNamespace(num_ranges=lambda: 1, range_by_idx=lambda index: scope)
                reader = SimpleNamespace(load_report=lambda path: report)
                with patch.object(optimization, "pc_values", return_value=[(1, 1)]):
                    sample = optimization.read_sat_profile(
                        self.root / "report.ncu-rep", reader, optimization.source_text_digest("source\n")
                    )[0]
                self.assertEqual(
                    sample["ratios"][f"combined_local_global_{operation}"]["data_wavefronts_per_request"], expected
                )
                self.assertEqual({key: sample["values"][key] for key in counts}, counts)
                self.assertEqual(
                    sample["invalid_counter_values"], sorted(key for key, value in counts.items() if value < 0)
                )

    def test_agreement_rejects_corrupted_report_and_wrong_frame_with_valid_hash(self):
        evidence = {
            "schema": "grain-optimization-v1",
            "mode": "g3-verify-opt",
            "status": "passed",
            "preset": "linux-release",
            "variant": "scalar",
            "block": "32x4",
            "binary_sha256": "binary",
            "manifest_sha256": "manifest",
            "commands": [{"returncode": 0, "drained": True}],
            "observations": {},
        }
        for name in self.manifest["workloads"]:
            row = optimization.selected_row(self.manifest, "linux-release", name, 0.0)
            member = self.member(row, "verify", "scalar", 1)
            path = self.root / member["report"]
            raw = json.loads(path.read_text())
            raw.update(frame_identities=80, bit_mismatches=0, replay_hash_mismatches=0, nonfinite=0, inactive_frames=0)
            raw["observations"] = [
                {
                    "identity": identity,
                    "order": "forward" if index < 80 else "reverse",
                    "time": identity // 2 - 10 + (identity % 2) * 0.5,
                    "bit_mismatches": 0,
                    "nonfinite": 0,
                    "first_mismatch": None,
                    "replay_hash_mismatch": False,
                    "changed": [1, 1, 1],
                }
                for index, identity in enumerate(list(range(80)) + list(reversed(range(80))))
            ]
            path.write_text(json.dumps(raw))
            member["sha256"] = execution.digest(path)
            evidence["observations"][row["id"]] = member
        receipt = self.root / "result.json"
        receipt.write_text(json.dumps(evidence))
        args = (self.root, receipt, self.manifest, "linux-release", "binary", "manifest", "scalar", "32x4")
        optimization.validate_agreement(*args)
        member = next(iter(evidence["observations"].values()))
        path = self.root / member["report"]
        raw = json.loads(path.read_text())
        raw["observations"][0]["time"] = 0
        path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "member changed"):
            optimization.validate_agreement(*args)
        member["sha256"] = execution.digest(path)
        receipt.write_text(json.dumps(evidence))
        with self.assertRaisesRegex(ValueError, "frame identity"):
            optimization.validate_agreement(*args)

    def test_profile_recovery_binds_closed_capture_to_candidate_command_and_bytes(self):
        row = optimization.selected_row(self.manifest, "linux-release", "gold-uhd-36-default-direct", 0.0)
        member = self.member(row, "profile", "scalar", 1)
        report = self.root / member["report"]
        raw = report.parent / "profile.ncu-rep"
        raw.write_bytes(b"retained capture")
        candidate = {
            "schema": "grain-optimization-v1",
            "mode": "g3-profile",
            "preset": "linux-release",
            "binary_sha256": "binary",
            "manifest_sha256": "manifest",
            "variant": "scalar",
            "block": "32x4",
            "profiler": str(self.root / "ncu"),
            "profiler_sha256": "profiler",
            "sections": list(optimization.SECTIONS),
        }
        build = self.root / "out/build/linux-release"
        command = optimization.profiler_command(
            candidate["profiler"],
            report.parent,
            [
                build / "tests/bin/JuicerGrainPrototype",
                "stage",
                build / "tests/Resources",
                report.parent / "request.json",
            ],
        )
        prior = {
            **candidate,
            "status": "failed",
            "commands": [
                {"name": report.parent.name, "command": list(map(str, command)), "returncode": 0, "drained": True}
            ],
        }
        source = self.root / "result.json"
        source.write_text(json.dumps(prior))
        directory = self.root / "continued"
        directory.mkdir()
        member.update(
            report="../" + member["report"],
            profile_sha256=execution.digest(raw),
            capture_receipt=str(source),
            hash_binding="readback",
        )
        self.assertEqual(optimization.profile_origin(source, candidate), prior)
        continuation = self.root / "continued-result.json"
        continuation.write_text(
            json.dumps(
                {
                    **candidate,
                    "status": "failed",
                    "commands": [],
                    "continued_from": {"result": str(source), "sha256": execution.digest(source)},
                }
            )
        )
        captures, held = optimization.profile_captures(self.root, continuation, candidate)
        self.assertEqual(set(captures), {report.parent.name})
        self.assertEqual(captures[report.parent.name][0], source)
        self.assertEqual(set(held), {source, continuation})
        args = (self.root, source, prior, candidate, self.manifest, row)
        self.assertEqual(optimization.retained_profile(*args, member=member, directory=directory), (report, raw))
        with self.assertRaisesRegex(ValueError, "closed profile"):
            optimization.profile_origin(source, {**candidate, "block": "16x8"})
        wrong = copy.deepcopy(prior)
        wrong["commands"][0]["command"][wrong["commands"][0]["command"].index("--launch-count") + 1] = "4"
        with self.assertRaisesRegex(ValueError, "command differs"):
            optimization.retained_profile(self.root, source, wrong, candidate, self.manifest, row)
        with self.assertRaisesRegex(ValueError, "member differs"):
            optimization.retained_profile(
                *args, member={**member, "capture_receipt": str(self.root / "other.json")}, directory=directory
            )
        raw.write_bytes(b"changed capture")
        with self.assertRaisesRegex(ValueError, "member differs"):
            optimization.retained_profile(*args, member=member, directory=directory)
        prior["commands"][0]["drained"] = False
        source.write_text(json.dumps(prior))
        with self.assertRaisesRegex(ValueError, "incomplete commands"):
            optimization.profile_origin(source, candidate)

    def test_profile_rejects_missing_launch_and_wrong_kernel(self):
        values = {
            "ID": "0",
            "Kernel Name": "apply_scalar",
            "launch__registers_per_thread": "64",
            "launch__block_dim_x": "32",
            "launch__block_dim_y": "4",
            "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": "3",
            "smsp__warps_eligible.avg.per_cycle_active": "1",
            "l1tex__t_sector_hit_rate.pct": "50",
            "lts__t_sector_hit_rate.pct": "99",
            "derived__local_spilling_requests": "0",
        }

        def encoded(rows):
            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=list(values))
            writer.writeheader()
            writer.writerows(rows)
            return stream.getvalue()

        rows = [{**values, "ID": str(i)} for i in range(3)]
        self.assertEqual(len(optimization.profile_metrics(encoded(rows), "scalar", "32x4")), 3)
        path = self.root / "native.csv"
        path.write_bytes(encoded(rows).replace("apply_scalar", "apply_scalar\xa0kernel").encode("cp1252"))
        with patch("grain_optimization.locale.getencoding", return_value="cp1252"):
            decoded = optimization.csv_metrics(path, "scalar", "32x4")
        self.assertEqual(decoded["metrics_encoding"], "cp1252")
        self.assertEqual(len(decoded["metrics"]), 3)
        self.assertIn("\xa0", decoded["metrics"][0]["kernel"])
        with self.assertRaisesRegex(ValueError, "membership"):
            optimization.profile_metrics(encoded(rows[:2]), "scalar", "32x4")
        wrong = copy.deepcopy(rows)
        wrong[1]["Kernel Name"] = "apply_reference"
        with self.assertRaisesRegex(ValueError, "kernel differs"):
            optimization.profile_metrics(encoded(wrong), "scalar", "32x4")


if __name__ == "__main__":
    unittest.main()
