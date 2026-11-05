"""Pure record validation for Python outcomes and current CTest membership."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import qualification
import verify
from execution import NativeRun, OutputLease


class PythonResultTests(unittest.TestCase):
    function = "sample.TestCase.test_rows"

    def discovery(self, *, rows=True, skip=False, reason=""):
        return {"functions": [{"id": self.function, "skip": skip,
                               "skip_reason": reason, "rows": rows}]}

    def function_events(self, *, outcome="passed", reason=None):
        events = [{"id": self.function, "event": "start"},
                  {"id": self.function, "event": "terminal", "outcome": outcome}]
        if reason is not None:
            events[-1]["reason"] = reason
        return b"".join((json.dumps(event) + "\n").encode() for event in events)

    def row_events(self, ids=("positive/first", "negative/control")):
        caller = self.function + "/subtests"
        events = [{"caller": caller, "event": "membership", "ids": list(ids)}]
        for case_id in ids:
            events.extend(({"caller": caller, "id": case_id, "event": "start"},
                           {"caller": caller, "id": case_id, "event": "terminal", "outcome": "passed"}))
        return b"".join((json.dumps(event) + "\n").encode() for event in events)

    def test_function_and_declared_rows_report_counts(self):
        result = qualification.validate_python_result(
            self.discovery(), self.function_events(), self.row_events())
        self.assertEqual(result, {"python_functions": 1, "python_rows": 2, "python_skips": []})

    def test_missing_function_outcome_is_rejected(self):
        payload = self.function_events().splitlines(keepends=True)[0]
        with self.assertRaisesRegex(ValueError, "missing Python function outcomes"):
            qualification.validate_python_result(self.discovery(), payload, self.row_events())

    def test_missing_required_row_terminal_is_rejected(self):
        rows = self.row_events().splitlines(keepends=True)
        payload = b"".join(rows[:-1])
        with self.assertRaisesRegex(ValueError, "missing case outcomes"):
            qualification.validate_python_result(self.discovery(), self.function_events(), payload)

    def test_missing_required_cohort_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing or unexpected required cohort"):
            qualification.validate_python_result(self.discovery(), self.function_events(), b"")

    def test_duplicate_function_event_is_rejected(self):
        payload = self.function_events().splitlines(keepends=True)
        with self.assertRaisesRegex(ValueError, "duplicate or unordered Python function outcome"):
            qualification.validate_python_result(
                self.discovery(), b"".join((payload[0], payload[0], *payload[1:])), self.row_events())

    def test_duplicate_row_terminal_is_rejected(self):
        rows = self.row_events().splitlines(keepends=True)
        with self.assertRaisesRegex(ValueError, "duplicate or unordered case event"):
            qualification.validate_python_result(
                self.discovery(), self.function_events(), b"".join((*rows, rows[-1])))

    def test_truncated_function_records_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "truncated Python function outcomes"):
            qualification.validate_python_result(
                self.discovery(), self.function_events()[:-1], self.row_events())

    def test_truncated_row_records_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "truncated case output"):
            qualification.validate_python_result(
                self.discovery(), self.function_events(), self.row_events()[:-1])

    def test_unexpected_runtime_skip_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "failed or unexpected skipped Python function"):
            qualification.validate_python_result(
                self.discovery(rows=False), self.function_events(outcome="skipped", reason="runtime"), b"")

    def test_static_platform_skip_is_counted_and_reason_must_match(self):
        discovery = self.discovery(rows=True, skip=True, reason="requires Linux")
        payload = self.function_events(outcome="skipped", reason="requires Linux")
        self.assertEqual(qualification.validate_python_result(discovery, payload, b""),
                         {"python_functions": 0, "python_rows": 0,
                          "python_skips": [self.function]})
        with self.assertRaisesRegex(ValueError, "unexpected Python skip outcome"):
            qualification.validate_python_result(
                discovery, self.function_events(outcome="skipped", reason="different reason"), b"")


class PythonWorkerTests(unittest.TestCase):
    def run_worker(self, source):
        with tempfile.TemporaryDirectory(prefix="python worker controls ") as folder:
            root = Path(folder).resolve()
            preset = "linux-debug" if os.name == "posix" else "windows-clang-debug"
            directory = root / "bundle"
            directory.mkdir()
            script = root / "fixture_case.py"
            script.write_text(source, encoding="utf-8")
            key = "quality-worker-control"
            environment = dict(os.environ)
            environment["FJ_HOST_BUNDLE"] = str(directory)
            command = [sys.executable, "-B", str(Path(verify.__file__).resolve()), "python-tests",
                       "--preset", preset, "--key", key, "--", sys.executable, "-B", str(script)]
            run = NativeRun(root, directory)
            build = root / "out/build" / preset
            with OutputLease(root, build):
                try:
                    run.command("python-suite", command, env=environment, timeout=120)
                except subprocess.CalledProcessError as error:
                    returncode = error.returncode
                else:
                    returncode = 0
                execution_path = directory / (key + ".execution.json")
                execution = json.loads(execution_path.read_text(encoding="utf-8"))
                log = (directory / "python-suite.log").read_text(encoding="utf-8", errors="replace")
                paths = {suffix: directory / (key + suffix) for suffix in (
                    ".discovery.json", ".functions.jsonl", ".rows.jsonl")}
                payloads = {suffix: path.read_bytes() for suffix, path in paths.items() if path.exists()}
                return returncode, execution, payloads, log

    @staticmethod
    def fixture_source(body):
        return (
            "import unittest\n"
            "class FixtureRows(unittest.TestCase):\n"
            "    def test_required_rows(self):\n"
            "        self.subtest_cases = ('positive/row', 'negative/control')\n"
            + body +
            "FixtureRows.test_required_rows.required_subtests = True\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )

    def test_public_worker_accepts_complete_two_row_table(self):
        source = self.fixture_source(
            "        with self.subTest(case='positive/row'):\n"
            "            self.assertEqual(2 + 2, 4)\n"
            "        with self.subTest(case='negative/control'):\n"
            "            with self.assertRaises(ValueError):\n"
            "                int('not an integer')\n")
        code, execution, payloads, log = self.run_worker(source)
        self.assertEqual(code, 0, log)
        self.assertTrue(execution["outcomes_valid"], log)
        discovery = json.loads(payloads[".discovery.json"])
        counts = qualification.validate_python_result(
            discovery, payloads[".functions.jsonl"], payloads[".rows.jsonl"])
        self.assertEqual(counts["python_rows"], 2)

    def test_public_worker_rejects_one_omitted_required_row(self):
        source = self.fixture_source(
            "        with self.subTest(case='positive/row'):\n"
            "            self.assertEqual(2 + 2, 4)\n")
        code, execution, payloads, log = self.run_worker(source)
        self.assertEqual(code, 1, log)
        self.assertFalse(execution["outcomes_valid"])
        self.assertIn("missing case outcomes", execution.get("error", ""))
        self.assertIn(b"negative/control", payloads[".rows.jsonl"])

    def test_public_worker_rejects_an_omitted_required_row_cohort(self):
        source = self.fixture_source("        self.assertTrue(True)\n")
        code, execution, _, log = self.run_worker(source)
        self.assertEqual(code, 1, log)
        self.assertFalse(execution["outcomes_valid"])
        self.assertIn("required cohort", execution.get("error", ""))

    def test_public_worker_rejects_unexpected_runtime_skip(self):
        source = (
            "import unittest\n"
            "class FixtureSkip(unittest.TestCase):\n"
            "    def test_runtime_skip(self):\n"
            "        self.skipTest('runtime decision')\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )
        code, execution, _, log = self.run_worker(source)
        self.assertEqual(code, 1, log)
        self.assertFalse(execution["outcomes_valid"])
        self.assertIn("unexpected skipped Python function", execution.get("error", ""))

    def test_public_worker_crash_has_no_valid_outcome_record(self):
        code, execution, payloads, log = self.run_worker("raise SystemExit(7)\n")
        self.assertEqual(code, 7, log)
        self.assertFalse(execution["outcomes_valid"])
        self.assertNotIn(".discovery.json", payloads)


class MembershipTests(unittest.TestCase):
    @staticmethod
    def inventory(*rows):
        return {"tests": [{"name": name, "properties": [{"name": "LABELS", "value": labels}]}
                           for name, labels in rows]}

    def test_relevance_and_runtime_selection_are_orthogonal(self):
        inventory = self.inventory(("behavior", ["host", "product"]),
                                   ("boundary", ["host", "boundary"]),
                                   ("runner", ["host", "tooling"]),
                                   ("data", ["host", "corpus"]),
                                   ("device", ["gpu", "product"]))
        groups = ["product", "boundary", "tooling", "corpus"]
        self.assertEqual(qualification.membership(inventory, groups, "all"),
                         {"product": ["behavior", "device"], "boundary": ["boundary"],
                          "tooling": ["runner"], "corpus": ["data"]})
        self.assertEqual(qualification.membership(inventory, ["product"], "host"),
                         {"product": ["behavior"]})

    def test_empty_inventory_fails(self):
        with self.assertRaisesRegex(ValueError, "required checkpoint selection is empty"):
            qualification.membership(self.inventory(), ["product"], "host")

    def test_missing_runtime_prerequisite_fails(self):
        with self.assertRaisesRegex(ValueError, "one relevance and a runtime prerequisite"):
            qualification.membership(self.inventory(("a", ["product"])), ["product"], "host")

    def test_multiple_relevance_labels_fail(self):
        with self.assertRaisesRegex(ValueError, "one relevance and a runtime prerequisite"):
            qualification.membership(self.inventory(("a", ["host", "product", "tooling"])),
                                     ["product"], "host")

    def test_duplicate_registration_fails(self):
        with self.assertRaisesRegex(ValueError, "duplicate CTest registration"):
            qualification.membership(self.inventory(("a", ["host", "product"]),
                                                    ("a", ["host", "product"])),
                                     ["product"], "host")

    def test_empty_group_selection_fails(self):
        with self.assertRaisesRegex(ValueError, "invalid checkpoint selection"):
            qualification.membership(self.inventory(("a", ["host", "product"])), [], "host")

    def test_selection_with_missing_group_membership_fails(self):
        with self.assertRaisesRegex(ValueError, "required checkpoint selection is empty"):
            qualification.membership(self.inventory(("a", ["host", "product"])),
                                     ["product", "boundary"], "host")

    def test_disabled_and_globally_serialized_required_cases_fail(self):
        inventory = self.inventory(("a", ["host", "boundary"]))
        inventory["tests"][0]["properties"].append({"name": "DISABLED", "value": True})
        with self.assertRaisesRegex(ValueError, "disabled or globally serialized"):
            qualification.membership(inventory, ["boundary"], "host")
        inventory["tests"][0]["properties"].pop()
        inventory["tests"][0]["properties"].append({"name": "RUN_SERIAL", "value": True})
        with self.assertRaisesRegex(ValueError, "disabled or globally serialized"):
            qualification.membership(inventory, ["boundary"], "host")

    def test_resource_lock_is_an_owned_serialization_contract(self):
        inventory = self.inventory(("a", ["host", "product"]))
        inventory["tests"][0]["properties"].append({"name": "RESOURCE_LOCK", "value": ["fixture_owner"]})
        self.assertEqual(qualification.membership(inventory, ["product"], "host"), {"product": ["a"]})

    def test_scratch_roots_must_be_absolute_and_noncolliding(self):
        inventory = self.inventory(("a", ["host", "product"]), ("b", ["host", "boundary"]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inventory["tests"][0]["command"] = ["case", "--scratch-root", str(root / "shared")]
            inventory["tests"][1]["command"] = ["case", "--scratch-root", str(root / "shared")]
            with self.assertRaisesRegex(ValueError, "colliding scratch roots"):
                qualification.membership(inventory, ["product", "boundary"], "all")
            inventory["tests"][1]["command"][-1] = str(root / "other")
            self.assertEqual(qualification.membership(inventory, ["product", "boundary"], "all"),
                             {"product": ["a"], "boundary": ["b"]})
            inventory["tests"][1]["command"] = ["case", "--scratch-root", "relative"]
            with self.assertRaisesRegex(ValueError, "scratch root must be absolute"):
                qualification.membership(inventory, ["product", "boundary"], "all")


class CollectionTests(unittest.TestCase):
    def test_rust_artifact_description_requires_separate_variants(self):
        cases = ("valid", "missing-member", "archive-alias", "directory-alias",
                 "production-feature", "dropped-description-flag")
        self.subtest_cases = cases
        for case in cases:
            with self.subTest(case=case):
                artifacts = {"schema": "rust-artifacts-v1", "variants": {
                    variant: {"archive": f"/unavailable/{variant}/plugin.a",
                              "target_dir": f"/unavailable/{variant}", "features": features}
                    for variant, features in (("production", []), ("test-support", ["test-support"]))}}
                if case == "archive-alias":
                    artifacts["variants"]["test-support"]["archive"] = artifacts["variants"]["production"]["archive"]
                elif case == "directory-alias":
                    artifacts["variants"]["test-support"]["target_dir"] = artifacts["variants"]["production"]["target_dir"]
                elif case == "production-feature":
                    artifacts["variants"]["production"]["features"] = ["test-support"]
                payloads = {name: b"{}" for name in (
                    "registration.json", "target-discovery.json", "host-contracts.json", "host-targets.json",
                    "gpu-contracts.json", "gpu-targets.json", "python-contracts.json", "python-targets.json")}
                payloads.update({"candidate.json": json.dumps({
                    "head": "source", "preset": "linux-debug", "features": {"rust_artifacts": "rust-artifacts.json"}}).encode(),
                    "candidate.patch": b"source patch", "rust-artifacts.json": json.dumps(artifacts).encode()})
                if case == "missing-member":
                    del payloads["rust-artifacts.json"]
                result = {"status": "failed", "preset": "linux-debug", "scope": "focused", "runtime": "host",
                          "groups": [], "selection": [], "build_kind": "product", "candidate_head": "source",
                          "shipping_isolation": False, "rust_artifacts": case != "dropped-description-flag",
                          "members": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                                      for name, data in payloads.items()}}
                if case == "valid":
                    self.assertEqual(qualification.validate_tests_result(result, payloads)["status"], "failed")
                else:
                    with self.assertRaises(ValueError):
                        qualification.validate_tests_result(result, payloads)

    def test_quality_omitted_operation_and_undrained_command_are_rejected(self):
        checks = {"schema": "native-quality-checks-v1", "status": "passed",
                  "expected_checks": ["rust-quality", "native-boundary-enforcement"],
                  "completed_checks": ["native-boundary-enforcement"],
                  "expected_commands": ["clippy"], "commands": [], "python_cases": ["quality-boundary"]}
        with self.assertRaisesRegex(ValueError, "missing or duplicate quality operation"):
            qualification.validate_quality_checks(checks)
        checks["completed_checks"] = checks["expected_checks"]
        checks["commands"] = [{"name": "clippy", "returncode": 0, "drained": False, "process_status": "completed"}]
        with self.assertRaisesRegex(ValueError, "undrained quality operation"):
            qualification.validate_quality_checks(checks)

    def test_focused_saved_passes_cannot_form_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            options = SimpleNamespace(bundles=[Path("first"), Path("second")], output=Path(folder) / "report.json",
                                      selection=["product", "boundary"], runtime="host", presets=["linux-debug"])
            summaries = [{"status": "passed", "preset": "linux-debug", "scope": "focused",
                          "groups": [group], "runtime": "host", "candidate_source": "same"}
                         for group in ("product", "boundary")]
            with patch.object(qualification, "read_native_result", side_effect=summaries):
                with self.assertRaisesRegex(ValueError, "one complete fresh execution"):
                    qualification.collect_bundles(options)
            self.assertFalse(options.output.exists())

    def test_checkpoint_requires_same_candidate_and_all_requested_presets(self):
        with tempfile.TemporaryDirectory() as folder:
            options = SimpleNamespace(bundles=[Path("first"), Path("second")], output=Path(folder) / "report.json",
                                      selection=["boundary"], runtime="host", presets=["linux-debug", "windows-clang-debug"])
            summaries = [{"status": "passed", "preset": preset, "scope": "checkpoint", "groups": ["boundary"],
                          "runtime": "host", "candidate_source": preset} for preset in options.presets]
            with patch.object(qualification, "read_native_result", side_effect=summaries):
                with self.assertRaisesRegex(ValueError, "different source candidates"):
                    qualification.collect_bundles(options)
            self.assertFalse(options.output.exists())
            with patch.object(qualification, "read_native_result", return_value=summaries[0]):
                with self.assertRaisesRegex(ValueError, "missing requested native preset"):
                    qualification.collect_bundles(options)

    def test_copied_collector_has_no_execution_source_build_or_tool_dependency(self):
        with tempfile.TemporaryDirectory(prefix="offline collection ü ") as folder:
            root = Path(folder).resolve()
            bundle = root / "bundle"
            bundle.mkdir()
            payloads = {"candidate.json": json.dumps({"head": "candidate", "preset": "linux-debug",
                                                     "inputs": {"/unavailable/source": "identity"}}).encode(),
                        "candidate.patch": b"reconstructible source patch\n"}
            result = {"schema": "native-build-v1", "status": "passed", "preset": "linux-debug", "scope": "build",
                      "candidate_head": "candidate", "candidate_unchanged": True, "shipping_isolation": False,
                      "commands": [{"returncode": 0, "drained": True, "process_status": "completed"}],
                      "members": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                                  for name, data in payloads.items()}}
            completed = os.environ.get("FJ_NATIVE_COMPLETED_BUNDLE")
            if completed:
                original = Path(completed)
                result = json.loads((original / "result.json").read_bytes())
                self.assertEqual(result["schema"], "native-tests-v1")
                qualification.tests_result_members(result)
                payloads = {name: (original / name).read_bytes() for name in result["members"]}
            for name, data in payloads.items():
                (bundle / name).write_bytes(data)
            (bundle / "result.json").write_text(json.dumps(result))
            for module in ("verify.py", "qualification.py"):
                (root / module).write_bytes((Path(verify.__file__).parent / module).read_bytes())
            guard = root / "guard.py"
            guard.write_text(
                "import argparse, contextlib, json, os, pathlib, re, shutil, subprocess, sys, time, uuid\n"
                "import qualification\n"
                "root = pathlib.Path(__file__).parent.resolve()\n"
                "code = compile((root/'verify.py').read_bytes(), str(root/'verify.py'), 'exec')\n"
                "def audit(event, args):\n"
                "    if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn'):\n"
                "        raise AssertionError('collector launched a tool')\n"
                "    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):\n"
                "        if not pathlib.Path(os.fsdecode(args[0])).resolve().is_relative_to(root):\n"
                "            raise AssertionError('collector reopened original source/build/tools: '+str(args[0]))\n"
                "sys.addaudithook(audit)\n"
                "sys.argv = ['verify.py','collect-qualification','--bundles',str(root/'bundle'),\n"
                "            '--output',str(root/'report.json')]\n"
                "exec(code, {'__name__':'__main__','__file__':str(root/'verify.py')})\n", encoding="utf-8")
            run = NativeRun(root, root)
            with OutputLease(root, root / "out/build/linux-debug"):
                try:
                    run.command("offline-collector", [sys.executable, "-B", str(guard)], timeout=30)
                except subprocess.CalledProcessError:
                    self.fail((root / "offline-collector.log").read_text(encoding="utf-8"))
            report = json.loads((root / "report.json").read_bytes())
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["scope"], "reported-runs")


CollectionTests.test_rust_artifact_description_requires_separate_variants.required_subtests = True
