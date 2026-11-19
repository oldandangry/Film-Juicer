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

from preview_configuration import PREVIEW_CASES, configure_preview_option

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import qualification
import verify
from execution import NativeRun, OutputLease


class PythonResultTests(unittest.TestCase):
    function = "sample.TestCase.test_rows"

    def test_rust_execution_features_match_the_resolved_package(self):
        cargo = {
            "packages": [
                {
                    "id": "bridge",
                    "name": "film-juicer-plugin",
                    "targets": [{"name": "bridge", "test": True, "kind": ["staticlib"]}],
                }
            ],
            "resolve": {"nodes": [{"id": "bridge", "features": ["test-support", "grain-preview"]}]},
        }
        execution = {
            "schema": "rust-test-execution-v2",
            "package": "film-juicer-plugin",
            "features": ["grain-preview", "test-support"],
        }
        qualification.validate_rust_execution_features(execution, "bridge", cargo)
        with self.assertRaisesRegex(ValueError, "features differ"):
            qualification.validate_rust_execution_features({**execution, "features": ["test-support"]}, "bridge", cargo)
        qualification.validate_rust_execution_features({"features": "test-support"}, "bridge", cargo)
        binaries = {"rust-binaries": {"bridge": {"package-id": "bridge", "binary-name": "bridge"}}}
        suite = {
            "package-id": "bridge",
            "testcases": {"abi": {"ignored": False, "filter-match": {"status": "matches"}}},
        }
        discovery = {"rust-suites": {"bridge": suite}}
        self.assertEqual(
            qualification.rust_membership(discovery, "bridge", cargo, binaries), ({("bridge", "abi")}, set())
        )
        suite["package-id"] = "core"
        with self.assertRaisesRegex(ValueError, "discovery package ID differs"):
            qualification.rust_membership(discovery, "bridge", cargo, binaries)

    def test_clippy_receipt_derives_required_forms_from_raw_cmake_value(self):
        baseline = (("film-juicer-core", ""), ("film-juicer-plugin", ""), ("film-juicer-plugin", "test-support"))
        preview = (
            ("film-juicer-core", "grain-preview"),
            ("film-juicer-plugin", "grain-preview"),
            ("film-juicer-plugin", "grain-preview,test-support"),
        )
        cases = (
            ("enabled", "TRUE", baseline + preview, ""),
            ("disabled", "OFF", baseline, ""),
            ("missing-preview", "1", baseline, "missing required Clippy forms"),
            ("unexpected-preview", "OFF", baseline + preview, "unexpected Clippy form"),
            ("missing-binding", "ON", baseline + preview, "missing Rust quality binding"),
            ("wrong-target", "ON", baseline + preview, "unexpected Clippy form"),
            ("debug-plugin", "ON", tuple(form for form in baseline + preview if form[0].endswith("-plugin")), ""),
            ("unconfigured", None, baseline, ""),
            ("historical", None, baseline, ""),
        )
        self.subtest_cases = tuple(case for case, *_ in cases)
        for case, raw, forms, error in cases:
            with self.subTest(case=case):
                debug = case == "debug-plugin"
                preset = "linux-debug" if debug else "linux-release"
                profile = "dev" if debug else "release"
                mode = "development" if debug else "qualification"
                label_prefix = "quality-clippy" if debug else "clippy"
                prefix = [
                    "cargo",
                    "clippy",
                    "--locked",
                    "--target",
                    "wrong-target" if case == "wrong-target" else "x86_64-unknown-linux-gnu",
                    "--target-dir",
                    "/cargo",
                ]
                commands = [
                    {
                        "name": "-".join((label_prefix, package, *filter(None, features.split(",")), profile)),
                        "command": [
                            *prefix,
                            "-p",
                            package,
                            "--all-targets",
                            *(["--features", features] if features else []),
                            *([] if debug else ["--release"]),
                            "--",
                            "-D",
                            "warnings",
                        ],
                        "returncode": 0,
                        "drained": True,
                        "process_status": "completed",
                    }
                    for package, features in forms
                ]
                packages = ["film-juicer-plugin"] if debug else ["film-juicer-core", "film-juicer-plugin"]
                checks = {
                    "schema": "native-quality-checks-v1" if case == "historical" else "native-quality-checks-v2",
                    "status": "passed",
                    "preset": preset,
                    "selected": ["cmake/JuicerRust.cmake"],
                    "excluded": [],
                    "mode": mode,
                    "expected_checks": ["rust-quality"],
                    "completed_checks": ["rust-quality"],
                    "expected_commands": [command["name"] for command in commands],
                    "commands": commands,
                    "python_cases": ["quality-boundary"],
                }
                if case not in {"missing-binding", "historical"}:
                    checks["rust_quality"] = {"JUICER_GRAIN_PREVIEW": raw, "packages": packages}
                result, payloads = self.standalone_quality(checks)
                if error:
                    with self.assertRaisesRegex(ValueError, error):
                        qualification.validate_quality_result(result, payloads)
                else:
                    self.assertEqual(qualification.validate_quality_result(result, payloads)["status"], "passed")

    test_clippy_receipt_derives_required_forms_from_raw_cmake_value.required_subtests = True

    @staticmethod
    def operation(name, command):
        return {"name": name, "command": command, "returncode": 0, "drained": True, "process_status": "completed"}

    def quality_checks(self, preset, selected, mode, raw, packages):
        prefix = [
            "cargo",
            "clippy",
            "--locked",
            "--target",
            qualification.PRESET_TARGETS[preset],
            "--target-dir",
            "/cargo",
        ]
        commands = [self.operation("clippy-version", ["cargo", "clippy", "--version"])]
        commands += [
            self.operation(name, prefix + args) for name, args in qualification.clippy_forms(preset, packages, raw)
        ]
        expected = ["source-hygiene", "git-diff-check", "native-boundary-enforcement"]
        if packages:
            expected.append("rust-quality")
        return {
            "schema": "native-quality-checks-v2",
            "status": "passed",
            "preset": preset,
            "selected": selected,
            "excluded": [],
            "mode": mode,
            "expected_checks": expected,
            "completed_checks": expected.copy(),
            "expected_commands": [item["name"] for item in commands],
            "commands": commands,
            "rust_quality": {"JUICER_GRAIN_PREVIEW": raw, "packages": list(packages)} if packages else None,
            "python_cases": ["quality-boundary"],
        }

    def python_payloads(self, key, **execution):
        return {
            key + ".execution.json": json.dumps({"returncode": 0, "outcomes_valid": True, **execution}).encode(),
            key + ".discovery.json": json.dumps(self.discovery()).encode(),
            key + ".functions.jsonl": self.function_events(),
            key + ".rows.jsonl": self.row_events(),
        }

    def quality_payloads(self, checks, **candidate):
        return {
            "candidate.json": json.dumps(
                {"head": "source", "preset": checks["preset"], "inputs": {"source": "id"}, **candidate}
            ).encode(),
            "candidate.patch": b"source patch",
            "quality.checks.json": json.dumps(checks).encode(),
            **self.python_payloads("quality-boundary"),
        }

    @staticmethod
    def result_members(payloads):
        return {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in payloads.items()
        }

    def standalone_quality(self, checks, **overrides):
        payloads = self.quality_payloads(checks)
        result = {
            **{key: checks[key] for key in ("status", "preset", "selected", "excluded", "mode", "commands")},
            "candidate_head": "source",
            "candidate_unchanged": True,
            "shipping_isolation": False,
            "members": self.result_members(payloads),
            **overrides,
        }
        return result, payloads

    def test_standalone_rejects_omitted_rust_quality_scope(self):
        checks = self.quality_checks(
            "linux-release", ["cmake/JuicerRust.cmake"], "qualification", "OFF", qualification.RUST_QUALITY_PACKAGES
        )
        checks["expected_checks"].remove("rust-quality")
        checks["completed_checks"].remove("rust-quality")
        del checks["rust_quality"]
        with self.assertRaisesRegex(ValueError, "declared Rust quality differs"):
            qualification.validate_quality_result(*self.standalone_quality(checks))

    def test_standalone_rejects_development_mode_for_qualification(self):
        checks = self.quality_checks(
            "windows-clang-release", ["cmake/JuicerRust.cmake"], "development", "1", ["film-juicer-core"]
        )
        with self.assertRaisesRegex(ValueError, "quality mode differs from the standalone result"):
            qualification.validate_quality_result(*self.standalone_quality(checks, mode="qualification"))
        checks["mode"] = "qualification"
        with self.assertRaisesRegex(ValueError, "Rust quality packages differ"):
            qualification.validate_quality_result(*self.standalone_quality(checks))

    def test_standalone_preserves_non_rust_and_forced_qualification(self):
        checks = self.quality_checks("linux-release", ["CONTRIBUTING.md"], "qualification", None, [])
        qualification.validate_quality_result(*self.standalone_quality(checks, qualification_requested=False))
        qualification.validate_quality_result(*self.standalone_quality(checks))
        with self.assertRaisesRegex(ValueError, "declared Rust quality differs"):
            qualification.validate_quality_result(*self.standalone_quality(checks, qualification_requested=True))
        checks = self.quality_checks(
            "linux-release", ["CONTRIBUTING.md"], "qualification", None, qualification.RUST_QUALITY_PACKAGES
        )
        qualification.validate_quality_result(*self.standalone_quality(checks, qualification_requested=True))
        qualification.validate_quality_result(*self.standalone_quality(checks))
        checks["expected_checks"].remove("rust-quality")
        checks["completed_checks"].remove("rust-quality")
        del checks["rust_quality"]
        with self.assertRaisesRegex(ValueError, "unexpected Clippy form"):
            qualification.validate_quality_result(*self.standalone_quality(checks, qualification_requested=False))

    def embedded_quality(self, checks, raw, packages):
        preset = checks["preset"]
        test = "Quality.Checkpoint"
        command = ["python", "test_checkpoint.py"]
        key = "host-" + hashlib.sha256(json.dumps(command).encode()).hexdigest()[:16]
        invocation = verify.quality_command(
            preset,
            {"files": checks["selected"], "compiler_consumers": ["src/ScanRoute.cpp"], "rust_packages": packages},
        )
        payloads = {
            **self.quality_payloads(
                checks, configuration={"CMAKE_BUILD_TYPE": "Debug", "BUILD_TESTING": "ON", "JUICER_GRAIN_PREVIEW": raw}
            ),
            **self.python_payloads(key, command=command),
            "registration.json": json.dumps(
                {"tests": [{"name": test, "command": ["python-tests", "--", *command]}]}
            ).encode(),
            "ctest.xml": f'<testsuite><testcase name="{test}" status="run"/></testsuite>'.encode(),
            "python-contracts.json": json.dumps(
                {
                    "schema": "python-contracts-v1",
                    "registrations": {test: {"source": "tests/quality/test_checkpoint.py", "tooling": True}},
                }
            ).encode(),
            **{
                name + ".json": b"{}"
                for name in (
                    "target-discovery",
                    "host-contracts",
                    "gpu-contracts",
                    "host-targets",
                    "gpu-targets",
                    "python-targets",
                )
            },
        }
        result = {
            "preset": preset,
            "status": "passed",
            "scope": "focused",
            "runtime": "host",
            "groups": [],
            "build_kind": "tooling",
            "candidate_head": "source",
            "candidate_unchanged": True,
            "shipping_isolation": False,
            "selection": [test],
            "quality_files": checks["selected"],
            "case_files": {test: {"key": key, "protocol": "python", "rows": False}},
            "commands": [self.operation("quality", invocation)],
            "members": self.result_members(payloads),
        }
        return result, payloads

    def test_embedded_binds_quality_to_the_verifier_request(self):
        core, plugin = qualification.RUST_QUALITY_PACKAGES
        cases = (
            (
                "two-package-request-narrowed",
                (core, plugin),
                (core,),
                "development",
                "TRUE",
                "Rust quality packages differ",
            ),
            ("default-package-request-narrowed", (), (core,), "development", "TRUE", "Rust quality packages differ"),
            (
                "changed-mode",
                (core, plugin),
                (core, plugin),
                "qualification",
                "TRUE",
                "quality mode differs from the verifier invocation",
            ),
            ("matching-two-packages", (core, plugin), (core, plugin), "development", "TRUE", ""),
            ("matching-default", (), (core, plugin), "development", "TRUE", ""),
            ("matching-core-only", (core,), (core,), "development", "TRUE", ""),
            ("matching-plugin-only", (plugin,), (plugin,), "development", "TRUE", ""),
            ("foreign-enabled-spelling", (core,), (core,), "development", "ON", "quality preview value differs"),
            ("off-is-not-null", (core,), (core,), "development", "OFF", "quality preview value differs"),
        )
        self.subtest_cases = tuple(case for case, *_ in cases)
        for case, requested, packages, mode, raw, error in cases:
            with self.subTest(case=case):
                checks = self.quality_checks("linux-debug", ["rust/film-juicer-core/src/route.rs"], mode, raw, packages)
                bundle = self.embedded_quality(checks, None if case == "off-is-not-null" else "TRUE", requested)
                if error:
                    with self.assertRaisesRegex(ValueError, error):
                        qualification.validate_tests_result(*bundle)
                else:
                    self.assertEqual(qualification.validate_tests_result(*bundle)["status"], "passed")

    test_embedded_binds_quality_to_the_verifier_request.required_subtests = True

    def discovery(self, *, rows=True, skip=False, reason=""):
        return {"functions": [{"id": self.function, "skip": skip, "skip_reason": reason, "rows": rows}]}

    def function_events(self, *, outcome="passed", reason=None):
        events = [
            {"id": self.function, "event": "start"},
            {"id": self.function, "event": "terminal", "outcome": outcome},
        ]
        if reason is not None:
            events[-1]["reason"] = reason
        return b"".join((json.dumps(event) + "\n").encode() for event in events)

    def row_events(self, ids=("positive/first", "negative/control")):
        caller = self.function + "/subtests"
        events = [{"caller": caller, "event": "membership", "ids": list(ids)}]
        for case_id in ids:
            events.extend(
                (
                    {"caller": caller, "id": case_id, "event": "start"},
                    {"caller": caller, "id": case_id, "event": "terminal", "outcome": "passed"},
                )
            )
        return b"".join((json.dumps(event) + "\n").encode() for event in events)

    def test_function_and_declared_rows_report_counts(self):
        result = qualification.validate_python_result(self.discovery(), self.function_events(), self.row_events())
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
                self.discovery(), b"".join((payload[0], payload[0], *payload[1:])), self.row_events()
            )

    def test_duplicate_row_terminal_is_rejected(self):
        rows = self.row_events().splitlines(keepends=True)
        with self.assertRaisesRegex(ValueError, "duplicate or unordered case event"):
            qualification.validate_python_result(self.discovery(), self.function_events(), b"".join((*rows, rows[-1])))

    def test_truncated_function_records_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "truncated Python function outcomes"):
            qualification.validate_python_result(self.discovery(), self.function_events()[:-1], self.row_events())

    def test_truncated_row_records_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "truncated case output"):
            qualification.validate_python_result(self.discovery(), self.function_events(), self.row_events()[:-1])

    def test_unexpected_runtime_skip_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "failed or unexpected skipped Python function"):
            qualification.validate_python_result(
                self.discovery(rows=False), self.function_events(outcome="skipped", reason="runtime"), b""
            )

    def test_static_platform_skip_is_counted_and_reason_must_match(self):
        discovery = self.discovery(rows=True, skip=True, reason="requires Linux")
        payload = self.function_events(outcome="skipped", reason="requires Linux")
        self.assertEqual(
            qualification.validate_python_result(discovery, payload, b""),
            {"python_functions": 0, "python_rows": 0, "python_skips": [self.function]},
        )
        with self.assertRaisesRegex(ValueError, "unexpected Python skip outcome"):
            qualification.validate_python_result(
                discovery, self.function_events(outcome="skipped", reason="different reason"), b""
            )


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
            command = [
                sys.executable,
                "-B",
                str(Path(verify.__file__).resolve()),
                "python-tests",
                "--preset",
                preset,
                "--key",
                key,
                "--",
                sys.executable,
                "-B",
                str(script),
            ]
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
                paths = {
                    suffix: directory / (key + suffix)
                    for suffix in (".discovery.json", ".functions.jsonl", ".rows.jsonl")
                }
                payloads = {suffix: path.read_bytes() for suffix, path in paths.items() if path.exists()}
                return returncode, execution, payloads, log

    @staticmethod
    def fixture_source(body):
        return (
            "import unittest\n"
            "class FixtureRows(unittest.TestCase):\n"
            "    def test_required_rows(self):\n"
            "        self.subtest_cases = ('positive/row', 'negative/control')\n"
            + body
            + "FixtureRows.test_required_rows.required_subtests = True\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )

    def test_public_worker_accepts_complete_two_row_table(self):
        source = self.fixture_source(
            "        with self.subTest(case='positive/row'):\n"
            "            self.assertEqual(2 + 2, 4)\n"
            "        with self.subTest(case='negative/control'):\n"
            "            with self.assertRaises(ValueError):\n"
            "                int('not an integer')\n"
        )
        code, execution, payloads, log = self.run_worker(source)
        self.assertEqual(code, 0, log)
        self.assertTrue(execution["outcomes_valid"], log)
        discovery = json.loads(payloads[".discovery.json"])
        counts = qualification.validate_python_result(discovery, payloads[".functions.jsonl"], payloads[".rows.jsonl"])
        self.assertEqual(counts["python_rows"], 2)

    def test_public_worker_rejects_one_omitted_required_row(self):
        source = self.fixture_source(
            "        with self.subTest(case='positive/row'):\n            self.assertEqual(2 + 2, 4)\n"
        )
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
        return {"tests": [{"name": name, "properties": [{"name": "LABELS", "value": labels}]} for name, labels in rows]}

    def test_relevance_and_runtime_selection_are_orthogonal(self):
        inventory = self.inventory(
            ("behavior", ["host", "product"]),
            ("boundary", ["host", "boundary"]),
            ("runner", ["host", "tooling"]),
            ("data", ["host", "corpus"]),
            ("device", ["gpu", "product"]),
        )
        groups = ["product", "boundary", "tooling", "corpus"]
        self.assertEqual(
            qualification.membership(inventory, groups, "all"),
            {"product": ["behavior", "device"], "boundary": ["boundary"], "tooling": ["runner"], "corpus": ["data"]},
        )
        self.assertEqual(qualification.membership(inventory, ["product"], "host"), {"product": ["behavior"]})

    def test_empty_inventory_fails(self):
        with self.assertRaisesRegex(ValueError, "required checkpoint selection is empty"):
            qualification.membership(self.inventory(), ["product"], "host")

    def test_missing_runtime_prerequisite_fails(self):
        with self.assertRaisesRegex(ValueError, "one relevance and a runtime prerequisite"):
            qualification.membership(self.inventory(("a", ["product"])), ["product"], "host")

    def test_multiple_relevance_labels_fail(self):
        with self.assertRaisesRegex(ValueError, "one relevance and a runtime prerequisite"):
            qualification.membership(self.inventory(("a", ["host", "product", "tooling"])), ["product"], "host")

    def test_duplicate_registration_fails(self):
        with self.assertRaisesRegex(ValueError, "duplicate CTest registration"):
            qualification.membership(
                self.inventory(("a", ["host", "product"]), ("a", ["host", "product"])), ["product"], "host"
            )

    def test_empty_group_selection_fails(self):
        with self.assertRaisesRegex(ValueError, "invalid checkpoint selection"):
            qualification.membership(self.inventory(("a", ["host", "product"])), [], "host")

    def test_selection_with_missing_group_membership_fails(self):
        with self.assertRaisesRegex(ValueError, "required checkpoint selection is empty"):
            qualification.membership(self.inventory(("a", ["host", "product"])), ["product", "boundary"], "host")

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
            self.assertEqual(
                qualification.membership(inventory, ["product", "boundary"], "all"),
                {"product": ["a"], "boundary": ["b"]},
            )
            inventory["tests"][1]["command"] = ["case", "--scratch-root", "relative"]
            with self.assertRaisesRegex(ValueError, "scratch root must be absolute"):
                qualification.membership(inventory, ["product", "boundary"], "all")


class CollectionTests(unittest.TestCase):
    def test_rust_artifact_description_requires_separate_variants(self):
        cases = (
            ("historical-valid", None, False),
            ("v1-preview-valid", "ON", False),
            ("missing-member", None, False),
            ("archive-alias", None, False),
            ("directory-alias", None, False),
            ("production-feature", None, False),
            ("dropped-description-flag", None, False),
            ("v2-test-support-feature", "ON", True),
            *((f"v2-{value}-{claim}", value, True) for value in PREVIEW_CASES for claim in ("valid", "mismatch")),
        )
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        build = root / "build"
        self.subtest_cases = tuple(case for case, _, _ in cases)
        for case, value, v2 in cases:
            with self.subTest(case=case):
                configuration = {}
                preview = False
                if value is not None:
                    preview = configure_preview_option(root, build, value)
                    raw = verify.cache_values(build)["JUICER_GRAIN_PREVIEW"]
                    self.assertEqual(raw, value)
                    self.assertEqual(preview, value != "OFF")
                    configuration["configuration"] = {"JUICER_GRAIN_PREVIEW": raw}
                preview_claimed = v2 and (preview != case.endswith("-mismatch"))
                artifacts = {
                    "schema": "rust-artifacts-v2" if v2 else "rust-artifacts-v1",
                    "variants": {
                        variant: {
                            "archive": f"/unavailable/{variant}/plugin.a",
                            "target_dir": f"/unavailable/{variant}",
                            "features": features,
                        }
                        for variant, features in (
                            ("production", ["grain-preview"] if preview_claimed else []),
                            ("test-support", ["test-support"]),
                        )
                    },
                }
                if case == "archive-alias":
                    artifacts["variants"]["test-support"]["archive"] = artifacts["variants"]["production"]["archive"]
                elif case == "directory-alias":
                    artifacts["variants"]["test-support"]["target_dir"] = artifacts["variants"]["production"][
                        "target_dir"
                    ]
                elif case == "production-feature":
                    artifacts["variants"]["production"]["features"] = ["test-support"]
                elif case == "v2-test-support-feature":
                    artifacts["variants"]["test-support"]["features"] = ["grain-preview", "test-support"]
                payloads = {
                    name: b"{}"
                    for name in (
                        "registration.json",
                        "target-discovery.json",
                        "host-contracts.json",
                        "host-targets.json",
                        "gpu-contracts.json",
                        "gpu-targets.json",
                        "python-contracts.json",
                        "python-targets.json",
                    )
                }
                payloads.update(
                    {
                        "candidate.json": json.dumps(
                            {
                                "head": "source",
                                "preset": "linux-debug",
                                **configuration,
                                "features": {"rust_artifacts": "rust-artifacts.json"},
                            }
                        ).encode(),
                        "candidate.patch": b"source patch",
                        "rust-artifacts.json": json.dumps(artifacts).encode(),
                    }
                )
                if case == "missing-member":
                    del payloads["rust-artifacts.json"]
                result = {
                    "status": "failed",
                    "preset": "linux-debug",
                    "scope": "focused",
                    "runtime": "host",
                    "groups": [],
                    "selection": [],
                    "build_kind": "product",
                    "candidate_head": "source",
                    "shipping_isolation": False,
                    "rust_artifacts": case != "dropped-description-flag",
                    "members": {
                        name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                        for name, data in payloads.items()
                    },
                }
                if case.endswith("-valid"):
                    self.assertEqual(qualification.validate_tests_result(result, payloads)["status"], "failed")
                elif v2:
                    with self.assertRaisesRegex(ValueError, "invalid separate Rust artifact description"):
                        qualification.validate_tests_result(result, payloads)
                else:
                    with self.assertRaises(ValueError):
                        qualification.validate_tests_result(result, payloads)

    def test_quality_omitted_operation_and_undrained_command_are_rejected(self):
        checks = {
            "schema": "native-quality-checks-v1",
            "status": "passed",
            "expected_checks": ["rust-quality", "native-boundary-enforcement"],
            "completed_checks": ["native-boundary-enforcement"],
            "expected_commands": ["clippy"],
            "commands": [],
            "python_cases": ["quality-boundary"],
        }
        with self.assertRaisesRegex(ValueError, "missing or duplicate quality operation"):
            qualification.validate_quality_checks(checks)
        checks["completed_checks"] = checks["expected_checks"]
        checks["commands"] = [{"name": "clippy", "returncode": 0, "drained": False, "process_status": "completed"}]
        with self.assertRaisesRegex(ValueError, "undrained quality operation"):
            qualification.validate_quality_checks(checks)

    def test_focused_saved_passes_cannot_form_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            options = SimpleNamespace(
                bundles=[Path("first"), Path("second")],
                output=Path(folder) / "report.json",
                selection=["product", "boundary"],
                runtime="host",
                presets=["linux-debug"],
            )
            summaries = [
                {
                    "status": "passed",
                    "preset": "linux-debug",
                    "scope": "focused",
                    "groups": [group],
                    "runtime": "host",
                    "candidate_source": "same",
                }
                for group in ("product", "boundary")
            ]
            with patch.object(qualification, "read_native_result", side_effect=summaries):
                with self.assertRaisesRegex(ValueError, "one complete fresh execution"):
                    qualification.collect_bundles(options)
            self.assertFalse(options.output.exists())

    def test_checkpoint_requires_same_candidate_and_all_requested_presets(self):
        with tempfile.TemporaryDirectory() as folder:
            options = SimpleNamespace(
                bundles=[Path("first"), Path("second")],
                output=Path(folder) / "report.json",
                selection=["boundary"],
                runtime="host",
                presets=["linux-debug", "windows-clang-debug"],
            )
            summaries = [
                {
                    "status": "passed",
                    "preset": preset,
                    "scope": "checkpoint",
                    "groups": ["boundary"],
                    "runtime": "host",
                    "candidate_source": preset,
                }
                for preset in options.presets
            ]
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
            payloads = {
                "candidate.json": json.dumps(
                    {"head": "candidate", "preset": "linux-debug", "inputs": {"/unavailable/source": "identity"}}
                ).encode(),
                "candidate.patch": b"reconstructible source patch\n",
            }
            result = {
                "schema": "native-build-v1",
                "status": "passed",
                "preset": "linux-debug",
                "scope": "build",
                "candidate_head": "candidate",
                "candidate_unchanged": True,
                "shipping_isolation": False,
                "commands": [{"returncode": 0, "drained": True, "process_status": "completed"}],
                "members": {
                    name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                    for name, data in payloads.items()
                },
            }
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
                "exec(code, {'__name__':'__main__','__file__':str(root/'verify.py')})\n",
                encoding="utf-8",
            )
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
