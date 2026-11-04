"""Host-domain membership/outcomes are data checks, with no product rebuilds."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qualification import (
    OPTICS_FUNCTION, case_events, framework_terminals, domain_result_members, native_registration,
    python_registration, rust_case_events, rust_membership, tests_result_members,
    validate_domain_result,
)


def encoded(value):
    return json.dumps(value).encode()


class HostResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        command = ["/unavailable/build/Consumer", "fixture"]
        key = "host-" + hashlib.sha256(json.dumps(command, ensure_ascii=False).encode()).hexdigest()[:16]
        contract = {"schema": "native-host-contracts-v1", "required": [],
                    "native_cohorts": {}, "gtest_cohorts": {}, "rust_cohorts": {},
                    "targets": {"Consumer": {"protocol": "custom", "registrations": 1}}}
        registration = {"tests": [{"name": "Native.Host.Rows", "command": ["/unavailable/tools/python", "verify.py", "native-case", "--protocol", "custom", "--", *command],
                                    "properties": [{"name": "LABELS", "value": ["host"]}]}]}
        cls.payloads = {
            "candidate.json": encoded({"head": "source", "preset": "linux-debug", "inputs": {"source": "identity"},
                                       "configuration": {"BUILD_TESTING": "ON", "CMAKE_BUILD_TYPE": "Debug"}}),
            "candidate.patch": b"reconstructible patch\n", "host-contracts.json": encoded(contract),
            "host-targets.json": encoded({"Consumer": command[0]}), "registration.json": encoded(registration),
            "target-discovery.json": b"{}", "ctest.xml": b'<testsuite><testcase name="Native.Host.Rows" status="run"/></testsuite>',
            key + ".list.jsonl": b'{"caller":"cpp","id":"fixture-row"}\n',
            key + ".jsonl": b'{"caller":"cpp","id":"fixture-row","event":"start"}\n{"caller":"cpp","id":"fixture-row","event":"terminal","outcome":"passed"}\n',
            key + ".execution.json": encoded({"command": command, "returncode": 0, "outcomes_valid": True}),
        }
        cls.result = {"schema": "native-host-v2", "preset": "linux-debug", "status": "passed", "scope": "focused",
                      "candidate_head": "source", "candidate_unchanged": True, "shipping_isolation": False,
                      "selection": ["Native.Host.Rows"], "case_files": {"Native.Host.Rows": {"key": key, "protocol": "custom", "rows": False}},
                      "commands": [{"returncode": 0, "drained": True, "process_status": "completed"}], "members": {}}
        actual = os.environ.get("FJ_HOST_COMPLETED_BUNDLE")
        if actual:
            directory = Path(actual)
            cls.result = json.loads((directory / "result.json").read_bytes())
            cls.payloads = {name: (directory / name).read_bytes() for name in cls.result["members"]}
        cls.template_result, cls.template_payloads = cls.result, cls.payloads

    def setUp(self):
        self.result, self.payloads = copy.deepcopy(self.template_result), dict(self.template_payloads)
        self.custom = next(e["key"] for e in self.result["case_files"].values() if e["protocol"] == "custom")

    def identities(self):
        self.result["members"] = {n: {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)} for n, b in self.payloads.items()}

    def check(self):
        self.identities()
        return validate_domain_result(self.result, self.payloads)

    def test_complete_host_results(self):
        self.assertEqual(self.check()["status"], "passed")

    def test_missing_registration(self):
        document = json.loads(self.payloads["registration.json"])
        document["tests"] = [t for t in document["tests"] if t["name"] != self.result["selection"][0]]
        self.payloads["registration.json"] = encoded(document)
        with self.assertRaisesRegex(ValueError, "missing"):
            self.check()

    @staticmethod
    def registration_row(name, target, protocol="custom", *, runtime="host"):
        command = ["python", "verify.py", "native-case", "--preset", "linux-debug",
                   "--protocol", protocol]
        if runtime != "host":
            command.extend(("--runtime", runtime))
        command.extend(("--", "/build/" + target))
        return {"name": name, "command": command,
                "properties": [{"name": "LABELS", "value": [runtime]}]}

    def test_cuda_free_native_registration_accepts_only_selected_eligible_target(self):
        contract = {
            "schema": "native-host-contracts-v1",
            "required": ["Ffi.Host.Abi"],
            "cuda_free_required": ["Ffi.Host.Abi"],
            "targets": {
                "JuicerCudaAbiTests": {"protocol": "custom", "registrations": 1, "cuda_free": True},
                "JuicerFilmRecipeTests": {"protocol": "custom", "registrations": 1},
            },
        }
        paths = {"JuicerCudaAbiTests": "/build/JuicerCudaAbiTests"}
        registration = {"tests": [self.registration_row("Ffi.Host.Abi", "JuicerCudaAbiTests")]}
        found = native_registration(contract, paths, registration, runtime="host", cuda_free=True)
        self.assertEqual(set(found), {"Ffi.Host.Abi"})
        self.assertEqual(found["Ffi.Host.Abi"]["target"], "JuicerCudaAbiTests")

    def test_cuda_free_native_registration_rejects_missing_eligible_target(self):
        contract = {
            "schema": "native-host-contracts-v1", "required": [],
            "cuda_free_required": [],
            "targets": {"JuicerHashContractTests": {
                "protocol": "gtest", "registrations": 1, "cuda_free": True}},
        }
        registration = {"tests": [self.registration_row(
            "Hash.Host.Contract", "JuicerHashContractTests", "gtest")]}
        with self.assertRaisesRegex(ValueError, "target contract|CUDA-free|eligible"):
            native_registration(contract, {}, registration, runtime="host", cuda_free=True)

    def test_cuda_free_native_registration_rejects_extra_product_target(self):
        contract = {
            "schema": "native-host-contracts-v1", "required": [],
            "cuda_free_required": ["Ffi.Host.Abi"],
            "targets": {
                "JuicerCudaAbiTests": {"protocol": "custom", "registrations": 1, "cuda_free": True},
                "JuicerFilmRecipeTests": {"protocol": "custom", "registrations": 1},
            },
        }
        paths = {"JuicerCudaAbiTests": "/build/JuicerCudaAbiTests",
                 "JuicerFilmRecipeTests": "/build/JuicerFilmRecipeTests"}
        registration = {"tests": [
            self.registration_row("Ffi.Host.Abi", "JuicerCudaAbiTests"),
            self.registration_row("Ffi.Host.FilmRecipe", "JuicerFilmRecipeTests"),
        ]}
        with self.assertRaisesRegex(ValueError, "target contract|CUDA-free|eligible"):
            native_registration(contract, paths, registration, runtime="host", cuda_free=True)

    def test_cuda_free_native_registration_requires_every_declared_registration(self):
        contract = {
            "schema": "native-host-contracts-v1", "required": [],
            "cuda_free_required": ["Ffi.Host.Abi"],
            "targets": {"JuicerCudaAbiTests": {
                "protocol": "custom", "registrations": 1, "cuda_free": True}},
        }
        paths = {"JuicerCudaAbiTests": "/build/JuicerCudaAbiTests"}
        with self.assertRaisesRegex(ValueError, "missing"):
            native_registration(contract, paths, {"tests": []}, runtime="host", cuda_free=True)

    def test_cuda_free_gpu_registration_requires_an_empty_closure(self):
        contract = {"schema": "native-gpu-contracts-v1", "required": ["Gpu.Test"],
                    "cuda_free_required": [],
                    "targets": {"GpuTests": {"protocol": "gtest", "registrations": 1}}}
        self.assertEqual(native_registration(contract, {}, {"tests": []},
                                             runtime="gpu", cuda_free=True), {})
        with self.assertRaisesRegex(ValueError, "target contract|CUDA-free|eligible"):
            native_registration(contract, {"GpuTests": "/build/GpuTests"}, {"tests": []},
                                runtime="gpu", cuda_free=True)

    def test_full_native_registration_still_requires_all_targets_and_counts(self):
        contract = {"schema": "native-host-contracts-v1",
                    "required": ["Ffi.Host.Abi", "Route.Host.Bridge"],
                    "targets": {
                        "JuicerCudaAbiTests": {"protocol": "custom", "registrations": 1},
                        "JuicerRouteBridgeTests": {"protocol": "custom", "registrations": 1},
                    }}
        paths = {name: "/build/" + name for name in contract["targets"]}
        rows = [self.registration_row("Ffi.Host.Abi", "JuicerCudaAbiTests"),
                self.registration_row("Route.Host.Bridge", "JuicerRouteBridgeTests")]
        self.assertEqual(set(native_registration(contract, paths, {"tests": rows})),
                         {"Ffi.Host.Abi", "Route.Host.Bridge"})
        with self.assertRaisesRegex(ValueError, "registration"):
            native_registration(contract, paths, {"tests": rows[:1]})

    def test_cuda_free_python_registration_accepts_tooling_and_marked_rows(self):
        contract = {"schema": "python-contracts-v1", "registrations": {
            "Quality.Tooling": {"source": "tests/quality/test_tooling.py", "tooling": True},
            "Quality.NativeBoundary": {"source": "tests/ffi/test_native_boundary.py", "cuda_free": True},
            "Quality.Product": {"source": "tests/quality/test_product.py"},
        }}
        inventory = {"tests": [
            {"name": "Quality.Tooling", "command": ["python", "verify.py", "python-tests", "--",
                                                          "-m", "unittest", "discover", "-s",
                                                          "tests/quality", "-p", "test_tooling.py"]},
            {"name": "Quality.NativeBoundary", "command": ["python", "verify.py", "python-tests", "--",
                                                              "-m", "unittest", "discover", "-s",
                                                              "tests/ffi", "-p", "test_native_boundary.py"]},
        ]}
        self.assertEqual(set(python_registration(contract, inventory, "linux-debug", cuda_free=True)),
                         {"Quality.Tooling", "Quality.NativeBoundary"})

    def test_cuda_free_python_registration_rejects_unmarked_product_row(self):
        contract = {"schema": "python-contracts-v1", "registrations": {
            "Quality.Tooling": {"source": "tests/quality/test_tooling.py", "tooling": True},
            "Quality.Product": {"source": "tests/quality/test_product.py"},
        }}
        inventory = {"tests": [
            {"name": "Quality.Tooling", "command": ["python", "verify.py", "python-tests", "--",
                                                          "-m", "unittest", "discover", "-s",
                                                          "tests/quality", "-p", "test_tooling.py"]},
            {"name": "Quality.Product", "command": ["python", "verify.py", "python-tests", "--",
                                                          "-m", "unittest", "discover", "-s",
                                                          "tests/quality", "-p", "test_product.py"]},
        ]}
        with self.assertRaisesRegex(ValueError, "unexpected Python registration"):
            python_registration(contract, inventory, "linux-debug", cuda_free=True)

    def test_full_python_registration_keeps_product_contract(self):
        contract = {"schema": "python-contracts-v1", "registrations": {
            "Quality.Tooling": {"source": "tests/quality/test_tooling.py", "tooling": True},
            "Quality.Product": {"source": "tests/quality/test_product.py"},
        }}
        inventory = {"tests": [
            {"name": "Quality.Tooling", "command": ["python", "verify.py", "python-tests", "--",
                                                          "-m", "unittest", "discover", "-s",
                                                          "tests/quality", "-p", "test_tooling.py"]},
            {"name": "Quality.Product", "command": ["python", "verify.py", "python-tests", "--",
                                                          "-m", "unittest", "discover", "-s",
                                                          "tests/quality", "-p", "test_product.py"]},
        ]}
        self.assertEqual(set(python_registration(contract, inventory, "linux-debug")),
                         {"Quality.Tooling", "Quality.Product"})

    def test_cuda_free_results_are_focused_and_cannot_claim_checkpoint(self):
        names = {"candidate.json", "candidate.patch", "registration.json", "target-discovery.json"}
        for runtime in ("host", "gpu", "python"):
            names.update({runtime + "-contracts.json", runtime + "-targets.json"})
        result = {"preset": "linux-debug", "scope": "focused", "runtime": "host",
                  "build_kind": "cuda-free", "shipping_isolation": False,
                  "groups": [], "selection": [], "status": "failed", "members": {
                      name: {"bytes": 0, "sha256": "0" * 64} for name in names}}
        self.assertEqual(set(tests_result_members(result)), names)
        result["scope"] = "checkpoint"
        with self.assertRaisesRegex(ValueError, "CUDA-free|focused"):
            tests_result_members(result)

    def test_missing_custom_row_terminal(self):
        name = self.custom + ".jsonl"
        self.payloads[name] = b"\n".join(self.payloads[name].splitlines()[:-1]) + b"\n"
        with self.assertRaisesRegex(ValueError, "missing"):
            self.check()

    def test_duplicate_custom_terminal(self):
        name = self.custom + ".jsonl"
        self.payloads[name] += self.payloads[name].splitlines()[-1] + b"\n"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.check()

    def test_failed_custom_row_despite_zero_exit(self):
        name = self.custom + ".jsonl"
        self.payloads[name] = self.payloads[name].replace(b'"passed"', b'"failed"', 1)
        with self.assertRaisesRegex(ValueError, "failed"):
            self.check()

    def test_truncated_custom_row(self):
        name = self.custom + ".jsonl"
        self.payloads[name] = self.payloads[name][:-3]
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.check()

    def test_missing_case_file_rejected_before_payload_reads(self):
        self.identities()
        del self.result["members"][self.custom + ".jsonl"]
        with patch.object(Path, "read_bytes", side_effect=AssertionError("payload reopened")):
            with self.assertRaisesRegex(ValueError, "missing"):
                domain_result_members(self.result)

    def test_incorrect_native_protocol(self):
        document = json.loads(self.payloads["registration.json"])
        command = next(t["command"] for t in document["tests"] if "native-case" in t.get("command", []))
        command[command.index("--protocol") + 1] = "unknown"
        self.payloads["registration.json"] = encoded(document)
        with self.assertRaisesRegex(ValueError, "protocol"):
            self.check()

    def test_nonzero_command(self):
        self.result["commands"][-1]["returncode"] = 7
        with self.assertRaisesRegex(ValueError, "failed command"):
            self.check()

    def test_missing_native_caller(self):
        name = self.custom + ".list.jsonl"
        self.payloads[name] += b'{"caller":"missing-caller","id":"required-row"}\n'
        with self.assertRaisesRegex(ValueError, "missing"):
            self.check()

    def test_missing_ctest_terminal(self):
        self.payloads["ctest.xml"] = b"<testsuite/>"
        with self.assertRaisesRegex(ValueError, "missing"):
            self.check()

    def test_undrained_command(self):
        self.result["commands"][-1]["drained"] = False
        with self.assertRaisesRegex(ValueError, "undrained"):
            self.check()

    def test_required_framework_skip(self):
        with self.assertRaisesRegex(ValueError, "skipped"):
            framework_terminals(b'<testsuites><testsuite name="Suite"><testcase name="Case" status="run"><skipped/></testcase></testsuite></testsuites>', {("Suite", "Case")})

    def test_missing_framework_function(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            framework_terminals(b'<testsuites><testsuite name="Suite"/></testsuites>', {("Suite", "Case")})

    def test_duplicate_framework_function(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            framework_terminals(b'<testsuites><testsuite name="Suite"><testcase name="Case" status="run"/><testcase name="Case" status="run"/></testsuite></testsuites>', {("Suite", "Case")})

    def test_copied_reader_is_offline_and_opens_members_once(self):
        self.identities()
        with tempfile.TemporaryDirectory(prefix="host results ü ") as scratch:
            root = Path(scratch)
            bundle = root / "relocated bundle ü"
            bundle.mkdir()
            for name, data in self.payloads.items():
                (bundle / name).write_bytes(data)
            (bundle / "result.json").write_bytes(encoded(self.result))
            shutil.copyfile(Path(__file__).resolve().parents[2] / "scripts/qualification.py", root / "reader.py")
            script = r'''
import collections, json, pathlib, runpy, sys
root = pathlib.Path(sys.argv[1]).resolve()
# Import stdlib before the guard; the reader itself may open only bundled files.
import argparse, hashlib, xml.etree.ElementTree, time, pkgutil, __future__
argparse.ArgumentParser()
opened = collections.Counter()
def audit(event, args):
    if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.exec'):
        raise RuntimeError('offline reader attempted execution')
    if event == 'open' and isinstance(args[0], (str, bytes)):
        p = pathlib.Path(args[0]).resolve()
        if p != root / 'reader.py' and p.parent != root / 'relocated bundle ü':
            raise RuntimeError('offline reader left bundle: ' + str(p))
        if p.suffix.lower() in ('.exe', '.dll', '.so', '.a', '.lib'):
            raise RuntimeError('executable bytes opened')
        opened[str(p)] += 1
sys.addaudithook(audit)
sys.argv = [str(root / 'reader.py'), '--bundle', str(root / 'relocated bundle ü')]
try:
    runpy.run_path(str(root / 'reader.py'), run_name='__main__')
except SystemExit as exc:
    if exc.code != 0:
        raise
for p in (root / 'relocated bundle ü').iterdir():
    assert opened[str(p.resolve())] == 1, (p, opened)
'''
            completed = subprocess.run([sys.executable, "-B", "-c", script, str(root)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_windows_ads_rejected_before_payload_io(self):
        self.identities()
        result = self.result
        self.subtest_cases = ("payload.json:stream", "C:payload.json", "C:/payload.json")
        for name in ("payload.json:stream", "C:payload.json", "C:/payload.json"):
            mutated = copy.deepcopy(result)
            mutated["members"][name] = {"bytes": 0, "sha256": "0" * 64}
            with self.subTest(case=name, name=name), patch.object(Path, "read_bytes", side_effect=AssertionError("payload reopened")):
                with self.assertRaisesRegex(ValueError, "unconfined"):
                    domain_result_members(mutated)

    test_windows_ads_rejected_before_payload_io.required_subtests = True

class CohortTests(unittest.TestCase):
    def events(self, *, rust=False):
        context = {"caller": "Suite.Function/parameters"}
        if rust:
            context = {"caller": "rust", "binary": "core", "function": "fixture", "cohort": "parameters"}
        rows = [{**context, "event": "membership", "ids": ["first", "second"]}]
        for case_id in ("first", "second"):
            rows.extend([{**context, "id": case_id, "event": "start"},
                         {**context, "id": case_id, "event": "terminal", "outcome": "passed"}])
        if rust:
            rows.append({**context, "event": "cohort-terminal", "outcome": "passed"})
        return rows

    def payload(self, rows):
        return b"".join(encoded(r) + b"\n" for r in rows)

    def test_native_parameter_membership(self):
        self.assertEqual(case_events(self.payload(self.events()), cohorts={"Suite.Function": 1}), 2)

    def test_independent_listing_and_cohort_agree(self):
        expected = {("Suite.Function/parameters", case_id) for case_id in ("first", "second")}
        self.assertEqual(case_events(self.payload(self.events()), expected, cohorts=1), 2)

    def test_missing_entire_required_cohort(self):
        with self.assertRaisesRegex(ValueError, "required cohort"):
            case_events(b"", cohorts=1)

    def test_skipped_parameter(self):
        with self.assertRaisesRegex(ValueError, "missing case"):
            case_events(self.payload(self.events()[:-2]), cohorts=1)

    def test_missing_declaration(self):
        with self.assertRaisesRegex(ValueError, "unexpected case"):
            case_events(self.payload(self.events()[1:]), cohorts=1)

    def test_wrong_framework_context(self):
        with self.assertRaisesRegex(ValueError, "context"):
            case_events(self.payload(self.events()), cohorts={"Other.Function": 1})

    def test_failed_late_assertion(self):
        rows = self.events()
        rows[-1]["outcome"] = "failed"
        with self.assertRaisesRegex(ValueError, "failed"):
            case_events(self.payload(rows), cohorts=1)

    def test_rust_cohort_requires_selected_function(self):
        rows = self.events(rust=True)
        self.assertEqual(case_events(self.payload(rows), cohorts={"core": 1}, functions={("core", "fixture")}), 2)
        with self.assertRaisesRegex(ValueError, "context"):
            case_events(self.payload(rows), cohorts={"core": 1}, functions={("core", "other")})

    def test_rust_worker_requires_all_declared_binary_cohorts(self):
        rows = self.payload(self.events(rust=True))
        contract = {"rust_cohorts": {"core": 1}}
        self.assertEqual(rust_case_events(rows, contract, {("core", "fixture")}), 2)
        contract["rust_cohorts"]["core::other"] = 0
        with self.assertRaisesRegex(ValueError, "binary"):
            rust_case_events(rows, contract, {("core", "fixture")})

    def test_ordinary_rust_worker_rejects_ambient_pilot_filter(self):
        from argparse import Namespace
        from verify import rust_tests
        with patch.dict(os.environ, {"FJ_RUST_FILTER": OPTICS_FUNCTION}, clear=True):
            with self.assertRaisesRegex(ValueError, "pilot environment is retired"):
                rust_tests(Namespace(package="core"))

    def test_rust_completion_after_all_rows(self):
        rows = self.events(rust=True)
        with self.assertRaisesRegex(ValueError, "completion"):
            case_events(self.payload(rows[:-1]), cohorts={"core": 1}, functions={("core", "fixture")})
        rows.insert(1, rows.pop())
        with self.assertRaisesRegex(ValueError, "premature"):
            case_events(self.payload(rows), cohorts={"core": 1}, functions={("core", "fixture")})

    def test_inherited_gtest_selection_is_removed(self):
        from verify import gtest_environment
        env = {"PATH": "tools", "GTEST_FILTER": "nothing", "GTEST_REPEAT": "0", "GTEST_TOTAL_SHARDS": "2",
               "GTEST_SHARD_INDEX": "1", "GTEST_ALSO_RUN_DISABLED_TESTS": "1", "TESTBRIDGE_TEST_ONLY": "nothing"}
        self.assertEqual(gtest_environment(env), {"PATH": "tools"})



class RustMembershipTests(unittest.TestCase):
    def setUp(self):
        self.cargo = {"packages": [
            {"name": "film-juicer-core", "id": "core", "targets": [
                {"name": "film_juicer_core", "test": True, "kind": ["lib"]},
                {"name": "integration", "test": True, "kind": ["test"]}]},
            {"name": "film-juicer-plugin", "id": "bridge", "targets": [
                {"name": "film_juicer_plugin", "test": True, "kind": ["staticlib"]}]},
        ]}
        self.binaries = {"rust-binaries": {
            "core": {"package-id": "core", "binary-name": "film_juicer_core"},
            "core::integration": {"package-id": "core", "binary-name": "integration"},
            "bridge": {"package-id": "bridge", "binary-name": "film_juicer_plugin"},
        }}
        def case(matches):
            return {"ignored": False, "filter-match": {"status": "matches" if matches else "mismatch"}}
        self.discovery = {"rust-suites": {
            "core": {"testcases": {OPTICS_FUNCTION: case(True)}},
            "core::integration": {"testcases": {"fixture": case(True)}},
            "bridge": {"testcases": {"abi": case(False)}},
        }}

    def test_complete_package_and_filtered_peer(self):
        selected, filtered = rust_membership(self.discovery, "core", self.cargo, self.binaries)
        self.assertEqual(len(selected), 2)
        self.assertEqual(filtered, {("bridge", "abi")})

    def test_missing_integration_binary(self):
        del self.binaries["rust-binaries"]["core::integration"]
        del self.discovery["rust-suites"]["core::integration"]
        with self.assertRaisesRegex(ValueError, "binary"):
            rust_membership(self.discovery, "core", self.cargo, self.binaries)


    def test_required_function_filtered_out(self):
        self.discovery["rust-suites"]["core"]["testcases"][OPTICS_FUNCTION]["filter-match"]["status"] = "mismatch"
        with self.assertRaisesRegex(ValueError, "function"):
            rust_membership(self.discovery, "core", self.cargo, self.binaries)

    def test_ignored_function(self):
        self.discovery["rust-suites"]["core"]["testcases"][OPTICS_FUNCTION]["ignored"] = True
        with self.assertRaisesRegex(ValueError, "ignored"):
            rust_membership(self.discovery, "core", self.cargo, self.binaries)


@unittest.skipUnless(os.environ.get("FJ_HOST_NATIVE_ABI"), "requires the current CMake-built native ABI consumer")
class NativePathTests(unittest.TestCase):
    def test_real_native_writer_uses_unicode_result_path(self):
        from execution import NativeRun, OutputLease
        root = Path(__file__).resolve().parents[2]
        preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        directory = root / "out/validation" / preset / "native-host-controls" / ("results space ü 汉字 " + uuid.uuid4().hex)
        with OutputLease(root, root / "out/build" / preset):
            directory.mkdir(parents=True)
            run = NativeRun(root, directory)
            try:
                run.command("native-unicode", [sys.executable, "-B", str(root / "scripts/verify.py"),
                            "native-case", "--preset", preset, "--protocol", "custom", "--",
                            os.environ["FJ_HOST_NATIVE_ABI"]], env=dict(os.environ, FJ_HOST_BUNDLE=str(directory), FJ_HOST_TEST="Ffi.Host.CudaAbi"))
                execution = json.loads(next(directory.glob("host-*.execution.json")).read_bytes())
                self.assertTrue(execution["outcomes_valid"])
                self.assertTrue(all(c["drained"] for c in run.commands))
            except BaseException:
                print("Native Unicode control retained:", directory, file=sys.stderr)
                raise
            else:
                shutil.rmtree(directory)


@unittest.skipUnless(os.environ.get("FJ_HOST_NATIVE_GTEST"), "requires the current CMake-built GoogleTest consumer")
class NativeWriterFailureTests(unittest.TestCase):
    def assert_output_failure(self, output):
        from execution import NativeRun, OutputLease
        root = Path(__file__).resolve().parents[2]
        preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        directory = root / "out/validation" / preset / "native-host-controls" / ("writer-failure-" + uuid.uuid4().hex)
        with OutputLease(root, root / "out/build" / preset):
            directory.mkdir(parents=True)
            run = NativeRun(root, directory)
            try:
                with self.assertRaises(subprocess.CalledProcessError):
                    run.command("writer-error", [os.environ["FJ_HOST_NATIVE_GTEST"],
                                "--gtest_filter=CudaFailure.RuntimeCodesSelectRecoveryWithoutText"],
                                env=dict(os.environ, FJ_HOST_CASES=str(directory / output)), timeout=10)
                self.assertTrue(run.commands[0]["drained"])
                self.assertIn("native case", (directory / "writer-error.log").read_text())
                self.assertFalse((directory / "result.json").exists())
            except BaseException:
                print("Native writer control retained:", directory, file=sys.stderr)
                raise
            else:
                shutil.rmtree(directory)

    def test_output_open_failure_fails_the_native_process(self):
        self.assert_output_failure("missing/rows.jsonl")

    @unittest.skipUnless(os.name == "posix", "requires the POSIX full output device")
    def test_output_flush_failure_fails_the_native_process(self):
        self.assert_output_failure("/dev/full")


@unittest.skipUnless(os.environ.get("FJ_HOST_NATIVE_GTEST"), "requires the current CMake-built GoogleTest consumer")
class NativeFilterTests(unittest.TestCase):
    def test_inherited_filter_repeat_and_shards_cannot_hide_functions(self):
        from execution import NativeRun, OutputLease
        from qualification import gtest_discovery
        root = Path(__file__).resolve().parents[2]
        preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        directory = root / "out/validation" / preset / "native-host-controls" / ("filter-" + uuid.uuid4().hex)
        with OutputLease(root, root / "out/build" / preset):
            directory.mkdir(parents=True)
            run = NativeRun(root, directory)
            environment = dict(os.environ, FJ_HOST_BUNDLE=str(directory), GTEST_FILTER="CudaFailure.MissingNativeCodeRemainsPreparation",
                               GTEST_REPEAT="0", GTEST_TOTAL_SHARDS="2", GTEST_SHARD_INDEX="1", TESTBRIDGE_TEST_ONLY="nothing")
            try:
                run.command("native-filter", [sys.executable, "-B", str(root / "scripts/verify.py"), "native-case",
                            "--preset", preset, "--protocol", "gtest", "--", os.environ["FJ_HOST_NATIVE_GTEST"]], env=environment)
                execution = json.loads(next(directory.glob("host-*.execution.json")).read_bytes())
                selected = gtest_discovery(next(directory.glob("host-*.discovery.json")).read_bytes())
                self.assertTrue(execution["outcomes_valid"])
                self.assertIn(("CudaFailure", "RuntimeCodesSelectRecoveryWithoutText"), selected)
                self.assertGreater(len(selected), 1)
                self.assertTrue(all(c["drained"] for c in run.commands))
            except BaseException:
                print("Native filter control retained:", directory, file=sys.stderr)
                raise
            else:
                shutil.rmtree(directory)



if __name__ == "__main__":
    unittest.main()
