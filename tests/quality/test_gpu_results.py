"""GPU-domain membership and exclusion evidence are offline data checks."""
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
from qualification import framework_terminals, gtest_discovery, domain_result_members, validate_domain_result


def encoded(value):
    return json.dumps(value).encode()


def synthetic_bundle():
    routes = {
        "Resource.Gpu.NegativeDirect": "negative-direct",
        "Resource.Gpu.NegativePrint": "negative-print",
    }
    commands = {
        name: ["/unavailable/build/GpuRoutes", route]
        for name, route in routes.items()
    }
    targets = {"GpuRoutes": "/unavailable/build/GpuRoutes"}
    contract = {
        "schema": "native-gpu-contracts-v1",
        "required": list(routes),
        "native_cohorts": {name: 1 for name in routes},
        "gtest_cohorts": {},
        "rust_cohorts": {},
        "targets": {"GpuRoutes": {"protocol": "custom", "registrations": len(routes), "rows": []}},
    }
    registrations = []
    case_files = {}
    payloads = {
        "candidate.json": encoded({
            "head": "synthetic-source",
            "preset": "linux-debug",
            "inputs": {"Resources/profiles/fixture.json": "source-identity"},
            "configuration": {"BUILD_TESTING": "ON", "CMAKE_BUILD_TYPE": "Debug"},
        }),
        "candidate.patch": b"synthetic candidate patch\n",
        "gpu-contracts.json": encoded(contract),
        "gpu-targets.json": encoded(targets),
        "target-discovery.json": encoded({}),
        "gpu-exclusion.json": encoded({"physical_gpu": "excluded"}),
        "ctest.xml": encoded({}),
    }
    ctest_cases = "".join(f'<testcase name="{name}" status="run"/>' for name in routes)
    payloads["ctest.xml"] = f"<testsuite>{ctest_cases}</testsuite>".encode()

    for name, route in routes.items():
        native_command = commands[name]
        key = "host-" + hashlib.sha256(json.dumps(native_command, ensure_ascii=False).encode()).hexdigest()[:16]
        launcher = [
            "/unavailable/tools/python", "verify.py", "native-case", "--runtime", "gpu",
            "--protocol", "custom", "--", *native_command,
        ]
        registrations.append({
            "name": name,
            "command": launcher,
            "properties": [
                {"name": "LABELS", "value": ["gpu"]},
                {"name": "TIMEOUT", "value": 600},
                {"name": "RESOURCE_LOCK", "value": ["juicer_gpu"]},
            ],
        })
        case_files[name] = {"key": key, "protocol": "custom", "rows": False}
        caller, case_id = "cpp", route
        payloads[key + ".list.jsonl"] = encoded({"caller": caller, "id": case_id}) + b"\n"
        events = [
            {"caller": caller, "event": "membership", "ids": [case_id]},
            {"caller": caller, "id": case_id, "event": "start"},
            {"caller": caller, "id": case_id, "event": "terminal", "outcome": "passed"},
        ]
        payloads[key + ".jsonl"] = b"".join(encoded(event) + b"\n" for event in events)
        payloads[key + ".execution.json"] = encoded({
            "command": native_command, "returncode": 0, "outcomes_valid": True,
        })

    payloads["registration.json"] = encoded({"tests": registrations})
    result = {
        "schema": "native-gpu-v1",
        "preset": "linux-debug",
        "status": "passed",
        "scope": "gpu-domains",
        "candidate_head": "synthetic-source",
        "candidate_unchanged": True,
        "shipping_isolation": False,
        "selection": list(routes),
        "case_files": case_files,
        "commands": [
            {"name": "gpu-exclusion", "returncode": 0, "drained": True, "process_status": "completed"},
            {"name": "ctest", "returncode": 0, "drained": True, "process_status": "completed"},
        ],
        "physical_gpu": {"contender_excluded": True, "drained_while_locked": True},
        "members": {},
    }
    return result, payloads


class GpuResultTests(unittest.TestCase):
    def test_only_explicit_disabled_benchmark_can_be_excluded(self):
        discovered = encoded({"testsuites": [{"name": "Suite", "testsuite": [
            {"name": "Correctness"}, {"name": "DISABLED_Timing"}]}]})
        with self.assertRaisesRegex(ValueError, "disabled"):
            gtest_discovery(discovered)
        excluded = {("Suite", "DISABLED_Timing")}
        self.assertEqual(gtest_discovery(discovered, excluded=excluded) - excluded, {("Suite", "Correctness")})
        with self.assertRaisesRegex(ValueError, "disabled"):
            gtest_discovery(discovered, excluded=excluded | {("Suite", "Correctness")})

    def test_declared_benchmark_requires_suppressed_terminal(self):
        expected, excluded = {("Suite", "Correctness")}, {("Suite", "DISABLED_Timing")}
        xml = b'<testsuites><testsuite name="Suite"><testcase name="Correctness" status="run"/><testcase name="DISABLED_Timing" status="notrun" result="suppressed"/></testsuite></testsuites>'
        framework_terminals(xml, expected, filtered=excluded)
        with self.assertRaisesRegex(ValueError, "terminal"):
            framework_terminals(xml.replace(b'status="notrun"', b'status="run"'), expected, filtered=excluded)
        with self.assertRaisesRegex(ValueError, "skipped"):
            framework_terminals(xml.replace(b'name="Correctness" status="run"', b'name="Correctness" status="notrun"'), expected, filtered=excluded)

    @classmethod
    def setUpClass(cls):
        cls.result, cls.payloads = synthetic_bundle()
        cls.actual_bundle = os.environ.get("FJ_GPU_COMPLETED_BUNDLE")
        cls.actual_result_bytes = None
        cls.actual_payloads = None
        if cls.actual_bundle:
            directory = Path(cls.actual_bundle)
            cls.actual_result_bytes = (directory / "result.json").read_bytes()
            actual_result = json.loads(cls.actual_result_bytes)
            cls.actual_payloads = {
                name: (directory / name).read_bytes() for name in actual_result["members"]
            }
            cls.actual_result = actual_result

    def setUp(self):
        self.result = copy.deepcopy(self.template_result)
        self.payloads = dict(self.template_payloads)
        self.routes = list(self.result["selection"])

    def reset_fixture(self):
        self.result, self.payloads = synthetic_bundle()
        self.routes = list(self.result["selection"])

    @property
    def template_result(self):
        return type(self).result

    @property
    def template_payloads(self):
        return type(self).payloads

    def identities(self):
        self.result["members"] = {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            for name, data in self.payloads.items()
        }

    def check(self):
        self.identities()
        return validate_domain_result(self.result, self.payloads)

    def key_for(self, test_name):
        return self.result["case_files"][test_name]["key"]

    def registration(self, test_name):
        document = json.loads(self.payloads["registration.json"])
        return document, next(test for test in document["tests"] if test["name"] == test_name)

    def write_registration(self, document):
        self.payloads["registration.json"] = encoded(document)

    def test_complete_gpu_result(self):
        summary = self.check()
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["scope"], "gpu-domains")
        self.assertEqual(summary["ctest_cases"], 2)
        self.assertEqual(summary["custom_cases"], 2)

    def test_missing_required_member_is_rejected_before_payload_io(self):
        self.identities()
        del self.result["members"]["gpu-exclusion.json"]
        with patch.object(Path, "read_bytes", side_effect=AssertionError("payload reopened")):
            with self.assertRaisesRegex(ValueError, "missing required host evidence"):
                domain_result_members(self.result)

    def test_missing_registration_and_missing_required_route(self):
        name = self.routes[1]
        document = json.loads(self.payloads["registration.json"])
        document["tests"] = [test for test in document["tests"] if test["name"] != name]
        self.payloads["registration.json"] = encoded(document)
        with self.assertRaisesRegex(ValueError, "missing"):
            self.check()

        self.result, self.payloads = synthetic_bundle()
        self.routes = list(self.result["selection"])
        contract = json.loads(self.payloads["gpu-contracts.json"])
        contract["required"].append("Resource.Gpu.PositiveDirect")
        self.payloads["gpu-contracts.json"] = encoded(contract)
        with self.assertRaisesRegex(ValueError, "missing"):
            self.check()

    def test_gpu_domain_scope_requires_every_registered_route(self):
        omitted = self.routes.pop()
        self.result["selection"].remove(omitted)
        del self.result["case_files"][omitted]
        with self.assertRaisesRegex(ValueError, "missing selected host domain"):
            self.check()

    def test_incorrect_host_or_gpu_runtime_is_rejected(self):
        document, test = self.registration(self.routes[0])
        next(p for p in test["properties"] if p["name"] == "LABELS")["value"] = ["host", "gpu"]
        self.write_registration(document)
        with self.assertRaisesRegex(ValueError, "incorrect native host runtime"):
            self.check()

        self.reset_fixture()
        document, test = self.registration(self.routes[0])
        command = test["command"]
        command[command.index("--runtime") + 1] = "host"
        self.write_registration(document)
        with self.assertRaisesRegex(ValueError, "launcher runtime"):
            self.check()

    def test_zero_missing_and_unbounded_timeout_are_rejected(self):
        document, test = self.registration(self.routes[0])
        next(p for p in test["properties"] if p["name"] == "TIMEOUT")["value"] = 0
        self.write_registration(document)
        with self.assertRaisesRegex(ValueError, "finite timeout"):
            self.check()

        self.reset_fixture()
        document, test = self.registration(self.routes[0])
        test["properties"] = [p for p in test["properties"] if p["name"] != "TIMEOUT"]
        self.write_registration(document)
        with self.assertRaisesRegex(ValueError, "finite timeout"):
            self.check()

        self.reset_fixture()
        document, test = self.registration(self.routes[0])
        next(p for p in test["properties"] if p["name"] == "TIMEOUT")["value"] = float("inf")
        self.write_registration(document)
        with self.assertRaisesRegex(ValueError, "finite timeout"):
            self.check()

    def test_missing_or_wrong_physical_gpu_lock_is_rejected(self):
        document, test = self.registration(self.routes[0])
        test["properties"] = [p for p in test["properties"] if p["name"] != "RESOURCE_LOCK"]
        self.write_registration(document)
        with self.assertRaisesRegex(ValueError, "physical-device lock"):
            self.check()

        self.reset_fixture()
        document, test = self.registration(self.routes[0])
        next(p for p in test["properties"] if p["name"] == "RESOURCE_LOCK")["value"] = ["other_gpu"]
        self.write_registration(document)
        with self.assertRaisesRegex(ValueError, "physical-device lock"):
            self.check()

    def test_gpu_exclusion_metadata_command_and_drainage_are_required(self):
        del self.result["physical_gpu"]
        with self.assertRaisesRegex(ValueError, "physical-device exclusion"):
            self.check()

        self.result, self.payloads = synthetic_bundle()
        self.routes = list(self.result["selection"])
        del self.payloads["gpu-exclusion.json"]
        with self.assertRaisesRegex(ValueError, "missing required host evidence"):
            self.check()

        self.result, self.payloads = synthetic_bundle()
        self.routes = list(self.result["selection"])
        self.payloads["gpu-exclusion.json"] = encoded({"physical_gpu": "contended"})
        with self.assertRaisesRegex(ValueError, "physical-device exclusion"):
            self.check()

        self.result, self.payloads = synthetic_bundle()
        self.routes = list(self.result["selection"])
        self.result["commands"] = [command for command in self.result["commands"] if command["name"] != "gpu-exclusion"]
        with self.assertRaisesRegex(ValueError, "physical-device exclusion"):
            self.check()

        self.reset_fixture()
        self.result["commands"][0]["returncode"] = 1
        with self.assertRaisesRegex(ValueError, "failed command"):
            self.check()

        self.result, self.payloads = synthetic_bundle()
        self.routes = list(self.result["selection"])
        self.result["physical_gpu"]["drained_while_locked"] = False
        with self.assertRaisesRegex(ValueError, "physical-device exclusion"):
            self.check()

        self.result, self.payloads = synthetic_bundle()
        self.routes = list(self.result["selection"])
        self.result["commands"][1]["drained"] = False
        with self.assertRaisesRegex(ValueError, "undrained"):
            self.check()

    def test_candidate_runtime_resources_are_required(self):
        candidate = json.loads(self.payloads["candidate.json"])
        candidate["inputs"] = {"src/Film.cpp": "source-identity"}
        self.payloads["candidate.json"] = encoded(candidate)
        with self.assertRaisesRegex(ValueError, "runtime resources"):
            self.check()

    def test_omitted_row_and_whole_cohort_are_rejected(self):
        key = self.key_for(self.routes[0])
        name = key + ".jsonl"
        rows = self.payloads[name].splitlines()
        self.payloads[name] = b"\n".join(rows[:-1]) + b"\n"
        with self.assertRaisesRegex(ValueError, "missing case outcomes"):
            self.check()

        self.result, self.payloads = synthetic_bundle()
        self.routes = list(self.result["selection"])
        key = self.key_for(self.routes[0])
        name = key + ".jsonl"
        self.payloads[name] = b"\n".join(row for row in self.payloads[name].splitlines() if b'"membership"' not in row) + b"\n"
        with self.assertRaisesRegex(ValueError, "required cohort"):
            self.check()

    def test_failed_or_truncated_case_events_are_rejected(self):
        key = self.key_for(self.routes[0])
        name = key + ".jsonl"
        self.payloads[name] = self.payloads[name].replace(b'"passed"', b'"failed"', 1)
        with self.assertRaisesRegex(ValueError, "failed or skipped"):
            self.check()

        self.result, self.payloads = synthetic_bundle()
        self.routes = list(self.result["selection"])
        key = self.key_for(self.routes[0])
        name = key + ".jsonl"
        self.payloads[name] = self.payloads[name][:-3]
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.check()

    def test_copied_reader_is_offline_and_opens_each_member_once(self):
        if self.actual_bundle:
            result = self.actual_result
            payloads = self.actual_payloads
            result_bytes = self.actual_result_bytes
        else:
            self.identities()
            result = self.result
            payloads = self.payloads
            result_bytes = encoded(result)

        with tempfile.TemporaryDirectory(prefix="GPU results ü ") as scratch:
            root = Path(scratch)
            bundle = root / "relocated bundle ü"
            bundle.mkdir()
            for name, data in payloads.items():
                (bundle / name).write_bytes(data)
            (bundle / "result.json").write_bytes(result_bytes)
            shutil.copyfile(Path(__file__).resolve().parents[2] / "scripts/qualification.py", root / "reader.py")
            script = r'''
import collections, json, pathlib, runpy, sys
root = pathlib.Path(sys.argv[1]).resolve()
import argparse, hashlib, stat, time, xml.etree.ElementTree, pkgutil, __future__
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


@unittest.skipUnless(os.environ.get("FJ_GPU_NATIVE_PROBE"), "requires the current native GPU Processor consumer")
class NativeGpuWriterTests(unittest.TestCase):
    def assert_output_failure(self, output):
        from execution import NativeRun, OutputLease
        root = Path(__file__).resolve().parents[2]
        preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        directory = root / "out/validation" / preset / "native-gpu-controls" / ("writer-failure-" + uuid.uuid4().hex)
        with OutputLease(root, root / "out/build" / preset), OutputLease(root):
            directory.mkdir(parents=True)
            run = NativeRun(root, directory)
            try:
                with self.assertRaises(subprocess.CalledProcessError):
                    run.command("gpu-writer-error", [os.environ["FJ_GPU_NATIVE_PROBE"], "--accepted-clean-captures"],
                                env=dict(os.environ, FJ_HOST_CASES=str(directory / output)), timeout=120)
                self.assertTrue(run.commands[0]["drained"])
                self.assertIn("native case results output", (directory / "gpu-writer-error.log").read_text())
                self.assertFalse((directory / "result.json").exists())
            except BaseException:
                print("Native GPU writer control retained:", directory, file=sys.stderr)
                raise
            else:
                shutil.rmtree(directory)

    def test_output_open_failure_fails_the_processor(self):
        self.assert_output_failure("missing/rows.jsonl")

    @unittest.skipUnless(os.name == "posix", "requires the POSIX full output device")
    def test_output_flush_failure_fails_the_processor(self):
        self.assert_output_failure("/dev/full")


if __name__ == "__main__":
    unittest.main()
