"""Step 1 completeness is checked as data; OS ownership gets small real witnesses."""

from __future__ import annotations

import copy
import hashlib
import json
import io
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import signal
import time
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import execution
import qualification
from qualification import (
    OPTICS_FUNCTION,
    OPTICS_TESTS,
    candidate_source_digest,
    read_native_result,
    validate_build_result,
    validate_native_result,
)


class NativeResultTests(unittest.TestCase):
    read_bytes = 0
    written_bytes = 0

    @classmethod
    def tearDownClass(cls):
        print(
            f"Reader mutation I/O: {cls.read_bytes} logical bundle bytes read; {cls.written_bytes} bytes written (Python/OS I/O excluded)."
        )

    @classmethod
    def setUpClass(cls):
        cls.read_bytes = cls.written_bytes = 0
        cls.rows = [f"row-{i}" for i in range(553)]
        cls.operations = [{"caller": "cpp", "id": "layout"}, {"caller": "c11", "id": "raw:null"}]
        native = [{"caller": c, "id": r} for c in ("c11", "cpp") for r in cls.rows] + cls.operations

        def events(cases):
            return b"".join(
                (
                    json.dumps({**r, "event": event, **({"outcome": "passed"} if event == "terminal" else {})}) + "\n"
                ).encode()
                for r in cases
                for event in ("start", "terminal")
            )

        candidate = {
            "head": "source-baseline",
            "preset": "linux-debug",
            "configuration": {"CMAKE_BUILD_TYPE": "Debug", "BUILD_TESTING": "ON"},
            "inputs": {"source.cpp": "identity"},
            "fixtures": {"rows": cls.rows},
            "original_source": "/unavailable/source",
            "original_build": "/unavailable/build",
        }
        cls.payloads = {
            "candidate.json": json.dumps(candidate).encode(),
            "candidate.patch": b"a reconstructible patch\n",
            "native-list.jsonl": b"".join((json.dumps(r) + "\n").encode() for r in native),
            "discovery.json": json.dumps(
                {
                    "rust-suites": {
                        "film-juicer-core": {
                            "testcases": {
                                OPTICS_FUNCTION: {"ignored": False, "filter-match": {"status": "matches"}},
                                "outside_selection": {"ignored": False, "filter-match": {"status": "mismatch"}},
                            }
                        }
                    }
                }
            ).encode(),
            "rust.jsonl": events([{"caller": "rust", "id": r} for r in cls.rows]),
            "native.jsonl": events(native),
            "nextest.xml": (
                '<testsuites><testsuite name="film-juicer-core"><testcase name="' + OPTICS_FUNCTION + '"/>'
                '<testcase name="outside_selection"><skipped/></testcase></testsuite></testsuites>'
            ).encode(),
            "ctest.xml": (
                "<testsuite>" + "".join(f'<testcase name="{n}" status="run"/>' for n in OPTICS_TESTS) + "</testsuite>"
            ).encode(),
            "nextest.toml": b'nextest-version = "0.9.143"\n',
            "rust-execution.json": b'{"commands":[{"returncode":0},{"returncode":0}]}',
        }
        cls.result = {
            "schema": "native-spatial-optics-v1",
            "preset": "linux-debug",
            "status": "passed",
            "candidate_head": "source-baseline",
            "shipping_isolation": False,
            "candidate_unchanged": True,
            "membership": {
                "rows": cls.rows,
                "native_operations": cls.operations,
                "ctest": OPTICS_TESTS,
                "rust_function": OPTICS_FUNCTION,
            },
            "commands": [{"returncode": 0, "drained": True, "process_status": "completed"}],
            "members": {},
        }

        actual = os.environ.get("FJ_OPTICS_COMPLETED_BUNDLE")
        if actual:
            source = Path(actual)
            raw = (source / "result.json").read_bytes()
            cls.result = json.loads(raw)
            cls.payloads = {name: (source / name).read_bytes() for name in cls.result["members"]}
            cls.read_bytes += len(raw) + sum(map(len, cls.payloads.values()))
            cls.rows = cls.result["membership"]["rows"]
            cls.operations = cls.result["membership"]["native_operations"]
        cls.template_result = cls.result
        cls.template_payloads = cls.payloads

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="optics results ü ")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        original_read = Path.read_bytes
        original_write = Path.write_bytes

        def read(path):
            data = original_read(path)
            type(self).read_bytes += len(data)
            return data

        def write(path, data):
            size = original_write(path, data)
            type(self).written_bytes += size
            return size

        for method, replacement in (("read_bytes", read), ("write_bytes", write)):
            patcher = patch.object(Path, method, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.bundle = self.root / "relocated bundle ü"
        self.bundle.mkdir()
        self.result = copy.deepcopy(type(self).template_result)
        self.payloads = dict(type(self).template_payloads)  # Bytes remain immutable.
        self.rows = self.result["membership"]["rows"]
        self.operations = self.result["membership"]["native_operations"]

    def save(self):
        self.result["members"] = {}
        for name, data in self.payloads.items():
            (self.bundle / name).write_bytes(data)
            self.result["members"][name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        (self.bundle / "result.json").write_bytes(json.dumps(self.result).encode())

    def production_fixture(self, preset="linux-release"):
        commands = [
            {"name": name, "returncode": 0}
            for name in (
                "native-symbols",
                "rust-symbols",
                "module-exports",
                "film-compile",
                "film-values",
                "accepted-output-color-compile",
                "accepted-output-color-values",
                "prepared-render",
                "module-symbols" if preset == "linux-release" else "pdb-symbols",
            )
        ]
        commands_data = json.dumps(commands).encode()
        rows = [
            {
                "case_id": case_id,
                "accepted_fixture_match": True,
                "diagnostic": "",
                "status": 0,
                "api": 0,
                "native_code": 0,
            }
            for case_id in (
                "negative-direct/cold",
                "negative-direct/warm",
                "negative-print/cold",
                "negative-print/warm",
            )
        ]
        report = {
            "schema": "production-contract-v1",
            "preset": preset,
            "build_testing": "OFF",
            "test_support": "OFF",
            "exports": ["OfxGetNumberOfPlugins", "OfxGetPlugin"],
            "hidden_device_load": True,
            "commands": {"sha256": hashlib.sha256(commands_data).hexdigest()},
            "scanned": [{"symbols_and_fault_strings_absent": True} for _ in range(4)],
            "probes": [
                {"name": name, "linked_sha256": "linked", "before_run_sha256": "linked", "after_run_sha256": "linked"}
                for name in ("film", "accepted-output-color")
            ],
            "prepared_render": {
                "executable_sha256_before": "prepared",
                "executable_sha256_after": "prepared",
                "rows": rows,
            },
            "original_source": "/unavailable/source",
            "original_build": "/unavailable/build",
        }
        candidate = {
            "head": "source-baseline",
            "preset": preset,
            "configuration": {"BUILD_TESTING": "OFF"},
            "inputs": {"CMakeLists.txt": "identity"},
            "original_source": "/unavailable/source",
            "original_build": "/unavailable/build",
        }
        payloads = {
            "candidate.json": json.dumps(candidate).encode(),
            "candidate.patch": b"patch",
            "production-contract.json": json.dumps(report).encode(),
            "production-commands.json": commands_data,
        }
        result = {
            "schema": "native-build-v1",
            "preset": preset,
            "status": "passed",
            "candidate_head": "source-baseline",
            "candidate_unchanged": True,
            "shipping_isolation": False,
            "production_contract": True,
            "commands": [
                {"name": "production-contract", "returncode": 0, "drained": True, "process_status": "completed"}
            ],
            "members": {},
        }
        return result, payloads

    def production_v2_fixture(self, preset="linux-release", *, module_record=None, rust_artifacts=None):
        result, payloads = self.production_fixture(preset)
        names = (
            "native-symbols",
            "rust-symbols",
            "module-exports",
            "module-symbols",
            "ofx-probe-build",
            "ofx-probe",
            "cuda-abi-probe-build",
            "cuda-abi-probe",
            "module-probe",
            "prepared-render",
        )
        empty = hashlib.sha256(b"").hexdigest()
        module, ofx, cuda, render, fixture = (str(index) * 64 for index in range(1, 6))
        stage_file, build_file = "module", "build-module"
        ofx_file, cuda_file, render_file = "ofx", "cuda", "render"
        windows = preset.startswith("windows")
        pdb_file = "pdb"
        native_archive, rust_archive = "native-symbols", "rust-symbols"
        if module_record is not None:
            stage_file, build_file = module_record["stage"], module_record["build"]
            module = module_record["stage_sha256"]
            build_module = PureWindowsPath(build_file)
            pdb_file = str(build_module.with_suffix(".pdb"))
            binaries = build_module.parent / "tests" / "bin"
            ofx_file = str(binaries / "JuicerProductionOfxProbe.exe")
            cuda_file = str(binaries / "JuicerProductionCudaAbiProbe.exe")
            render_file = str(binaries / "JuicerProductionPreparedRenderProbe.exe")
            native_archive = rust_artifacts["native_archive"]
            rust_archive = rust_artifacts["variants"]["production"]["archive"]
        symbol_file = pdb_file if windows else stage_file
        scan_files = dict(zip(names[:4], (native_archive, rust_archive, stage_file, symbol_file), strict=True))
        commands = [
            {
                "name": name,
                "returncode": 0,
                "process_status": "completed",
                "drained": True,
                "command": [name],
                "stdout_sha256": empty,
                "stderr_sha256": empty,
            }
            for name in names
        ]
        by_name = {command["name"]: command for command in commands}
        for name, file in scan_files.items():
            by_name[name]["command"] = ["scan", file]
        by_name["ofx-probe"]["command"] = [ofx_file, stage_file]
        by_name["cuda-abi-probe"]["command"] = [cuda_file]
        by_name["module-probe"]["command"] = ["probe", stage_file]
        by_name["prepared-render"]["command"] = [render_file, stage_file, "resources", "fixture", preset, "updates"]
        commands_data = json.dumps(commands).encode()
        rows = []
        for route in ("negative-direct", "negative-print"):
            channels = 3 if route == "negative-direct" else 4
            for render_pass in ("cold", "warm"):
                rows.append(
                    {
                        "case_id": f"{route}/{render_pass}",
                        "route": route,
                        "pass": render_pass,
                        "input_shape": [5, 7, channels],
                        "output_shape": [5, 7, channels],
                        "ofx_status": 0,
                        "messages": [],
                        "input_sha256": empty,
                        "output_f32_words_sha256": empty,
                        "samples_compared": 35 * channels,
                        "accepted_fixture_match": True,
                        "finite_output": True,
                        "alpha_preserved": True,
                        "padding_preserved": True,
                        "stream_completed": True,
                        "readback_complete": True,
                        "context_preserved": True,
                        "context_restored": True,
                        "source_image_releases": 1,
                        "output_image_releases": 1,
                    }
                )

        def identity(value):
            return {"sha256_before": value, "sha256_after": value}

        report = {
            "schema": "production-contract-v2",
            "preset": preset,
            "build_testing": "OFF",
            "test_support": "OFF",
            "hidden_device_load": True,
            "exports": ["OfxGetNumberOfPlugins", "OfxGetPlugin"],
            "commands": {"sha256": hashlib.sha256(commands_data).hexdigest()},
            "scanned": [
                {
                    "name": name,
                    "file": file,
                    "sha256": module if file == stage_file else empty,
                    "symbols_and_fault_strings_absent": True,
                    "legacy_symbols_absent": True,
                }
                for name, file in scan_files.items()
            ],
            "build_inputs": {
                stage_file: identity(module),
                build_file: identity(module),
                pdb_file: identity(empty),
                native_archive: identity(empty),
                rust_archive: identity(empty),
                ofx_file: identity(ofx),
                cuda_file: identity(cuda),
                render_file: identity(render),
            },
            "consumed_inputs": {"fixture": identity(fixture)},
            "loaded_module": {"file": stage_file, "build_file": build_file, "sha256": module},
            "probes": [
                {
                    "name": name,
                    "executable": path,
                    "linked_sha256": value,
                    "before_run_sha256": value,
                    "after_run_sha256": value,
                }
                for name, path, value in (("ofx", ofx_file, ofx), ("cuda-abi", cuda_file, cuda))
            ],
            "prepared_render": {
                "module": stage_file,
                "module_sha256": module,
                "executable": render_file,
                "executable_sha256_before": render,
                "executable_sha256_after": render,
                "fixture": "fixture",
                "fixture_sha256": fixture,
                "command": by_name["prepared-render"]["command"],
                "stdout_sha256": empty,
                "stderr_sha256": empty,
                "rows": rows,
                "lifecycle": {
                    "module_load_status": 0,
                    "effects_created": 2,
                    "effects_destroyed": 2,
                    "destroy_statuses": [0, 0],
                    "module_unload_status": 0,
                    "context_restored": True,
                },
            },
        }
        payloads["production-contract.json"] = json.dumps(report).encode()
        payloads["production-commands.json"] = commands_data
        return result, payloads

    def production_v3_fixture(self, preset="linux-release", root=None):
        result, payloads = self.production_v2_fixture(preset)
        root = (
            PureWindowsPath(root or "C:/checkout")
            if preset.startswith("windows-")
            else PurePosixPath(root or "/checkout")
        )
        arch = "Win64" if preset.startswith("windows-") else "Linux-x86-64"
        stage = root / "out" / "stage" / preset / "juicer.ofx.bundle" / "Contents" / arch / "juicer.ofx"
        build = root / "out" / "build" / preset / "juicer.ofx"
        binaries = build.parent / "tests" / "bin"
        paths = {
            "module": str(stage),
            "build-module": str(build),
            "ofx": str(binaries / ("JuicerProductionOfxProbe" + (".exe" if preset.startswith("windows-") else ""))),
            "cuda": str(
                binaries / ("JuicerProductionCudaAbiProbe" + (".exe" if preset.startswith("windows-") else ""))
            ),
            "render": str(
                binaries / ("JuicerProductionPreparedRenderProbe" + (".exe" if preset.startswith("windows-") else ""))
            ),
            "fixture": str(root / "tests" / "ofx" / "fixtures" / "spectral-v1" / f"ordinary-{preset}.json"),
            "pdb": str(build.with_suffix(".pdb")),
            "native-symbols": str(root / "rust" / "target" / preset / "libfilm_juicer_native.a"),
            "rust-symbols": str(root / "rust" / "target" / preset / "libfilm_juicer_core.a"),
        }

        def replace(value):
            if isinstance(value, str):
                return paths.get(value, value)
            if isinstance(value, list):
                return [replace(item) for item in value]
            if isinstance(value, dict):
                return {key: replace(item) for key, item in value.items()}
            return value

        report = replace(self.production_report(payloads))
        commands = replace(json.loads(payloads["production-commands.json"]))
        command_names = (
            "native-symbols",
            "rust-symbols",
            "module-exports",
            "module-symbols",
            "ofx-probe-build",
            "ofx-probe",
            "cuda-abi-probe-build",
            "cuda-abi-probe",
            "module-probe",
            "prepared-render",
        )
        for row, name in zip(commands, command_names, strict=True):
            row["name"] = name
        for row, name in zip(report["scanned"], command_names[:4], strict=True):
            row["name"] = name
        report["build_inputs"] = {paths.get(key, key): value for key, value in report["build_inputs"].items()}
        report["probes"][0]["name"] = "ofx"
        report["probes"][1]["name"] = "cuda-abi"
        pin = "f" * 64
        collector = str(root / "tests" / "ffi" / "production_contract.py")
        fixture = paths["fixture"]
        report["schema"] = "production-contract-v3"
        report["collector"] = collector
        report["consumed_inputs"].pop("fixture", None)
        report["consumed_inputs"][fixture] = {"sha256_before": pin, "sha256_after": pin}
        report["consumed_inputs"][collector] = {"sha256_before": "e" * 64, "sha256_after": "e" * 64}
        prepared = report["prepared_render"]
        prepared["module"] = paths["module"]
        prepared["fixture"] = fixture
        prepared["fixture_sha256"] = pin
        resources = (
            str(PureWindowsPath(paths["module"]).parent.parent / "Resources")
            if preset.startswith("windows-")
            else str(PurePosixPath(paths["module"]).parent.parent / "Resources")
        )
        expected_command = [paths["render"], paths["module"], resources, fixture, preset]
        prepared["command"] = expected_command
        by_name = {row["name"]: row for row in commands}
        by_name["prepared-render"]["command"] = expected_command
        report["commands"]["sha256"] = hashlib.sha256(json.dumps(commands).encode()).hexdigest()
        payloads["production-contract.json"] = json.dumps(report).encode()
        payloads["production-commands.json"] = json.dumps(commands).encode()
        return result, payloads

    def write_bundle(self, result, payloads, *, omitted=frozenset()):
        members = {}
        for name, data in payloads.items():
            (self.bundle / name).write_bytes(data)
            if name not in omitted:
                members[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        result["members"] = members
        raw = json.dumps(result).encode()
        (self.bundle / "result.json").write_bytes(raw)
        return raw

    @staticmethod
    def production_report(payloads):
        return json.loads(payloads["production-contract.json"])

    @staticmethod
    def update_production_report(payloads, report):
        payloads["production-contract.json"] = json.dumps(report).encode()

    def reject(self):
        # Semantic mutations exercise the same pure validator as publication and
        # the reader, without rewriting an entire fixture bundle for every row.
        self.result["members"] = {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in self.payloads.items()
        }
        with self.assertRaises((ValueError, KeyError, OSError)):
            validate_native_result(self.result, self.payloads)

    def test_complete_results_and_expired_optional_diagnostics(self):
        self.save()
        result = read_native_result(self.bundle)
        self.assertEqual(result["rust_rows"], 553)
        self.assertEqual(result["native_rows"], 1106)
        self.assertFalse(result["shipping_isolation"])
        self.assertEqual(result["bookkeeping_read_bytes"], sum(p.stat().st_size for p in self.bundle.iterdir()))

    def test_damaged_member_rejected_by_offline_reader(self):
        self.save()
        (self.bundle / "rust.jsonl").write_bytes(b"damaged\n")
        with self.assertRaisesRegex(ValueError, "damaged bundle member"):
            read_native_result(self.bundle)

    def test_gpu_witness_cannot_pass_without_shared_exclusion(self):
        from qualification import GPU_WITNESS_TEST

        self.result["gpu_witness"] = {
            "test": GPU_WITNESS_TEST,
            "scope": "containment-and-shared-lock-only",
            "status": "passed",
            "contender_excluded": False,
            "drained_while_locked": True,
        }
        self.payloads["gpu-witness.xml"] = (
            '<testsuite><testcase name="' + GPU_WITNESS_TEST + '" status="run"/></testsuite>'
        ).encode()
        self.reject()

    def test_gpu_witness_requires_its_case_results(self):
        from qualification import GPU_WITNESS_TEST

        self.result["gpu_witness"] = {
            "test": GPU_WITNESS_TEST,
            "scope": "containment-and-shared-lock-only",
            "status": "passed",
            "contender_excluded": True,
            "drained_while_locked": True,
        }
        self.payloads.pop("gpu-witness.xml", None)
        self.reject()

    def test_missing_rust_function(self):
        self.payloads["discovery.json"] = b'{"rust-suites":{}}'
        self.reject()

    def test_missing_rust_row(self):
        self.payloads["rust.jsonl"] = b"\n".join(self.payloads["rust.jsonl"].splitlines()[:-2]) + b"\n"
        self.reject()

    def test_missing_native_caller(self):
        self.payloads["native.jsonl"] = (
            b"\n".join(
                line for line in self.payloads["native.jsonl"].splitlines() if json.loads(line)["caller"] != "c11"
            )
            + b"\n"
        )
        self.reject()

    def test_missing_native_row(self):
        self.payloads["native.jsonl"] = b"\n".join(self.payloads["native.jsonl"].splitlines()[2:]) + b"\n"
        self.reject()

    def test_missing_named_operation(self):
        self.payloads["native.jsonl"] = b"\n".join(self.payloads["native.jsonl"].splitlines()[:-2]) + b"\n"
        self.reject()

    def test_native_listing_omits_row(self):
        self.payloads["native-list.jsonl"] = b"\n".join(self.payloads["native-list.jsonl"].splitlines()[1:]) + b"\n"
        self.reject()

    def test_duplicate_terminal(self):
        self.payloads["rust.jsonl"] += self.payloads["rust.jsonl"].splitlines()[-1] + b"\n"
        self.reject()

    def test_crash_before_terminal(self):
        self.payloads["rust.jsonl"] = b"\n".join(self.payloads["rust.jsonl"].splitlines()[:-1]) + b"\n"
        self.reject()

    def test_truncated_output(self):
        self.payloads["native.jsonl"] = self.payloads["native.jsonl"][:-5]
        self.reject()

    def test_nonzero_exit(self):
        self.result["commands"][0]["returncode"] = 9
        self.reject()

    def test_ignored_selected_function(self):
        document = json.loads(self.payloads["discovery.json"])
        for suite in document["rust-suites"].values():
            suite["testcases"][OPTICS_FUNCTION]["ignored"] = True
        self.payloads["discovery.json"] = json.dumps(document).encode()
        self.reject()

    def test_unexpected_skip(self):
        import xml.etree.ElementTree as ET

        tree = ET.fromstring(self.payloads["nextest.xml"])
        for case in tree.iter("testcase"):
            if case.get("name") == OPTICS_FUNCTION:
                ET.SubElement(case, "skipped")
        self.payloads["nextest.xml"] = ET.tostring(tree)
        self.reject()

    def test_failed_row_despite_zero_exit(self):
        self.payloads["native.jsonl"] = self.payloads["native.jsonl"].replace(b'"passed"', b'"failed"', 1)
        self.reject()

    def test_missing_junit(self):
        del self.payloads["nextest.xml"]
        self.reject()

    def test_candidate_mismatch(self):
        self.result["candidate_head"] = "different"
        self.reject()

    def test_actual_build_type_mismatch(self):
        candidate = json.loads(self.payloads["candidate.json"])
        candidate["configuration"]["CMAKE_BUILD_TYPE"] = "Release"
        self.payloads["candidate.json"] = json.dumps(candidate).encode()
        self.reject()

    def test_failed_bundle_collection_exits_nonzero(self):
        import verify

        self.result["status"] = "failed"
        self.save()
        output = self.root / "report.json"
        with patch.object(
            sys, "argv", ["verify.py", "collect-qualification", "--bundles", str(self.bundle), "--output", str(output)]
        ):
            self.assertEqual(verify.main(), 1)
        self.assertEqual(json.loads(output.read_bytes())["status"], "failed")

    def test_invalid_membership_rejected_before_payload_reads(self):
        self.save()
        self.result["membership"]["rows"] = self.rows[:-1]
        raw = json.dumps(self.result).encode()
        with patch.object(Path, "read_bytes", side_effect=AssertionError("payload read")):
            with self.assertRaisesRegex(ValueError, "membership"):
                read_native_result(self.bundle, result_bytes=raw)

    def test_boolean_member_size_rejected_before_payload_reads(self):
        self.payloads["candidate.patch"] = b"x"
        self.save()
        self.assertEqual((self.bundle / "candidate.patch").stat().st_size, 1)
        self.result["members"]["candidate.patch"]["bytes"] = True
        raw = json.dumps(self.result).encode()
        with patch.object(Path, "read_bytes", side_effect=AssertionError("payload read")):
            with self.assertRaisesRegex(ValueError, "member identity"):
                read_native_result(self.bundle, result_bytes=raw)

    def test_physical_member_size_rejected_before_payload_reads(self):
        self.save()
        raw = (self.bundle / "result.json").read_bytes()
        member = self.bundle / "candidate.patch"
        member.write_bytes(member.read_bytes() + b"x")
        with patch.object(Path, "read_bytes", side_effect=AssertionError("payload read")):
            with self.assertRaisesRegex(ValueError, "damaged bundle member size: candidate.patch"):
                read_native_result(self.bundle, result_bytes=raw)

    def test_completed_production_off_report_is_read_from_bundle_only(self):
        result, payloads = self.production_fixture()
        raw = self.write_bundle(result, payloads)
        summary = read_native_result(self.bundle, result_bytes=raw)
        self.assertEqual(summary["production_off"], True)
        self.assertEqual(summary["prepared_render_cases"], 4)

    def test_completed_v2_production_report_is_read_from_bundle_only(self):
        self.subtest_cases = ("linux-release", "windows-clang-release")
        for preset in self.subtest_cases:
            with self.subTest(case=preset):
                result, payloads = self.production_v2_fixture(preset)
                summary = read_native_result(self.bundle, result_bytes=self.write_bundle(result, payloads))
                self.assertEqual(summary["prepared_render_cases"], 4)

    def test_completed_v3_production_report_accepts_relocated_windows_and_linux_sources(self):
        sources = {"linux-release": "/relocated/source", "windows-clang-release": "D:/relocated/source"}
        self.subtest_cases = tuple(sources)
        for preset, root in sources.items():
            with self.subTest(case=preset):
                with patch.dict(qualification.SPECTRAL_PRODUCTION_FIXTURE_SHA256, {preset: "f" * 64}, clear=True):
                    result, payloads = self.production_v3_fixture(preset, root)
                    summary = read_native_result(self.bundle, result_bytes=self.write_bundle(result, payloads))
                    self.assertEqual(summary["prepared_render_cases"], 4)

    def test_production_v3_rejects_unreviewed_path_digest_and_argv_variants(self):
        self.subtest_cases = (
            "wrong-resource",
            "wrong-fixture-digest",
            "self-consistent-fixture",
            "stale-supplement",
            "wrong-preset",
            "argv-mismatch",
        )
        with patch.dict(qualification.SPECTRAL_PRODUCTION_FIXTURE_SHA256, {"linux-release": "f" * 64}, clear=True):
            for case in self.subtest_cases:
                with self.subTest(case=case):
                    result, payloads = self.production_v3_fixture("linux-release")
                    report = self.production_report(payloads)
                    prepared = report["prepared_render"]
                    commands = json.loads(payloads["production-commands.json"])
                    logged = next(row for row in commands if row["name"] == "prepared-render")
                    if case == "wrong-resource":
                        prepared["command"][2] = "/relocated/source/Resources-other"
                    elif case == "wrong-fixture-digest":
                        prepared["fixture_sha256"] = "a" * 64
                        fixture = prepared["fixture"]
                        report["consumed_inputs"][fixture] = {"sha256_before": "a" * 64, "sha256_after": "a" * 64}
                    elif case == "self-consistent-fixture":
                        fixture = "/relocated/source/tests/ofx/fixtures/spectral-v1/ordinary-other.json"
                        report["consumed_inputs"][fixture] = report["consumed_inputs"].pop(prepared["fixture"])
                        prepared["fixture"] = fixture
                        prepared["command"][3] = fixture
                    elif case == "stale-supplement":
                        prepared["command"].append("/relocated/source/product-updates.json")
                        report["consumed_inputs"][prepared["command"][-1]] = {
                            "sha256_before": "b" * 64,
                            "sha256_after": "b" * 64,
                        }
                    elif case == "wrong-preset":
                        prepared["command"][4] = "windows-clang-release"
                    else:
                        logged["command"][4] = "windows-clang-release"
                    if case != "argv-mismatch":
                        logged["command"] = prepared["command"]
                    payloads["production-commands.json"] = json.dumps(commands).encode()
                    report["commands"]["sha256"] = hashlib.sha256(payloads["production-commands.json"]).hexdigest()
                    payloads["production-contract.json"] = json.dumps(report).encode()
                    with self.assertRaisesRegex(ValueError, "spectral v3|production render identity"):
                        read_native_result(self.bundle, result_bytes=self.write_bundle(result, payloads))

    def test_production_v2_rejects_invalid_observations(self):
        cases = (
            "missing-row",
            "duplicate-row",
            "reordered-rows",
            "route",
            "pass",
            "input-shape",
            "output-shape",
            "shape-boolean",
            "count",
            "count-boolean",
            "status",
            "status-boolean",
            "messages",
            "input-digest",
            "output-digest",
            "warm-output",
            "fixture-match",
            "finite",
            "alpha",
            "padding",
            "stream",
            "readback",
            "context",
            "restored",
            "source-release-zero",
            "source-release-two",
            "output-release-zero",
            "output-release-two",
            "release-boolean",
            "lifecycle-load",
            "lifecycle-created",
            "lifecycle-destroyed",
            "destroy-status",
            "destroy-boolean",
            "lifecycle-unload",
            "lifecycle-restored",
            "module-hash",
            "probe-hash",
            "render-hash",
            "input-hash",
            "fixture-hash",
            "scan-hash",
            "missing-field",
            "legacy-field",
            "legacy-submitted",
            "legacy-state-release",
            "legacy-view-release",
            "legacy-status",
            "legacy-api",
            "legacy-native-code",
            "forbidden-symbol",
            "legacy-symbol",
            "hidden-export",
            "missing-command",
            "duplicate-command",
            "failed-command",
            "boolean-command",
            "undrained-command",
            "unfinished-command",
            "changed-command",
            "changed-stdout",
        )
        self.subtest_cases = cases
        for case in cases:
            with self.subTest(case=case):
                result, payloads = self.production_v2_fixture()
                report = self.production_report(payloads)
                prepared = report["prepared_render"]
                row = prepared["rows"][0]
                life = prepared["lifecycle"]
                commands = json.loads(payloads["production-commands.json"])
                if case == "missing-row":
                    prepared["rows"].pop()
                elif case == "duplicate-row":
                    prepared["rows"][1] = copy.deepcopy(row)
                elif case == "reordered-rows":
                    prepared["rows"].reverse()
                elif case == "route":
                    row["route"] = "negative-print"
                elif case == "pass":
                    row["pass"] = "warm"
                elif case == "input-shape":
                    row["input_shape"] = [5, 7, 4]
                elif case == "output-shape":
                    row["output_shape"] = [5, 7, 4]
                elif case == "shape-boolean":
                    row["input_shape"] = [5, 7, True]
                elif case == "count":
                    row["samples_compared"] = 104
                elif case == "count-boolean":
                    row["samples_compared"] = True
                elif case == "status":
                    row["ofx_status"] = 1
                elif case == "status-boolean":
                    row["ofx_status"] = False
                elif case == "messages":
                    row["messages"] = ["unexpected"]
                elif case == "input-digest":
                    row["input_sha256"] = "invalid"
                elif case == "output-digest":
                    row["output_f32_words_sha256"] = "invalid"
                elif case == "warm-output":
                    prepared["rows"][1]["output_f32_words_sha256"] = "a" * 64
                elif case in {
                    "fixture-match",
                    "finite",
                    "alpha",
                    "padding",
                    "stream",
                    "readback",
                    "context",
                    "restored",
                }:
                    key = {
                        "fixture-match": "accepted_fixture_match",
                        "finite": "finite_output",
                        "alpha": "alpha_preserved",
                        "padding": "padding_preserved",
                        "stream": "stream_completed",
                        "readback": "readback_complete",
                        "context": "context_preserved",
                        "restored": "context_restored",
                    }[case]
                    row[key] = False
                elif case.startswith("source-release-"):
                    row["source_image_releases"] = int(case.rsplit("-", 1)[1] == "two") * 2
                elif case.startswith("output-release-"):
                    row["output_image_releases"] = int(case.rsplit("-", 1)[1] == "two") * 2
                elif case == "release-boolean":
                    row["source_image_releases"] = True
                elif case == "lifecycle-load":
                    life["module_load_status"] = 1
                elif case == "lifecycle-created":
                    life["effects_created"] = 1
                elif case == "lifecycle-destroyed":
                    life["effects_destroyed"] = 1
                elif case == "destroy-status":
                    life["destroy_statuses"] = [0, 1]
                elif case == "destroy-boolean":
                    life["destroy_statuses"] = [0, False]
                elif case == "lifecycle-unload":
                    life["module_unload_status"] = 1
                elif case == "lifecycle-restored":
                    life["context_restored"] = False
                elif case == "module-hash":
                    report["loaded_module"]["sha256"] = "a" * 64
                elif case == "probe-hash":
                    report["probes"][0]["after_run_sha256"] = "a" * 64
                elif case == "render-hash":
                    prepared["executable_sha256_after"] = "a" * 64
                elif case == "input-hash":
                    report["consumed_inputs"]["fixture"]["sha256_after"] = "a" * 64
                elif case == "fixture-hash":
                    prepared["fixture_sha256"] = "a" * 64
                elif case == "scan-hash":
                    report["scanned"][0]["sha256"] = "a" * 64
                elif case == "missing-field":
                    del row["finite_output"]
                elif case == "legacy-field":
                    row["recipe_hash"] = 0
                elif case == "legacy-submitted":
                    row["submitted"] = True
                elif case == "legacy-state-release":
                    row["state_released_after_submit"] = True
                elif case == "legacy-view-release":
                    row["views_released_after_submit"] = True
                elif case == "legacy-status":
                    row["status"] = 0
                elif case == "legacy-api":
                    row["api"] = 0
                elif case == "legacy-native-code":
                    row["native_code"] = 0
                elif case == "forbidden-symbol":
                    report["scanned"][0]["symbols_and_fault_strings_absent"] = False
                elif case == "legacy-symbol":
                    report["scanned"][0]["legacy_symbols_absent"] = False
                elif case == "hidden-export":
                    report["exports"] = ["OfxGetPlugin"]
                elif case == "missing-command":
                    commands.pop()
                elif case == "duplicate-command":
                    commands[1]["name"] = commands[0]["name"]
                elif case == "failed-command":
                    commands[0]["returncode"] = 1
                elif case == "boolean-command":
                    commands[0]["returncode"] = False
                elif case == "undrained-command":
                    commands[0]["drained"] = False
                elif case == "unfinished-command":
                    commands[0]["process_status"] = "running"
                elif case == "changed-command":
                    commands[-1]["command"] = ["changed"]
                elif case == "changed-stdout":
                    commands[-1]["stdout_sha256"] = "a" * 64
                payloads["production-commands.json"] = json.dumps(commands).encode()
                report["commands"]["sha256"] = hashlib.sha256(payloads["production-commands.json"]).hexdigest()
                self.update_production_report(payloads, report)
                result["members"] = {
                    name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                    for name, data in payloads.items()
                }
                with self.assertRaises((ValueError, KeyError)):
                    validate_build_result(result, payloads)

    def test_production_report_rejects_missing_or_failed_command(self):
        self.subtest_cases = ("missing", "failed")
        for failed in (False, True):
            with self.subTest(case="failed" if failed else "missing"):
                result, payloads = self.production_fixture()
                commands = json.loads(payloads["production-commands.json"])
                if failed:
                    commands[0]["returncode"] = 1
                    message = "missing or failed production command"
                else:
                    commands.pop()
                    message = "missing or failed production command"
                payloads["production-commands.json"] = json.dumps(commands).encode()
                report = self.production_report(payloads)
                report["commands"]["sha256"] = hashlib.sha256(payloads["production-commands.json"]).hexdigest()
                self.update_production_report(payloads, report)
                result["members"] = {
                    name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                    for name, data in payloads.items()
                }
                with self.assertRaisesRegex(ValueError, message):
                    validate_build_result(result, payloads)

    def test_production_report_rejects_changed_probe(self):
        result, payloads = self.production_fixture()
        report = self.production_report(payloads)
        report["probes"][0]["after_run_sha256"] = "changed"
        self.update_production_report(payloads, report)
        result["members"] = {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in payloads.items()
        }
        with self.assertRaisesRegex(ValueError, "changed or missing production C11 probe"):
            validate_build_result(result, payloads)

    def test_production_report_rejects_missing_or_failed_prepared_row(self):
        self.subtest_cases = ("missing", "failed")
        for failed in (False, True):
            with self.subTest(case="failed" if failed else "missing"):
                result, payloads = self.production_fixture()
                report = self.production_report(payloads)
                if failed:
                    report["prepared_render"]["rows"][0]["accepted_fixture_match"] = False
                else:
                    report["prepared_render"]["rows"].pop()
                self.update_production_report(payloads, report)
                result["members"] = {
                    name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                    for name, data in payloads.items()
                }
                with self.assertRaisesRegex(ValueError, "incomplete or failed production prepared rendering"):
                    validate_build_result(result, payloads)

    def test_production_report_rejects_testing_on_or_wrong_preset(self):
        self.subtest_cases = ("testing-on", "wrong-preset")
        for mutation in self.subtest_cases:
            with self.subTest(case=mutation):
                result, payloads = self.production_fixture()
                report = self.production_report(payloads)
                if mutation == "testing-on":
                    candidate = json.loads(payloads["candidate.json"])
                    candidate["configuration"]["BUILD_TESTING"] = "ON"
                    payloads["candidate.json"] = json.dumps(candidate).encode()
                    message = "production contract requires a testing-OFF candidate"
                else:
                    report["preset"] = "linux-debug"
                    self.update_production_report(payloads, report)
                    message = "incomplete production OFF assertions"
                result["members"] = {
                    name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                    for name, data in payloads.items()
                }
                with self.assertRaisesRegex(ValueError, message):
                    validate_build_result(result, payloads)

    def test_missing_production_payload_rejected_before_payload_reads(self):
        result, payloads = self.production_fixture()
        raw = self.write_bundle(result, payloads, omitted={"production-commands.json"})
        with patch.object(Path, "read_bytes", side_effect=AssertionError("payload read")):
            with self.assertRaisesRegex(ValueError, "missing required native run members"):
                read_native_result(self.bundle, result_bytes=raw)

    def module_producer_fixture(
        self, preview="OFF", *, explicit_preview=True, testing="ON", raw_preview=None, stage_name=None
    ):
        source = "C:/checkout"
        preset = "windows-clang-release"
        build = f"{source}/out/build/{preset}"
        stage = f"{source}/out/stage/{preset}"
        if stage_name is not None:
            stage += f"/{stage_name}"
        cmake = "C:/tools/cmake.exe"
        python = "C:/venv/python.exe"
        targets = ["bundle-archive", "JuicerOfxModuleTests"]
        module_name = "juicer-grain-preview.ofx" if preview == "ON" else "juicer.ofx"
        module = f"{build}/{module_name}"
        module_stage = f"{stage}/portra-grain-preview" if preview == "ON" else stage
        configure = [cmake, "--preset", preset, f"-DBUILD_TESTING={testing}", f"-DPython3_EXECUTABLE={python}"]
        if explicit_preview:
            configure.append(f"-DJUICER_GRAIN_PREVIEW:BOOL={preview}")
        if stage_name is not None:
            configure.append(f"-DCMAKE_INSTALL_PREFIX:PATH={stage}")
        variants = {"production": ["grain-preview"] if preview == "ON" else []}
        if testing == "ON":
            variants["test-support"] = ["test-support"]
        rust_variants = {}
        for name, features in variants.items():
            directory = f"{build}/cargo/{name}"
            rust_variants[name] = {
                "target_dir": directory,
                "archive": f"{directory}/x86_64-pc-windows-msvc/release/film_juicer_plugin.lib",
                "feature_input": f"{directory}/features.txt",
                "stamp": f"{directory}/build.stamp",
                "features": features,
            }
        sha256 = "a" * 64
        candidate = {
            "head": "source-baseline",
            "preset": preset,
            "inputs": {"CMakeLists.txt": {"git_blob": "b" * 40}},
            "configuration": {
                "CMAKE_HOME_DIRECTORY": source,
                "CMAKE_INSTALL_PREFIX": stage,
                "CMAKE_BUILD_TYPE": "Release",
                "CMAKE_CXX_COMPILER": "C:/tools/clang-cl.exe",
                "CMAKE_CUDA_COMPILER": "C:/tools/nvcc.exe",
                "BUILD_TESTING": testing,
                "JUICER_GRAIN_PREVIEW": preview if raw_preview is None else raw_preview,
                "JUICER_GRAIN_PREVIEW_CONFIGURED": preview,
                "JUICER_GRAIN_MOTION_REPLAY_CONFIGURED": "OFF",
            },
            "version_commands": [
                {"name": "cmake", "command": [cmake, "--version"], "returncode": 0, "output": "cmake version 4.3.3"}
            ],
        }
        payloads = {
            "candidate.json": json.dumps(candidate).encode(),
            "candidate.patch": b"",
            "production-module.json": json.dumps(
                {
                    "build": module,
                    "stage": f"{module_stage}/{module_name}.bundle/Contents/Win64/{module_name}",
                    "build_sha256": sha256,
                    "stage_sha256": sha256,
                }
            ).encode(),
            "rust-artifacts.json": json.dumps(
                {
                    "schema": "rust-artifacts-v2",
                    "build": build,
                    "native_archive": f"{build}/juicer_core.lib",
                    "consumers": {"juicer": {"binary": module, "variant": "production"}},
                    "variants": rust_variants,
                }
            ).encode(),
        }
        result = {
            "schema": "native-build-v1",
            "status": "passed",
            "preset": preset,
            "candidate_head": candidate["head"],
            "candidate_unchanged": True,
            "shipping_isolation": False,
            "production_module": True,
            "targets": targets,
            "commands": [
                {
                    "name": name,
                    "command": command,
                    "directory": source,
                    "returncode": 0,
                    "drained": True,
                    "process_status": "completed",
                }
                for name, command in (
                    ("configure", configure),
                    ("build", [cmake, "--build", build, "--target", *targets, "--", "-k0"]),
                    (
                        "versions",
                        [
                            python,
                            "-B",
                            f"{source}/scripts/verify.py",
                            "tool-versions",
                            "--tool",
                            "cmake",
                            cmake,
                        ],
                    ),
                )
            ],
        }
        return result, payloads

    def test_module_producer_binds_named_stage(self):
        self.subtest_cases = ("ordinary", "preview", "foreign", "different-cached-stage")
        for case in self.subtest_cases:
            with self.subTest(case=case):
                preview = "ON" if case == "preview" else "OFF"
                stage_name = "outside/bundle" if case == "foreign" else "scanner-candidate"
                result, payloads = self.module_producer_fixture(preview, stage_name=stage_name)
                if case == "different-cached-stage":
                    result["commands"][0]["command"][-1] += "-different"
                self.write_bundle(result, payloads)
                if case in {"ordinary", "preview"}:
                    self.assertEqual(read_native_result(self.bundle)["status"], "passed")
                else:
                    with self.assertRaisesRegex(ValueError, "invalid named stage"):
                        read_native_result(self.bundle)

    test_module_producer_binds_named_stage.required_subtests = True

    def test_module_producer_requires_configured_cmake_build(self):
        result, payloads = self.module_producer_fixture()
        self.write_bundle(result, payloads)
        self.assertEqual(read_native_result(self.bundle)["status"], "passed")
        self.subtest_cases = (
            "build-dir",
            "working-dir",
            "cmake",
            "targets",
            "extra-configure",
            "missing-build",
            "missing-versions",
        )
        for case in self.subtest_cases:
            with self.subTest(case=case):
                altered = copy.deepcopy(result)
                operations = altered["commands"]
                operation = operations[1]
                diagnostic = "CMake producer build command differs"
                if case == "build-dir":
                    operation["command"][2] = "C:/checkout/out/build/windows-clang-debug"
                elif case == "working-dir":
                    operation["directory"] = "C:/other"
                elif case == "cmake":
                    operation["command"][0] = "C:/other/cmake.exe"
                elif case == "targets":
                    operation["command"][5] = "JuicerOfxRenderTests"
                elif case == "extra-configure":
                    extra = copy.deepcopy(operations[0])
                    extra["name"] = "reconfigure"
                    extra["command"][-1] = "-DJUICER_GRAIN_PREVIEW:BOOL=ON"
                    operations.insert(1, extra)
                    diagnostic = "unexpected CMake producer command sequence"
                else:
                    operations.pop(1 if case == "missing-build" else 2)
                    diagnostic = "unexpected CMake producer command sequence"
                (self.bundle / "result.json").write_text(json.dumps(altered), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, diagnostic):
                    read_native_result(self.bundle)

    test_module_producer_requires_configured_cmake_build.required_subtests = True

    def test_module_producer_requires_configured_cmake_options(self):
        passing = {
            "explicit-ordinary": ("OFF", True, "ON", None),
            "explicit-preview": ("ON", True, "ON", None),
            "cached-ordinary": ("OFF", False, "ON", None),
            "cached-preview": ("ON", False, "ON", None),
            "cached-false-alias": ("OFF", False, "ON", "FaLsE"),
            "cached-true-alias": ("ON", False, "ON", "TRUE"),
            "cached-notfound": ("OFF", False, "ON", "preview-NOTFOUND"),
            "testing-off": ("OFF", True, "OFF", None),
        }
        rejected = {
            "contradictory-preview": (5, ["-DJUICER_GRAIN_PREVIEW:BOOL=ON"]),
            "redirected-build": (6, ["-B", "C:/checkout/out/build/windows-clang-debug"]),
            "redirected-source": (6, ["-S", "C:/other"]),
            "contradictory-testing": (3, ["-DBUILD_TESTING=OFF"]),
            "substituted-python": (4, ["-DPython3_EXECUTABLE=C:/other/python.exe"]),
            "overridden-profile": (6, ["-DCMAKE_BUILD_TYPE=Debug"]),
        }
        self.subtest_cases = (
            *passing,
            *rejected,
            "null-argument",
            "null-directory",
            "foreign-directory",
            "cached-disagreement",
            "explicit-disagreement",
        )
        for case, (preview, explicit_preview, testing, raw_preview) in passing.items():
            with self.subTest(case=case):
                result, payloads = self.module_producer_fixture(
                    preview, explicit_preview=explicit_preview, testing=testing, raw_preview=raw_preview
                )
                self.write_bundle(result, payloads)
                self.assertEqual(read_native_result(self.bundle)["status"], "passed")

        result, payloads = self.module_producer_fixture()
        self.write_bundle(result, payloads)
        for case, (index, operands) in rejected.items():
            with self.subTest(case=case):
                altered = copy.deepcopy(result)
                command = altered["commands"][0]["command"]
                command[index : index + 1] = operands
                (self.bundle / "result.json").write_text(json.dumps(altered), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "CMake producer configure command differs"):
                    read_native_result(self.bundle)

        with self.subTest(case="null-argument"):
            altered = copy.deepcopy(result)
            altered["commands"][0]["command"][4] = None
            (self.bundle / "result.json").write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid CMake producer configure command arguments"):
                read_native_result(self.bundle)
        for case, directory, diagnostic in (
            ("null-directory", None, "invalid CMake producer configure working directory"),
            ("foreign-directory", "C:/other", "CMake producer configure command differs"),
        ):
            with self.subTest(case=case):
                altered = copy.deepcopy(result)
                altered["commands"][0]["directory"] = directory
                (self.bundle / "result.json").write_text(json.dumps(altered), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, diagnostic):
                    read_native_result(self.bundle)
        for explicit, name in ((False, "cached-disagreement"), (True, "explicit-disagreement")):
            with self.subTest(case=name):
                result, payloads = self.module_producer_fixture("ON", explicit_preview=explicit, raw_preview="OFF")
                self.write_bundle(result, payloads)
                with self.assertRaisesRegex(ValueError, "cached preview value disagrees"):
                    read_native_result(self.bundle)

    test_module_producer_requires_configured_cmake_options.required_subtests = True

    def module_producer_contract_fixture(self):
        result, payloads = self.module_producer_fixture(testing="OFF")
        module = json.loads(payloads["production-module.json"])
        artifacts = json.loads(payloads["rust-artifacts.json"])
        _, contract = self.production_v2_fixture(result["preset"], module_record=module, rust_artifacts=artifacts)
        output = PureWindowsPath(f"C:/checkout/out/validation/{result['preset']}/results/run/production")
        report = json.loads(contract["production-contract.json"])
        report["commands"]["file"] = str(output / "commands.json")
        payloads["production-contract.json"] = json.dumps(report).encode()
        payloads["production-commands.json"] = contract["production-commands.json"]
        targets = result["targets"]
        targets.extend(["juicer", "JuicerProductionPreparedRenderProbe"])
        result["commands"][1]["command"][-2:-2] = targets[-2:]
        result["commands"].append(
            {
                "name": "production-contract",
                "command": [
                    "C:/venv/python.exe",
                    "-B",
                    "C:/checkout/tests/ffi/production_contract.py",
                    "--preset",
                    result["preset"],
                    "--artifacts",
                    output.as_posix(),
                ],
                "directory": "C:/checkout",
                "returncode": 0,
                "drained": True,
                "process_status": "completed",
            }
        )
        result["production_contract"] = True
        return result, payloads

    def test_module_producer_accepts_final_production_contract_only_when_requested(self):
        result, payloads = self.module_producer_contract_fixture()
        output = PureWindowsPath(result["commands"][-1]["command"][-1])
        self.write_bundle(result, payloads)
        # The bundle lives in a temporary directory unrelated to the recorded producer output.
        summary = read_native_result(self.bundle)
        self.assertTrue(summary["production_off"])
        self.assertEqual(summary["prepared_render_cases"], 4)
        command_error = "production contract command differs from the configured producer"
        artifacts_error = "production contract artifacts path differs from the configured producer"
        rejected = {
            "not-requested": ("unexpected CMake producer command sequence", None, None),
            "configure-as-contract": (command_error, slice(None), result["commands"][0]["command"]),
            "wrong-python": (command_error, 0, "C:/other/python.exe"),
            "wrong-script": (command_error, 2, "C:/other/tests/ffi/production_contract.py"),
            "wrong-preset": (command_error, 4, "linux-release"),
            "null-artifacts": (r"commands\[3\]\.command\[6\]", 6, None),
            "foreign-artifacts": (
                artifacts_error,
                6,
                "C:/other/out/validation/windows-clang-release/results/run/production",
            ),
            "wrong-artifacts-leaf": (
                artifacts_error,
                6,
                "C:/checkout/out/validation/windows-clang-release/results/run/other",
            ),
            "foreign-run-artifacts": (
                "production contract commands file differs from the recorded artifacts path",
                6,
                (output.parent.with_name("another-run") / output.name).as_posix(),
            ),
        }
        self.subtest_cases = tuple(rejected)
        for case, (diagnostic, operand, replacement) in rejected.items():
            with self.subTest(case=case):
                altered = copy.deepcopy(result)
                if operand is None:
                    altered["production_contract"] = False
                else:
                    altered["commands"][-1]["command"][operand] = replacement
                (self.bundle / "result.json").write_text(json.dumps(altered), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, diagnostic):
                    read_native_result(self.bundle)

    test_module_producer_accepts_final_production_contract_only_when_requested.required_subtests = True

    def test_module_producer_contract_binds_loaded_module(self):
        result, payloads = self.module_producer_contract_fixture()
        module = json.loads(payloads["production-module.json"])
        module.update(build_sha256="b" * 64, stage_sha256="b" * 64)
        wrong_digest = {**payloads, "production-module.json": json.dumps(module).encode()}

        report = json.loads(payloads["production-contract.json"])
        report["loaded_module"]["build_file"] = report["loaded_module"]["file"]
        wrong_build = {**payloads, "production-contract.json": json.dumps(report).encode()}

        module = json.loads(payloads["production-module.json"])
        module["stage"] = "C:/other-stage/juicer.ofx"
        artifacts = json.loads(payloads["rust-artifacts.json"])
        _, contract = self.production_v2_fixture(result["preset"], module_record=module, rust_artifacts=artifacts)
        report = json.loads(contract["production-contract.json"])
        report["commands"]["file"] = str(PureWindowsPath(result["commands"][-1]["command"][-1]) / "commands.json")
        wrong_stage = {
            **payloads,
            "production-contract.json": json.dumps(report).encode(),
            "production-commands.json": contract["production-commands.json"],
        }

        report = json.loads(payloads["production-contract.json"])
        report["variant"] = "grain-preview"
        trace = json.loads(payloads["production-commands.json"])
        next(row["command"] for row in trace if row["name"] == "ofx-probe").append("preview")
        commands = json.dumps(trace).encode()
        report["commands"]["sha256"] = hashlib.sha256(commands).hexdigest()
        wrong_variant = {
            **payloads,
            "production-contract.json": json.dumps(report).encode(),
            "production-commands.json": commands,
        }
        module_error = "production contract loaded module differs from the producing build and stage record"
        cases = (
            ("foreign-digests", wrong_digest, module_error),
            ("substituted-build", wrong_build, module_error),
            ("foreign-stage", wrong_stage, module_error),
            ("wrong-variant", wrong_variant, "production contract variant differs from the configured producer"),
        )
        self.subtest_cases = tuple(name for name, _, _ in cases)
        for case, altered, diagnostic in cases:
            with self.subTest(case=case):
                self.write_bundle(result, altered)
                with self.assertRaisesRegex(ValueError, diagnostic):
                    read_native_result(self.bundle)

    test_module_producer_contract_binds_loaded_module.required_subtests = True

    def test_module_producer_contract_requires_build_and_probe_inputs(self):
        result, payloads = self.module_producer_contract_fixture()
        altered = copy.deepcopy(result)
        target = "JuicerProductionPreparedRenderProbe"
        altered["targets"].remove(target)
        altered["commands"][1]["command"].remove(target)
        self.subtest_cases = ("missing-target", "foreign-archive", "foreign-render-probe")
        with self.subTest(case="missing-target"):
            self.write_bundle(altered, payloads)
            with self.assertRaisesRegex(ValueError, "missing required production contract targets"):
                read_native_result(self.bundle)

        cases = (
            ("foreign-archive", "rust-symbols", -1, "scan files differ from the producing build"),
            ("foreign-render-probe", "prepared-render", 0, "probe executables differ from the producing build"),
        )
        for case, name, operand, diagnostic in cases:
            with self.subTest(case=case):
                report = json.loads(payloads["production-contract.json"])
                trace = json.loads(payloads["production-commands.json"])
                command = next(row["command"] for row in trace if row["name"] == name)
                original = command[operand]
                substituted = str(PureWindowsPath(original).with_name("foreign-" + PureWindowsPath(original).name))
                report["build_inputs"][substituted] = report["build_inputs"].pop(original)
                command[operand] = substituted
                if name == "rust-symbols":
                    next(row for row in report["scanned"] if row["name"] == name)["file"] = substituted
                else:
                    report["prepared_render"].update(executable=substituted, command=command)
                evidence = json.dumps(trace).encode()
                report["commands"]["sha256"] = hashlib.sha256(evidence).hexdigest()
                self.write_bundle(
                    result,
                    {
                        **payloads,
                        "production-contract.json": json.dumps(report).encode(),
                        "production-commands.json": evidence,
                    },
                )
                with self.assertRaisesRegex(ValueError, diagnostic):
                    read_native_result(self.bundle)

    test_module_producer_contract_requires_build_and_probe_inputs.required_subtests = True

    def test_module_producer_rejects_foreign_production_archives(self):
        result, payloads = self.module_producer_fixture()
        original = json.loads(payloads["rust-artifacts.json"])
        archive = original["variants"]["production"]["archive"]
        alternatives = {
            "test-support": original["variants"]["test-support"]["archive"],
            "wrong-target": archive.replace("/x86_64-pc-windows-msvc/", "/x86_64-unknown-linux-gnu/"),
            "wrong-profile": archive.replace("/release/", "/debug/"),
            "wrong-build": archive.replace("/windows-clang-release/", "/windows-clang-debug/"),
        }
        self.subtest_cases = tuple(alternatives)
        for case, wrong_archive in alternatives.items():
            with self.subTest(case=case):
                artifacts = copy.deepcopy(original)
                artifacts["variants"]["production"]["archive"] = wrong_archive
                payloads["rust-artifacts.json"] = json.dumps(artifacts).encode()
                self.write_bundle(result, payloads)
                with self.assertRaisesRegex(
                    ValueError, "Rust artifact paths differ from the configured build: production"
                ):
                    read_native_result(self.bundle)

    test_module_producer_rejects_foreign_production_archives.required_subtests = True

    def preview_probe_producer_fixture(self):
        result, payloads = self.module_producer_fixture("ON", explicit_preview=False)
        target = "JuicerProductionPreparedRenderProbe"
        result["targets"].append(target)
        result["commands"][1]["command"].insert(-2, target)
        probe = {
            "path": str(PureWindowsPath("C:/checkout/out/build/windows-clang-release/tests/bin") / (target + ".exe")),
            "sha256": "d" * 64,
        }
        payloads["paired-probe.json"] = json.dumps(probe).encode()
        return result, payloads

    def paired_producer_fixture(self):
        preview_result, preview_payloads = self.preview_probe_producer_fixture()
        _, ordinary_payloads = self.module_producer_fixture()
        candidate = json.loads(preview_payloads["candidate.json"])
        del candidate["version_commands"]
        patch = preview_payloads["candidate.patch"]
        preset = candidate["preset"]
        modules = {}
        for role, producer in (("ordinary", ordinary_payloads), ("preview", preview_payloads)):
            module = json.loads(producer["production-module.json"])
            if role == "preview":
                module.update(build_sha256="b" * 64, stage_sha256="b" * 64)
            modules[role] = {
                "preset": preset,
                "configuration": json.loads(producer["candidate.json"])["configuration"],
                "production_module": module,
                "candidate_source": candidate_source_digest(candidate, patch),
                "receipt": f"C:/checkout/out/validation/{preset}/results/{role}",
                "result_sha256": "c" * 64,
            }
        probe = json.loads(preview_payloads["paired-probe.json"])
        modules["preview"].update(paired_probe=probe, targets=preview_result["targets"])
        command = [
            probe["path"],
            "--grain-preview-paired-discovery",
            modules["preview"]["production_module"]["stage"],
            modules["ordinary"]["production_module"]["stage"],
        ]
        caller, case_id = "grain-preview-paired-discovery", "ordinary-and-preview"
        payloads = {
            "candidate.json": json.dumps(candidate).encode(),
            "candidate.patch": patch,
            "production-modules.json": json.dumps(modules).encode(),
            "paired-probe.json": json.dumps(probe).encode(),
            "paired-list.jsonl": (json.dumps({"caller": caller, "id": case_id}) + "\n").encode(),
            "paired-run.jsonl": b"".join(
                (json.dumps({"caller": caller, **row}) + "\n").encode()
                for row in (
                    {"event": "membership", "ids": [case_id]},
                    {"event": "start", "id": case_id},
                    {"event": "terminal", "id": case_id, "outcome": "passed"},
                )
            ),
        }
        result = {
            "schema": "native-paired-discovery-v1",
            "test": "Ffi.Host.GrainPreviewPairedDiscovery",
            "scope": "focused",
            "status": "passed",
            "preset": preset,
            "candidate_head": candidate["head"],
            "candidate_unchanged": True,
            "shipping_isolation": False,
            "commands": [
                {
                    "name": name,
                    "directory": "C:/checkout",
                    "command": argv,
                    "returncode": 0,
                    "drained": True,
                    "process_status": "completed",
                }
                for name, argv in (("paired-list", [*command, "--list-cases"]), ("paired-run", command))
            ],
        }
        return result, payloads

    def test_module_producer_pair_binds_embedded_cached_preview(self):
        result, payloads = self.paired_producer_fixture()
        self.write_bundle(result, payloads)
        self.assertEqual(read_native_result(self.bundle)["custom_cases"], 1)
        modules = json.loads(payloads["production-modules.json"])
        modules["preview"]["configuration"]["JUICER_GRAIN_PREVIEW"] = "OFF"
        payloads["production-modules.json"] = json.dumps(modules).encode()
        self.write_bundle(result, payloads)
        with self.assertRaisesRegex(ValueError, "cached preview value disagrees"):
            read_native_result(self.bundle)

    def test_module_producer_pair_requires_probe_target_and_identity(self):
        result, payloads = self.paired_producer_fixture()
        probe = json.loads(payloads["paired-probe.json"])
        self.write_bundle(result, payloads)
        self.assertEqual(read_native_result(self.bundle)["paired_probe"], probe)

        self.subtest_cases = ("target", "preview-record", "pair-record", "digest", "list-executable", "run-executable")
        for case in self.subtest_cases:
            with self.subTest(case=case):
                altered = copy.deepcopy(result)
                members = dict(payloads)
                modules = json.loads(members["production-modules.json"])
                if case == "target":
                    modules["preview"]["targets"].remove("JuicerProductionPreparedRenderProbe")
                elif case == "preview-record":
                    del modules["preview"]["paired_probe"]
                elif case == "pair-record":
                    del members["paired-probe.json"]
                elif case == "digest":
                    members["paired-probe.json"] = json.dumps({**probe, "sha256": "e" * 64}).encode()
                else:
                    index = 0 if case == "list-executable" else 1
                    altered["commands"][index]["command"][0] = "C:/stale/probe.exe"
                members["production-modules.json"] = json.dumps(modules).encode()
                self.write_bundle(altered, members)
                with self.assertRaisesRegex(ValueError, "probe identity|declared probe"):
                    read_native_result(self.bundle)

    test_module_producer_pair_requires_probe_target_and_identity.required_subtests = True

    def test_producer_and_pair_reject_wrong_json_field_types(self):
        # Follow the metadata, tool/artifact records, embedded producer records and
        # witness events consumed by the reader. Inputs are opaque source identities.
        fixtures = {
            "producer": self.preview_probe_producer_fixture(),
            "contract": self.module_producer_contract_fixture(),
            "pair": self.paired_producer_fixture(),
        }
        wrong_types = (None, 123, True, [], {}, "wrong")

        def fields(value, path=()):
            yield path, value
            if path == ("inputs",):
                return
            if isinstance(value, dict):
                for key, child in value.items():
                    yield from fields(child, (*path, key))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    yield from fields(child, (*path, index))

        for name, (result, payloads) in fixtures.items():
            if name == "producer":
                candidate = json.loads(payloads["candidate.json"])
                candidate["new_files_base64"] = {"fixture.txt": "YQ=="}
                payloads["candidate.json"] = json.dumps(candidate).encode()
            elif name == "contract":
                report = json.loads(payloads["production-contract.json"])
                report["variant"] = "production"
                payloads["production-contract.json"] = json.dumps(report).encode()
            self.write_bundle(result, payloads)
            self.assertEqual(read_native_result(self.bundle)["status"], "passed", name)
            metadata = json.loads((self.bundle / "result.json").read_bytes())
            documents = {"result.json": metadata}
            documents.update({key: json.loads(data) for key, data in payloads.items() if key.endswith(".json")})
            documents.update(
                {
                    key: [json.loads(line) for line in data.splitlines()]
                    for key, data in payloads.items()
                    if key.endswith(".jsonl")
                }
            )
            for member, document in documents.items():
                for path, value in fields(document):
                    if member.endswith(".jsonl") and not path:
                        continue
                    for replacement in wrong_types:
                        if type(replacement) is type(value):
                            continue
                        context = (name, member, path, type(replacement).__name__)
                        mutated = copy.deepcopy(document)
                        if path:
                            parent = mutated
                            for key in path[:-1]:
                                parent = parent[key]
                            parent[path[-1]] = replacement
                        else:
                            mutated = replacement
                        encoded = json.dumps(mutated).encode()
                        if member == "result.json":
                            with self.assertRaisesRegex(ValueError, r"^result\.json", msg=str(context)) as failed:
                                read_native_result(self.bundle, result_bytes=encoded)
                            self.assert_receipt_type_field(failed.exception, member, path)
                            continue
                        if member.endswith(".jsonl"):
                            encoded = b"".join(json.dumps(row).encode() + b"\n" for row in mutated)
                        spec = copy.deepcopy(metadata)
                        spec["members"][member] = {"bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}
                        (self.bundle / member).write_bytes(encoded)
                        with self.assertRaisesRegex(ValueError, member, msg=str(context)) as failed:
                            read_native_result(self.bundle, result_bytes=json.dumps(spec).encode())
                        self.assert_receipt_type_field(failed.exception, member, path)
                        (self.bundle / member).write_bytes(payloads[member])

    def assert_receipt_type_field(self, error, member, path):
        field = member + "".join(f"[{key}]" if isinstance(key, int) else f".{key}" for key in path)
        diagnostic = str(error)
        if " (invalid CMake producer configure " in diagnostic:
            start = diagnostic.index(" (invalid CMake producer configure ")
            diagnostic = diagnostic[:start] + diagnostic[diagnostic.index(")", start) + 1 :]
        self.assertTrue(diagnostic.startswith(field + ": expected "), str(error))

    def test_path_escape_rejected_before_payload_reads(self):
        self.save()
        self.result["members"]["../source"] = {"bytes": 0, "sha256": "0" * 64}
        with patch.object(Path, "read_bytes", side_effect=AssertionError("payload read")):
            with self.assertRaises(ValueError):
                read_native_result(self.bundle, result_bytes=json.dumps(self.result).encode())

    def test_missing_publication_cannot_pass(self):
        self.save()
        (self.bundle / "result.json").unlink()
        with self.assertRaises(FileNotFoundError):
            read_native_result(self.bundle)

    def test_reader_copy_runs_offline_without_tools_or_original_paths(self):
        self.save()
        reader = self.root / "reader.py"
        shutil.copyfile(Path(__file__).resolve().parents[2] / "scripts/qualification.py", reader)
        code = r"""
import sys, os, runpy
from pathlib import Path
root = Path(sys.argv[1]).resolve()
stdlib = Path(os.__file__).resolve().parent
opens = []
def audit(event, args):
    if event in {'subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.exec'}:
        raise AssertionError('offline reader launched a tool')
    if event == 'open' and isinstance(args[0], (str, bytes)):
        p = Path(os.fsdecode(args[0])).resolve()
        if not (p.is_relative_to(root) or p.is_relative_to(stdlib)):
            raise AssertionError('offline reader accessed original machine: ' + str(p))
        if p.suffix.lower() in {'.exe', '.dll', '.a', '.lib', '.ofx'}:
            raise AssertionError('offline reader read executable bytes')
        opens.append(str(p))
sys.addaudithook(audit)
sys.argv = [str(root / 'reader.py'), '--bundle', str(root / 'relocated bundle ü')]
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as exc:
    if exc.code:
        raise
for p in (root / 'relocated bundle ü').iterdir():
    assert opens.count(str(p)) == 1, (p, opens.count(str(p)))
"""
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-X", "utf8", "-c", code, str(self.root)],
            capture_output=True,
            encoding="utf-8",
            timeout=20,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        type(self).read_bytes += json.loads(completed.stdout)["bookkeeping_read_bytes"]


class VersionProbeTests(unittest.TestCase):
    def test_failure_and_success_are_both_recorded_with_exact_arguments(self):
        import argparse
        import verify

        output = io.StringIO()
        commands = []

        def probe(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 7 if command[0] == "failed tool" else 0, "version output")

        with patch.object(verify.subprocess, "run", side_effect=probe), patch.object(sys, "stdout", output):
            status = verify.tool_versions(
                argparse.Namespace(tool=[("cargo", "failed tool"), ("rustc", "proxy path/rustc")])
            )
        records = json.loads(output.getvalue())
        self.assertEqual(status, 1)
        self.assertEqual([r["returncode"] for r in records], [7, 0])
        self.assertIn(["proxy path/rustc", "-Vv"], commands)
        self.assertIn(["failed tool", "--version"], commands)

    def test_missing_tool_is_a_failed_probe_and_keeps_other_results(self):
        import argparse
        import verify

        output = io.StringIO()

        def probe(command, **kwargs):
            if command[0] == "missing":
                raise FileNotFoundError("missing tool")
            return subprocess.CompletedProcess(command, 0, "version output")

        with patch.object(verify.subprocess, "run", side_effect=probe), patch.object(sys, "stdout", output):
            status = verify.tool_versions(argparse.Namespace(tool=[("cargo", "missing"), ("cmake", "present")]))
        records = json.loads(output.getvalue())
        self.assertEqual(status, 1)
        self.assertIn("error", records[0])
        self.assertEqual(records[1]["returncode"], 0)


class OpticsSourceTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="optics source ü ")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.literal = "tests/film/optics-fixtures/literal.json"
        self.git("init", "-q")
        policy = Path(__file__).resolve().parents[2] / "scripts/source_file_policy.json"
        (self.root / "scripts").mkdir()
        (self.root / "scripts/source_file_policy.json").write_bytes(policy.read_bytes())
        for name in (
            "src/producer.h",
            "rust/producer.rs",
            "tests/ffi/consumer.inc",
            self.literal,
            "tests/route/development.json",
            "tests/quality/test_native_results.py",
        ):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("baseline\n")
        self.git("add", ".")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "baseline")

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root)

    def capture(self):
        from verify import capture_native_source

        return capture_native_source(self.root, self.literal)

    @unittest.skipUnless(os.name == "posix", "POSIX executable mode control")
    def test_new_input_modes_follow_git_filemode_policy(self):
        source = self.root / "src/new_header.h"
        source.write_text("new source\n")
        self.git("config", "core.filemode", "false")
        source.chmod(0o644)
        first = self.capture()[0]
        source.chmod(0o755)
        second = self.capture()[0]
        self.assertEqual(first, second)
        self.assertEqual(second["inputs"]["src/new_header.h"]["mode"], "100644")
        self.git("config", "core.filemode", "true")
        self.assertEqual(self.capture()[0]["inputs"]["src/new_header.h"]["mode"], "100755")

    def test_tracked_inputs_use_git_without_reopening_each_source(self):
        policy = self.root / "scripts/source_file_policy.json"
        data = policy.read_bytes()

        def read(path):
            if path == policy:
                return data
            raise AssertionError("tracked source reread")

        with patch.object(Path, "read_bytes", read):
            first = self.capture()
            second = self.capture()
        self.assertEqual(first[:2], second[:2])
        self.assertEqual(first[2]["source_read_bytes"], len(data))
        self.assertIn("tests/route/development.json", first[0]["inputs"])

    def test_tracked_dirty_patch_detects_same_length_edit_and_reconstructs(self):
        before = self.capture()
        (self.root / "src/producer.h").write_text("modified\n")
        after = self.capture()
        self.assertNotEqual(before[1], after[1])
        self.assertEqual(after[0]["head"], before[0]["head"])
        subprocess.run(["git", "apply", "--reverse", "--check"], cwd=self.root, input=after[1], check=True)

    def test_preparation_retains_new_host_entry_and_preset_bytes(self):
        import base64

        paths = ("tests/host/CMakeLists.txt", "tests/host/CMakePresets.json")
        before = self.capture()
        for name in paths:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"new host configuration\n")
        source, _, _ = self.capture()
        self.assertNotEqual(source, before[0])
        for name in paths:
            self.assertIn(name, source["inputs"])
            self.assertEqual(base64.b64decode(source["new_files_base64"][name]), b"new host configuration\n")

    def test_preparation_nested_cmake_edits_are_reconstructible(self):
        paths = ("tests/host/CMakeLists.txt", "tests/host/CMakePresets.json", "tests/host/nested/CMakeLists.txt")
        for name in paths:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"baseline host configuration\n")
        self.git("add", "tests/host")
        self.git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "host configuration baseline",
        )
        before = self.capture()
        self.subtest_cases = paths
        for name in paths:
            with self.subTest(case=name):
                (self.root / name).write_bytes(b"modified host configuration\n")
                after = self.capture()
                self.assertNotEqual(before[1], after[1])
                self.assertIn(("diff --git a/" + name + " b/" + name).encode(), after[1])
                subprocess.run(["git", "apply", "--reverse", "--check"], cwd=self.root, input=after[1], check=True)
                (self.root / name).write_bytes(b"baseline host configuration\n")

    test_preparation_nested_cmake_edits_are_reconstructible.required_subtests = True

    def test_rapid_same_size_edit_after_staging_and_capture_is_detected(self):
        path = self.root / "src/producer.h"
        same_second = 0
        for _ in range(8):
            path.write_bytes(b"entry words\n")
            self.git("add", "src/producer.h")
            before = self.capture()
            timestamp = path.stat().st_mtime_ns // 1_000_000_000
            path.write_bytes(b"other words\n")
            same_second += timestamp == path.stat().st_mtime_ns // 1_000_000_000
            self.assertNotEqual(before[1], self.capture()[1])
        self.assertGreater(same_second, 0, "no same-second mutation was exercised")

    def test_new_input_bytes_and_deletion_change_candidate(self):
        import base64

        before = self.capture()
        path = self.root / "rust/new.rs"
        path.write_bytes(b"new source\n")
        after = self.capture()
        self.assertNotEqual(before[0], after[0])
        self.assertEqual(base64.b64decode(after[0]["new_files_base64"]["rust/new.rs"]), path.read_bytes())
        path.write_bytes(b"changed source\n")
        self.assertNotEqual(after[0], self.capture()[0])
        (self.root / "src/producer.h").unlink()
        self.assertNotEqual(before[1], self.capture()[1])

    def test_staged_deletion_remains_in_reconstructible_patch(self):
        self.git("rm", "-q", "src/producer.h")
        source, patch_bytes, _ = self.capture()
        self.assertNotIn("src/producer.h", source["inputs"])
        self.assertIn(b"deleted file mode", patch_bytes)
        subprocess.run(["git", "apply", "--reverse", "--check"], cwd=self.root, input=patch_bytes, check=True)

    def test_unrelated_scientific_payload_is_outside_pilot_inputs(self):
        before = self.capture()
        path = self.root / "tests/unrelated/array.json"
        path.parent.mkdir(parents=True)
        path.write_text("unrelated\n")
        self.assertEqual(before[:2], self.capture()[:2])

    def test_linked_support_in_other_test_domain_is_captured(self):
        before = self.capture()
        path = self.root / "tests/dir/boundary_test_hooks.cpp"
        path.parent.mkdir(parents=True)
        path.write_text("linked support\n")
        self.git("add", str(path.relative_to(self.root)))
        after = self.capture()
        self.assertIn("tests/dir/boundary_test_hooks.cpp", after[0]["inputs"])
        self.assertNotEqual(before[1], after[1])
        path.write_text("changed support\n")
        self.assertNotEqual(after[1], self.capture()[1])

    def test_consumed_linker_maps_are_captured_and_detect_changed_bytes(self):
        for name in ("tests/benchmarks/measurement.exports", "tests/benchmarks/measurement.def"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("export original\n")
            before, _, _ = self.capture()
            self.assertIn(name, before["new_files_base64"])
            self.git("add", name)
            before, patch, _ = self.capture()
            path.write_text("export replacement\n")
            changed, updated, _ = self.capture()
            self.assertIn(name, before["inputs"])
            self.assertIn(name, changed["inputs"])
            self.assertNotEqual(patch, updated)

    def test_supported_untracked_headers_are_preserved_and_compared(self):
        import base64

        self.subtest_cases = tuple(f"extension/{extension}" for extension in (".cuh", ".hh", ".hxx", ".inl", ".ixx"))
        for extension in (".cuh", ".hh", ".hxx", ".inl", ".ixx"):
            with self.subTest(case=f"extension/{extension}", extension=extension):
                path = self.root / ("src/new" + extension)
                path.write_bytes(b"new header\n")
                before = self.capture()
                name = path.relative_to(self.root).as_posix()
                self.assertEqual(base64.b64decode(before[0]["new_files_base64"][name]), path.read_bytes())
                path.write_bytes(b"changed header\n")
                self.assertNotEqual(before[0], self.capture()[0])
                path.unlink()

    @unittest.skipUnless(os.name == "posix", "native symlink worktree witness")
    def test_unstaged_regular_file_to_symlink_is_rejected(self):
        target = self.root / "target"
        target.write_text("external bytes\n")
        path = self.root / "src/producer.h"
        path.unlink()
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "worktree source must be a regular file"):
            self.capture()

    @unittest.skipUnless(os.name == "posix", "native linked-directory witness")
    def test_linked_source_directory_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "producer.h").write_text("outside bytes\n")
        shutil.rmtree(self.root / "src")
        (self.root / "src").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "worktree source must be a regular file"):
            self.capture()

    @unittest.skipUnless(os.name == "posix", "native nested linked-directory witness")
    def test_new_linked_test_directory_is_rejected_before_suffix_filter(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "new.h").write_text("outside bytes\n")
        (self.root / "tests/new_support").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "worktree source must be a regular file"):
            self.capture()

    @unittest.skipUnless(os.name == "nt", "Windows junction source witness")
    def test_windows_source_junction_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "producer.h").write_text("outside bytes\n")
        shutil.rmtree(self.root / "src")
        subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(self.root / "src"), str(outside)],
            check=True,
            capture_output=True,
        )
        with self.assertRaisesRegex(ValueError, "worktree source must be a regular file"):
            self.capture()

    def test_gpu_witness_records_resource_inputs_without_copying_arrays(self):
        from verify import capture_native_source

        path = self.root / "Resources/profiles/selected.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"resource bytes\n")
        self.git("add", "Resources")
        before = capture_native_source(self.root, self.literal, gpu_witness=True)
        self.assertIn("Resources/profiles/selected.json", before[0]["inputs"])
        path.write_bytes(b"changed resource bytes\n")
        after = capture_native_source(self.root, self.literal, gpu_witness=True)
        self.assertNotEqual(before[1], after[1])

    def test_index_flags_cannot_hide_selected_mutation(self):
        self.subtest_cases = ("flag/assume-unchanged", "flag/skip-worktree")
        for flag in ("assume-unchanged", "skip-worktree"):
            with self.subTest(case=f"flag/{flag}", flag=flag):
                self.git("update-index", "--" + flag, "src/producer.h")
                with self.assertRaisesRegex(ValueError, "ordinary, merged Git tracking"):
                    self.capture()
                self.git("update-index", "--no-" + flag, "src/producer.h")

    test_supported_untracked_headers_are_preserved_and_compared.required_subtests = True
    test_index_flags_cannot_hide_selected_mutation.required_subtests = True


class NativeOwnerTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="native owner ü ")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name).resolve()
        self.directory = self.root / "result"
        self.directory.mkdir()

    def command(self, code):
        python = str(Path(sys.base_prefix) / "python.exe") if os.name == "nt" else sys.executable
        return [python, "-c", code]

    @unittest.skipUnless(os.name == "nt", "Windows compiler telemetry job policy")
    def test_native_compiler_cleanup_drains_its_exact_telemetry_child(self):
        vctip = self.root / "vctip.exe"
        shutil.copyfile(Path(os.environ["SystemRoot"]) / "System32/ping.exe", vctip)
        code = f"import subprocess;subprocess.Popen([{str(vctip)!r},'-n','30','127.0.0.1'])"
        with execution.OutputLease(self.root, self.root / "build"):
            run = execution.NativeRun(self.root, self.directory)
            run.command("compile", self.command(code), timeout=5, vctip_executable=vctip)
            self.assertTrue(run.commands[0]["drained"])
            self.assertEqual(len(run.commands[0]["vctip_terminated"]), 1)
            self.assertEqual(Path(run.commands[0]["vctip_terminated"][0]["path"]), vctip)
            self.assertEqual(run.commands[0]["vctip_cleanup"]["grace_seconds"], 1)
            run.command("test", self.command("pass"), timeout=5)
            self.assertNotIn("vctip_cleanup", run.commands[1])

    def test_failure_records_exit_and_drains_without_legacy_admission(self):
        with (
            execution.OutputLease(self.root, self.root / "build"),
            patch.object(execution, "digest", side_effect=AssertionError("tool hashing")),
        ):
            run = execution.NativeRun(self.root, self.directory)
            with self.assertRaises(subprocess.CalledProcessError):
                run.command("failure", self.command("raise SystemExit(7)"))
            self.assertEqual(run.commands[0]["returncode"], 7)
            self.assertTrue(run.commands[0]["drained"])
            self.assertFalse((self.directory / "result.json").exists())

    def test_successful_parent_waits_for_descendant(self):
        child = "import time; time.sleep(.15); print('descendant finished',flush=True)"
        code = "import subprocess,sys;subprocess.Popen([sys.executable,'-c'," + repr(child) + "])"
        with execution.OutputLease(self.root, self.root / "build"):
            run = execution.NativeRun(self.root, self.directory)
            log = run.command("descendant", self.command(code), timeout=10)
            self.assertIn("descendant finished", log.read_text())
            self.assertTrue(run.commands[0]["drained"])

    def test_cancellation_drains_and_does_not_publish(self):
        with execution.OutputLease(self.root, self.root / "build"):
            run = execution.NativeRun(self.root, self.directory)
            with self.assertRaises(subprocess.TimeoutExpired):
                run.command(
                    "cancel",
                    self.command(
                        "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);time.sleep(30)"
                    ),
                    timeout=0.2,
                )
            self.assertTrue(run.commands[0]["drained"])
            self.assertFalse((self.directory / "result.json").exists())

    def test_owner_cancellation_drains_started_descendant_on_both_hosts(self):
        marker = self.root / "descendant started"
        child = "import pathlib,time;pathlib.Path(" + repr(str(marker)) + ").write_text('ready');time.sleep(30)"
        command = (
            "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c'," + repr(child) + "]);time.sleep(30)"
        )
        original = execution.subprocess.Popen

        def launch(*args, **kwargs):
            process = original(*args, **kwargs)
            wait = process.wait
            first = True

            def cancel(timeout=None):
                nonlocal first
                if first:
                    first = False
                    deadline = time.monotonic() + 10
                    while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
                        time.sleep(0.02)
                    self.assertTrue(marker.exists(), "real descendant did not start")
                    raise KeyboardInterrupt
                return wait(timeout=timeout)

            process.wait = cancel
            return process

        with execution.OutputLease(self.root, self.root / "build"):
            run = execution.NativeRun(self.root, self.directory)
            with patch.object(execution.subprocess, "Popen", side_effect=launch):
                with self.assertRaises(KeyboardInterrupt):
                    run.command("cancel-owner", self.command(command))
            self.assertTrue(run.commands[0]["drained"])
            self.assertEqual(run.commands[0]["process_status"], "interrupted")
            self.assertFalse((self.directory / "result.json").exists())

    @unittest.skipUnless(os.name == "posix", "Step 1 Linux owner interruption witness")
    def test_manual_owner_interruption_drains_descendants(self):
        started = self.root / "started"
        child = "import pathlib,time;pathlib.Path(" + repr(str(started)) + ").write_text('started');time.sleep(30)"
        owner = """
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from execution import NativeRun,OutputLease
root=Path(sys.argv[2])
with OutputLease(root,root/'build'):
    run=NativeRun(root,root/'result')
    try:
        run.command('interrupt',[sys.executable,'-c',sys.argv[3]])
    except KeyboardInterrupt:
        pass
    (root/'observed.json').write_text(json.dumps(run.commands))
"""
        process = subprocess.Popen(
            [sys.executable, "-B", "-c", owner, str(Path(execution.__file__).parent), str(self.root), child],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            deadline = time.monotonic() + 10
            while not started.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(started.exists(), "contained child did not start")
            process.send_signal(signal.SIGINT)
            _, error = process.communicate(timeout=25)
            self.assertEqual(process.returncode, 0, error.decode())
            records = json.loads((self.root / "observed.json").read_text())
            self.assertTrue(records[0]["drained"])
            self.assertEqual(records[0]["process_status"], "interrupted")
            self.assertFalse((self.directory / "result.json").exists())
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=20)

    def test_crash_after_case_start_never_publishes(self):
        events = self.directory / "partial.jsonl"
        code = "import pathlib,os;pathlib.Path(" + repr(str(events)) + ").write_text('start\\n');os._exit(9)"
        with execution.OutputLease(self.root, self.root / "build"):
            run = execution.NativeRun(self.root, self.directory)
            with self.assertRaises(subprocess.CalledProcessError):
                run.command("crash", self.command(code))
            self.assertEqual(events.read_text(), "start\n")
            self.assertTrue(run.commands[0]["drained"])
            self.assertFalse((self.directory / "result.json").exists())

    def test_uncertain_drainage_blocks_owner_and_publication(self):
        lease = execution.OutputLease(self.root, self.root / "build")
        with lease:
            run = execution.NativeRun(self.root, self.directory)

            def uncertain(*args):
                args[-1]["drained"] = False
                raise RuntimeError("uncertain child")

            with patch.object(execution, "run_process_tree", side_effect=uncertain):
                with self.assertRaises(RuntimeError):
                    run.command("uncertain", self.command("pass"))
            self.assertTrue(lease.blocked)
            with self.assertRaises(RuntimeError):
                run.publish({"status": "passed"})
        self.assertEqual(json.loads((lease.path / "owner.json").read_text())["status"], "blocked")

    def test_atomic_publication_failure_leaves_no_completed_result(self):
        with execution.OutputLease(self.root, self.root / "build"):
            run = execution.NativeRun(self.root, self.directory)
            with patch.object(Path, "replace", side_effect=OSError("interrupted publication")):
                with self.assertRaises(OSError):
                    run.publish({"status": "passed"})
            self.assertFalse((self.directory / "result.json").exists())
            run.publish({"status": "failed"})
            with self.assertRaises(ValueError):
                run.publish({"status": "passed"})

    def test_gpu_cancellation_drains_before_releasing_both_owners(self):
        with execution.OutputLease(self.root, self.root / "build") as build_owner:
            with execution.OutputLease(self.root) as gpu_owner:
                run = execution.NativeRun(self.root, self.directory)
                with self.assertRaises(subprocess.TimeoutExpired):
                    run.command(
                        "gpu-cancel",
                        self.command(
                            "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);time.sleep(30)"
                        ),
                        timeout=0.2,
                    )
                self.assertTrue(run.commands[0]["drained"])
                self.assertTrue((gpu_owner.path / "owner.json").exists())
                self.assertTrue((build_owner.path / "owner.json").exists())
                self.assertFalse((self.directory / "result.json").exists())
            self.assertFalse(gpu_owner.path.exists())
        self.assertFalse(build_owner.path.exists())

    def test_uncertain_gpu_descendants_block_both_owners(self):
        with execution.OutputLease(self.root, self.root / "build") as build_owner:
            with execution.OutputLease(self.root) as gpu_owner:
                run = execution.NativeRun(self.root, self.directory)

                def uncertain(*args):
                    args[-1]["drained"] = False
                    raise RuntimeError("uncertain GPU child")

                with patch.object(execution, "run_process_tree", side_effect=uncertain):
                    with self.assertRaisesRegex(RuntimeError, "uncertain"):
                        run.command("gpu-uncertain", self.command("pass"))
                self.assertTrue(build_owner.blocked and gpu_owner.blocked)
                with self.assertRaises(RuntimeError):
                    run.publish({"status": "passed"})
        for owner in (build_owner, gpu_owner):
            self.assertEqual(json.loads((owner.path / "owner.json").read_text())["status"], "blocked")


NativeResultTests.test_production_report_rejects_missing_or_failed_command.required_subtests = True

NativeResultTests.test_production_v2_rejects_invalid_observations.required_subtests = True

NativeResultTests.test_completed_v2_production_report_is_read_from_bundle_only.required_subtests = True
NativeResultTests.test_completed_v3_production_report_accepts_relocated_windows_and_linux_sources.required_subtests = (
    True
)
NativeResultTests.test_production_v3_rejects_unreviewed_path_digest_and_argv_variants.required_subtests = True

NativeResultTests.test_production_report_rejects_missing_or_failed_prepared_row.required_subtests = True

NativeResultTests.test_production_report_rejects_testing_on_or_wrong_preset.required_subtests = True


if __name__ == "__main__":
    unittest.main()
