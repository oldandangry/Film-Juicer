"""Offline OFX measurement evidence validation."""

import json
import hashlib
import importlib
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import ofx


def sample(lane, row):
    width, height = ofx.HALATION_EXTENTS.get(row, (6048, 4032))
    renders = []
    phases = [(p, 0) for p in ["cold"] + ["warmup"] * 10 + ["warm"] * 30 + ["changed", "changed-reuse"] * 3]
    phases += [(p, c) for c in (1, 2) for p in ["cycle-cold"] + ["cycle-warmup"] * 10 + ["cycle-end"]]
    counts = {}
    for phase, cycle in phases:
        index = counts.get((phase, cycle), 0)
        counts[phase, cycle] = index + 1
        cold = phase in ("cold", "changed", "cycle-cold")
        renders.append(
            {
                "phase": phase,
                "cycle": cycle,
                "index": index,
                "callback_status": 0,
                "callback_ns": 100,
                "context_preserved": True,
                "output_checked": True,
                "output_valid": True,
                "stream_interval_ms": 1,
                "releases_valid": True,
                "construction": {
                    "starts": int(cold),
                    "pairs": int(cold),
                    "incomplete": 0,
                    "outcome": "success" if cold else "reused",
                    "duration_ns": 10 if cold else None,
                },
            }
        )
    endpoints = [("before-load", -1, 0), ("after-sample", 0, 30), ("after-unload", -1, 0)]
    endpoints += [(n, c, 0) for n in ("after-create", "after-cold", "after-warmups", "after-destroy") for c in range(3)]
    endpoints += [("after-edit", 0, i) for i in range(3)]
    memory = [{"name": n, "cycle": c, "index": i, "rss_bytes": 1, "peak_rss_bytes": 1} for n, c, i in endpoints]
    keys = (
        "device_reserved_bytes",
        "device_committed_bytes",
        "device_retiring_bytes",
        "device_charged_bytes",
        "device_records",
        "device_peak_reserved_bytes",
        "device_peak_committed_bytes",
        "device_peak_retiring_bytes",
        "device_peak_charged_bytes",
        "device_peak_records",
        "pinned_upload_pool_physical_bytes",
        "pinned_upload_pool_peak_bytes",
        "pinned_upload_pool_state_bytes",
        "pinned_upload_pool_state_counts",
    )
    for endpoint in memory:
        endpoint["observation"] = {
            key: ([0, 0, 0, 0] if key.endswith(("state_bytes", "state_counts")) else 0) for key in keys
        }
        endpoint["observation"].update({key: 1 for key in keys if "_peak_" in key})
        endpoint["observation"].update({"device_id": 0, "context_opaque": 1, "context_epoch": 1})
    record = {
        "schema": 1,
        "status": "passed",
        "mode": "measure",
        "lane": lane,
        "workload": row,
        "width": width,
        "height": height,
        "controls": {"x": [1]},
        "control_types": {"x": "double"},
        "device": {"ordinal": 0, "context_opaque": 1},
        "renders": renders,
        "memory": memory,
        "edits": [{"gamma": g, "status": 0, "edit_ns": 1} for g in (1.1, 1.0, 1.1)],
        "timings": {
            "module_open_ns": 1,
            "load_ns": 1,
            "describe_ns": 1,
            "unload_ns": 1,
            "create_ns": [1] * 3,
            "author_ns": [1] * 3,
            "destroy_ns": [1] * 3,
        },
        "driver_owned": {"device_buffer_bytes": 1, "host_buffer_bytes": 1},
        "lifecycle": {
            "module_load_status": 0,
            "module_unload_status": 0,
            "context_restored": True,
            "module_closed": True,
            "callback_error": False,
            "effects_created": 3,
            "effects_destroyed": 3,
        },
        "drained": {"device_charged_bytes": 0, "device_records": 0, "pinned_upload_pool_physical_bytes": 0},
    }
    if row in ofx.MATCHED_HIGHLIGHT_WORKLOADS:
        record.update(
            controls={"Exposure": [6.0]},
            advanced_halation_available=True,
            halation_boost_ev=4.0 if row.endswith("-boost") else 0.0,
            halation_protection_ev=0.0,
        )
    return record


def production_evidence(preset, module_hash, consumed_path, *, module=None, snapshot=b"{}"):
    digest = ofx.execution.digest(consumed_path)
    contract = {
        "preset": preset,
        "build_testing": "OFF",
        "test_support": "OFF",
        "exports": ["OfxGetNumberOfPlugins", "OfxGetPlugin"],
        "loaded_module": {"sha256": module_hash},
        "consumed_inputs": {str(consumed_path): {"sha256_before": digest, "sha256_after": digest}},
    }
    if module is not None:
        contract["loaded_module"]["file"] = str(module)
    contract_bytes = json.dumps(contract).encode()
    patch = b""

    def member(data):
        return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    result = {
        "status": "passed",
        "preset": preset,
        "production_contract": True,
        "candidate_unchanged": True,
        "commands": [{"drained": True, "returncode": 0, "process_status": "completed"}],
        "members": {
            "production-contract.json": member(contract_bytes),
            "candidate.json": member(snapshot),
            "candidate.patch": member(patch),
        },
    }
    return json.dumps(result).encode(), contract_bytes, snapshot, patch


class MeasurementReportTests(unittest.TestCase):
    def test_positive_print_route_selects_print_layout(self):
        # The route reader operates entirely on the ABI tag.
        with patch.dict(sys.modules, {"lldb": object()}):
            reader = importlib.import_module("observe_seed")
        for tag, route, medium in (
            (0, "NegativeDirectScan", "Direct"),
            (1, "NegativePrintScan", "Print"),
            (2, "PositiveDirectScan", "Direct"),
            (3, "PositivePrintScan", "Print"),
        ):
            self.assertEqual(reader.scan_medium(tag, route), medium)
        with self.assertRaisesRegex(ValueError, "expected scanner route NegativePrintScan"):
            reader.scan_medium(3, "NegativePrintScan")
        with self.assertRaisesRegex(ValueError, "unknown native scanner route"):
            reader.scan_medium(4, "PositivePrintScan")
        direct_only = {"layout": {"scanner_payload": {"DirectPipelineRunParams": {"width": [0, 4]}}}}
        with patch.object(reader, "settings", direct_only), patch.object(reader, "argument", return_value=0):
            with self.assertRaisesRegex(KeyError, "PrintPipelineRunParams"):
                reader.observe_materialized_density(None, {"scan_medium": reader.scan_medium(3, "PositivePrintScan")})

    def test_replay_rejects_the_same_companion_and_a_byte_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "companion.ofx"
            path.write_bytes(b"companion")
            arm = {"module": str(path)}
            with self.assertRaisesRegex(ValueError, "distinct module files and contents"):
                ofx.validate_distinct_replay_modules(arm, arm)
            copy = path.with_name("shipping.ofx")
            copy.write_bytes(path.read_bytes())
            shipping = {"module": str(copy)}
            with self.assertRaisesRegex(ValueError, "distinct module files and contents"):
                ofx.validate_distinct_replay_modules(shipping, arm)
            copy.write_bytes(b"shipping")
            ofx.validate_distinct_replay_modules(shipping, arm)

    def make_replay_producer(self, preset, name, targets):
        root = self.root.resolve() / name
        build = root / "out/build" / preset
        receipt = root / "receipt"
        policy = root / "scripts/source_file_policy.json"
        policy.parent.mkdir(parents=True)
        policy.write_text('{"supportedExtensions": [".cu"]}', encoding="utf-8")
        (root / ".gitattributes").write_bytes(b"*.def text eol=lf\n*.exports text eol=lf\n")
        controls = {
            target: control
            for target in targets
            if (control := ofx.qualification.scanner_replay_linker_control(target, preset)) is not None
        }
        for control in controls.values():
            path = root / control
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"export original\r\n")
        for args in (
            ["init", "-q"],
            ["add", "."],
            ["-c", "user.name=Test", "-c", "user.email=test@example.test", "commit", "-qm", "Fixture"],
        ):
            subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
        candidate, source_patch, _ = ofx.verify.capture_native_source(
            root, "scripts/verify.py", domain_tests=True, raw_sha256_inputs=controls.values()
        )
        candidate.update(
            preset=preset,
            configuration={
                "CMAKE_HOME_DIRECTORY": str(root),
                "CMAKE_BUILD_TYPE": "Release",
                "JUICER_GRAIN_PREVIEW_CONFIGURED": "OFF",
            },
        )
        receipt.mkdir()
        source, patch_file = receipt / "candidate.json", receipt / "candidate.patch"
        ofx.execution.write_json(source, candidate)
        patch_file.write_bytes(source_patch)

        def file(path, contents):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
            return path

        native = file(build / "native.a", b"native")
        variants = {
            role: {
                "archive": str(file(build / "cargo" / role / "rust.a", role.encode())),
                "features": ["test-support"] if role == "test-support" else [],
            }
            for role in ("production", "test-support")
        }
        resources = {"fixture": hashlib.sha256(b"resource").hexdigest()}
        modules, consumers, queries, cargo_records = {}, {}, {}, {}
        for target in targets:
            contents = build / target / "Contents"
            binary = file(contents / "arch/module.ofx", f"{name}/{target}".encode())
            file(contents / "Resources/fixture", b"resource")
            obj = file(build / f"{target}.o", target.encode())
            role = "production" if target == "juicer" else "test-support"
            archive = variants[role]["archive"]
            consumers[target] = {"binary": str(binary), "variant": role}
            cargo = build / "tests/rust-artifacts.json"
            if target == "JuicerOfxSpectralReplayModule":
                cargo = build / "tests/spectral-replay/artifacts.json"
                cargo.parent.mkdir(parents=True, exist_ok=True)
                ofx.execution.write_json(
                    cargo,
                    {
                        "variant": "ordinary",
                        "rust_features": ["test-support"],
                        "module": str(binary),
                        "rust_archive": archive,
                        "native_archive": str(native),
                    },
                )
            cargo_records[target] = cargo
            modules[target] = binary
            output = binary.relative_to(build).as_posix()
            queries[output] = "\n".join(
                [
                    f"{output}:",
                    f"  input: CXX_MODULE_LIBRARY_LINKER__{target}_Release",
                    f"    {obj.name}",
                    f"    | {native}",
                    f"    | {archive}",
                    *([f"    | {root / controls[target]}"] if target in controls else []),
                    "    || order-only-stamp",
                    "  outputs:",
                    f"    {target}",
                ]
            )
        cargo = build / "tests/rust-artifacts.json"
        cargo.parent.mkdir(exist_ok=True)
        ofx.execution.write_json(cargo, {"consumers": consumers, "variants": variants, "native_archive": str(native)})
        with patch.object(ofx.verify.subprocess, "check_output", side_effect=lambda command, **_: queries[command[-1]]):
            ofx.verify.capture_scanner_replay_links(build, receipt, candidate, source_patch, targets, "ninja")
        result = receipt / "result.json"
        cmake = str(root / "cmake")
        ofx.execution.write_json(
            result,
            {
                "schema": "native-build-v1",
                "status": "passed",
                "preset": preset,
                "shipping_isolation": False,
                "candidate_head": candidate["head"],
                "candidate_unchanged": True,
                "scanner_replay_targets": list(targets),
                "commands": [
                    {
                        "name": name,
                        "command": command,
                        "directory": str(root),
                        "drained": True,
                        "returncode": 0,
                        "process_status": "completed",
                    }
                    for name, command in (
                        ("configure", [cmake, "--preset", preset]),
                        ("build", [cmake, "--build", str(build), "--target", *targets, "--", "-k0"]),
                    )
                ],
                "members": {
                    file.name: {"bytes": file.stat().st_size, "sha256": ofx.execution.digest(file)}
                    for file in receipt.iterdir()
                },
            },
        )
        declarations = {
            target: {
                "module": str(module),
                "module_sha256": ofx.execution.digest(module),
                "build_binding": {
                    "producer": str(result),
                    "source": str(source),
                    "patch": str(patch_file),
                    "cargo": str(cargo_records[target]),
                },
            }
            for target, module in modules.items()
        }
        return declarations, resources

    def test_replay_rejects_source_and_patch_from_another_producer(self):
        preset = "windows-clang-release" if os.name == "nt" else "linux-release"
        target = "JuicerOfxMeasurementModule"
        declarations, _ = self.make_replay_producer(preset, "cubic", (target,))
        declaration = declarations[target]
        binding = declaration["build_binding"]
        cargo = Path(binding["cargo"])
        candidate = json.loads(Path(binding["source"]).read_bytes())
        roles = json.loads(cargo.read_bytes())
        rust = Path(roles["variants"]["test-support"]["archive"])
        native = Path(roles["native_archive"])
        evidence = ofx.scanner_replay_provenance(declaration, "cubic", True, preset)
        self.assertEqual(evidence["files"]["rust_archive"]["sha256"], ofx.execution.digest(rust))
        self.assertEqual(evidence["files"]["native_archive"]["sha256"], ofx.execution.digest(native))
        self.assertEqual(Path(evidence["files"]["cargo"]["path"]).parent, Path(binding["producer"]).parent)

        unrelated_source, unrelated_patch = self.root / "spectral.json", self.root / "spectral.patch"
        ofx.execution.write_json(unrelated_source, {**candidate, "head": "2" * 40})
        unrelated_patch.write_bytes(b"spectral patch")
        with patch.dict(binding, source=str(unrelated_source), patch=str(unrelated_patch)):
            with self.assertRaisesRegex(ValueError, "source snapshot or patch differs from the producing build"):
                ofx.scanner_replay_provenance(declaration, "cubic", True, preset)
        cargo.write_bytes(b"new configure")
        with self.assertRaisesRegex(ValueError, "Cargo role record differs from the producing build"):
            ofx.scanner_replay_provenance(declaration, "cubic", True, preset)
        del binding["cargo"]
        self.assertEqual(ofx.scanner_replay_provenance(declaration, "cubic", True, preset), evidence)

    def test_replay_requires_tracked_linker_control_bytes_to_match(self):
        preset = "windows-clang-release" if os.name == "nt" else "linux-release"
        measurement = "JuicerOfxMeasurementModule"
        spectral = "JuicerOfxSpectralReplayModule"
        controls = (
            {measurement: "tests/benchmarks/measurement.def"}
            if os.name == "nt"
            else {
                "juicer": "cmake/juicer.exports",
                measurement: "tests/benchmarks/measurement.exports",
                spectral: "tests/benchmarks/spectral_replay.exports",
            }
        )
        declarations, _ = self.make_replay_producer(preset, "tracked-links", ("juicer", measurement, spectral))
        receipt = Path(declarations["juicer"]["build_binding"]["producer"]).parent
        result = json.loads((receipt / "result.json").read_bytes())
        payloads = {name: (receipt / name).read_bytes() for name in result["members"]}
        candidate = json.loads(payloads["candidate.json"])
        record = json.loads(payloads["scanner-replay-links.json"])
        root = Path(candidate["configuration"]["CMAKE_HOME_DIRECTORY"])

        def validate():
            return ofx.qualification.validate_scanner_replay_links(
                result, candidate, {**payloads, "scanner-replay-links.json": json.dumps(record).encode()}
            )

        self.assertEqual(validate(), record["targets"])
        self.subtest_cases = tuple(
            f"{target}/{defect}"
            for target in controls
            for defect in ("missing-source-digest", "mismatched-link-digest")
        )
        for target, control in controls.items():
            identity = candidate["inputs"][control]
            self.assertIn("git_blob", identity)
            self.assertEqual(identity["worktree_sha256"], hashlib.sha256(b"export original\r\n").hexdigest())
            self.assertEqual(record["targets"][target]["inputs"][str(root / control)], identity["worktree_sha256"])
            error = f"{target} replay linker control digest"
            with self.subTest(case=f"{target}/missing-source-digest"):
                tracked_only = {key: identity[key] for key in ("git_blob", "mode")}
                with patch.dict(candidate["inputs"], {control: tracked_only}):
                    source_identity = ofx.qualification.candidate_source_digest(candidate, payloads["candidate.patch"])
                    with patch.dict(record, candidate_source=source_identity):
                        with self.assertRaisesRegex(ValueError, error):
                            validate()
            with self.subTest(case=f"{target}/mismatched-link-digest"):
                with patch.dict(record["targets"][target]["inputs"], {str(root / control): "0" * 64}):
                    with self.assertRaisesRegex(ValueError, error):
                        validate()

    test_replay_requires_tracked_linker_control_bytes_to_match.required_subtests = True

    def test_replay_runner_rejects_wrong_fourth_module_before_any_child(self):
        preset = "windows-clang-release" if os.name == "nt" else "linux-release"
        workload = "ordinary-negative-direct-current-1080p"
        entry = ofx.SCANNER_SCHEDULE_ENTRIES["direct", "materialized"]
        arms = {}
        for method, companion in (
            ("cubic", "JuicerOfxMeasurementModule"),
            ("spectral", "JuicerOfxSpectralReplayModule"),
        ):
            declarations, resources = self.make_replay_producer(preset, method, ("juicer", companion))
            shipping = declarations["juicer"]
            arms[method] = {**shipping, "companion": declarations[companion]}
            for declaration in (arms[method], arms[method]["companion"]):
                declaration.update(
                    points={},
                    current_grain={},
                    schedule_points={entry: {"symbol": entry}},
                    symbols=declaration["module"],
                    symbols_sha256=declaration["module_sha256"],
                    uuid="test-module",
                    layout={},
                )
        root = self.root.resolve() / "cubic"
        results = root / "out/validation" / preset / "results"
        results.mkdir(parents=True)
        driver = (
            root
            / "out/build"
            / preset
            / "tests/bin"
            / ("JuicerOfxMeasurement.exe" if os.name == "nt" else "JuicerOfxMeasurement")
        )
        driver.parent.mkdir()
        driver.write_bytes(b"render driver fixture")
        inputs = root / "inputs.json"
        sha = ofx.execution.digest(driver)
        ofx.execution.write_json(
            inputs,
            {
                "variants": {"ordinary": {**arms, "resources": resources}},
                "row_bindings": {
                    workload: {
                        method: {"module_sha256": arm["module_sha256"], "schedule": "materialized", "grid": 17}
                        for method, arm in arms.items()
                    }
                },
                "debugger": str(driver),
                "debugger_sha256": sha,
                "driver_sha256": sha,
            },
        )

        def replay(run_id):
            command = [
                "ofx.py",
                "scanner-paired",
                "--preset",
                preset,
                "--mode",
                "replay",
                "--inputs",
                str(inputs),
                "--run-id",
                run_id,
                "--workloads",
                workload,
            ]
            with patch.object(sys, "argv", command):
                ofx.main()

        fourth_module = Path(arms["spectral"]["companion"]["module"])
        original = fourth_module.read_bytes()
        fourth_module.write_bytes(b"unrelated companion contents")
        with (
            patch.object(ofx, "ROOT", root),
            patch.object(
                ofx.execution.NativeRun, "command", side_effect=AssertionError("render child intercepted")
            ) as child,
        ):
            with self.assertRaisesRegex(
                ValueError, "JuicerOfxSpectralReplayModule link output differs from supplied module"
            ):
                replay("wrong-fourth")
            child.assert_not_called()
            failed = json.loads((results / "wrong-fourth/result.json").read_bytes())
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["commands"], [])
            fourth_module.write_bytes(original)
            with self.assertRaisesRegex(AssertionError, "render child intercepted"):
                replay("valid-four")
            child.assert_called_once()

    def test_controlled_print_rejects_a_different_live_field_or_paper(self):
        observation = {
            side: [{"density": {"sha256": "paper"}, "glare": {"sha256": side}}] for side in ("reference", "candidate")
        }
        self.assertFalse(ofx.print_input_verdict(observation, "natural")["pre_white_glare_identical"])
        with self.assertRaisesRegex(ValueError, "controlled Print pre-white glare field differs"):
            ofx.print_input_verdict(observation, "constant")
        observation["candidate"][0]["glare"]["sha256"] = "reference"
        self.assertEqual(ofx.print_input_verdict(observation, "constant")["claim"], "matched-inputs")
        observation["candidate"][0]["density"]["sha256"] = "other-paper"
        with self.assertRaisesRegex(ValueError, "live Print paper C/M/Y differs"):
            ofx.print_input_verdict(observation, "constant")

    def test_frozen_photo_poll_must_consume_exact_last_event_and_flag(self):
        owner = {"resources": 11, "context": 12, "device": 0, "epoch": 13}
        held = {
            **owner,
            "event": 14,
            "host": 15,
            "epoch_at_production": 13,
            "producer": [2, 9],
            "success": True,
            "handoff": True,
        }
        draw = {"context": 12, "stream": 16, "submission": 2, "token": 9, "retained": held}
        poll = {
            **owner,
            "stream": 16,
            "before": [101, 102],
            "after": [101, 101],
            "entry": [15, 14],
            "queries": [{"event": 14, "status": 0, "completed_flag": 0}],
            "destroyed": [14],
            "success": True,
            "status": 0,
        }
        reporter = {
            "context": 12,
            "stream": 16,
            "submission": 3,
            "token": 10,
            "prepare": {"stream": 16, "outcome": 0},
            "poll": poll,
        }
        ofx.spectral_replay_poll(draw, {**reporter, "submission": 2, "token": 9}, frozen_photo=True)
        ofx.spectral_replay_poll(draw, reporter)
        held["producer"] = []
        with self.assertRaisesRegex(ValueError, "exact preceding owner event"):
            ofx.spectral_replay_poll(draw, reporter)
        ofx.spectral_replay_poll(draw, reporter, legacy_producer=True)
        for key, value in (("event", 17), ("completed_flag", 1)):
            with patch.dict(poll["queries"][0], {key: value}):
                with self.assertRaisesRegex(ValueError, "exact preceding owner event"):
                    ofx.spectral_replay_poll(draw, reporter, legacy_producer=True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_run(self, preset, *, qualified=False, workloads=None):
        selected = list(ofx.ROWS) if workloads is None else workloads
        directory = self.root / preset
        directory.mkdir()
        members = []
        commands = []
        inputs = {"driver": {"path": "driver", "sha256": "driver-hash"}, "overlay": {"sha256": "overlay-hash"}}
        for side in ofx.SIDES:
            module_hash = "ship-hash" if side == "candidate" and qualified else ofx.SHIPPING[preset][side]
            inputs[side] = {
                "shipping": {"path": side + "-shipping", "sha256": module_hash},
                "observed": {"path": side + "-observed", "sha256": "observer-hash"},
                "resources": {"path": side + "-resources", "files": {"x": "hash"}},
                "observed_resources": {"path": side + "-observed-resources", "files": {"x": "hash"}},
            }
        identities = {"driver": "driver-hash", "overlay": "overlay-hash"}
        for side in ofx.SIDES:
            identities[side] = {
                "shipping": inputs[side]["shipping"]["sha256"],
                "observed": "observer-hash",
                "resources": {"x": "hash"},
                "observed_resources": {"x": "hash"},
            }
        for name, value in (("candidate.patch", ""), ("candidate.json", "{}"), ("inputs.json", json.dumps(inputs))):
            (directory / name).write_text(value)
        original = (
            PureWindowsPath("C:/campaign/run") if preset.startswith("windows") else PurePosixPath("/campaign/run")
        )
        for lane, row, pair, side in (
            (lane, row, pair, side)
            for lane in ofx.LANES
            for row in selected
            for pair in range(3)
            for side in (ofx.SIDES if pair != 1 else ofx.SIDES[::-1])
        ):
            name = f"{lane}-{row}-{pair}-{side}.json"
            record = sample(lane, row)
            resource_key = "resources" if lane == "shipping" else "observed_resources"
            record["resources"] = inputs[side][resource_key]["path"]
            (directory / name).write_text(json.dumps(record))
            process = name.removesuffix(".json") + ".process.json"
            (directory / process).write_text(json.dumps({"status": "completed", "drained": True, "returncode": 0}))
            commands.append(
                {
                    "name": name.removesuffix(".json"),
                    "drained": True,
                    "returncode": 0,
                    "command": [
                        "driver",
                        "--mode",
                        "measure",
                        "--module",
                        inputs[side][lane]["path"],
                        "--resources",
                        inputs[side][resource_key]["path"],
                        "--lane",
                        lane,
                        "--workload",
                        row,
                        "--output",
                        str(original / name),
                    ],
                }
            )
            members.append(
                {"lane": lane, "row": row, "pair": pair, "side": side, "file": name, "drained": True, "returncode": 0}
            )
        evidence = {p.name: ofx.execution.digest(p) for p in directory.iterdir()}
        (directory / "result.json").write_text(
            json.dumps(
                {
                    "run_directory": str(original),
                    "source": {},
                    "commands": commands,
                    "evidence": evidence,
                    "status": "passed",
                    "operation": "measure",
                    "preset": preset,
                    "samples": members,
                    "source_unchanged": True,
                    "identities_unchanged": True,
                    "identities": identities,
                    "workloads": selected,
                }
            )
        )
        return directory

    def make_matched_highlight_run(self, preset):
        selected = list(ofx.MATCHED_HIGHLIGHT_WORKLOADS)
        directory = self.make_run(preset, workloads=selected)
        result = json.loads((directory / "result.json").read_text())
        inputs = json.loads((directory / "inputs.json").read_text())
        module_dir = "Win64" if preset.startswith("windows") else "Linux-x86-64"
        baseline_shipping = self.root / preset / "baseline-shipping.ofx.bundle"
        baseline_observed = self.root / preset / "baseline-observed.ofx.bundle"
        candidate_shipping = self.root / preset / "candidate-shipping.ofx.bundle"
        candidate_observed = self.root / preset / "candidate-observed.ofx.bundle"

        def side_inputs(shipping_bundle, observed_bundle, shipping_hash, observed_hash):
            return {
                "shipping": {
                    "path": str(shipping_bundle / "Contents" / module_dir / "juicer.ofx"),
                    "sha256": shipping_hash,
                },
                "observed": {
                    "path": str(observed_bundle / "Contents" / module_dir / "JuicerOfxMeasurementModule.ofx"),
                    "sha256": observed_hash,
                },
                "resources": {"path": str(shipping_bundle / "Contents" / "Resources"), "files": {"x": "hash"}},
                "observed_resources": {
                    "path": str(observed_bundle / "Contents" / "Resources"),
                    "files": {"x": "hash"},
                },
            }

        inputs["reference"] = side_inputs(baseline_shipping, baseline_observed, "baseline-ship", "baseline-observed")
        inputs["candidate"] = side_inputs(
            candidate_shipping, candidate_observed, "candidate-ship", "candidate-observed"
        )
        baseline_manifest = {
            preset: {
                "shipping": str(baseline_shipping),
                "observed": str(baseline_observed),
                "observed_module_sha256": "baseline-observed",
                "build_result": str(self.root / preset / "baseline-observed-build" / "result.json"),
                "members": {
                    f"Contents/{module_dir}/juicer.ofx": "baseline-ship",
                    f"Contents/{module_dir}/JuicerOfxMeasurementModule.ofx": "baseline-observed",
                    "Contents/Resources/x": "hash",
                },
            }
        }
        manifest_path = self.root / preset / "baseline-artifacts.json"
        manifest_bytes = json.dumps(baseline_manifest).encode()
        manifest_path.write_bytes(manifest_bytes)
        inputs["matched_highlight_baseline"] = {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        }
        for side in ofx.SIDES:
            module_values = inputs[side]
            result["identities"][side] = {
                "shipping": module_values["shipping"]["sha256"],
                "observed": module_values["observed"]["sha256"],
                "resources": module_values["resources"]["files"],
                "observed_resources": module_values["observed_resources"]["files"],
            }
        result["identities"]["matched_highlight_baseline"] = inputs["matched_highlight_baseline"]["sha256"]
        for member, command in zip(result["samples"], result["commands"]):
            lane = member["lane"]
            side = member["side"]
            resources_key = "resources" if lane == "shipping" else "observed_resources"
            module_path = inputs[side][lane]["path"]
            resources_path = inputs[side][resources_key]["path"]
            sample_path = directory / member["file"]
            sample_data = json.loads(sample_path.read_text())
            sample_data["resources"] = resources_path
            sample_path.write_text(json.dumps(sample_data))
            command["command"][4] = module_path
            command["command"][6] = resources_path
        (directory / "inputs.json").write_text(json.dumps(inputs))

        baseline_build_directory = manifest_path.parent / "baseline-observed-build"
        baseline_build_directory.mkdir()
        baseline_snapshot = json.dumps({"head": "baseline", "inputs": {}}).encode()
        baseline_patch = b"baseline observation source patch"

        def member(data):
            return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

        baseline_build_result = {
            "schema": "native-build-v1",
            "status": "passed",
            "preset": preset,
            "targets": ["JuicerOfxMeasurementModule"],
            "commands": [{"returncode": 0, "drained": True, "process_status": "completed"}],
            "members": {"candidate.json": member(baseline_snapshot), "candidate.patch": member(baseline_patch)},
        }
        baseline_build_bytes = json.dumps(baseline_build_result).encode()
        (directory / "matched-highlight-baseline-artifacts.json").write_bytes(manifest_bytes)
        (directory / "matched-highlight-observed-build-result.json").write_bytes(baseline_build_bytes)
        (directory / "matched-highlight-observed-candidate.json").write_bytes(baseline_snapshot)
        (directory / "matched-highlight-observed-candidate.patch").write_bytes(baseline_patch)

        consumed = self.root / "consumed"
        consumed.write_text("qualified module source")
        reference_production = production_evidence(preset, "baseline-ship", consumed)
        candidate_production = production_evidence(preset, "candidate-ship", consumed)
        for names, proof in ((ofx.REFERENCE_PROOF, reference_production),):
            for (name, _), data in zip(names, proof):
                (directory / name).write_bytes(data)
        candidate_names = (
            "candidate-production-result.json",
            "candidate-production-contract.json",
            "candidate-production-source.json",
            "candidate-production-source.patch",
        )
        for name, data in zip(candidate_names, candidate_production):
            (directory / name).write_bytes(data)
        result["comparison"] = "matched-highlight"
        result["reference_qualification"] = ofx.reference_qualification(
            reference_production, preset, "baseline-ship", check_inputs=False, comparison="matched-highlight"
        )
        result["candidate_qualification"] = ofx.candidate_qualification(
            *candidate_production, preset, "candidate-ship", check_inputs=False
        )
        result["matched_highlight_baseline_binding"] = ofx.validate_matched_highlight_baseline(
            inputs,
            preset,
            result["identities"],
            manifest_bytes=manifest_bytes,
            build_result_bytes=baseline_build_bytes,
            build_snapshot_bytes=baseline_snapshot,
            build_patch_bytes=baseline_patch,
        )["binding"]
        result["evidence"] = {
            path.name: ofx.execution.digest(path)
            for path in directory.iterdir()
            if path.is_file() and path.name != "result.json"
        }
        (directory / "result.json").write_text(json.dumps(result))
        return directory

    def test_identity_mismatch(self):
        p = self.root / "file"
        p.write_text("x")
        with self.assertRaises(ValueError):
            ofx.identities({"driver": {"path": str(p), "sha256": "wrong"}})
        production = production_evidence("linux-release", "candidate-hash", p)
        binding = ofx.candidate_qualification(*production, "linux-release", "candidate-hash", check_inputs=True)
        self.assertEqual(binding["module_sha256"], "candidate-hash")
        with self.assertRaises(ValueError):
            ofx.candidate_qualification(*production, "linux-release", "other-hash", check_inputs=True)
        p.write_text("changed")
        with self.assertRaises(ValueError):
            ofx.candidate_qualification(*production, "linux-release", "candidate-hash", check_inputs=True)

    def test_g3_shipping_source_checks_unstaged_render_changes(self):
        def git(*arguments):
            subprocess.run(["git", *arguments], cwd=self.root, check=True, capture_output=True)

        def capture():
            return ofx.verify.capture_native_source(self.root, "tests/benchmarks/ofx.py", domain_tests=True)[:2]

        for name in ("scripts", "src/Cuda", "tests/benchmarks"):
            (self.root / name).mkdir(parents=True)
        (self.root / "scripts/source_file_policy.json").write_text('{"supportedExtensions": [".cu"]}', encoding="utf-8")
        render = self.root / "src/Cuda/Render.cu"
        benchmark = self.root / "tests/benchmarks/ofx.py"
        render.write_text("original\n", encoding="utf-8")
        benchmark.write_text("original\n", encoding="utf-8")
        git("init", "-q")
        git("add", ".")
        git("-c", "user.name=Test", "-c", "user.email=test@example.test", "commit", "-qm", "Fixture")
        original, original_patch = capture()
        snapshot = json.dumps(original).encode()
        benchmark.write_text("test-only edit\n", encoding="utf-8")
        test_only, test_patch = capture()
        self.assertNotEqual(test_patch, original_patch)
        ofx.g3_production_source(snapshot, original_patch, test_only, test_patch, root=self.root)
        render.write_text("changed renderer\n", encoding="utf-8")
        source, patch = capture()
        self.assertEqual(source["inputs"]["src/Cuda/Render.cu"], original["inputs"]["src/Cuda/Render.cu"])
        with self.assertRaisesRegex(ValueError, "product source or membership differs"):
            ofx.g3_production_source(snapshot, original_patch, source, patch, root=self.root)
        render.write_text("original\n", encoding="utf-8")
        (render.parent / "New.cu").write_text("additional source\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "product source or membership differs"):
            ofx.g3_production_source(snapshot, original_patch, *capture(), root=self.root)

    def test_g3_shipping_binding_checks_relocated_sibling_inventory(self):
        module = self.root / "qualified/Contents/native/juicer.ofx"
        relocated = self.root / "relocated/Contents/native/juicer.ofx"
        driver = self.root / "driver"
        driver.write_bytes(b"driver")
        for path in (module, relocated):
            path.parent.mkdir(parents=True)
            path.write_bytes(b"module")
            resources = path.parent.parent / "Resources"
            resources.mkdir()
            (resources / "profile").write_bytes(b"profile")
        qualified = module.parent.parent / "Resources"
        resources = relocated.parent.parent / "Resources"
        resource = resources / "profile"
        preset = "windows-clang-release" if PureWindowsPath(str(module)).drive else "linux-release"
        source = {"head": "base", "inputs": {}, "new_files_base64": {}}
        module_hash = ofx.inventory(module)
        proof = production_evidence(
            preset, module_hash, qualified / "profile", module=module, snapshot=json.dumps(source).encode()
        )

        def bind():
            inputs = {
                "driver": {"path": str(driver), "sha256": ofx.inventory(driver)},
                "candidate": {
                    "shipping": {"path": str(relocated), "sha256": module_hash},
                    "resources": {"path": str(resources), "files": ofx.inventory(resources)},
                },
            }
            selected = ofx.g3_identities(inputs)["candidate"]
            return ofx.g3_production_binding(proof, preset, selected, source, b"", root=self.root)

        self.assertEqual(bind()["module_sha256"], module_hash)
        resource.write_bytes(b"altered")
        with self.assertRaisesRegex(ValueError, "Resources differ"):
            bind()
        resource.unlink()
        (resources / "other").write_bytes(b"profile")
        with self.assertRaisesRegex(ValueError, "Resources differ"):
            bind()
        resource.write_bytes(b"profile")
        with self.assertRaisesRegex(ValueError, "Resources differ"):
            bind()

    def test_missing_pair(self):
        p = self.make_run("linux-release")
        result = json.loads((p / "result.json").read_text())
        result["samples"].pop()
        (p / "result.json").write_text(json.dumps(result))
        with self.assertRaises(ValueError):
            ofx.report_run(p)

    def test_missing_workload(self):
        p = self.make_run("linux-release")
        result = json.loads((p / "result.json").read_text())
        result["samples"] = [m for m in result["samples"] if m["row"] != ofx.ROWS[0]]
        (p / "result.json").write_text(json.dumps(result))
        with self.assertRaises(ValueError):
            ofx.report_run(p)

    def test_failed_or_undrained_child(self):
        p = self.make_run("linux-release")
        result = json.loads((p / "result.json").read_text())
        result["samples"][0]["drained"] = False
        (p / "result.json").write_text(json.dumps(result))
        with self.assertRaises(ValueError):
            ofx.report_run(p)

    def test_matched_highlight_report_requires_both_closures_and_all_controls(self):
        linux = self.make_matched_highlight_run("linux-release")
        windows = self.make_matched_highlight_run("windows-clang-release")
        report = ofx.offline_report([linux, windows])
        self.assertEqual(report["comparison"], "matched-highlight")
        self.assertEqual(sum(key.startswith("shipping/") for key in report["comparisons"]["linux-release"]["rows"]), 10)

        result = json.loads((linux / "result.json").read_text())
        result["reference_qualification"] = None
        (linux / "result.json").write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "reference production closure"):
            ofx.report_run(linux)

    def test_matched_highlight_requires_candidate_closure_even_for_fixed_candidate_hash(self):
        run = self.make_matched_highlight_run("linux-release")
        result = json.loads((run / "result.json").read_text())
        inputs_path = run / "inputs.json"
        inputs = json.loads(inputs_path.read_text())
        fixed_candidate_hash = ofx.SHIPPING["linux-release"]["candidate"]
        inputs["candidate"]["shipping"]["sha256"] = fixed_candidate_hash
        inputs_path.write_text(json.dumps(inputs))
        result["identities"]["candidate"]["shipping"] = fixed_candidate_hash
        result["candidate_qualification"] = None
        result["evidence"]["inputs.json"] = ofx.execution.digest(inputs_path)
        (run / "result.json").write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "candidate production qualification"):
            ofx.report_run(run)

    def test_matched_highlight_rejects_missing_zero_control_and_wrong_exposure(self):
        run = self.make_matched_highlight_run("linux-release")
        sample_path = run / "shipping-halation-1080p-direct-boost-zero-0-reference.json"
        original_sample = json.loads(sample_path.read_text())
        original_bytes = sample_path.read_bytes()
        cases = (("halation_boost_ev", 4.0), ("controls", {"Exposure": [5.0]}))
        self.subtest_cases = tuple(field for field, _ in cases)
        for field, value in cases:
            sample_data = dict(original_sample)
            sample_data[field] = value
            sample_path.write_text(json.dumps(sample_data))
            result = json.loads((run / "result.json").read_text())
            result["evidence"][sample_path.name] = ofx.execution.digest(sample_path)
            (run / "result.json").write_text(json.dumps(result))
            with self.subTest(case=field), self.assertRaisesRegex(ValueError, "exposure 6"):
                ofx.report_run(run)
            sample_path.write_bytes(original_bytes)
            result["evidence"][sample_path.name] = ofx.execution.digest(sample_path)
            (run / "result.json").write_text(json.dumps(result))

        result = json.loads((run / "result.json").read_text())
        result["workloads"].remove("halation-uhd-direct-boost-zero")
        (run / "result.json").write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "all eight boosted and two zero-boost"):
            ofx.report_run(run)

    def test_incomplete_memory(self):
        out_of_order = sample("observed", ofx.ROWS[0])
        out_of_order["renders"][0], out_of_order["renders"][1] = out_of_order["renders"][1], out_of_order["renders"][0]
        with self.assertRaises(ValueError):
            ofx.validate_sample(out_of_order, "measure", "observed", ofx.ROWS[0])
        s = sample("observed", ofx.ROWS[0])
        s["memory"][0]["rss_bytes"] = None
        with self.assertRaises(ValueError):
            ofx.validate_sample(s, "measure", "observed", ofx.ROWS[0])

    def test_complete_offline_report(self):
        linux = self.make_run("linux-release", qualified=True)
        with self.assertRaises(ValueError):
            ofx.report_run(linux)
        unqualified = json.loads((linux / "result.json").read_text())
        unqualified["candidate_qualification"] = None
        (linux / "result.json").write_text(json.dumps(unqualified))
        with self.assertRaises(ValueError):
            ofx.report_run(linux)
        consumed = self.root / "consumed"
        consumed.write_text("source")
        production = production_evidence("linux-release", "ship-hash", consumed)
        result = json.loads((linux / "result.json").read_text())
        names = (
            "candidate-production-result.json",
            "candidate-production-contract.json",
            "candidate-production-source.json",
            "candidate-production-source.patch",
        )
        for name, data in zip(names, production):
            (linux / name).write_bytes(data)
            result["evidence"][name] = hashlib.sha256(data).hexdigest()
        result["candidate_qualification"] = ofx.candidate_qualification(
            *production, "linux-release", "ship-hash", check_inputs=True
        )
        (linux / "result.json").write_text(json.dumps(result))
        windows = self.make_run("windows-clang-release")
        report = ofx.offline_report([linux, windows])
        contract_file = linux / "candidate-production-contract.json"
        contract_file.write_bytes(contract_file.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            ofx.offline_report([linux, windows])
        self.assertEqual(len(report["comparisons"]), 2)
        self.assertEqual(len(report["comparisons"]["linux-release"]["rows"]), 8)
        for path in (linux, windows):
            result = json.loads((path / "result.json").read_text())
            result["workloads"] = ["negative-print-grain"]
            result["commands"] = [
                c
                for c in result["commands"]
                if c["name"].startswith(("shipping-negative-print-grain-", "observed-negative-print-grain-"))
            ]
            result["samples"] = [m for m in result["samples"] if m["row"] == "negative-print-grain"]
            (path / "result.json").write_text(json.dumps(result))
        contract_file.write_bytes(production[1])
        focused = ofx.offline_report([linux, windows])
        self.assertEqual(len(focused["comparisons"]["linux-release"]["rows"]), 2)
        result["workloads"] = ["negative-print-grain", "negative-print-grain"]
        (windows / "result.json").write_text(json.dumps(result))
        with self.assertRaises(ValueError):
            ofx.offline_report([linux, windows])


MeasurementReportTests.test_matched_highlight_rejects_missing_zero_control_and_wrong_exposure.required_subtests = True


if __name__ == "__main__":
    unittest.main()
