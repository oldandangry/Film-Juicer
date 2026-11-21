"""Pure record validation for Python outcomes and current CTest membership."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path, PureWindowsPath
import re
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
            "schema": "native-quality-v1",
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

    def test_current_replay_compiler_receipt_requires_exact_commands_and_exclusions(self):
        cases = (
            ("excluded", False, ""),
            ("mixed", False, ""),
            ("mixed-without-ordinary-command", False, "missing ordinary compiler analysis"),
            ("mixed-without-ordinary-format-and-command", False, "missing formatting for a selected native file"),
            (
                "mixed-with-substituted-policy",
                False,
                "retained source differs from candidate: scripts/source_file_policy.json",
            ),
            ("mixed-with-working-copy-policy", False, ""),
            ("mixed-with-patched-policy", False, ""),
            ("mixed-with-inline-command", False, ""),
            ("analyzed", True, ""),
            ("missing-exclusion", False, "missing or invented replay header compiler analysis"),
            ("invented-exclusion", True, "missing or unexpected replay compiler command"),
            ("wrong-exclusion", False, "incorrect compiler exclusion"),
            ("missing-evidence", True, "missing or invented replay header compiler analysis"),
            ("downgraded-evidence", True, "requires compiler analysis evidence"),
            ("missing-command", True, "missing or unexpected replay compiler command"),
            ("wrong-compiler-variant", True, "incorrect motion replay compile variant"),
            ("wrong-replay-define", True, "incorrect motion replay compile variant"),
            ("wrong-database-path", True, "did not use the retained database"),
            ("wrong-sibling-database", True, "did not use the retained database"),
            ("missing-run-identity", True, "missing replay compiler result directory identity"),
            ("current-excluded", False, ""),
            ("current-analyzed", True, ""),
            ("current-extra-consumer", True, ""),
            ("current-extra-missing-command", True, "missing or unexpected replay compiler command"),
            ("current-extra-wrong-variant", True, "incorrect motion replay compile variant"),
            ("current-changed-database", True, "differs from its recorded database execution"),
            ("current-missing-execution", True, "missing or unexpected replay compiler execution evidence"),
            ("current-wrong-sibling-database", True, "did not use the retained database"),
            ("current-coordinated-sibling", True, "differs from its recorded database execution"),
            (
                "current-false-replay-off",
                False,
                "configured replay availability disagrees with CMake artifact consumers",
            ),
            ("current-foreign-artifacts", True, "product artifact description differs from the configured build"),
            ("current-missing-artifacts", False, "missing CMake artifact description for replay quality"),
            ("current-downgraded-evidence", True, "requires compiler analysis evidence"),
            ("linked-analyzed", True, ""),
            ("linked-excluded", False, ""),
            ("linked-stage-only", False, ""),
            ("linked-testing-off", False, ""),
            ("linked-false-replay-off", False, "configured replay availability disagrees with the native preset"),
            ("linked-missing-stage-proof", True, "missing Film grain stage compilation database"),
            ("linked-missing-stage-variants", True, "missing configured Film grain stage consumers"),
            ("linked-wrong-stage-hook", True, "incorrect configured Film grain stage hook define"),
            ("linked-wrong-stage-probe", True, "Film grain stage probes differ"),
            ("linked-missing-stage-command", True, "missing ordinary compiler analysis"),
            ("linked-wrong-stage-database", True, "missing ordinary compiler analysis"),
            ("linked-wrong-stage-cwd", True, "missing ordinary compiler analysis"),
            ("linked-inline-stage-command", True, "missing ordinary compiler analysis"),
            ("linked-mixed-with-inline-command", True, "missing ordinary compiler analysis"),
            ("linked-mixed-with-cuda-command", True, ""),
            ("linked-mixed-with-cuda-wrong-cwd", True, "missing ordinary compiler analysis"),
            ("linked-excluded-shared-stage", False, "missing or invented replay header compiler analysis"),
            ("linked-missing-density-exclusion", False, "missing or invented replay header compiler analysis"),
            ("linked-missing-density", True, "invalid motion replay compilation database"),
            ("linked-wrong-density-variant", True, "incorrect motion replay compile variant"),
            ("linked-obsolete-executor", True, "incorrect motion replay compile variant"),
            ("linked-wrong-density-coverage", True, "invalid replay header compiler execution coverage"),
            ("linked-missing-shared-coverage", True, "missing required replay header compiler execution coverage"),
            (
                "linked-missing-render-shared-coverage",
                True,
                "missing required replay header compiler execution coverage",
            ),
            ("linked-downgraded-contract", True, "differs from the recorded compiler analysis contract"),
        )
        self.subtest_cases = tuple(case for case, *_ in cases)
        header = "tests/grain/motion-replay.h"
        render = "tests/ffi/production-ofx-render.cpp"
        executor = "src/Cuda/JuicerCudaExecutor.cpp"
        density = "tests/grain/motion-density.cpp"
        stage = "src/Cuda/JuicerCudaFilmGrainStage.h"
        stage_source = "src/Cuda/JuicerCudaFilmGrainStage.cpp"
        extra_source = "tests/grain/film_oracle.cpp"
        root = "/configured"
        preset = "linux-debug"
        build = root + "/out/build/" + preset
        database_path = root + "/out/validation/" + preset + "/results/fixture"
        source_policy = Path(verify.__file__).with_name("source_file_policy.json").read_bytes()
        policy_source = "scripts/source_file_policy.json"

        def git_blob(data):
            header = b"blob " + str(len(data)).encode() + b"\0"
            return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()

        policy_blob = git_blob(source_policy)
        configuration = {
            "CMAKE_HOME_DIRECTORY": root,
            "CMAKE_BUILD_TYPE": "Debug",
            "BUILD_TESTING": "ON",
            "JUICER_GRAIN_PREVIEW": "TRUE",
            "JUICER_GRAIN_PREVIEW_CONFIGURED": "ON",
        }
        for case_id, replay, error in cases:
            with self.subTest(case=case_id):
                linked = case_id.startswith("linked-")
                current = linked or case_id.startswith("current-")
                case = case_id.removeprefix("current-").removeprefix("linked-")
                required_sources = (render, density if linked else executor)
                sources = (
                    sorted((*required_sources, extra_source) if case.startswith("extra-") else required_sources)
                    if current
                    else required_sources
                )
                if linked and case == "missing-density":
                    sources.remove(density)
                if linked and case == "obsolete-executor":
                    sources = sorted((*sources, executor))
                for embedded in (False,) if linked and case == "testing-off" else (False, True):
                    prefix = "quality-" if embedded else ""
                    cuda = case in {"mixed-with-cuda-command", "mixed-with-cuda-wrong-cwd"}
                    ordinary = "src/Cuda/Scan/JuicerCudaScanPipeline.cu" if cuda else "native/ofx_exports.cpp"
                    selected = [ordinary, header] if case.startswith("mixed") else [header]
                    if linked:
                        selected = [stage] if case == "stage-only" else sorted((header, density, stage))
                        if case.startswith("mixed"):
                            selected = sorted((*selected, ordinary))
                    formatted = (
                        [header]
                        if case in {"mixed-without-ordinary-format-and-command", "mixed-with-substituted-policy"}
                        else selected
                    )
                    checks = self.quality_checks(preset, selected, "development", None, [])
                    checks["schema"] = "native-quality-checks-v4" if current else "native-quality-checks-v3"
                    if linked and case != "downgraded-contract":
                        checks["schema"] = "native-quality-checks-v5"
                    checks["expected_checks"].append("native-quality")
                    checks["completed_checks"].append("native-quality")
                    commands = [
                        self.operation(
                            prefix + "clang-format", ["fmt", "--dry-run", "--Werror", "--style=file", *formatted]
                        ),
                        self.operation(prefix + "clang-tidy-config", ["tidy", "--verify-config"]),
                    ]
                    if case in {"mixed", "mixed-with-working-copy-policy", "mixed-with-patched-policy"}:
                        commands.append(
                            self.operation(
                                prefix + "clang-tidy-1-ofx-exports-cpp",
                                ["tidy", "-p", build, "--warnings-as-errors=*", ordinary],
                            )
                        )
                    if cuda or case == "mixed-with-inline-command":
                        commands.append(
                            {
                                **self.operation(
                                    prefix + "clang-tidy-cuda-1-inline",
                                    [
                                        "tidy",
                                        "--warnings-as-errors=*",
                                        f"--config-file={root}/.clang-tidy",
                                        root + "/" + ordinary,
                                        "--",
                                        "-std=c++20",
                                    ],
                                ),
                                "directory": build
                                if case in {"mixed-with-cuda-command", "mixed-with-inline-command"}
                                else root,
                            }
                        )
                    stage_database = []
                    if linked:
                        for index, source in enumerate((executor, stage_source), 1):
                            if source == stage_source and case == "missing-stage-command":
                                continue
                            database_dir = (
                                database_path if source == stage_source and case == "wrong-stage-database" else build
                            )
                            command = ["tidy", "-p", database_dir, "--warnings-as-errors=*", source]
                            if source == stage_source and case == "inline-stage-command":
                                command = [
                                    "tidy",
                                    "--warnings-as-errors=*",
                                    f"--config-file={root}/.clang-tidy",
                                    source,
                                    "--",
                                ]
                            commands.append(
                                {
                                    **self.operation(prefix + f"clang-tidy-{index}-stage", command),
                                    "directory": build
                                    if source == stage_source and case == "wrong-stage-cwd"
                                    else root,
                                }
                            )
                        owners = {
                            f"CMakeFiles/juicer_native.dir/{executor}.o": (executor, []),
                            f"CMakeFiles/juicer_native.dir/{stage_source}.o": (stage_source, []),
                        }
                        if case not in {"testing-off", "missing-stage-variants"}:
                            flag = 0 if case == "wrong-stage-hook" else 1
                            owners.update(
                                {
                                    f"tests/CMakeFiles/juicer_render_test_objects.dir/__/{executor}.o": (
                                        executor,
                                        [f"-DJUICER_CUDA_RENDER_TEST_HOOK={flag}"],
                                    ),
                                    f"tests/CMakeFiles/juicer_executor_failure_test_objects.dir/__/{executor}.o": (
                                        executor,
                                        ["-DJUICER_EXECUTOR_FAILURE_TEST_HOOK=1"],
                                    ),
                                }
                            )
                        for index, (output, (source, defines)) in enumerate(sorted(owners.items()), 1):
                            file = root + "/" + source
                            stage_database.append(
                                {
                                    "file": file,
                                    "directory": build,
                                    "output": build + "/" + output,
                                    "arguments": ["c++", *defines, "-o", output, "-c", file],
                                }
                            )
                            probe = ["c++", "-M", "-MF", "-", "-MT", "juicer-dependencies", *defines, file]
                            if case == "wrong-stage-probe" and index == len(owners):
                                probe.insert(-1, "-DOTHER=1")
                            commands.append(
                                {
                                    **self.operation(prefix + f"film-grain-stage-{index}-dependencies", probe),
                                    "directory": build,
                                }
                            )
                    analysis = {
                        source: {"excluded": qualification.MOTION_REPLAY_EXCLUSION}
                        for source in selected
                        if source in {header, density}
                    }
                    if linked and case == "excluded-shared-stage":
                        analysis[stage] = {"excluded": qualification.MOTION_REPLAY_EXCLUSION}
                    if linked and case == "missing-density-exclusion":
                        del analysis[density]
                    database = []
                    coverage = {}
                    if replay:
                        labels = [
                            prefix + qualification.motion_replay_tidy_label(source, legacy=not current)
                            for source in sources
                        ]
                        analysis[header] = {"commands": labels}
                        for label, source in zip(labels, sources):
                            file = root + "/" + source
                            target = "JuicerGrainMotionReplay"
                            if (
                                (case == "wrong-compiler-variant" and source == sources[0])
                                or (case == "extra-wrong-variant" and source == extra_source)
                                or (case == "wrong-density-variant" and source == density)
                            ):
                                target = "JuicerProductionOfxProbe"
                            output = f"CMakeFiles/{target}.dir/{source}.o"
                            flags = (
                                []
                                if case == "wrong-replay-define" and source == sources[0]
                                else ["-DJUICER_GRAIN_MOTION_REPLAY=1"]
                            )
                            database.append(
                                {
                                    "file": file,
                                    "directory": build,
                                    "output": build + "/" + output,
                                    "arguments": ["c++", *flags, "-o", output, "-c", file],
                                }
                            )
                            if (case == "missing-command" and source == sources[0]) or (
                                case == "extra-missing-command" and source == extra_source
                            ):
                                continue
                            used_database = database_path
                            if case == "wrong-database-path":
                                used_database = build
                            if case in {"wrong-sibling-database", "coordinated-sibling"}:
                                used_database = database_path.removesuffix("fixture") + "wrong-sibling"
                            command = [
                                "tidy",
                                "-p",
                                used_database,
                                "--warnings-as-errors=*",
                                file,
                            ]
                            commands.append({**self.operation(label, command), "directory": root})
                            coverage[source] = [header]
                            if linked and source == density:
                                coverage[source].append(density)
                                if case != "missing-shared-coverage":
                                    coverage[source].append(stage)
                            if linked and source == render and case != "missing-render-shared-coverage":
                                coverage[source].append(stage)
                            if linked and case == "wrong-density-coverage" and source == render:
                                coverage[source].append(density)
                        if linked:
                            analysis = {
                                file: {
                                    "commands": [
                                        label for label, source in zip(labels, sources) if file in coverage[source]
                                    ]
                                }
                                for file in selected
                                if file != ordinary
                            }
                    if case == "invented-exclusion":
                        analysis[header] = {"excluded": qualification.MOTION_REPLAY_EXCLUSION}
                    if case == "wrong-exclusion":
                        analysis[header] = {"excluded": "unknown"}
                    if case == "missing-exclusion":
                        del analysis[header]
                    if case == "downgraded-evidence":
                        checks["schema"] = "native-quality-checks-v3" if current else "native-quality-checks-v2"
                    if case != "missing-evidence":
                        checks["replay_compiler_analysis"] = analysis
                    checks["commands"] = commands
                    checks["expected_commands"] = [command["name"] for command in commands]
                    if embedded:
                        result, payloads = self.embedded_quality(checks, "TRUE", [])
                        result["schema"] = "native-tests-v6" if current else "native-tests-v5"
                    else:
                        result, payloads = self.standalone_quality(
                            checks, schema="native-quality-v3" if current else "native-quality-v2", scope="quality"
                        )
                    if case != "missing-run-identity":
                        result["run_id"] = "wrong-sibling" if case == "coordinated-sibling" else "fixture"
                    candidate = json.loads(payloads["candidate.json"])
                    candidate["configuration"] = {
                        **configuration,
                        "JUICER_GRAIN_MOTION_REPLAY_CONFIGURED": "ON" if replay else "OFF",
                    }
                    if linked and not replay and case not in {"false-replay-off", "testing-off"}:
                        candidate["configuration"]["JUICER_GRAIN_PREVIEW"] = "OFF"
                        candidate["configuration"]["JUICER_GRAIN_PREVIEW_CONFIGURED"] = "OFF"
                    if linked and case == "testing-off":
                        candidate["configuration"]["BUILD_TESTING"] = "OFF"
                    if stage_database and case != "missing-stage-proof":
                        payloads["film-grain-stage-compile.json"] = json.dumps(stage_database).encode()
                    if current and case != "missing-artifacts":
                        artifacts = {
                            "schema": "rust-artifacts-v2",
                            "build": build if case != "foreign-artifacts" else "/foreign" + build,
                            "consumers": {
                                "JuicerGrainMotionReplay": {
                                    "binary": build + "/grain-motion/bin/JuicerGrainMotionReplay"
                                }
                            }
                            if replay or case == "false-replay-off"
                            else {},
                            "variants": {
                                "production": {
                                    "archive": "/production/lib.a",
                                    "target_dir": "/production",
                                    "features": ["grain-preview"]
                                    if candidate["configuration"]["JUICER_GRAIN_PREVIEW_CONFIGURED"] == "ON"
                                    else [],
                                },
                                "test-support": {
                                    "archive": "/test/lib.a",
                                    "target_dir": "/test",
                                    "features": ["test-support"],
                                },
                            },
                        }
                        payloads["rust-artifacts.json"] = json.dumps(artifacts).encode()
                        if embedded:
                            result["rust_artifacts"] = True
                            candidate["features"] = {"rust_artifacts": "rust-artifacts.json"}
                    candidate["inputs"][policy_source] = {"git_blob": policy_blob, "mode": "100644"}
                    policy = source_policy
                    if case == "mixed-with-substituted-policy":
                        substituted = json.loads(policy)
                        substituted["supportedExtensions"].remove(".cpp")
                        policy = json.dumps(substituted).encode()
                    if case == "mixed-with-working-copy-policy":
                        policy = source_policy + b"\n"
                        candidate["inputs"][policy_source]["worktree_sha256"] = hashlib.sha256(policy).hexdigest()
                    if case == "mixed-with-patched-policy":
                        updated = source_policy + b"\n"
                        payloads["candidate.patch"] = (
                            f"diff --git a/{policy_source} b/{policy_source}\n"
                            f"index {policy_blob}..{git_blob(updated)} 100644\n"
                        ).encode()
                        policy = updated.replace(b"\n", b"\r\n")
                    payloads["candidate.json"] = json.dumps(candidate).encode()
                    payloads["source_file_policy.json"] = policy
                    if database:
                        database_bytes = json.dumps(database).encode()
                        if current:
                            database_identity = {
                                "path": database_path + "/compile_commands.json",
                                "sha256": hashlib.sha256(database_bytes).hexdigest(),
                            }
                            for label, source in zip(labels, sources):
                                if case == "missing-execution" and source == sources[0]:
                                    continue
                                payloads[label + qualification.REPLAY_EXECUTION_SUFFIX] = json.dumps(
                                    {
                                        "command": [
                                            "tidy",
                                            "-p",
                                            database_path,
                                            "--warnings-as-errors=*",
                                            root + "/" + source,
                                        ],
                                        "directory": root,
                                        "headers": coverage.get(source, [header]),
                                        "database": database_identity,
                                    }
                                ).encode()
                        if case == "changed-database":
                            database[0]["arguments"].append("-DNON_REPLAY_INPUT=1")
                            database_bytes = json.dumps(database).encode()
                        payloads["compile_commands.json"] = database_bytes
                    result["members"] = self.result_members(payloads)
                    with tempfile.TemporaryDirectory() as directory:
                        bundle = Path(directory)
                        for name, data in payloads.items():
                            (bundle / name).write_bytes(data)
                        (bundle / "result.json").write_text(json.dumps(result), encoding="utf-8")
                        if error:
                            with self.assertRaisesRegex(ValueError, error):
                                qualification.read_native_result(bundle)
                        else:
                            summary = qualification.read_native_result(bundle)
                            self.assertEqual(summary["replay_compiler_analysis"], analysis)

    test_current_replay_compiler_receipt_requires_exact_commands_and_exclusions.required_subtests = True

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
            "schema": "native-tests-v1",
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


class RustReceiptBundleTests(unittest.TestCase):
    publication = "state::tests::film_proxy_publication_uses_full_grid_and_encloses_its_densities"
    target = "x86_64-unknown-linux-gnu"

    def test_offline_windows_rustc_probe_requires_verbose_native_host(self):
        source = PureWindowsPath("C:/configured")
        programs = {
            "cmake": ("cmake.exe", "--version", "cmake version 4.3"),
            "ctest": ("ctest.exe", "--version", "ctest version 4.3"),
            "cargo": ("cargo.exe", "--version", "cargo 1.98.1 fixture"),
            "nextest": ("cargo-nextest.exe", "--version", "cargo-nextest 0.9.143 fixture"),
        }
        cases = (
            ("invalid-argument", "--not-a-rustc-option", "x86_64-pc-windows-msvc", "rustc probe must use -Vv"),
            ("foreign-host", "-Vv", self.target, "rustc host differs from configured native target"),
        )
        self.subtest_cases = tuple(name for name, _, _, _ in cases)
        for name, argument, host, message in cases:
            with self.subTest(case=name):
                programs["rustc"] = ("rustc.exe", argument, f"rustc 1.98.1 fixture\nhost: {host}")
                records = [
                    {
                        "name": tool,
                        "command": [str(source / "bin" / executable), flag],
                        "returncode": 0,
                        "output": output,
                    }
                    for tool, (executable, flag, output) in programs.items()
                ]
                candidate = {
                    "preset": "windows-clang-release",
                    "version_commands": records,
                    "versions": {record["name"]: record["output"] for record in records},
                }
                probe = {
                    "directory": str(source),
                    "command": [
                        str(source / "bin" / "python.exe"),
                        "-B",
                        str(source / "scripts" / "verify.py"),
                        "tool-versions",
                        *(arg for row in records for arg in ("--tool", row["name"], row["command"][0])),
                    ],
                }
                with self.assertRaisesRegex(ValueError, message):
                    qualification.rust_test_tool_paths(candidate, probe, source, PureWindowsPath)

    test_offline_windows_rustc_probe_requires_verbose_native_host.required_subtests = True

    def bundle(
        self,
        root,
        *,
        preview=True,
        build_kind="product",
        selected=("Rust.Core", "Rust.Bridge"),
        historical=False,
        legacy=False,
        with_junit_digest=False,
        with_junit_id=False,
        defect="",
    ):
        with_junit_digest = with_junit_digest or with_junit_id
        features = ["grain-preview", "test-support"] if preview and build_kind == "product" else ["test-support"]
        downgraded = defect.startswith("schema-downgrade")
        stripped = defect in {"schema-downgrade-all", "schema-downgrade-no-artifact"}
        claimed = ["test-support"] if downgraded or defect in {"features", "combined"} else features
        request = {
            "build_kind": build_kind,
            "grain_preview": preview and build_kind == "product",
            "target": self.target,
            "profile": "debug",
            "cargo_features": "film-juicer-plugin/test-support"
            + (",film-juicer-plugin/grain-preview" if preview and build_kind == "product" else ""),
        }
        names = ("film-juicer-core", "film-juicer-plugin")
        binary_names = ("film_juicer_core", "film_juicer_plugin")
        source = "/configured"
        rust_source = "/foreign" if defect == "foreign-checkout" else source
        build = source + "/out/build/linux-debug"
        if build_kind == "cuda-free":
            build += "/host-tests"
        run_dir = f"{source}/out/validation/linux-debug/results/fixture"
        run_ids = {
            "Rust.Core": "27bd2e7d-132c-4df9-9359-9cb6eb35b7c1",
            "Rust.Bridge": "31a5cd0b-a04e-47de-87d8-1662de8f3b2f",
        }
        target_dir = rust_source + "/out/build/linux-debug/cargo/test-support"
        paths = [f"{target_dir}/metadata/binaries.json", f"{target_dir}/metadata/cargo.json"]
        ids = {name: f"path+file://{rust_source}/rust/{name}#0.1.0" for name in names}
        cargo = {
            "workspace_root": "/foreign" if defect == "foreign-metadata" else rust_source,
            "target_directory": "/foreign/cargo/test-support" if defect == "foreign-metadata" else target_dir,
            "workspace_members": list(ids.values()),
            "packages": [
                {
                    "id": ids[name],
                    "name": name,
                    "manifest_path": f"{rust_source}/rust/{name}/Cargo.toml",
                    "targets": [{"name": binary, "test": True, "kind": ["lib"]}],
                }
                for name, binary in zip(names, binary_names)
            ],
            "resolve": {"nodes": [{"id": ids[name], "features": claimed} for name in names]},
        }
        platform = {"triple": self.target}
        reported_target = {"triple": "x86_64-pc-windows-msvc"} if defect == "prepared-platforms" else platform
        prepared_names = names[1:] if defect == "bridge-incomplete" else names
        binaries = {
            "rust-build-meta": {
                "target-directory": target_dir,
                "base-output-directories": [self.target + "/debug"],
                "target-platform": self.target,
                "target-platforms": [reported_target],
                "platforms": {"host": {"platform": platform}, "targets": [{"platform": reported_target}]},
            },
            "rust-binaries": {
                name: {
                    "binary-id": name,
                    "binary-name": binary,
                    "package-id": ids[name],
                    "kind": "lib",
                    "binary-path": f"{target_dir}/{self.target}/debug/deps/{binary}-123",
                    "build-platform": "target",
                }
                for name, binary in zip(names, binary_names)
                if name in prepared_names
            },
        }
        if defect == "aliased-binaries":
            binaries["rust-binaries"][names[1]]["binary-path"] = binaries["rust-binaries"][names[0]]["binary-path"]
        tests = []
        documents = {}
        for registration, package, short in zip(("Rust.Core", "Rust.Bridge"), names, ("core", "bridge")):
            command = [
                "/foreign/missing.exe" if defect == "foreign-python" and short == "core" else "/python",
                f"{rust_source}/scripts/verify.py",
                "rust-tests",
                "--preset",
                "linux-debug",
                "--cargo",
                "/cargo",
                "--nextest",
                "/nextest",
                "--binaries-metadata",
                paths[0],
                "--cargo-metadata",
                paths[1],
                "--config-file",
                f"{rust_source}/.config/nextest.toml",
                "--package",
                short,
            ]
            if not historical and not stripped and defect != "schema-downgrade-execution":
                command += ["--request", json.dumps(request)]
            tests.append(
                {
                    "name": registration,
                    "command": command,
                    "properties": [{"name": "LABELS", "value": ["host", "product"]}],
                }
            )
            if registration not in selected:
                continue
            suites = {}
            xml = []
            suite_tests = 0
            case_start = "2026-11-19T18:10:00+01:00"
            if short == "bridge" and "Rust.Core" in selected:
                case_start = "2026-11-19T18:10:42+01:00"
                if defect in {"early-bridge-report", "early-bridge-digest"}:
                    case_start = "2026-11-19T18:10:12+01:00"
            if defect == "stale-case-timestamps" and short == "core":
                case_start = "2026-11-19T17:10:00+01:00"
            for owner in prepared_names:
                # Reproduce the 340 + 41 function shape, including the explicit publication witness.
                count = 340 if owner == names[0] else 41
                cases = [f"test_{i:03}" for i in range(count)]
                if owner == names[0]:
                    cases[0] = qualification.OPTICS_FUNCTION
                    cases[1] = self.publication
                    if (
                        not request["grain_preview"]
                        or downgraded
                        or defect in {"missing", "combined", "discovery-only"}
                    ):
                        cases.remove(self.publication)
                matched = owner == package
                suite = {
                    **binaries["rust-binaries"][owner],
                    "package-name": owner,
                    "status": "listed",
                    "testcases": {
                        case: {"ignored": False, "filter-match": {"status": "matches" if matched else "mismatch"}}
                        for case in cases
                    },
                }
                if defect == "filtered" and owner == names[0] and matched:
                    suite["testcases"][self.publication]["filter-match"]["status"] = "mismatch"
                suites[owner] = suite
                xml_cases = cases + ([self.publication] if owner == names[0] and defect == "discovery-only" else [])
                xml.append(f'<testsuite name="{owner}" tests="{len(xml_cases)}">')
                suite_tests += len(xml_cases)
                for case in xml_cases:
                    terminal = "" if matched else "<skipped/>"
                    if case == self.publication and matched and defect in {"skipped", "failed"}:
                        terminal = "<skipped/>" if defect == "skipped" else "<failure/>"
                    timestamp = f' timestamp="{case_start}"' if matched else ""
                    xml.append(f'<testcase name="{case}"{timestamp}>{terminal}</testcase>')
                xml.append("</testsuite>")
            common = ["/nextest", "nextest"]
            config_dir = "/foreign/old-run" if defect == "foreign-nextest-config" and short == "core" else run_dir
            junit_dir = "/foreign/old-run" if defect == "foreign-nextest-junit" and short == "core" else run_dir
            metadata = [
                "--config-file",
                f"{config_dir}/{registration}.nextest.toml",
                "--user-config-file",
                "none",
                "--profile",
                "default",
                "--binaries-metadata",
                paths[0],
                "--cargo-metadata",
                paths[1],
                "-E",
                f"package(={package})",
            ]
            execution = {
                "schema": "rust-test-execution-v2" if historical else "rust-test-execution-v3",
                "registration": registration,
                "package": package,
                "cargo": "/cargo",
                "features": claimed,
                "required_binaries": [[ids[package], binary_names[names.index(package)]]],
                "discovered_binaries": list(prepared_names),
                "doctests": {"status": "none", "source_files_scanned": 1, "code_fence_sources": []},
                "commands": [
                    {"command": [*common, operation, *metadata, *suffix], "returncode": 0}
                    for operation, suffix in (
                        ("list", ["--message-format", "json"]),
                        ("run", ["--test-threads", "1", "--retries", "0", "--no-fail-fast", "--no-tests", "fail"]),
                    )
                ],
            }
            if legacy or downgraded:
                execution.pop("schema")
                execution["features"] = "test-support"
            if not historical and (not downgraded or defect == "schema-downgrade-execution"):
                execution.update({"request": request, "package_id": ids[package]})
            if defect == "binding" and registration == "Rust.Core":
                execution.pop("request")
            if defect == "package" and registration == "Rust.Core":
                execution["package_id"] = ids[names[1]]
            if defect == "binary" and registration == "Rust.Core":
                suites[names[0]]["binary-path"] = "/wrong/binary"
            discovery_build = binaries["rust-build-meta"]
            if defect == "discovery-build":
                discovery_build = {
                    **discovery_build,
                    "target-platform": "x86_64-pc-windows-msvc",
                    "base-output-directories": ["x86_64-pc-windows-msvc/release"],
                }
            junit_start = case_start
            if defect == "stale-case-timestamps" and short == "core":
                junit_start = "2026-11-19T18:10:00+01:00"
            if defect == "stale-junit" and short == "core":
                junit_start = "2026-11-19T17:10:00+01:00"
            duration = "90" if defect == "foreign-junit-and-digest" and short == "core" else "30"
            junit_id = (
                "c152d1c9-d1b0-4e78-b084-ab6f73332167"
                if defect == "foreign-junit-run-id" and short == "core"
                else run_ids[registration]
            )
            junit = (
                f'<testsuites tests="{suite_tests}" timestamp="{junit_start}" time="{duration}" uuid="{junit_id}">'
                + "".join(xml)
                + "</testsuites>"
            )
            if with_junit_digest:
                execution["schema"] = "rust-test-execution-v4"
                if defect != "missing-junit-digest" or short != "core":
                    execution["junit_sha256"] = hashlib.sha256(junit.encode()).hexdigest()
                if defect == "stale-junit-digest" and short == "core":
                    junit = junit.replace("T18:10:00", "T18:10:12")
                if defect == "downgraded-junit-schema":
                    execution["schema"] = "rust-test-execution-v3"
            documents.update(
                {
                    registration + ".discovery.json": {"rust-suites": suites, "rust-build-meta": discovery_build},
                    registration + ".execution.json": execution,
                    registration + ".xml": junit,
                    registration + ".nextest.toml": (
                        f'[profile.default.junit]\npath = "{junit_dir}/{registration}.xml"\n'
                    ),
                    registration + ".rows.jsonl": "",
                }
            )
        host = {
            "schema": "native-host-contracts-v1",
            "targets": {},
            "required": [],
            "rust_cohorts": {name: 0 for name in names},
        }
        if (
            not historical
            and not stripped
            and defect
            not in {
                "schema-downgrade-registration",
                "schema-downgrade-execution",
            }
        ):
            host["grain_preview"] = {"rust_core_publication": self.publication}
        if build_kind == "cuda-free":
            host["targets"] = {"probe": {"cuda_free": True, "protocol": "custom", "registrations": 1}}
            host["required"] = host["cuda_free_required"] = ["Probe"]
            tests.append(
                {
                    "name": "Probe",
                    "command": ["native-case", "--protocol", "custom", "--", "/probe"],
                    "properties": [{"name": "LABELS", "value": ["host", "product"]}],
                }
            )
        candidate = {
            "head": "source",
            "preset": "linux-debug",
            "inputs": {"source": "identity"},
            "configuration": {
                "BUILD_TESTING": "ON",
                "CMAKE_BUILD_TYPE": "Debug",
                "JUICER_GRAIN_PREVIEW": "ON" if preview else "OFF",
                "CMAKE_HOME_DIRECTORY": source + "/tests/host" if build_kind == "cuda-free" else source,
            },
        }
        if (
            not historical
            and not stripped
            and defect
            not in {
                "candidate-binding",
                "schema-downgrade-registration",
                "schema-downgrade-execution",
            }
        ):
            candidate["configuration"]["rust_tests"] = request
        tools = {
            "cmake": "/qualified/cmake",
            "ctest": "/qualified/ctest",
            "cargo": "/cargo",
            "rustc": "/rustc",
            "nextest": "/nextest",
        }
        if defect == "foreign-cargo-tool":
            tools["cargo"] = "/foreign/cargo"
            tools["rustc"] = "/foreign/rustc"
        if defect == "foreign-nextest-tool":
            tools["nextest"] = "/foreign/nextest"
        candidate["versions"] = {
            "cmake": "cmake version 4.3",
            "ctest": "ctest version 4.3",
            "cargo": "cargo 1.98.1 fixture",
            "rustc": f"rustc 1.98.1 fixture\nhost: {self.target}",
            "nextest": "cargo-nextest 0.9.143 fixture",
        }
        if defect.startswith("unqualified-"):
            tool = defect.removeprefix("unqualified-")
            label = "cargo-nextest" if tool == "nextest" else tool
            candidate["versions"][tool] = f"{label} 0.0.1 fixture"
        candidate["version_commands"] = [
            {
                "name": name,
                "command": [program, "-Vv" if name == "rustc" else "--version"],
                "returncode": 0,
                "output": candidate["versions"][name],
            }
            for name, program in tools.items()
        ]
        artifact_schema = None
        if defect == "legacy-preview":
            artifact_schema = "rust-artifacts-v1"
        elif build_kind == "product" and defect not in {"missing-artifact", "schema-downgrade-no-artifact"}:
            artifact_schema = "rust-artifacts-v2"
        if artifact_schema:
            candidate["features"] = {"rust_artifacts": "rust-artifacts.json"}
            archive_build = "C:/foreign/out/build/windows-clang-release" if defect == "foreign-artifact" else build
            archive_target = "x86_64-pc-windows-msvc" if defect == "foreign-artifact" else self.target
            archive_name = "film_juicer_plugin.lib" if defect == "foreign-artifact" else "libfilm_juicer_plugin.a"
            archive_profile = "release" if defect in {"foreign-artifact", "wrong-artifact-profile"} else "debug"
            production = ["grain-preview"] if artifact_schema == "rust-artifacts-v2" and preview else []
            variants = {}
            for variant, variant_features in (("production", production), ("test-support", ["test-support"])):
                directory = f"{archive_build}/cargo/{variant}"
                variants[variant] = {
                    "archive": f"{directory}/{archive_target}/{archive_profile}/{archive_name}",
                    "target_dir": directory,
                    "feature_input": f"{directory}/features.txt",
                    "stamp": f"{directory}/build.stamp",
                    "features": variant_features,
                }
            documents["rust-artifacts.json"] = {
                "schema": artifact_schema,
                "build": archive_build,
                "native_archive": f"{archive_build}/{'juicer_core.lib' if defect == 'foreign-artifact' else 'libjuicer_core.a'}",
                "consumers": {"juicer": {"binary": f"{archive_build}/juicer.ofx", "variant": "production"}},
                "variants": variants,
            }
        ctest_cases = []
        for name in selected:
            witness = ""
            if with_junit_id and (defect != "missing-junit-run-id" or name != "Rust.Core"):
                witness = f'<properties><property name="juicer_nextest_run_id" value="{run_ids[name]}"/></properties>'
            ctest_cases.append(f'<testcase name="{name}" status="run" time="40">{witness}</testcase>')
        documents.update(
            {
                "candidate.json": candidate,
                "candidate.patch": "patch",
                "registration.json": {"tests": tests},
                "target-discovery.json": {},
                "host-contracts.json": host,
                "host-targets.json": {"probe": "/probe"} if build_kind == "cuda-free" else {},
                "gpu-contracts.json": {"schema": "native-gpu-contracts-v1", "targets": {}, "required": []},
                "gpu-targets.json": {},
                "python-contracts.json": {"schema": "python-contracts-v1", "registrations": {}},
                "python-targets.json": {},
                "rust-cargo.json": cargo,
                "rust-binaries.json": binaries,
                "ctest.xml": '<testsuites><testsuite timestamp="2026-11-19T18:10:00" time="84">'
                + "".join(ctest_cases)
                + "</testsuite></testsuites>",
            }
        )
        payloads = {
            name: (value if isinstance(value, str) else json.dumps(value)).encode() for name, value in documents.items()
        }
        invoked = ("Rust.Bridge",) if defect == "outer-selection" else selected
        build_target = "juicer_rust_test_support_build" if defect == "outer-build-target" else "juicer_rust_tests"
        schema = "native-tests-v1" if historical or downgraded else "native-tests-v2"
        if with_junit_digest and defect != "downgraded-junit-schema":
            schema = "native-tests-v3"
        if with_junit_id and defect != "downgraded-junit-id-schema":
            schema = "native-tests-v4"
        result = {
            "schema": schema,
            "status": "passed",
            "preset": "linux-debug",
            "scope": "focused",
            "runtime": "host",
            "groups": [],
            "selection": list(selected),
            "build_kind": build_kind,
            "candidate_head": "source",
            "candidate_unchanged": True,
            "shipping_isolation": False,
            "case_files": {},
            "rust_artifacts": bool(artifact_schema),
            "commands": [
                {
                    "name": "registration",
                    "command": [
                        tools["ctest"],
                        "--test-dir",
                        "/foreign/build" if defect == "foreign-inventory" else build,
                        "--show-only=json-v1",
                    ],
                    "directory": source,
                    "returncode": 0,
                    "drained": True,
                    "process_status": "completed",
                },
                {
                    "name": "build",
                    "command": [
                        "/foreign/cmake" if defect == "foreign-cmake" else tools["cmake"],
                        "--build",
                        build,
                        "--target",
                        build_target,
                        "--",
                        "-k0",
                    ],
                    "directory": source,
                    "returncode": 0,
                    "drained": True,
                    "process_status": "completed",
                },
                {
                    "name": "versions",
                    "command": [
                        "/python",
                        "-B",
                        f"{source}/scripts/verify.py",
                        "tool-versions",
                        "--tool",
                        "cmake",
                        tools["cmake"],
                        "--tool",
                        "ctest",
                        "/foreign/ctest" if defect == "foreign-version-probe" else tools["ctest"],
                        "--tool",
                        "cargo",
                        tools["cargo"],
                        "--tool",
                        "rustc",
                        tools["rustc"],
                        "--tool",
                        "nextest",
                        tools["nextest"],
                    ],
                    "directory": source,
                    "returncode": 0,
                    "drained": True,
                    "process_status": "completed",
                },
                {
                    "name": "ctest-host",
                    "command": [
                        "/foreign/ctest" if defect == "foreign-ctest" else tools["ctest"],
                        "--test-dir",
                        "/foreign/out/build/linux-debug" if defect == "outer-build" else build,
                        "--no-tests=error",
                        "--output-on-failure",
                        "--output-junit",
                        run_dir + "/ctest-host.xml",
                        "-j",
                        "1",
                        "-R",
                        "^(" + "|".join(map(re.escape, sorted(invoked))) + ")$",
                    ],
                    "directory": source,
                    "returncode": 0,
                    "drained": True,
                    "process_status": "completed",
                },
            ],
            "members": {
                name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in payloads.items()
            },
        }
        for name, data in payloads.items():
            (root / name).write_bytes(data)
        (root / "result.json").write_text(json.dumps(result))

    def test_current_product_registration_uses_configured_preview_and_replay(self):
        cases = (
            ("preview-off", False, False, ""),
            ("preview-no-replay", True, False, ""),
            ("preview-with-replay", True, True, ""),
            (
                "drop-preview",
                True,
                True,
                "missing configured product registration contract: Ffi.Gpu.GrainPreviewStaged",
            ),
            ("drop-replay", True, True, "missing configured product registration contract: Grain.Gpu.FilmOfxApertures"),
            ("invent-replay", True, False, "optional product registration in the base catalog"),
            ("missing-configuration", True, True, "missing configured product preview and replay state"),
            ("swapped-owners", True, True, "optional product registration differs from its owning command"),
            ("aperture-wrong-selector", True, True, "optional product registration differs from its owning command"),
            ("aperture-wrong-runtime", True, True, "optional product registration differs from its owning command"),
            ("weak-aperture-catalog", True, True, "retained source differs from candidate: tests/gpu-contracts.json"),
            ("substituted-executable", True, True, "optional product executable differs from configured target output"),
            ("false-replay-off", True, False, "configured replay availability disagrees with CMake artifact consumers"),
            ("false-replay-on", True, True, "configured replay availability disagrees with CMake artifact consumers"),
        )
        self.subtest_cases = tuple(case for case, *_ in cases)
        root = Path(verify.__file__).parent.parent
        aperture = "Grain.Gpu.FilmOfxApertures"
        staged = "Ffi.Gpu.GrainPreviewStaged"
        selectors = {
            "Ffi.Host.GrainPreviewDiscovery": "--grain-preview-discovery",
            staged: "--grain-preview-witness",
            aperture: "--grain-preview-apertures",
        }
        with tempfile.TemporaryDirectory() as directory:
            for case, preview, replay, error in cases:
                with self.subTest(case=case):
                    bundle = Path(directory) / case
                    bundle.mkdir()
                    self.bundle(bundle, preview=preview, selected=("Rust.Bridge",), with_junit_id=True)
                    result = json.loads((bundle / "result.json").read_bytes())
                    payloads = {name: (bundle / name).read_bytes() for name in result["members"]}
                    candidate = json.loads(payloads["candidate.json"])
                    candidate["configuration"].update(
                        JUICER_GRAIN_PREVIEW_CONFIGURED="ON" if preview else "OFF",
                        JUICER_GRAIN_MOTION_REPLAY_CONFIGURED="ON" if replay else "OFF",
                    )
                    if case == "missing-configuration":
                        del candidate["configuration"]["JUICER_GRAIN_MOTION_REPLAY_CONFIGURED"]
                    inventory = json.loads(payloads["registration.json"])
                    for runtime in ("host", "gpu"):
                        member = f"{runtime}-contracts.json"
                        catalog = json.loads(payloads[member])
                        source = json.loads((root / "tests" / member).read_bytes())
                        paths = {}
                        properties = [{"name": "LABELS", "value": [runtime]}]

                        def register(target, name):
                            paths[target] = "/configured/out/build/linux-debug/" + target
                            inventory["tests"].append(
                                {
                                    "name": name,
                                    "command": [
                                        "native-case",
                                        "--runtime",
                                        runtime,
                                        "--protocol",
                                        "custom",
                                        "--",
                                        paths[target],
                                        selectors.get(name, name),
                                    ],
                                    "properties": properties,
                                }
                            )

                        sections = [("grain_preview", preview, runtime == "gpu" and case == "drop-preview")]
                        if runtime == "gpu":
                            properties += [
                                {"name": "TIMEOUT", "value": 0},
                                {"name": "RESOURCE_LOCK", "value": ["juicer_gpu"]},
                            ]
                            target = "JuicerProductionPreparedRenderProbe"
                            name = "Ffi.Gpu.ProductionPreparedRender"
                            catalog["targets"][target] = source["targets"][target]
                            catalog["required"].append(name)
                            register(target, name)
                            sections.append(("grain_preview_motion_replay", replay, case == "drop-replay"))
                        for section, enabled, dropped in sections:
                            if dropped:
                                catalog.pop(section, None)
                                continue
                            scope = catalog[section] = source[section]
                            if runtime == "host":
                                # Historical preview receipts declared a single paired discovery case.
                                discovery = "Ffi.Host.GrainPreviewDiscovery"
                                scope = catalog[section] = {
                                    "rust_core_publication": scope["rust_core_publication"],
                                    "targets": scope["targets"],
                                    "required": [discovery],
                                    "native_cohorts": {discovery: 1},
                                    "native_membership": {
                                        discovery: {
                                            "listed": 1,
                                            "terminal": 1,
                                            "required": [["grain-preview-discovery", "both-modules"]],
                                        }
                                    },
                                }
                            if section == "grain_preview_motion_replay" and case == "invent-replay":
                                catalog["required"].extend(scope["required"])
                                for field in ("targets", "native_cohorts", "native_membership"):
                                    catalog.setdefault(field, {}).update(scope[field])
                                enabled = True
                            if not enabled:
                                continue
                            (target,) = scope["targets"]
                            (name,) = scope["required"]
                            register(target, name)
                        payloads[member] = json.dumps(catalog).encode()
                        payloads[f"{runtime}-targets.json"] = json.dumps(paths).encode()
                    for runtime in ("host", "gpu", "python"):
                        member = f"{runtime}-contracts.json"
                        candidate["inputs"][f"tests/{member}"] = {
                            "worktree_sha256": hashlib.sha256(payloads[member]).hexdigest()
                        }
                    artifacts = json.loads(payloads["rust-artifacts.json"])
                    probe = "JuicerProductionPreparedRenderProbe"
                    replay_target = "JuicerGrainMotionReplay"
                    artifacts["consumers"][probe] = {"binary": f"{artifacts['build']}/{probe}", "variant": "none"}
                    if (replay and case != "false-replay-on") or case == "false-replay-off":
                        artifacts["consumers"][replay_target] = {
                            "binary": f"{artifacts['build']}/{replay_target}",
                            "variant": "production",
                        }
                    payloads["rust-artifacts.json"] = json.dumps(artifacts).encode()
                    if case == "weak-aperture-catalog":
                        catalog = json.loads(payloads["gpu-contracts.json"])
                        membership = catalog["grain_preview_motion_replay"]["native_membership"][aperture]
                        self.assertEqual(membership["listed"], 6)
                        membership.update(listed=1, terminal=1, required=membership["required"][:1])
                        payloads["gpu-contracts.json"] = json.dumps(catalog).encode()
                    if case in {
                        "preview-with-replay",
                        "swapped-owners",
                        "aperture-wrong-selector",
                        "aperture-wrong-runtime",
                        "substituted-executable",
                    }:
                        tests = {entry["name"]: entry for entry in inventory["tests"]}
                        if case == "swapped-owners":
                            tests[aperture]["command"], tests[staged]["command"] = (
                                tests[staged]["command"],
                                tests[aperture]["command"],
                            )
                        elif case == "aperture-wrong-selector":
                            tests[aperture]["command"][-1] = "--grain-preview-witness"
                        elif case == "aperture-wrong-runtime":
                            tests[aperture]["properties"] = [{"name": "LABELS", "value": ["host"]}]
                        catalog = qualification.configured_product_contract(
                            json.loads(payloads["gpu-contracts.json"]), "gpu", candidate["configuration"]
                        )
                        paths = json.loads(payloads["gpu-targets.json"])
                        if case == "substituted-executable":
                            paths[replay_target] = "/usr/bin/true"
                            native = tests[aperture]["command"]
                            native[native.index("--") + 1] = paths[replay_target]
                            payloads["gpu-targets.json"] = json.dumps(paths).encode()
                        options = {"runtime": "gpu", "artifact_consumers": artifacts["consumers"]}
                        if error:
                            with self.assertRaisesRegex(ValueError, error):
                                qualification.native_registration(catalog, paths, inventory, **options)
                        else:
                            registered = qualification.native_registration(catalog, paths, inventory, **options)
                            self.assertEqual(
                                registered[aperture]["command"],
                                [paths["JuicerGrainMotionReplay"], "--grain-preview-apertures"],
                            )
                            aperture_cases = catalog["native_membership"][aperture]
                            self.assertEqual(aperture_cases["listed"], 6)
                            self.assertEqual(aperture_cases["terminal"], 6)
                            self.assertEqual(len(aperture_cases["required"]), 6)
                    payloads["candidate.json"] = json.dumps(candidate).encode()
                    payloads["registration.json"] = json.dumps(inventory).encode()
                    result["schema"] = "native-tests-v6"
                    result["members"] = PythonResultTests.result_members(payloads)
                    for name, data in payloads.items():
                        (bundle / name).write_bytes(data)
                    (bundle / "result.json").write_text(json.dumps(result), encoding="utf-8")
                    if error:
                        # The complete reader checks the host target map before the GPU inventory.
                        offline_error = "unowned native host executable" if case == "aperture-wrong-runtime" else error
                        with self.assertRaisesRegex(ValueError, offline_error):
                            qualification.read_native_result(bundle)
                    else:
                        summary = qualification.read_native_result(bundle)
                        self.assertEqual(summary["ctest_cases"], 1)
                        self.assertEqual(summary["custom_cases"], 0)
                        self.assertEqual("Grain.Gpu.FilmOfxApertures" in summary["unselected_registrations"], replay)

    test_current_product_registration_uses_configured_preview_and_replay.required_subtests = True

    def test_preview_host_witness_requires_own_command_and_cases(self):
        from copy import deepcopy

        root = Path(verify.__file__).parent.parent
        host = json.loads((root / "tests/host-contracts.json").read_bytes())
        preset = "linux-debug"
        build = "/configured/out/build/linux-debug"
        configuration = {
            "BUILD_TESTING": "ON",
            "JUICER_GRAIN_PREVIEW": "ON",
            "JUICER_GRAIN_PREVIEW_CONFIGURED": "ON",
            "JUICER_GRAIN_MOTION_REPLAY_CONFIGURED": "ON",
            "CMAKE_BUILD_TYPE": "Debug",
            "CMAKE_HOME_DIRECTORY": "/configured",
            "CMAKE_INSTALL_PREFIX": "/configured/out/stage/linux-debug",
        }
        module = "juicer-grain-preview.ofx"
        built = f"{build}/{module}"
        staged = (
            f"/configured/out/stage/linux-debug/portra-grain-preview/{module}.bundle/Contents/Linux-x86-64/{module}"
        )
        discovery = "Ffi.Host.GrainPreviewDiscovery"
        python = "/configured/venv/bin/python"
        commands = {
            discovery: ("JuicerProductionPreparedRenderProbe", ["--grain-preview-discovery", staged, built]),
            "Ofx.Host.GrainPreviewModule": (
                "JuicerOfxModuleTests",
                ["--grain-preview", "/configured/tests/ofx/descriptor_manifest_grain_preview.json"],
            ),
            "Ofx.Host.GrainPreviewInstance": (
                "JuicerOfxInstanceTests",
                ["--grain-preview", "--reference", "/configured/tests/ofx/fixtures/s6-d/instance.json"],
            ),
            "Ofx.Host.GrainPreviewRender": (
                "JuicerOfxRenderTests",
                ["--grain-preview", "--reference", "/configured/tests/ofx/fixtures/s6-d/render.json"],
            ),
        }
        targets = {target: f"{build}/tests/bin/{target}" for target, _ in commands.values()}
        host["required"] = []
        host["targets"] = {
            target: host["targets"][target] for target in targets if target != "JuicerProductionPreparedRenderProbe"
        }
        contract = qualification.configured_product_contract(host, "host", configuration)
        consumers = {
            **{target: {"binary": binary, "variant": "production"} for target, binary in targets.items()},
            "juicer": {"binary": built, "variant": "production"},
        }
        tests = {
            name: {
                "name": name,
                "command": [
                    python,
                    "-B",
                    "/configured/scripts/verify.py",
                    "native-case",
                    "--preset",
                    preset,
                    "--protocol",
                    "custom",
                    "--",
                    targets[owner],
                    *arguments,
                ],
                "properties": [{"name": "LABELS", "value": ["host"]}],
            }
            for name, (owner, arguments) in commands.items()
        }

        def register(cases):
            return qualification.native_registration(
                contract,
                targets,
                {"tests": list(cases.values())},
                artifact_consumers=consumers,
                preset=preset,
                configuration=configuration,
                launcher_python=python,
            )

        self.assertEqual(register(tests).keys(), tests.keys())
        for name in tests:
            wrong_launcher = deepcopy(tests)
            wrong_launcher[name]["command"][2] = "/configured/scripts/check-quality.py"
            with self.assertRaisesRegex(ValueError, "OFX witness launcher differs"):
                register(wrong_launcher)
        for index, argument in (
            (0, "/foreign/bin/python"),
            (1, "-u"),
            (3, "python-tests"),
            (5, "linux-release"),
            (6, "--runtime"),
            (7, "gtest"),
            (8, "--other"),
        ):
            wrong_launcher = deepcopy(tests)
            wrong_launcher[discovery]["command"][index] = argument
            with self.assertRaisesRegex(ValueError, "OFX witness launcher differs"):
                register(wrong_launcher)
        module_test = "Ofx.Host.GrainPreviewModule"
        swapped = deepcopy(tests)
        swapped[module_test]["command"] = tests["Ofx.Host.GrainPreviewRender"]["command"]
        with self.assertRaisesRegex(ValueError, "callback registration differs"):
            register(swapped)
        wrong_module = deepcopy(tests)
        wrong_module[discovery]["command"][-2:] = [built, staged]
        with self.assertRaisesRegex(ValueError, "discovery arguments differ"):
            register(wrong_module)
        wrong_manifest = deepcopy(tests)
        wrong_manifest[module_test]["command"][-1] = "/configured/tests/ofx/descriptor_manifest.json"
        with self.assertRaisesRegex(ValueError, "callback registration differs"):
            register(wrong_manifest)
        with self.assertRaisesRegex(ValueError, "callback registration differs"):
            register({name: row for name, row in tests.items() if name != module_test})
        qualification.native_membership(contract, discovery, {("grain-preview-discovery", "preview-module")}, 1)
        with self.assertRaisesRegex(ValueError, "missing or unexpected native membership"):
            qualification.native_membership(contract, discovery, {("grain-preview-discovery", "both-modules")}, 1)

    def test_offline_rust_preview_receipts(self):
        cases = (
            ("preview", {}, False),
            ("preview-junit-digest", {"with_junit_digest": True}, False),
            ("preview-junit-run-id", {"with_junit_id": True}, False),
            ("off", {"preview": False}, False),
            ("cuda-free", {"build_kind": "cuda-free"}, False),
            ("core-only", {"selected": ("Rust.Core",)}, False),
            ("bridge-only", {"selected": ("Rust.Bridge",)}, False),
            ("bridge-incomplete", {"selected": ("Rust.Bridge",), "defect": "bridge-incomplete"}, True),
            ("historical", {"historical": True}, False),
            ("legacy-label", {"historical": True, "legacy": True, "preview": False}, False),
            ("legacy-preview", {"historical": True, "legacy": True, "defect": "legacy-preview"}, False),
            ("historical-features", {"historical": True, "defect": "features"}, True),
            ("historical-missing", {"historical": True, "defect": "missing"}, True),
            ("historical-combined", {"historical": True, "defect": "combined"}, True),
            *(
                (name, {"with_junit_id": True, "defect": name}, True)
                for name in (
                    "foreign-junit-run-id",
                    "missing-junit-run-id",
                    "downgraded-junit-id-schema",
                    "foreign-artifact",
                    "wrong-artifact-profile",
                    "unqualified-cargo",
                    "unqualified-rustc",
                    "unqualified-nextest",
                )
            ),
            *(
                (name, {"with_junit_digest": True, "defect": name}, True)
                for name in (
                    "stale-junit-digest",
                    "missing-junit-digest",
                    "downgraded-junit-schema",
                    "foreign-junit-and-digest",
                    "early-bridge-digest",
                    "missing-artifact",
                )
            ),
            *(
                (name, {"defect": name}, True)
                for name in (
                    "features",
                    "missing",
                    "combined",
                    "discovery-only",
                    "package",
                    "binary",
                    "filtered",
                    "skipped",
                    "failed",
                    "binding",
                    "candidate-binding",
                    "foreign-checkout",
                    "foreign-metadata",
                    "outer-build",
                    "schema-downgrade",
                    "schema-downgrade-registration",
                    "schema-downgrade-execution",
                    "schema-downgrade-all",
                    "schema-downgrade-no-artifact",
                    "outer-selection",
                    "outer-build-target",
                    "foreign-nextest-config",
                    "foreign-nextest-junit",
                    "stale-junit",
                    "stale-case-timestamps",
                    "early-bridge-report",
                    "foreign-inventory",
                    "foreign-cmake",
                    "foreign-ctest",
                    "foreign-version-probe",
                    "foreign-python",
                    "foreign-cargo-tool",
                    "foreign-nextest-tool",
                    "prepared-platforms",
                    "discovery-build",
                    "aliased-binaries",
                )
            ),
        )
        self.subtest_cases = tuple(name for name, _, _ in cases)
        for name, options, rejected in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.bundle(root, **options)
                if rejected:
                    messages = {
                        "features": "workspace features",
                        "missing": "required Rust core publication",
                        "combined": "workspace features",
                        "discovery-only": "unexpected framework terminal",
                        "package": "execution binding",
                        "binary": "discovery differs",
                        "filtered": "package function",
                        "skipped": "required framework case skipped",
                        "failed": "duplicate/failed framework case",
                        "binding": "execution binding",
                        "candidate-binding": "Rust test request",
                        "foreign-checkout": "Rust CTest registration differs",
                        "foreign-metadata": "Cargo metadata differs from configured workspace or target directory",
                        "outer-build": "differs from the configured outer CTest invocation",
                        "schema-downgrade": "current Rust test bindings",
                        "schema-downgrade-registration": "current Rust test bindings",
                        "schema-downgrade-execution": "current Rust execution binding",
                        "schema-downgrade-all": "preview Rust execution requires resolved feature evidence",
                        "schema-downgrade-no-artifact": "product Rust receipt requires its separate artifact description",
                        "outer-selection": "outer CTest invocation",
                        "outer-build-target": "outer build did not prepare",
                        "foreign-nextest-config": "Rust nextest commands differ",
                        "foreign-nextest-junit": "Rust nextest JUnit path differs",
                        "stale-junit": "nextest JUnit timestamp is outside the retained CTest run",
                        "stale-case-timestamps": "nextest JUnit case timestamp is outside the reported run",
                        "early-bridge-report": "nextest JUnit starts before preceding CTest cases",
                        "early-bridge-digest": "nextest JUnit starts before preceding CTest cases",
                        "foreign-junit-and-digest": "nextest JUnit duration exceeds the retained CTest case",
                        "missing-artifact": "product Rust receipt requires its separate artifact description",
                        "foreign-junit-run-id": "Rust JUnit run ID differs",
                        "missing-junit-run-id": "Rust JUnit run ID differs",
                        "downgraded-junit-id-schema": "current Rust JUnit run ID in an older result",
                        "foreign-artifact": "Rust artifact manifest differs from the configured build",
                        "wrong-artifact-profile": "Rust artifact paths differ from the configured build",
                        "unqualified-cargo": "qualified Rust/Cargo",
                        "unqualified-rustc": "qualified Rust/Cargo",
                        "unqualified-nextest": "qualified Rust/Cargo",
                        "stale-junit-digest": "Rust JUnit digest differs",
                        "missing-junit-digest": "Rust JUnit digest differs",
                        "downgraded-junit-schema": "current Rust JUnit binding",
                        "foreign-inventory": "registration did not use the configured build and CTest executable",
                        "foreign-cmake": "outer build did not prepare",
                        "foreign-ctest": "outer CTest invocation",
                        "foreign-version-probe": "native tool probes differ",
                        "foreign-python": "Rust CTest registration differs",
                        "foreign-cargo-tool": "Rust CTest tools differ from version probes",
                        "foreign-nextest-tool": "Rust CTest tools differ from version probes",
                        "prepared-platforms": "prepared Rust binaries differ from configured target or profile",
                        "discovery-build": "discovery build metadata differs",
                        "bridge-incomplete": "missing or unexpected prepared Rust test binaries",
                        "aliased-binaries": "duplicate prepared Rust test executable path",
                        "historical-features": "workspace features",
                        "historical-missing": "required Rust core publication",
                        "historical-combined": "workspace features",
                    }
                    with self.assertRaisesRegex(ValueError, messages[name]):
                        qualification.read_native_result(root)
                else:
                    self.assertEqual(qualification.read_native_result(root)["status"], "passed")


RustReceiptBundleTests.test_offline_rust_preview_receipts.required_subtests = True


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
                self.assertIn(
                    result["schema"],
                    {
                        "native-tests-v1",
                        "native-tests-v2",
                        "native-tests-v3",
                        "native-tests-v4",
                        "native-tests-v5",
                        "native-tests-v6",
                    },
                )
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
