"""Checkpoint ownership, cross-caller reuse and exact CTest membership controls."""
from __future__ import annotations
import json
import hashlib
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import textwrap
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/ctest"))
from control_artifacts import (MARKER, current_fixture_blockers, read_fixture_marker, record_fixture,
                               release_fixture_hold, selected_paths)
import qualification
from execution import Campaign, OutputLease, digest, snapshot, tool_snapshot, write_json

from execution import read_json

ROOT = Path(__file__).resolve().parents[2]
SUITE_STARTED = time.monotonic()


def policy_fixture_mode(root, mode, *, early_on=False):
    """Maintained synthetic producer/admission modes; no real product qualification."""
    from verify import source_snapshot
    root = Path(root).resolve()
    environment = dict(os.environ)
    manifest = root / "out/policy-fixture.json"
    def state_file(preset):
        return root / "out/build" / preset / "verification-checkpoint.json"
    def read_state(preset):
        return json.loads(state_file(preset).read_text()) if state_file(preset).exists() else {}
    roles = {"linux-debug": ["product", "boundary", "tooling", "corpus"],
             "windows-clang-debug": ["product", "boundary", "tooling"],
             "linux-release": ["product", "boundary"], "windows-clang-release": ["product", "boundary"]}
    if mode == "original-early-on":
        return policy_fixture_mode(root, "original", early_on=True)
    if mode == "original-fault":
        actual_capture = qualification._policy_capture
        def omitted(*args, **kwargs):
            facts = actual_capture(*args, **kwargs)
            registrations = facts["subject"]["registrations"]
            if registrations:
                registrations.pop(next(path for path in registrations if path.replace("\\", "/").endswith("policy-leaf.cmake")))
            return facts
        # Explicit faulty-recorder seam. The real reader and native command run;
        # the original preparation retains the independently complete four files.
        with patch.object(qualification, "_policy_capture", side_effect=omitted):
            return policy_fixture_mode(root, "original")

    def prepare(preset, testing, campaign_name):
        build = root / "out/build" / preset
        campaign = Campaign(root, preset, campaign_name, ["prepare"])
        state = read_state(preset)
        with OutputLease(root, build):
            # The first configure is an ordinary owned bootstrap, as in the CLI fixture.
            if not (build / "CMakeCache.txt").exists():
                campaign.run("prepare", ["cmake", "--preset", preset, "-DBUILD_TESTING=" + testing],
                    inputs=sorted(qualification._policy_candidate(root)["files"]),
                    tools=[str(Path(__import__("shutil").which("cmake")).resolve())], env=environment, timeout=None)
            if testing == "ON":
                return qualification.prepare_checkpoint(campaign, build, state_file(preset), state, environment,
                    source_files=source_snapshot(root), allow_build=True, repeat_reason="explicit synthetic policy preparation")
            inputs = qualification.checkpoint_inputs(root, build, "prepare", source_files=source_snapshot(root))
            outputs = []
            def validate(path):
                from build_evidence import check_scoped_preparation
                outputs[:] = check_scoped_preparation(campaign, build, path, environment)["scoped_preparation"]["outputs"]
            return qualification.run_stage(campaign, state_file(preset), state, "prepare", ["cmake", "--build", str(build)],
                inputs=inputs, tools=qualification.checkpoint_tools(build, environment, "prepare"), environment=environment,
                timeout=None, action=lambda: qualification.prepare_build(campaign, "prepare", source=root, build=build,
                    inputs=[name for name in inputs if not name.startswith("out/build/")],
                    timeout=None, targets=["all"], configure_args=["-DBUILD_TESTING=OFF"],
                    env=environment, checkpoint=True)[1], validate=validate, payloads=outputs,
                refresh_inputs=lambda: qualification.checkpoint_inputs(root, build, "prepare"))

    def ctest_stage(preset, group, campaign_name, *, capture=False, support=False, required=None):
        build = root / "out/build" / preset
        state = read_state(preset)
        campaign = Campaign(root, preset, campaign_name, required or [group])
        name = "Tooling.Miniature" if group == "tooling" else "Policy." + group
        junit = campaign.directory / (group + ".xml")
        command = [*qualification.ctest_command(root, build), "-R", "^(" + re.escape(name) + ")$", "-LE", "gpu",
                   "--no-tests=error", "--output-on-failure", "--output-junit", str(junit)]
        inputs = qualification.checkpoint_inputs(root, build, group, source_files=source_snapshot(root))
        tools = qualification._policy_tools(build, environment, group)
        payloads = [junit] + ([] if group == "tooling" else [build / ("counter-" + group + ".txt")])
        dependency = qualification.preparation_identity(build, state["prepare"]) if group != "tooling" else None
        with OutputLease(root, build):
            result = qualification.run_stage(campaign, state_file(preset), state, group, command,
                inputs=inputs, tools=tools, environment=environment, timeout=None,
                semantic_command=[qualification.ctest_identity(command)],
                action=lambda: qualification.run_ctest_parts(campaign, group, [command], {"host": [name]}, build=build,
                    inputs=inputs, tools=tools, environment=environment, timeout=None, repeat_reason="",
                    state=state, state_path=state_file(preset), payloads=[junit]),
                validate=lambda paths: qualification.validate_ctest_stage(paths, {"host": [name]}, [name],
                    [qualification.ctest_identity(command)]), payloads=payloads,
                dependencies=[] if dependency is None else [dependency], ctest=True,
                bounded_capture=qualification._POLICY_RECIPE if capture else None, policy_support=support)
        if capture and "execution_contract" not in result:
            raise AssertionError("policy original did not capture: " + str(result.get("bounded_ineligible")))
        return result

    def quality_stage(preset, campaign_name, base):
        build = root / "out/build" / preset
        campaign = Campaign(root, preset, campaign_name, ["quality"])
        state = read_state(preset)
        with OutputLease(root, build):
            return qualification.run_stage(campaign, state_file(preset), state, "quality",
                [sys.executable, "-B", "scripts/check-quality.py", "--preset", preset, "--qualification", "--base", base],
                inputs=qualification.checkpoint_inputs(root, build, "product"),
                tools=qualification._policy_tools(build, environment, "product"), environment=environment, timeout=None,
                dependencies=[qualification.preparation_identity(build, state["prepare"])], synthetic_quality_base=base,
                synthetic_quality_recipe=qualification._POLICY_RECIPE, policy_support=True)

    def wrapper(preset, groups, stages, campaign_name, *, chronology=None, support_path=None):
        required = sorted(set(groups) | ({"prepare", "quality"} if "product" in groups else set()))
        campaign = None if support_path is not None else Campaign(root, preset, campaign_name, required)
        members = {group: ["Tooling.Miniature" if group == "tooling" else "Policy." + group]
                   for group in groups if group != "production"}
        record = {"mode": "checkpoint", "status": "passed", "scope_status": "passed", "reason": "synthetic policy fixture",
                  "preset": preset, "runtime": "all", "selection": sorted(groups), "required": required,
                  "source": source_snapshot(root), "stages": {name: stages[name] for name in required},
                  "membership": members, "runtime_membership": {name: {"host": names} for name, names in members.items()}}
        if chronology is not None:
            record["policy_chronology"] = chronology
        path = support_path if support_path is not None else campaign.directory / "qualification.json"
        write_json(path, record)
        return str(path)

    if mode == "original":
        base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        originals, stages, paths, off = {}, {}, [], {}
        for preset in ("linux-release", "windows-clang-release"):
            build = root / "out/build" / preset
            prepare(preset, "OFF", "policy-off-prepare")
            state = read_state(preset)
            campaign = Campaign(root, preset, "policy-off-original", ["production"])
            with OutputLease(root, build):
                retained_ref = qualification.policy_retain_off(campaign, state)
                retained = qualification._policy_off(retained_ref)
                command = [retained["binary"]["path"], str(root / "input.txt"), retained["counter"]]
                inputs = qualification.checkpoint_inputs(root, build, "production")
                result = qualification.run_stage(campaign, state_file(preset), state, "production", command,
                    inputs=inputs, tools=qualification._policy_tools(build, environment, "production", executable=command[0]),
                    environment=environment, timeout=None,
                    dependencies=[retained["preparation"]["identity"]], bounded_capture=qualification._POLICY_RECIPE,
                    payloads=[Path(retained_ref["path"]), Path(retained["counter"]),
                              *[Path(row["path"]) for row in retained["files"].values()]])
            if "execution_contract" not in result:
                raise AssertionError("OFF original did not capture: " + str(result.get("bounded_ineligible")))
            off[preset] = result
            originals[preset + ":production"] = result["execution_contract"]
            if early_on and preset == "linux-release":
                prepare(preset, "ON", "policy-original-on-prepare")
        on = {}
        for preset in qualification.PRESETS:
            prepare(preset, "ON", "policy-original-on-prepare")
            prepared = read_state(preset)["prepare"]
            preparation_campaign = Campaign(root, preset, "policy-original-on-prepare", ["prepare"])
            starts = [raw for raw in preparation_campaign.attempts("prepare")
                      if raw["identity"]["command"][:3] == ["cmake", "--preset", preset]
                      and "-DBUILD_TESTING=ON" in raw["identity"]["command"]]
            if not starts:
                raise AssertionError("original ON restoration has no actual configure-start receipt")
            start = starts[0]
            start_path = (root / start["output"]).with_name("result.json")
            on[preset] = {"preparation": prepared, "started": qualification._bounded_reference(start_path)}
            quality_stage(preset, "policy-original-quality", base)
            for group in ("product", "boundary", "corpus"):
                stage = ctest_stage(preset, group, "policy-original-" + group, capture=True)
                originals[preset + ":" + group] = stage["execution_contract"]
            ctest_stage(preset, "tooling", "policy-original-tooling", support=True)
            stages[preset] = read_state(preset)
        chronology = {"off": {preset: stage["execution_contract"] for preset, stage in off.items()},
                      "on": {preset: on[preset] for preset in off}}
        for preset in qualification.PRESETS:
            paths.append(wrapper(preset, roles[preset], stages[preset], "policy-original-wrapper"))
        for preset, stage in off.items():
            paths.append(wrapper(preset, ["production"], {"production": stage}, "policy-off-wrapper", chronology=chronology))
        pdb_files = sorted(str(path) for preset in qualification.PRESETS
                           for path in (root / "out/build" / preset).rglob("*.pdb") if path.is_file())
        if pdb_files:
            raise AssertionError("prospective ordinary policy build emitted PDBs: " + ", ".join(pdb_files))
        data = {"root": str(root), "base": base, "candidate": qualification._policy_candidate(root),
                "producer": qualification._policy_producer(root), "originals": originals, "original_stages": stages,
                "off_stages": off, "chronology": chronology, "ordinary_paths": paths,
                "actual_native": __import__("sysconfig").get_platform(), "counters": {str(p): p.read_bytes().hex()
                    for p in (root / "out").rglob("counter*.txt")}}
        write_json(manifest, data)
        return data
    data = json.loads(manifest.read_text())
    if mode == "support":
        support = {}
        for preset in qualification.PRESETS:
            final_name = "policy-admitted-valid-" + str(list(roles).index(preset))
            final_required = sorted(set(roles[preset]) | {"prepare", "quality"})
            in_final = "tooling" in roles[preset]
            campaign_name = final_name if in_final else "policy-current-tooling"
            ctest_stage(preset, "tooling", campaign_name, support=True,
                        required=final_required if in_final else None)
            current = read_state(preset)
            support_path = root / "out/validation" / preset / final_name / "tooling-support.json" if in_final else None
            support[preset] = {"tooling": wrapper(preset, ["tooling"], current, "policy-current-tooling-wrapper", support_path=support_path)}
            prepare(preset, "ON", "policy-current-prepare")
            quality_stage(preset, "policy-current-quality", data["base"])
            support[preset]["state"] = read_state(preset)
        data["support"] = support
        data["current_candidate"] = qualification._policy_candidate(root)
        data["current_producer"] = qualification._policy_producer(root)
        write_json(manifest, data)
        return data
    if mode in {"collect-old", "collect-current"}:
        paths = data["ordinary_paths"] if mode == "collect-old" else data["admitted_paths"]
        windows_python = os.environ.get("JUICER_WINDOWS_PYTHON")
        if not windows_python or os.name != "posix":
            raise ValueError("actual foreign F21 consumer requires Linux and its qualified Windows interpreter")
        before = {path: Path(qualification.mapped_windows_path(path)).read_bytes().hex() for path in data["counters"]}
        try:
            result = qualification.collect_qualification(root, [str(qualification.mapped_windows_path(path)) for path in paths],
                                                         windows_python=windows_python)
        except ValueError as error:
            if mode != "collect-old" or "missing" not in str(error):
                raise
            import traceback
            failure = {"mode": "actual-old-reader-foreign-collection", "status": "failed-as-expected",
                       "reader": {"path": str(root / "scripts/build_evidence.py"),
                                  "sha256": hashlib.sha256((root / "scripts/build_evidence.py").read_bytes()).hexdigest()},
                       "error": str(error), "traceback": traceback.format_exc(), "originals": data["originals"],
                       "counters_before": before, "native_identity_attempts": [str(p) for p in
                           (root / "out/validation/linux-debug").glob("collection-*")]}
            if "registration_files" not in failure["traceback"]:
                raise AssertionError("F21 failure did not reach the real registration reader") from error
            write_json(root / "out/f21-failed-collection.json", failure)
            return failure
        if mode == "collect-old":
            raise AssertionError("the actual old foreign registration reader unexpectedly collected")
        if before != {path: Path(qualification.mapped_windows_path(path)).read_bytes().hex() for path in data["counters"]}:
            raise AssertionError("foreign collection replayed an original child")
        write_json(root / "out/f21-current-collection.json", result)
        return result
    if mode == "injection":
        preset = "linux-debug"
        build = root / "out/build" / preset
        state = read_state(preset)
        original = json.loads(Path(data["originals"][preset + ":product"]["path"]).read_text())
        command = list(original["facts"]["stage_identity"]["command"][0])
        command[-1] = original["execution"]["junit"]["path"]
        campaign = Campaign(root, preset, "policy-injection-controls", ["product"])
        for key in ("LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH",
                    "CPATH", "C_INCLUDE_PATH", "GCC_EXEC_PREFIX", "COMPILER_PATH", "LIBRARY_PATH", "CL", "_CL_",
                    "CMAKE_TOOLCHAIN_FILE", "CMAKE_PROJECT_TOP_LEVEL_INCLUDES", "PATH", "INCLUDE", "LIB"):
            injected = dict(environment)
            injected[key] = (injected.get(key, "") + os.pathsep if key in {"PATH", "INCLUDE", "LIB"} else "") + str(root)
            with OutputLease(root, build):
                try:
                    qualification._policy_capture(campaign, state, "product", command,
                        qualification.checkpoint_inputs(root, build, "product"),
                        qualification._policy_tools(build, environment, "product"), injected)
                except ValueError as error:
                    if "injection" not in str(error) and "mutable miniature" not in str(error):
                        raise
                else:
                    raise AssertionError("unsupported project environment was admitted: " + key)
        return data
    if mode == "race":
        preset = "linux-debug"
        state = data["support"][preset]["state"].copy()
        for group in ("product", "boundary", "corpus"):
            state[group] = data["original_stages"][preset][group]
        write_json(state_file(preset), state)
        original_declaration = root / "out/policy-declarations/valid/0/declaration.json"
        declaration_path = root / "out/policy-declarations/race/declaration.json"
        declaration_path.parent.mkdir()
        declaration_path.write_bytes(original_declaration.read_bytes())
        writer = qualification._write_admission_record
        injected = False
        def newer_failure(path, record):
            nonlocal injected
            writer(path, record)
            if path.name == "admission.json" and not injected:
                injected = True
                latest = read_state(preset)
                latest["product"]["status"] = "failed"
                latest["product"]["error"] = "explicit current owner race injection"
                write_json(state_file(preset), latest)
        args = SimpleNamespace(reason="policy owner publication race", selection=roles[preset], preset=preset,
            runtime="all", campaign="policy-race", base=data["base"], repeat_reason="", prepare=False,
            reuse_declaration=str(declaration_path))
        try:
            with patch.object(qualification, "_write_admission_record", side_effect=newer_failure):
                qualification.qualify(root, args)
        except ValueError as error:
            if not injected or "owner_conflict" not in str(error):
                raise
            if read_state(preset)["product"]["status"] != "failed":
                raise AssertionError("admission overwrote the current failed owner") from error
            write_json(root / "out/policy-owner-race.json", {"status": "rejected", "error": str(error), "state": read_state(preset)})
            return data
        raise AssertionError("policy admission ignored a newer failed owner")
    if mode.startswith("admit"):
        variant = mode.removeprefix("admit").removeprefix(":")
        reuse_mode = variant == "reuse"
        selected_roles = {key: value.copy() for key, value in roles.items()}
        if variant and not reuse_mode:
            selected_roles["linux-debug"].remove("corpus")
            selected_roles[variant].append("corpus")
        paths = []
        for preset in qualification.PRESETS:
            if reuse_mode:
                state = read_state(preset)
            else:
                state = data["support"][preset]["state"].copy()
                for group in ("product", "boundary", "corpus"):
                    state[group] = data["original_stages"][preset][group]
                if preset in data["off_stages"]:
                    state["production"] = data["off_stages"][preset]
            write_json(state_file(preset), state)
        contexts = [(preset, groups) for preset, groups in selected_roles.items()]
        contexts += [(preset, ["production"]) for preset in ("linux-release", "windows-clang-release")]
        for index, (preset, groups) in enumerate(contexts):
            selected = sorted(set(groups) - {"tooling"})
            if reuse_mode:
                path = root / "out/policy-declarations/valid" / str(index) / "declaration.json"
            else:
                folder = root / "out/policy-declarations" / (variant or "valid") / str(index)
                folder.mkdir(parents=True)
                original_refs = [{"preset": preset, "group": group, "contract": data["originals"][preset + ":" + group]}
                                 for group in selected]
                quality = data["support"][preset]["state"]["quality"]
                declaration = {"protocol": "tooling-evidence-policy-transition-v1", "recipe": qualification._POLICY_RECIPE,
                    "original_candidate": data["candidate"], "current_candidate": data["current_candidate"],
                    "changes": [{"path": "scripts/build_evidence.py", "old_sha256": data["candidate"]["files"]["scripts/build_evidence.py"],
                        "new_sha256": data["current_candidate"]["files"]["scripts/build_evidence.py"], "groups": selected}],
                    "originals": original_refs, "compatibility": [{"path": "scripts/build_evidence.py", "group": group,
                        "role": "controller-only", "outcome_integrity": "preserved"} for group in selected],
                    "producer_transition": {"original": data["producer"], "current": data["current_producer"]},
                    "fresh_evidence": {"tooling": [qualification._bounded_reference(data["support"][preset]["tooling"])],
                        "quality": [{"path": quality["receipt"], "sha256": quality["sha256"]}], "quality_base": data["base"]},
                    "chronology": data["chronology"], "review": None, "owner_decision": None}
                core = qualification._bounded_hash({k: v for k, v in declaration.items() if k not in {"review", "owner_decision"}})
                for key, decision in (("review", "compatible"), ("owner_decision", "approve")):
                    path = folder / (key + ".json")
                    proof = {"protocol": "synthetic-policy-" + key + "-v1", "transition_sha256": core,
                             "original_candidate_sha256": qualification._bounded_hash(data["candidate"]),
                             "current_candidate_sha256": qualification._bounded_hash(data["current_candidate"]),
                             "originals": original_refs, "producer_transition": declaration["producer_transition"], "decision": decision}
                    path.write_bytes(qualification._bounded_bytes(proof))
                    declaration[key] = qualification._bounded_reference(path)
                path = folder / "declaration.json"
                path.write_bytes(qualification._bounded_bytes(declaration))
            campaign_variant = "valid-reuse" if reuse_mode else (variant or "valid")
            args = SimpleNamespace(reason="reviewed synthetic registration-reader transition", selection=groups,
                preset=preset, runtime="all", campaign="policy-admitted-" + campaign_variant + "-" + str(index),
                base=data["base"], repeat_reason="", prepare=False, reuse_declaration=str(path))
            import verify
            previous_root, previous_argv = verify.ROOT, sys.argv
            verify.ROOT = root
            sys.argv = ["verify.py", "qualification", "--preset", preset, "--runtime", "all", "--reason", args.reason,
                        "--base", data["base"], "--campaign", args.campaign, "--reuse-declaration", str(path),
                        "--selection", *groups]
            try:
                if verify.main() != 0:
                    raise AssertionError("actual policy qualification CLI did not complete")
            finally:
                verify.ROOT, sys.argv = previous_root, previous_argv
            paths.append(str(root / "out/validation" / preset / args.campaign / "qualification.json"))
        data["admitted_paths" + (":" + variant if variant else "")] = paths
        write_json(manifest, data)
        return data
    raise ValueError("unknown maintained policy fixture mode")


class PolicyFactsStorageTests(unittest.TestCase):
    def facts(self, owner, group, *, current_preparation=None):
        candidate = {"root": str(owner.parents[3]), "head": "1" * 40, "tree": "2" * 40,
                    "files": {"source/%03d.cpp" % index: "a" * 64 for index in range(64)}}
        producer = {"external": {"root": str(owner), "files": {}},
                    "reader": {"path": str(owner / "reader.py"), "sha256": "b" * 64},
                    "modules": {"qualification": "c" * 64}, "supervisor": {}, "native_supervisor": {},
                    "cache_policy": {}, "cache_absent": {}}
        preparation = {"identity": "d" * 64, "stage": {"receipt": str(owner / "prepare.json"), "sha256": "e" * 64}}
        support = {"tooling": {"path": str(owner / "tooling.json"), "sha256": "f" * 64},
                   "tooling_stage": {"receipt": str(owner / "tooling-result.json"), "sha256": "0" * 64},
                   "preparation": preparation,
                   "quality_stage": {"receipt": str(owner / "quality-result.json"), "sha256": "1" * 64},
                   "quality": {"path": str(owner / "quality.json"), "sha256": "2" * 64}}
        current = {"policy": {"recipe": "synthetic-policy"}, "candidate": candidate,
                   "context": {"group": group}, "subject": {"files": candidate["files"]},
                   "preparation": current_preparation or preparation, "producer": producer,
                   "stage_identity": {"command": [group], "inputs": candidate["files"]},
                   "counter": str(owner / ("counter-" + group))}
        original_facts = {"policy": current["policy"], "candidate": candidate,
                          "context": {"preset": "linux-debug", "group": group}, "subject": current["subject"],
                          "preparation": current["preparation"], "producer": producer,
                          "stage_identity": current["stage_identity"], "counter": current["counter"]}
        original_result = {"status": "passed", "receipt": str(owner / (group + "-result.json")),
                           "sha256": "8" * 64, "additional_receipts": {}, "payloads": {}}
        if group != "production":
            original_result["ctest_chunks"] = []
        original_path = owner / (group + "-original.json")
        original_document = {"protocol": "product-evidence-policy-v2", "assurance": qualification._POLICY_ASSURANCE,
            "facts": original_facts, "stage": {"name": group, "result": original_result}, "execution": {},
            "started_at": "2026-11-03T00:00:00+00:00", "finished_at": "2026-11-03T00:00:01+00:00",
            "launch": {}, "seals": {"entry": qualification._bounded_hash(original_facts),
                                      "terminal": qualification._bounded_hash(original_facts)},
            "producer_source": {"path": str(owner / "reader.py"), "sha256": "9" * 64}}
        original_path.write_bytes(qualification._bounded_bytes(original_document))
        original = qualification._bounded_reference(original_path)
        owner_record = {**original_result, "identity": original_facts["stage_identity"],
                        "execution_contract": original}
        return {"declaration": {"sha256": "3" * 64}, "original": original,
                "current": current, "comparison": {"sha256": "5" * 64}, "support": support,
                "owner": owner_record,
                "chronology": {"finished_at": "2026-11-03T00:00:00+00:00"}}

    def read_admission(self, reference, facts):
        with patch.object(qualification, "read_policy_contract",
                          side_effect=AssertionError("admission storage decode ran full contract validation")) as full_reader:
            result = qualification._policy_read_admission(reference)
            full_reader.assert_not_called()
            return result

    def write_admission(self, record, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(qualification._bounded_bytes(record))
        return qualification._bounded_reference(path)

    def test_campaign_local_shared_payload_roundtrips_and_saves_repeated_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            project = fixture / "project"
            owner = project / "out/validation/linux-debug/campaign"
            build = project / "out/build/linux-debug"
            owner.mkdir(parents=True)
            build.mkdir(parents=True)
            campaign = SimpleNamespace(directory=owner)
            admissions = []
            shared_paths = []
            semantic_facts = [self.facts(owner, "product"),
                               self.facts(owner, "production", current_preparation={
                                   "identity": "6" * 64, "stage": {"receipt": str(owner / "off.json"), "sha256": "7" * 64}})]
            for index, facts in enumerate(semantic_facts):
                record, path = qualification._policy_encode_admission(campaign, facts,
                    "2026-11-03T00:00:00+00:00", admission_root=build)
                reference = self.write_admission(record, path)
                restored, metadata = self.read_admission(reference, facts)
                self.assertEqual(restored, facts)
                self.assertEqual(metadata["owner_campaign"], str(owner))
                shared_paths.append(metadata["shared_facts"]["path"])
                self.assertEqual(record["current_preparation"]["mode"], "shared" if index == 0 else "override")
                admissions.append((record, reference))
            self.assertEqual(len(set(shared_paths)), 1)
            blobs = list((owner / "policy-facts").glob("*.json"))
            self.assertEqual(len(blobs), 1)
            shared_bytes = blobs[0].stat().st_size
            retained_bytes = shared_bytes + sum(Path(reference["path"]).stat().st_size for _, reference in admissions)
            legacy_bytes = sum(len(qualification._bounded_bytes({"protocol": "synthetic-policy-admission-v1",
                "assurance": qualification._POLICY_ASSURANCE, "admitted_at": "2026-11-03T00:00:00+00:00", **facts}))
                for facts in semantic_facts)
            self.assertLess(retained_bytes, legacy_bytes)
            self.assertLess(len(blobs[0].read_bytes()) + sum(len(qualification._bounded_bytes(record))
                for record, _ in admissions), legacy_bytes)
            stored_bytes = len(blobs[0].read_bytes()) + sum(Path(reference["path"]).stat().st_size
                for _, reference in admissions)
            print(f"policy_facts_storage_bytes legacy_v1={legacy_bytes} shared_blob={shared_bytes} "
                  f"admission_records={stored_bytes - shared_bytes} total_v2={stored_bytes} saved={legacy_bytes - stored_bytes}")

    def test_shared_payload_rejects_corruption_missing_blob_unknown_schema_and_wrong_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            project = fixture / "project"
            owner = project / "out/validation/linux-debug/campaign"
            build = project / "out/build/linux-debug"
            owner.mkdir(parents=True)
            build.mkdir(parents=True)
            campaign = SimpleNamespace(directory=owner)
            facts = self.facts(owner, "product")
            record, path = qualification._policy_encode_admission(campaign, facts,
                "2026-11-03T00:00:00+00:00", admission_root=build)
            reference = self.write_admission(record, path)
            blob = Path(record["shared_facts"]["path"])
            original = blob.read_bytes()
            blob.write_bytes(original + b"tamper")
            with self.assertRaisesRegex(ValueError, "policy evidence changed"):
                self.read_admission(reference, facts)
            blob.write_bytes(original)
            blob.unlink()
            with self.assertRaises((ValueError, OSError)):
                self.read_admission(reference, facts)
            blob.write_bytes(original)
            malformed = dict(record, unknown=True)
            bad = self.write_admission(malformed, fixture / "unknown.json")
            with self.assertRaisesRegex(ValueError, "malformed policy admission v2"):
                self.read_admission(bad, facts)
            wrong_owner = dict(record)
            wrong_owner["shared_facts"] = dict(record["shared_facts"],
                path=str(fixture / "sibling" / "policy-facts" / (record["shared_facts"]["sha256"] + ".json")))
            wrong = self.write_admission(wrong_owner, fixture / "wrong-owner.json")
            with self.assertRaisesRegex(ValueError, "outside its campaign owner"):
                self.read_admission(wrong, facts)

    def test_flat_retained_shared_payload_uses_logical_owner_and_strict_physical_leaf(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            owner = r"C:\r4-fixture\project\out\validation\linux-debug\qualification"
            record = {"protocol": "synthetic-policy-facts-v1", "owner_campaign": owner,
                      "candidate": {"root": owner, "head": "1" * 40, "tree": "2" * 40, "files": {}},
                      "producer": {"external": {"root": owner, "files": {}}, "reader": {}, "modules": {},
                                   "supervisor": {}, "native_supervisor": {}, "cache_policy": {}, "cache_absent": {}},
                      "support": {"tooling": {"path": "C:\\support\\tooling.json", "sha256": "3" * 64},
                                  "tooling_stage": {}, "preparation": {"identity": "4" * 64, "stage": {}},
                                  "quality_stage": {}, "quality": {"path": "C:\\support\\quality.json", "sha256": "5" * 64}}}
            content = qualification._bounded_bytes(record)
            checksum = hashlib.sha256(content).hexdigest()
            logical = owner + "\\policy-facts\\" + checksum + ".json"
            physical = fixture / "shared-facts.json"
            physical.write_bytes(content)
            def resolver(value):
                if str(value) == logical:
                    return physical
                raise FileNotFoundError(value)
            loaded = qualification._policy_read_shared_facts({"path": logical, "sha256": checksum}, owner, resolver)
            self.assertEqual(loaded, record)
            self.assertFalse(Path(owner).exists())
            if os.name == "posix":
                target = fixture / "target.json"
                target.write_bytes(content)
                physical.unlink()
                physical.symlink_to(target)
                with self.assertRaisesRegex(ValueError, "retargeted"):
                    qualification._policy_read_shared_facts({"path": logical, "sha256": checksum}, owner, resolver)

    def test_original_v1_envelope_keeps_canonical_bytes_and_nested_stage_compatibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            owner = Path(temporary) / "project/out/validation/linux-debug/campaign"
            owner.mkdir(parents=True)
            facts = self.facts(owner, "product")
            original_path = Path(facts["original"]["path"])
            document = json.loads(original_path.read_text())
            document["protocol"] = "product-evidence-policy-v1"
            document["stage"]["result"]["identity"] = document["facts"]["stage_identity"]
            document["stage"]["legacy_note"] = "older readers permitted this nested field"
            original_path.write_bytes(qualification._bounded_bytes(document))
            reference = qualification._bounded_reference(original_path)
            decoded, version = qualification._policy_decode_contract_envelope(reference)
            self.assertEqual(version, "product-evidence-policy-v1")
            self.assertEqual(decoded["stage"]["legacy_note"], document["stage"]["legacy_note"])
            original_path.write_text(json.dumps(document, indent=2) + "\n")
            noncanonical = qualification._bounded_reference(original_path)
            with self.assertRaisesRegex(ValueError, "malformed prospective policy original"):
                qualification._policy_decode_contract_envelope(noncanonical)

    def test_v2_admission_v1_original_owner_uses_full_reader_contract_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            owner = project / "out/validation/linux-debug/campaign"
            build = project / "out/build/linux-debug"
            owner.mkdir(parents=True)
            build.mkdir(parents=True)
            facts = self.facts(owner, "product")
            legacy_document = json.loads(Path(facts["original"]["path"]).read_text())
            legacy_document["protocol"] = "product-evidence-policy-v1"
            legacy_document["stage"]["result"]["identity"] = legacy_document["facts"]["stage_identity"]
            legacy_path = owner / "legacy-original.json"
            legacy_path.write_bytes(qualification._bounded_bytes(legacy_document))
            facts["original"] = qualification._bounded_reference(legacy_path)
            facts["owner"]["execution_contract"] = facts["original"]
            record, admission_path = qualification._policy_encode_admission(
                SimpleNamespace(directory=owner), facts, "2026-11-03T00:00:00+00:00", admission_root=build)
            reference = self.write_admission(record, admission_path)
            contract = {"facts": legacy_document["facts"], "stage": legacy_document["stage"]}
            with patch.object(qualification, "read_policy_contract", return_value={"contract": contract}) as reader:
                restored, metadata = qualification._policy_read_admission(reference)
            reader.assert_called_once_with(facts["original"], resolve=Path)
            self.assertEqual(restored, facts)
            self.assertEqual(metadata["protocol"], "synthetic-policy-admission-v2")

    def test_completion_binds_v2_admission_to_its_exact_creator_group(self):
        original_a = {"path": "/campaign/a.json", "sha256": "a" * 64}
        original_b = {"path": "/campaign/b.json", "sha256": "b" * 64}
        admission_a = {"path": "/build/admission/a.json", "sha256": "c" * 64}
        admission_b = {"path": "/build/admission/b.json", "sha256": "d" * 64}
        metadata = {"protocol": "synthetic-policy-admission-v2"}
        record = {"original": original_a}
        declaration = {"originals": [{"group": "product", "contract": original_a},
                                      {"group": "boundary", "contract": original_b}]}
        retained = {"stages": {"product": {"reuse_admission": admission_a},
                                "boundary": {"reuse_admission": admission_b}}}
        qualification._policy_validate_completion_admission_owner(
            admission_a, metadata, record, declaration, retained)
        retained["stages"]["product"]["reuse_admission"] = admission_b
        with self.assertRaisesRegex(ValueError, "creator wrapper does not retain this exact admission"):
            qualification._policy_validate_completion_admission_owner(
                admission_a, metadata, record, declaration, retained)
        with self.assertRaisesRegex(ValueError, "no unique original group"):
            qualification._policy_validate_completion_admission_owner(
                admission_a, metadata, record, {"originals": [declaration["originals"][1]]}, retained)
        qualification._policy_validate_completion_admission_owner(
            admission_a, {"protocol": "synthetic-policy-admission-v1"}, record,
            {"originals": []}, retained)

    def test_legacy_admission_and_completion_headers_keep_v1_meaning(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            wrapper = folder / "campaign/qualification.json"
            wrapper.parent.mkdir()
            wrapper.write_text("{}\n")
            wrapper_ref = qualification._bounded_reference(wrapper)
            record = {"protocol": "synthetic-policy-admission-v1", "assurance": qualification._POLICY_ASSURANCE,
                      "admitted_at": "2026-11-03T00:00:00+00:00", "declaration": {"old": 1},
                      "original": {"old": 2}, "current": {"old": 3}, "comparison": {"old": 4},
                      "support": {"old": 5}, "owner": {"old": 6}, "chronology": {"old": 7}}
            admission_path = folder / "build/verification-admissions/legacy/admission.json"
            admission_path.parent.mkdir(parents=True)
            admission_path.write_text(json.dumps(record, indent=2) + "\n")
            admission = qualification._bounded_reference(admission_path)
            semantic, metadata = qualification._policy_read_admission(admission)
            self.assertEqual(semantic, {key: value for key, value in record.items()
                                        if key not in {"protocol", "assurance", "admitted_at"}})
            self.assertEqual(metadata["protocol"], "synthetic-policy-admission-v1")
            completed = {"protocol": "synthetic-policy-completion-v1", "admission": admission, "wrapper": wrapper_ref}
            self.assertTrue(qualification._policy_validate_completion_header(admission, metadata, completed))
            unknown = {**completed, "shared_facts": {"path": "unused", "sha256": "0" * 64}}
            with self.assertRaisesRegex(ValueError, "malformed or conflicting policy completion"):
                qualification._policy_validate_completion_header(admission, metadata, unknown)

    def test_command_receipt_identity_change_between_retained_and_decoded_reads_is_rejected(self):
        import execution
        from execution import write_command_result

        with tempfile.TemporaryDirectory() as temporary:
            owner = Path(temporary)
            receipt = owner / "attempts/product-run/result.json"
            receipt.parent.mkdir(parents=True)
            identity = {"command": ["child"], "inputs": {"input.txt": "a" * 64},
                        "tools": {"/usr/bin/child": "b" * 64}, "tool_paths": ["/usr/bin/child"],
                        "environment": "c" * 64, "dependencies": []}
            write_command_result(receipt, {"stage": "product", "status": "passed", "identity": identity})
            retained = json.loads(receipt.read_text())
            self.assertTrue(qualification._policy_receipt_decode_matches(retained, read_json(receipt), receipt))
            tools_reference = retained["identity"]["tools_ref"]
            tools_blob = owner / "identities" / (tools_reference["sha256"] + ".json")
            tools_bytes = tools_blob.read_bytes()
            tools_document = json.loads(tools_bytes)
            tools_document["values"]["/usr/bin/child"] = "e" * 64
            tools_blob.write_bytes(execution._identity_map_bytes(tools_document["values"]))
            try:
                with self.assertRaisesRegex(ValueError, "content hash changed"):
                    read_json(receipt)
            finally:
                tools_blob.write_bytes(tools_bytes)
            for field, replacement in (("command", ["other-child"]), ("environment", "d" * 64)):
                changed = json.loads(json.dumps(retained))
                if field == "command":
                    changed["identity"][field][0] = replacement[0]
                else:
                    changed["identity"][field] = replacement
                receipt.write_bytes(qualification._bounded_bytes(changed))
                with self.subTest(field=field):
                    self.assertFalse(qualification._policy_receipt_decode_matches(retained, read_json(receipt), receipt))


class PolicyCollectionTests(unittest.TestCase):
    def setUp(self):
        import shutil
        self.preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        parent = Path(os.environ.get("JUICER_TEST_ARTIFACT_DIR", ROOT / "out/validation" / self.preset)) / "checkpoint-controls"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="policy-", dir=parent, delete=False)
        self.folder = Path(self.temporary.name).resolve()
        self.fixture_holds = set()
        self.retain = bool(os.environ.get("JUICER_TEST_RETAIN_FIXTURES"))
        self.root = self.folder / "project"
        self.root.mkdir()
        record_fixture(self.folder, self, "running", owner_root=self.root)
        def finish():
            outcome = record_fixture(self.folder, self, owner_root=self.root)
            blockers = current_fixture_blockers(self.root)
            if (outcome == "passed" and not self.retain
                    and not os.environ.get("JUICER_TEST_RETAIN_FIXTURES")
                    and not self.fixture_holds and not blockers):
                try:
                    self.temporary.cleanup()
                except BaseException:
                    record_fixture(self.folder, self, "failed", owner_root=self.root)
                    raise
            else:
                print("Retained policy fixture:", self.folder, file=sys.stderr, flush=True)
        self.addCleanup(finish)
        for name, template in qualification._policy_templates().items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(template, path)
        (self.root / "input.txt").write_bytes(b"value=7\n")
        (self.root / ".gitignore").write_bytes(b"out/\n__pycache__/\n")
        (self.root / "scripts/qualification.py").write_bytes(b"# Fixed policy fixture metadata; the selected reader is actual code.\n")
        reader = (ROOT / "scripts/build_evidence.py").read_text(encoding="utf-8")
        (self.root / "scripts/build_evidence.py").write_bytes(qualification._policy_reader_before(reader).encode("utf-8"))
        compiler = shutil.which("clang-cl.exe") if os.name == "nt" else "/usr/bin/gcc-13"
        self.assertIsNotNone(compiler)
        write_json(self.root / "CMakePresets.json", qualification._policy_presets(self.root, compiler))
        subprocess.run(["git", "init", "-q", str(self.root)], check=True, capture_output=True)
        subprocess.run(["git", "-c", "core.autocrlf=false", "add", "."], cwd=self.root, check=True, capture_output=True)
        subprocess.run(["git", "-c", "user.name=Control", "-c", "user.email=control@example.invalid",
                        "-c", "commit.gpgsign=false", "commit", "-qm", "prospective policy original"],
                       cwd=self.root, check=True, capture_output=True)

    @unittest.skipUnless(os.name == "nt", "the symbol-output contract is Windows-specific")
    def test_prospective_policy_recipe_disables_pdb_generation_and_keeps_v1_reader_recipe(self):
        import shutil

        compiler = shutil.which("clang-cl.exe")
        self.assertIsNotNone(compiler)
        current = qualification._policy_presets(self.root, compiler)
        previous = qualification._policy_presets(self.root, compiler, recipe=qualification._POLICY_RECIPE_V1)
        for presets, expect_symbols in ((current, False), (previous, True)):
            rows = {row["name"]: row["cacheVariables"] for row in presets["configurePresets"]}
            for name in ("windows-clang-debug", "windows-clang-release"):
                variables = rows[name]
                compiler_flags = variables["CMAKE_C_FLAGS_DEBUG"] + " " + variables["CMAKE_C_FLAGS_RELEASE"]
                linker_flags = variables["CMAKE_EXE_LINKER_FLAGS_DEBUG"] + " " + variables["CMAKE_EXE_LINKER_FLAGS_RELEASE"]
                self.assertEqual("/Z7" in compiler_flags, expect_symbols)
                self.assertEqual("/DEBUG" in linker_flags, expect_symbols)
        self.assertEqual(qualification._policy_policy()["recipe"], "native-c11-policy-miniature-v2")
        self.assertEqual(qualification._policy_policy(qualification._POLICY_RECIPE_V1)["recipe"],
                         qualification._POLICY_RECIPE_V1)
        environment = dict(os.environ)
        for preset in ("windows-clang-debug", "windows-clang-release"):
            with self.subTest(preset=preset):
                build = self.root / "out/build" / preset
                campaign = Campaign(self.root, preset, "v2-no-pdb-" + preset, ["prepare", "child"])
                qualification.prepare_build(campaign, "prepare", source=self.root, build=build,
                    inputs=qualification.checkpoint_inputs(self.root, build, "prepare"), timeout=120,
                    targets=["BoundedEvidenceChild"], configure_args=["-DBUILD_TESTING=ON"],
                    env=environment, checkpoint=True)
                binary = build / "BoundedEvidenceChild.exe"
                counter = build / "no-pdb-child-counter.txt"
                with OutputLease(self.root, build):
                    receipt = campaign.run("child", [str(binary), str(self.root / "input.txt"), str(counter)],
                        inputs=["input.txt"], tools=[str(binary)], env=environment, timeout=30)
                result = read_json(receipt)
                transcript = (self.root / result["output"]).read_text(encoding="utf-8")
                self.assertEqual(result["status"], "passed")
                self.assertEqual(counter.read_text(encoding="utf-8"), "1\n")
                self.assertIn("testing=ON", transcript)
                self.assertIn("result=14", transcript)
                pdb_files = [path for path in build.rglob("*")
                             if path.is_file() and path.suffix.casefold() == ".pdb"]
                self.assertEqual(pdb_files, [], f"v2 emitted PDB files under {build}: {pdb_files}")

    def test_policy_reader_source_is_one_immutable_blob_per_owner_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            owner_root = Path(temporary) / "project"
            owner_root.mkdir()
            receipt = owner_root / "out/validation/first/attempts/product/attempt/result.json"
            receipt.parent.mkdir(parents=True)
            receipt.write_bytes(b"{}\n")
            payload = b"original reader bytes\xff\n"
            first = qualification._write_policy_reader_source(owner_root, payload)
            other_receipt = owner_root / "out/validation/second/attempts/boundary/other/result.json"
            other_receipt.parent.mkdir(parents=True)
            other_receipt.write_bytes(b"{}\n")
            second = qualification._write_policy_reader_source(owner_root, payload)
            self.assertEqual(first, second)
            self.assertEqual(len(list((owner_root / "out/policy-reader-sources").glob("*.py"))), 1)
            self.assertEqual(Path(first["path"]).read_bytes(), payload)
            Path(first["path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "existing policy reader-source blob differs"):
                qualification._write_policy_reader_source(owner_root, payload)

    def mode(self, mode, *, root=None, success=True):
        root = root or self.root
        bootstrap = ("import importlib.util,pathlib,runpy,sys; "
            "source=pathlib.Path(sys.argv[1]); root=pathlib.Path(sys.argv[2]); "
            "sys.path.insert(0,str(source/'scripts')); "
            "spec=importlib.util.spec_from_file_location('build_evidence',root/'scripts/build_evidence.py'); "
            "module=importlib.util.module_from_spec(spec); sys.modules['build_evidence']=module; spec.loader.exec_module(module); "
            "scope=runpy.run_path(str(source/'tests/quality/test_checkpoint.py'),run_name='policy_fixture_helpers'); "
            "scope['policy_fixture_mode'](root,sys.argv[3])")
        log = self.folder / (mode.replace(":", "-") + ".log")
        with log.open("wb") as output:
            result = subprocess.run([sys.executable, "-B", "-c", bootstrap, str(ROOT), str(root), mode],
                                    stdout=output, stderr=subprocess.STDOUT, check=False)
        if success:
            self.assertEqual(result.returncode, 0, log.read_text(errors="replace")[-20000:])
        else:
            self.assertNotEqual(result.returncode, 0)
        return json.loads((root / "out/policy-fixture.json").read_text()) if (root / "out/policy-fixture.json").exists() else None

    def repair(self):
        (self.root / "scripts/build_evidence.py").write_bytes((ROOT / "scripts/build_evidence.py").read_bytes())

    def handoff(self, *, original=False):
        path = Path(os.environ["JUICER_POLICY_F21_HANDOFF"]).resolve(strict=True)
        record = qualification.identity_json(path.read_text())
        if set(record) != {"protocol", "root", "fixture", "data"} or record["protocol"] != "policy-f21-handoff-v1" or path.parent.name != "logs":
            raise ValueError("malformed retained F21 handoff")
        def local(name):
            return qualification.mapped_windows_path(name) if os.name == "posix" else Path(name)
        root, fixture = local(record["root"]).resolve(strict=True), local(record["fixture"]).resolve(strict=True)
        if (root != fixture / "project" or fixture.parent != path.parent.parent / "checkpoint-controls"
                or not fixture.name.startswith("policy-") or record["data"]["root"] != record["root"]
                or record["data"].get("actual_native") != "win-amd64"):
            raise ValueError("retained F21 root is outside its owned synthetic fixture")
        fixture_outcome = qualification.identity_json((fixture / MARKER).read_text(encoding="utf-8"))
        if (local(fixture_outcome.get("fixture", "")).resolve(strict=True) != fixture
                or fixture_outcome.get("outcome") != "passed"
                or fixture_outcome.get("holds") != ["policy-f21-downstream", "policy-f21-review"]
                or fixture_outcome.get("blocked_owners") != []):
            raise ValueError("retained F21 fixture lacks its passed consumer and review holds")
        expected = record["data"]["candidate"] if original else record["data"].get("current_candidate", record["data"]["candidate"])
        observed = qualification._policy_candidate(root)
        observed["root"] = expected["root"]
        if observed != expected:
            raise ValueError("retained F21 fixed source candidate drifted before fixture access")
        return path, record, root

    def original_bytes(self, data):
        paths = []
        for reference in data["originals"].values():
            document = json.loads(Path(reference["path"]).read_text())
            paths += [Path(reference["path"]), Path(document["producer_source"]["path"]),
                      *[Path(row["path"]) for row in document["execution"].values()]]
        return {str(p): p.read_bytes() for p in paths}

    def assert_counters(self, data):
        self.assertEqual({path: Path(path).read_bytes().hex() for path in data["counters"]}, data["counters"])
        self.assertTrue(all(bytes.fromhex(raw) == b"1\n" for raw in data["counters"].values()))

    def test_native_policy_composes_without_replay_and_rejects_role_context_faults(self):
        data = self.mode("original")
        originals = self.original_bytes(data)
        reference = data["originals"][self.preset + ":product"]
        contract_path = Path(reference["path"])
        contract_bytes = contract_path.read_bytes()
        stored_contract = json.loads(contract_bytes)
        self.assertEqual(stored_contract["protocol"], "product-evidence-policy-v2")
        self.assertIn("stage_identity", stored_contract["facts"])
        self.assertNotIn("identity", stored_contract["stage"]["result"])
        decoded_contract = qualification.read_policy_contract(reference)["contract"]
        raw_command = read_json(decoded_contract["execution"]["receipt"]["path"])
        self.assertEqual(decoded_contract["facts"]["stage_identity"]["inputs"], raw_command["identity"]["inputs"])
        self.assertEqual({key: value for key, value in decoded_contract["facts"]["stage_identity"]["tools"].items()
                          if value is not None}, raw_command["identity"]["tools"])
        self.assertIsNot(decoded_contract["facts"]["stage_identity"], decoded_contract["stage"]["result"]["identity"])
        independent_identity = json.loads(json.dumps(decoded_contract["stage"]["result"]["identity"]))
        decoded_contract["facts"]["stage_identity"]["inputs"]["mutation-probe"] = "f" * 64
        self.assertEqual(decoded_contract["stage"]["result"]["identity"], independent_identity)
        legacy_size = len(qualification._bounded_bytes(decoded_contract))
        self.assertLess(len(contract_bytes), legacy_size)
        raw_receipt_path = Path(decoded_contract["execution"]["receipt"]["path"])
        raw_receipt_bytes = raw_receipt_path.read_bytes()
        raw_receipt_path.write_bytes(raw_receipt_bytes + b"tamper")
        try:
            with self.assertRaises((ValueError, OSError)):
                qualification.read_policy_contract(reference)
        finally:
            raw_receipt_path.write_bytes(raw_receipt_bytes)
        for mutation in ("binary", "input", "counter", "working-directory", "extra-argument", "duplicate-property",
                         "missing-runnable", "extra-runnable"):
            document = json.loads(contract_bytes)
            registration = document["facts"]["subject"]["registration"]
            if mutation in {"binary", "input", "counter"}:
                operand = {"binary": 0, "input": 1, "counter": 2}[mutation]
                path = Path(registration["command"][operand])
                registration["command"][operand] = str(path.with_name("wrong-" + path.name))
            elif mutation == "working-directory":
                next(row for row in registration["properties"] if row["name"] == "WORKING_DIRECTORY")["value"] = str(self.root)
            elif mutation == "extra-argument":
                registration["command"].append("--unexpected")
            elif mutation == "duplicate-property":
                registration["properties"].append(dict(registration["properties"][0]))
            else:
                files = document["facts"]["subject"]["dependencies"]["files"]
                if mutation == "missing-runnable":
                    row = files[str(Path(registration["command"][0]))]
                    row["executable"] = False
                    row.pop("execute_bits", None)
                else:
                    cmake = next(path for path, row in files.items() if not row["executable"]
                                 and Path(path).name.casefold() in {"cmake", "cmake.exe"})
                    files[cmake]["executable"] = True
                    if os.name == "posix":
                        files[cmake]["execute_bits"] = Path(cmake).stat().st_mode & 0o111
            document["seals"] = dict.fromkeys(("entry", "terminal"), qualification._bounded_hash(document["facts"]))
            contract_path.write_bytes(qualification._bounded_bytes(document))
            try:
                changed = {"path": str(contract_path), "sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest()}
                diagnostic = ("policy runnable role membership differs" if mutation in {"missing-runnable", "extra-runnable"}
                              else "historical policy registration/command differs")
                with self.subTest(registration=mutation), self.assertRaisesRegex(ValueError, diagnostic):
                    qualification.read_policy_contract(changed)
            finally:
                contract_path.write_bytes(contract_bytes)
        # Ordinary policy receipts exercise the same collector before the transition.
        self.assertEqual(qualification.collect_qualification(self.root, data["ordinary_paths"])["status"], "passed")
        roles = [(index, group) for index, path in enumerate(data["ordinary_paths"])
                 for group in json.loads(Path(path).read_text())["selection"]]
        self.assertEqual(len(roles), 13)
        for index, group in roles:
            records = [json.loads(Path(p).read_text()) for p in data["ordinary_paths"]]
            record = records[index]
            record["selection"].remove(group)
            for lane in ("stages", "membership", "runtime_membership"):
                record[lane].pop(group, None)
            if group == "product":
                for extra in ("prepare", "quality"):
                    record["stages"].pop(extra)
            record["required"] = list(record["stages"])
            path = self.folder / ("missing-" + str(index) + "-" + group + ".json")
            write_json(path, record)
            with self.subTest(missing=(index, group)), self.assertRaises(ValueError):
                qualification.collect_qualification(self.root, [str(path) if j == index else p for j, p in enumerate(data["ordinary_paths"])])
            duplicate = json.loads(Path(data["ordinary_paths"][index]).read_text())
            duplicate["selection"].append(group)
            target = self.folder / ("duplicate-" + str(index) + "-" + group + ".json")
            write_json(target, duplicate)
            with self.subTest(duplicate=(index, group)), self.assertRaises(ValueError):
                qualification.collect_qualification(self.root, [str(target) if j == index else p for j, p in enumerate(data["ordinary_paths"])])
        for index, path in enumerate(data["ordinary_paths"]):
            for selection in ([p for j, p in enumerate(data["ordinary_paths"]) if j != index], [*data["ordinary_paths"], path]):
                with self.subTest(context=index), self.assertRaises(ValueError):
                    qualification.collect_qualification(self.root, selection)
        extra = json.loads(Path(data["ordinary_paths"][1]).read_text())
        extra["selection"].append("tooling")
        extra["stages"]["tooling"] = data["original_stages"][extra["preset"]]["tooling"]
        extra["required"].append("tooling")
        extra["membership"]["tooling"] = ["Tooling.Miniature"]
        extra["runtime_membership"]["tooling"] = {"host": ["Tooling.Miniature"]}
        target = self.folder / "extra-role.json"
        write_json(target, extra)
        with self.assertRaisesRegex(ValueError, "membership"):
            qualification.collect_qualification(self.root, [str(target) if j == 1 else p for j, p in enumerate(data["ordinary_paths"])])
        host = json.loads(Path(data["ordinary_paths"][0]).read_text())
        host["runtime"] = "host"
        target = self.folder / "host-only.json"
        write_json(target, host)
        with self.assertRaises(ValueError):
            qualification.collect_qualification(self.root, [str(target), *data["ordinary_paths"][1:]])
        self.repair()
        data = self.mode("support")
        self.mode("injection")
        data = self.mode("admit")
        for wrapper_path in data["admitted_paths"]:
            wrapper = json.loads(Path(wrapper_path).read_text())
            grouped_refs = [(name, result["reuse_admission"]) for name, result in wrapper["stages"].items()
                            if result.get("reuse_admission")]
            shared_refs = set()
            for name, reference in grouped_refs:
                stored = json.loads(Path(reference["path"]).read_text())
                self.assertEqual(stored["protocol"], "synthetic-policy-admission-v2")
                self.assertNotIn("owner", stored)
                self.assertEqual(Path(stored["owner_campaign"]), Path(wrapper_path).parent)
                self.assertEqual(set(stored["current"]), {"policy", "context", "subject", "stage_identity", "counter"})
                self.assertEqual(stored["current_preparation"]["mode"], "override" if name == "production" else "shared")
                shared_refs.add((stored["shared_facts"]["path"], stored["shared_facts"]["sha256"]))
                completion = json.loads(Path(reference["path"]).with_name("completion.json").read_text())
                self.assertEqual(completion["protocol"], "synthetic-policy-completion-v2")
                self.assertEqual(completion["shared_facts"], stored["shared_facts"])
            self.assertLessEqual(len(shared_refs), 1)
        result = qualification.collect_qualification(self.root, data["admitted_paths"])
        self.assertEqual(result["status"], "passed")
        self.assert_counters(data)
        self.assertEqual({path: Path(path).read_bytes() for path in originals}, originals)
        self.assertEqual(len(data["admitted_paths"]), 6)
        self.assertEqual({row["provenance"] for row in result["stages"]}, {"executed", "exact-reused", "admitted"})
        for path in data["admitted_paths"]:
            wrapper = json.loads(Path(path).read_text())
            self.assertTrue(any(value == "admitted" for value in wrapper["stage_provenance"].values()))

    def test_v2_shared_admissions_complete_and_collect_once(self):
        data = self.mode("original")
        self.repair()
        self.mode("support")
        data = self.mode("admit")
        prior_admissions = {}
        for wrapper_path in data["admitted_paths"]:
            wrapper = json.loads(Path(wrapper_path).read_text())
            shared_refs = set()
            grouped_refs = []
            for name, result in wrapper["stages"].items():
                reference = result.get("reuse_admission")
                if not reference:
                    continue
                grouped_refs.append((name, reference))
                stored = json.loads(Path(reference["path"]).read_text())
                self.assertEqual(stored["protocol"], "synthetic-policy-admission-v2")
                self.assertNotIn("owner", stored)
                self.assertEqual(Path(stored["owner_campaign"]), Path(wrapper_path).parent)
                self.assertEqual(stored["current_preparation"]["mode"],
                                 "override" if name == "production" else "shared")
                shared_refs.add((stored["shared_facts"]["path"], stored["shared_facts"]["sha256"]))
                completion = json.loads(Path(reference["path"]).with_name("completion.json").read_text())
                self.assertEqual(completion["protocol"], "synthetic-policy-completion-v2")
                self.assertEqual(completion["shared_facts"], stored["shared_facts"])
                prior_admissions[(Path(wrapper_path).parent.name, name)] = {
                    "reference": reference,
                    "owner_campaign": stored["owner_campaign"],
                    "admission": Path(reference["path"]).read_bytes(),
                    "completion": Path(reference["path"]).with_name("completion.json").read_bytes(),
                    "shared_path": stored["shared_facts"]["path"],
                    "shared": Path(stored["shared_facts"]["path"]).read_bytes(),
                }
            self.assertTrue(grouped_refs, f"wrapper had no admissions: {wrapper_path}")
            self.assertEqual(len(shared_refs), 1)
        collected = qualification.collect_qualification(self.root, data["admitted_paths"])
        self.assertEqual(collected["status"], "passed")
        self.assertIn("admitted", {row["provenance"] for row in collected["stages"]})
        self.assert_counters(data)
        for wrapper_path in data["admitted_paths"]:
            wrapper = json.loads(Path(wrapper_path).read_text())
            name, result = next((name, result) for name, result in wrapper["stages"].items()
                                if result.get("reuse_admission"))
            old_reference = result["reuse_admission"]
            semantic, metadata = qualification._policy_read_admission(old_reference)
            self.assertEqual(metadata["protocol"], "synthetic-policy-admission-v2")
            legacy_record = {"protocol": "synthetic-policy-admission-v1", "assurance": qualification._POLICY_ASSURANCE,
                             "admitted_at": json.loads(Path(old_reference["path"]).read_text())["admitted_at"],
                             **semantic}
            preset = wrapper["preset"]
            build = self.root / "out/build" / preset
            legacy_path = build / "verification-admissions" / qualification._bounded_hash(semantic) / "admission.json"
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
            legacy_path.write_bytes(qualification._bounded_bytes(legacy_record))
            legacy_reference = qualification._bounded_reference(legacy_path)
            legacy_completion = {"protocol": "synthetic-policy-completion-v1", "admission": legacy_reference,
                                 "wrapper": qualification._bounded_reference(Path(wrapper_path))}
            legacy_completion_path = legacy_path.with_name("completion.json")
            legacy_completion_path.write_bytes(qualification._bounded_bytes(legacy_completion))
            state_path = build / "verification-checkpoint.json"
            state = json.loads(state_path.read_text())
            state[name]["reuse_admission"] = legacy_reference
            write_json(state_path, state)
            prior = prior_admissions[(Path(wrapper_path).parent.name, name)]
            prior.update(reference=legacy_reference, admission=legacy_path.read_bytes(),
                         completion=legacy_completion_path.read_bytes(), protocol="synthetic-policy-admission-v1")
        reused = self.mode("admit:reuse")
        self.assertEqual(len(reused["admitted_paths:reuse"]), len(data["admitted_paths"]))
        for wrapper_path in reused["admitted_paths:reuse"]:
            wrapper = json.loads(Path(wrapper_path).read_text())
            self.assertIn("reuse", Path(wrapper_path).parent.name)
            for name, result in wrapper["stages"].items():
                reference = result.get("reuse_admission")
                if reference is None:
                    continue
                prior_campaign = Path(wrapper_path).parent.name.replace("-reuse", "")
                prior = prior_admissions[(prior_campaign, name)]
                self.assertEqual(reference, prior["reference"])
                self.assertEqual(Path(reference["path"]).read_bytes(), prior["admission"])
                self.assertEqual(Path(reference["path"]).with_name("completion.json").read_bytes(), prior["completion"])
                stored = json.loads(Path(reference["path"]).read_text())
                self.assertEqual(stored["protocol"], prior.get("protocol", "synthetic-policy-admission-v2"))
                if stored["protocol"] == "synthetic-policy-admission-v1":
                    self.assertNotIn("owner_campaign", stored)
                else:
                    self.assertEqual(stored["owner_campaign"], prior["owner_campaign"])
                    self.assertEqual(Path(prior["shared_path"]).read_bytes(), prior["shared"])
        reused_collection = qualification.collect_qualification(self.root, reused["admitted_paths:reuse"])
        self.assertEqual(reused_collection["status"], "passed")
        self.assertIn("admitted", {row["provenance"] for row in reused_collection["stages"]})
        self.assert_counters(reused)

    def test_wrong_preset_corpus_reaches_placement_for_ordinary_and_admitted(self):
        data = self.mode("original")
        for preset in ("windows-clang-debug", "linux-release", "windows-clang-release"):
            paths = []
            for index, original_path in enumerate(data["ordinary_paths"]):
                record = json.loads(Path(original_path).read_text())
                if record["preset"] == "linux-debug":
                    record["selection"].remove("corpus")
                    for key in ("stages", "membership", "runtime_membership"):
                        record[key].pop("corpus", None)
                if record["preset"] == preset and "production" not in record["selection"]:
                    record["selection"].append("corpus")
                    record["stages"]["corpus"] = data["original_stages"][preset]["corpus"]
                    record["membership"]["corpus"] = ["Policy.corpus"]
                    record["runtime_membership"]["corpus"] = {"host": ["Policy.corpus"]}
                record["required"] = list(record["stages"])
                target = self.folder / ("ordinary-" + preset + "-" + str(index) + ".json")
                write_json(target, record)
                paths.append(str(target))
            self.assertEqual(sum("corpus" in json.loads(Path(p).read_text())["selection"] for p in paths), 1)
            with self.subTest(kind="ordinary", preset=preset), self.assertRaisesRegex(ValueError, "corpus evidence is required only on Linux Debug"):
                qualification.collect_qualification(self.root, paths)
        self.repair()
        self.mode("support")
        for preset in ("windows-clang-debug", "linux-release", "windows-clang-release"):
            data = self.mode("admit:" + preset)
            paths = data["admitted_paths:" + preset]
            self.assertEqual(len(paths), 6)
            self.assertEqual(sum("corpus" in json.loads(Path(p).read_text())["selection"] for p in paths), 1)
            with self.subTest(kind="admitted", preset=preset), self.assertRaisesRegex(ValueError, "corpus evidence is required only on Linux Debug"):
                qualification.collect_qualification(self.root, paths)
            self.assert_counters(data)

    def test_off_retention_and_original_barrier_are_required(self):
        data = self.mode("original")
        self.assertEqual(qualification.collect_qualification(self.root, data["ordinary_paths"])["status"], "passed")
        for index in (4, 5):
            path = Path(data["ordinary_paths"][index])
            wrapper = json.loads(path.read_text())
            baseline = path.read_bytes()
            for mutation in ("missing", "early-on", "other-off"):
                damaged = json.loads(json.dumps(wrapper))
                if mutation == "missing":
                    damaged.pop("policy_chronology")
                elif mutation == "early-on":
                    damaged["policy_chronology"]["on"][damaged["preset"]] = {
                        "path": data["off_stages"][damaged["preset"]]["receipt"],
                        "sha256": data["off_stages"][damaged["preset"]]["sha256"]}
                else:
                    damaged["policy_chronology"]["off"][damaged["preset"]] = data["originals"][damaged["preset"] + ":product"]
                path.write_text(json.dumps(damaged))
                try:
                    with self.subTest(mutation=mutation, index=index), self.assertRaises(ValueError):
                        qualification.collect_qualification(self.root, data["ordinary_paths"])
                finally:
                    path.write_bytes(baseline)
            original = json.loads(Path(data["off_stages"][wrapper["preset"]]["execution_contract"]["path"]).read_text())
            off = json.loads(Path(original["facts"]["subject"]["off"]["path"]).read_text())
            configuration = off["files"][str(self.root / "out/build" / wrapper["preset"] / "verification-config-outputs.json")]
            prepared = off["preparation"]["stage"]
            qualification._policy_configuration(self.root, configuration, preparation=prepared)
            registry = json.loads(Path(configuration["path"]).read_text())
            configure = read_json(Path(registry["producer"]))
            vctip_cleanup = configure["identity"].get("vctip_cleanup")
            if os.name == "nt":
                self.assertIsInstance(vctip_cleanup, dict, "Windows configure must bind its VCTIP cleanup executable")
                auxiliary_tools = {vctip_cleanup["executable"]: digest(Path(vctip_cleanup["executable"]))}
            else:
                self.assertIsNone(vctip_cleanup, "POSIX configure must not declare Windows VCTIP cleanup")
                auxiliary_tools = {}
            configure_tools = configure["identity"]["tools"]
            self.assertEqual(configure_tools, auxiliary_tools,
                             "configure tools may contain only its declared VCTIP cleanup executable")
            self.assertFalse(any(Path(name.replace("\\", "/")).stem.casefold() == "cmake"
                                 for name in configure_tools),
                             "configure receipt must exercise its sealed CMake parent binding")
            cmake_tools = [name for name, checksum in prepared["identity"]["tools"].items()
                           if Path(name.replace("\\", "/")).stem.casefold() == "cmake"
                           and checksum == configure["identity"]["executable_sha256"]]
            self.assertEqual(len(cmake_tools), 1)
            cmake = cmake_tools[0]
            other_preset = next(preset for preset in data["off_stages"] if preset != wrapper["preset"])
            other_original = json.loads(Path(data["off_stages"][other_preset]["execution_contract"]["path"]).read_text())
            for mutation in ("missing", "ambiguous", "wrong-owner", "tool-hash", "receipt-hash"):
                anchor = json.loads(json.dumps(prepared))
                if mutation == "missing":
                    anchor = None
                elif mutation == "ambiguous":
                    duplicate = str(self.root / "out/anchor-controls/duplicate" / Path(cmake).name)
                    anchor["identity"]["tools"][duplicate] = configure["identity"]["executable_sha256"]
                elif mutation == "wrong-owner":
                    anchor = other_original["facts"]["preparation"]["stage"]
                elif mutation == "tool-hash":
                    anchor["identity"]["tools"][cmake] = "0" * 64
                else:
                    anchor["sha256"] = "0" * 64
                with self.subTest(parent_binding=mutation, preset=wrapper["preset"]), \
                        self.assertRaisesRegex(ValueError, "unique sealed tool binding|preparation tool anchor|policy evidence changed"):
                    qualification._policy_configuration(self.root, configuration, preparation=anchor)
            if os.name == "posix":
                alias = self.root / "out/anchor-controls" / wrapper["preset"] / "prepare-alias.json"
                alias.parent.mkdir(parents=True, exist_ok=True)
                alias.symlink_to(prepared["receipt"])
                anchor = json.loads(json.dumps(prepared))
                anchor["receipt"] = str(alias)
                try:
                    with self.subTest(parent_binding="retarget", preset=wrapper["preset"]), \
                            self.assertRaisesRegex(ValueError, "policy reference path was retargeted"):
                        qualification._policy_configuration(self.root, configuration, preparation=anchor)
                finally:
                    alias.unlink()
            target = Path(off["binary"]["path"])
            raw = target.read_bytes()
            on_binary = self.root / "out/build" / wrapper["preset"] / ("BoundedEvidenceChild.exe" if os.name == "nt" else "BoundedEvidenceChild")
            self.assertNotEqual(on_binary.read_bytes(), raw, "synthetic OFF/ON definition must affect the native binary")
            target.write_bytes(on_binary.read_bytes())
            try:
                with self.assertRaises(ValueError):
                    qualification.collect_qualification(self.root, data["ordinary_paths"])
            finally:
                target.write_bytes(raw)
        self.assert_counters(data)

    def test_policy_publication_current_failure_and_terminal_mutations_reject(self):
        data = self.mode("original")
        self.repair()
        self.mode("support")
        data = self.mode("admit")
        paths = data["admitted_paths"]
        wrapper = json.loads(Path(paths[0]).read_text())
        owner = self.root / "out/build" / wrapper["preset"] / "verification-checkpoint.json"
        owner_bytes = owner.read_bytes()
        for status in ("pending", "failed", "blocked"):
            state = json.loads(owner_bytes)
            state["product"]["status"] = status
            write_json(owner, state)
            try:
                with self.subTest(status=status), self.assertRaisesRegex(ValueError, "pending|owner"):
                    qualification.collect_qualification(self.root, paths)
            finally:
                owner.write_bytes(owner_bytes)
        state = json.loads(owner_bytes)
        state["product"]["sha256"] = "f" * 64
        write_json(owner, state)
        try:
            with self.assertRaisesRegex(ValueError, "owner|newer"):
                qualification.collect_qualification(self.root, paths)
        finally:
            owner.write_bytes(owner_bytes)
        admission = wrapper["stages"]["product"]["reuse_admission"]
        completion = Path(admission["path"]).with_name("completion.json")
        completion_bytes = completion.read_bytes()
        malformed = json.loads(completion_bytes)
        malformed["unknown"] = True
        completion.write_text(json.dumps(malformed))
        try:
            with self.assertRaisesRegex(ValueError, "completion"):
                qualification.collect_qualification(self.root, paths)
        finally:
            completion.write_bytes(completion_bytes)
        completion.unlink()
        try:
            with self.assertRaises((ValueError, OSError)):
                qualification.collect_qualification(self.root, paths)
        finally:
            completion.write_bytes(completion_bytes)
        original = json.loads(Path(data["originals"][wrapper["preset"] + ":product"]["path"]).read_text())
        for target in (Path(original["execution"]["process"]["path"]), self.root / "input.txt"):
            raw = target.read_bytes()
            validator = qualification.policy_collection_stage
            visits = 0
            def terminal(*args, **kwargs):
                nonlocal visits
                info = validator(*args, **kwargs)
                if args[3] == "product" and args[1].name == wrapper["preset"]:
                    visits += 1
                    if visits == 2:
                        target.write_bytes(raw + b"terminal mutation")
                return info
            try:
                with self.subTest(terminal=str(target)), patch.object(qualification, "policy_collection_stage", side_effect=terminal), \
                        self.assertRaises(ValueError):
                    qualification.collect_qualification(self.root, paths)
                self.assertEqual(visits, 2)
            finally:
                target.write_bytes(raw)
        self.assert_counters(data)
        self.mode("race")
        self.assert_counters(data)

    def test_original_off_barrier_rejects_real_early_on_restoration(self):
        data = self.mode("original-early-on")
        with self.assertRaisesRegex(ValueError, "both original OFF completions must precede either ON restoration"):
            qualification.collect_qualification(self.root, data["ordinary_paths"])
        self.assert_counters(data)

    def test_faulty_prospective_registration_closure_is_not_salvaged_by_repair(self):
        data = self.mode("original-fault")
        before = {path: Path(path).read_bytes() for path in data["counters"]}
        with self.assertRaisesRegex(ValueError, "omitted a generated registration consumer") as failed_collection:
            qualification.collect_qualification(self.root, data["ordinary_paths"])
        write_json(self.root / "out/faulty-original-failed-collection.json", {
            "status": "rejected-faulty-prospective-recorder", "error": str(failed_collection.exception),
            "originals": data["originals"], "scope": "actual native commands passed; injected recorder omitted one registration fact; reader unmocked"})
        # A separate real native child prints the expected result and exits 7.
        # A false historical passed label cannot turn its process proof into success.
        preset = self.preset
        binary = self.root / "out/build" / preset / ("BoundedEvidenceChild.exe" if os.name == "nt" else "BoundedEvidenceChild")
        campaign = Campaign(self.root, preset, "policy-false-pass", ["product"])
        original = json.loads(Path(data["originals"][preset + ":product"]["path"]).read_text())
        with OutputLease(self.root, binary.parent), self.assertRaises(subprocess.CalledProcessError):
            campaign.run("product", [str(binary), str(self.root / "input.txt"), str(campaign.directory / "failed-counter.txt"), "--fail"],
                inputs=list(original["facts"]["stage_identity"]["inputs"]),
                tools=[path for path in original["facts"]["stage_identity"]["tool_paths"] if Path(path).is_file()],
                env=dict(os.environ), timeout=None)
        failed_path = next((campaign.directory / "attempts").glob("product-*/result.json"))
        raw = json.loads(failed_path.read_text())
        self.assertEqual(raw["returncode"], 7)
        self.assertIn("result=14", (self.root / raw["output"]).read_text())
        stored_original = json.loads(Path(data["originals"][preset + ":product"]["path"]).read_text())
        false_claim = json.loads(json.dumps(stored_original))
        false_claim["stage"]["result"].update(receipt=str(failed_path), sha256=hashlib.sha256(failed_path.read_bytes()).hexdigest())
        false_claim["execution"]["receipt"] = {"path": str(failed_path), "sha256": hashlib.sha256(failed_path.read_bytes()).hexdigest()}
        for key, path in (("output", self.root / raw["output"]), ("process", Path(raw["process_tree"]["receipt"]))):
            false_claim["execution"][key] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        false_path = failed_path.with_name("execution-contract.json")
        false_path.write_bytes(qualification._bounded_bytes(false_claim))
        with self.assertRaisesRegex(ValueError, "raw command/process"):
            qualification.read_policy_contract({"path": str(false_path), "sha256": hashlib.sha256(false_path.read_bytes()).hexdigest()})
        self.repair()
        self.mode("support")
        self.mode("admit", success=False)
        failed = (self.folder / "admit.log").read_text(errors="replace")
        self.assertIn("omitted a generated registration consumer", failed)
        self.assertEqual({path: Path(path).read_bytes() for path in before}, before)
        self.assertFalse(list((self.root / "out").glob("build/*/verification-admissions/*/completion.json")))

    @unittest.skipUnless(os.name == "nt" and os.environ.get("JUICER_POLICY_F21_HANDOFF"),
                         "focused actual Windows F21 producer requires its reserved handoff path")
    def test_windows_f21_produces_trustworthy_originals(self):
        handoff = Path(os.environ["JUICER_POLICY_F21_HANDOFF"])
        if handoff.exists() or handoff.parent.resolve() != self.folder.parent.parent / "logs":
            raise ValueError("F21 producer requires a fresh handoff under its owned artifact logs")
        data = self.mode("original")
        handoff.write_text(json.dumps({"protocol": "policy-f21-handoff-v1", "root": str(self.root), "fixture": str(self.folder), "data": data}))
        self.assert_counters(data)
        self.fixture_holds.add("policy-f21-downstream")
        self.fixture_holds.add("policy-f21-review")

    @unittest.skipUnless(os.name == "nt" and os.environ.get("JUICER_POLICY_F21_HANDOFF"),
                         "focused actual Windows F21 continuation requires the retained native original")
    def test_windows_f21_admits_actual_reader_repair_without_replay(self):
        handoff_path, handoff, root = self.handoff(original=True)
        baseline = self.original_bytes(handoff["data"])
        (root / "scripts/build_evidence.py").write_bytes((ROOT / "scripts/build_evidence.py").read_bytes())
        self.mode("support", root=root)
        data = self.mode("admit", root=root)
        self.assert_counters(data)
        self.assertEqual({path: Path(path).read_bytes() for path in baseline}, baseline)
        handoff["data"] = data
        handoff_path.write_text(json.dumps(handoff))

    @unittest.skipUnless(os.name == "posix" and os.environ.get("JUICER_POLICY_F21_HANDOFF")
                         and os.environ.get("JUICER_WINDOWS_PYTHON"), "focused actual Windows originals/WSL old-reader consumer")
    def test_windows_f21_old_reader_fails_foreign_collection_only(self):
        _, _, root = self.handoff(original=True)
        self.mode("collect-old", root=root)
        failure = json.loads((root / "out/f21-failed-collection.json").read_text())
        self.assertIn("registration_files", failure["traceback"])
        self.assertEqual(failure["status"], "failed-as-expected")
        self.assertFalse((root / "out/f21-current-collection.json").exists())

    @unittest.skipUnless(os.name == "posix" and os.environ.get("JUICER_POLICY_F21_HANDOFF")
                         and os.environ.get("JUICER_WINDOWS_PYTHON"), "focused actual Windows originals/WSL repaired-reader consumer")
    def test_windows_f21_repaired_reader_collects_without_replay(self):
        from pathlib import PureWindowsPath
        _, handoff, root = self.handoff()
        def local(path):
            return qualification.mapped_windows_path(path) if PureWindowsPath(path).is_absolute() else Path(path)
        counters = {path: local(path).read_bytes() for path in handoff["data"]["counters"]}
        failure_before = (root / "out/f21-failed-collection.json").read_bytes()
        self.mode("collect-current", root=root)
        aggregate = json.loads((root / "out/f21-current-collection.json").read_text())
        self.assertEqual(aggregate["status"], "passed")
        records = [json.loads(local(path).read_text()) for path in handoff["data"]["admitted_paths"]]
        # Select a genuine alias from sealed admitted facts, independently of
        # both the partition generator and the native reader's response.
        alias = None
        for wrapper in records:
            for stage in wrapper["stages"].values():
                if not stage.get("reuse_admission"):
                    continue
                admitted = json.loads(local(stage["reuse_admission"]["path"]).read_text())
                subject = admitted["current"]["subject"]
                runtime = {PureWindowsPath(path) for path in subject["runtime"].values()}
                for path, row in subject["dependencies"]["files"].items():
                    target = row["resolved"]
                    if path != target and PureWindowsPath(path) == PureWindowsPath(target) and PureWindowsPath(path) not in runtime:
                        alias = path, target, {"resolved": target, "sha256": row["sha256"]}
                        break
                if alias is not None:
                    break
            if alias is not None:
                break
        self.assertIsNotNone(alias, "the real Windows cohort must include a nonruntime lexical case alias")
        lexical, target, binding = alias
        extra = qualification.policy_native_collection_bindings(root, records, local)
        lexical_proofs = {key for key, groups in extra.items() if groups["files"].get(lexical) == binding}
        target_proofs = {key for key, groups in extra.items() if groups["files"].get(target) == binding}
        self.assertTrue(lexical_proofs)
        self.assertTrue(target_proofs)
        self.assertFalse(lexical_proofs & target_proofs)
        for groups in extra.values():
            self.assertEqual(len(groups["files"]), len({PureWindowsPath(path) for path in groups["files"]}))
        entries = aggregate["native_identities"]
        self.assertEqual([entry["phase"] for entry in entries], ["entry", "terminal"])
        native_bytes = {path: Path(path).read_bytes() for entry in entries for path in entry["payloads"]}
        proof = sorted(lexical_proofs)[0]
        for entry in entries:
            raw = json.loads(Path(entry["receipt"]).read_text())
            native = raw["identity"]["windows_identity"]
            request = qualification.read_identity_request(root / native["request"], native["request_sha256"])
            response = json.loads((root / raw["output"]).read_text())
            # Preserve request ordering for the already validated preparation
            # maps; recompute every additional policy map from sealed facts.
            expected = {key: {group: {path: response["proofs"][key][group][path] for path in paths}
                              for group, paths in groups.items()} for key, groups in request["proofs"].items()}
            for key, groups in extra.items():
                self.assertEqual(expected[key], groups)
                expected[key] = groups
            self.assertEqual(qualification.native_identity_result(root, entry, expected), response)
            # These cloned expectations test request/response consumer integrity;
            # they do not represent filesystem mutations or new native reads.
            for mutation in ("missing-alias", "hash", "target"):
                damaged = json.loads(json.dumps(expected))
                row = damaged[proof]["files"][lexical]
                if mutation == "missing-alias":
                    del damaged[proof]["files"][lexical]
                    diagnostic = "command/request does not bind"
                elif mutation == "hash":
                    row["sha256"] = "0" * 64
                    diagnostic = "result does not match original proof maps"
                else:
                    row["resolved"] = str(PureWindowsPath(target).with_name("wrong-" + PureWindowsPath(target).name))
                    diagnostic = "result does not match original proof maps"
                with self.subTest(native_phase=entry["phase"], consumer_integrity=mutation), \
                        self.assertRaisesRegex(ValueError, diagnostic):
                    qualification.native_identity_result(root, entry, damaged)
        self.assertEqual({path: Path(path).read_bytes() for path in native_bytes}, native_bytes)
        self.assertEqual({path: local(path).read_bytes() for path in counters}, counters)
        self.assertEqual((root / "out/f21-failed-collection.json").read_bytes(), failure_before)
        review_closure = self.original_bytes(handoff["data"])
        release_fixture_hold(root.parent, "policy-f21-downstream", self.id(), resolve=local)
        fixture_outcome = read_fixture_marker(root.parent / MARKER)
        if fixture_outcome.get("holds") != ["policy-f21-review"]:
            raise ValueError("F21 review hold did not preserve the original cohort")
        if {path: Path(path).read_bytes() for path in review_closure} != review_closure:
            raise ValueError("F21 review-held original evidence changed after foreign consumption")


class BoundedExecutionTests(unittest.TestCase):
    def setUp(self):
        import shutil
        self.preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        parent = Path(os.environ.get("JUICER_TEST_ARTIFACT_DIR", ROOT / "out/validation" / self.preset))
        parent = parent / "checkpoint-controls"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="bounded-", dir=parent, delete=False)
        self.folder = Path(self.temporary.name).resolve()
        self.root = self.folder / "project"
        self.root.mkdir()
        self.build = self.root / "out/build" / self.preset
        self.build.mkdir(parents=True)
        self.environment = dict(os.environ)
        self.campaign = Campaign(self.root, self.preset, "bounded-original", ["prepare", "product"])
        self.state_path = self.build / "verification-checkpoint.json"
        self.state = {}
        self.lease = OutputLease(self.root, self.build)
        self.lease.__enter__()
        record_fixture(self.folder, self, "running", owner_root=self.root)

        def finish():
            self.lease.__exit__(None, None, None)
            outcome = record_fixture(self.folder, self, owner_root=self.root)
            if outcome == "passed" and not os.environ.get("JUICER_TEST_RETAIN_FIXTURES"):
                try:
                    self.temporary.cleanup()
                except BaseException:
                    record_fixture(self.folder, self, "failed", owner_root=self.root)
                    raise
            else:
                print("Retained bounded evidence fixture:", self.folder, file=sys.stderr, flush=True)
        self.addCleanup(finish)
        for name in ("CMakeLists.txt", "src/evidence_child.c", "input.txt"):
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "tests/quality/evidence" / name, self.root / name)
        (self.root / "scripts").mkdir()
        self.controller = self.root / "scripts/qualification.py"
        self.controller.write_text("# original outer controller\n")
        (self.root / ".gitignore").write_text("out/\n__pycache__/\n")
        cc = shutil.which("clang-cl.exe") if os.name == "nt" else "/usr/bin/gcc-13"
        self.assertIsNotNone(cc, "qualified native C compiler is missing")
        write_json(self.root / "CMakePresets.json", qualification._bounded_presets(self.root, self.preset, cc))
        subprocess.run(["git", "init", "-q", str(self.root)], check=True, capture_output=True)
        subprocess.run(["git", "-c", "core.autocrlf=false", "add", "."], cwd=self.root, check=True, capture_output=True)
        subprocess.run(["git", "-c", "user.name=Control", "-c", "user.email=control@example.invalid",
                        "-c", "commit.gpgsign=false", "commit", "-qm", "bounded fixture original"],
                       cwd=self.root, check=True, capture_output=True)
        self.inputs = sorted([".gitignore", "CMakeLists.txt", "CMakePresets.json", "src/evidence_child.c", "input.txt",
                              "scripts/qualification.py"])
        self.build_inputs = ["CMakeLists.txt", "CMakePresets.json", "src/evidence_child.c"]
        self.build_tools = [str(Path(shutil.which(name)).resolve()) for name in ("cmake", "ninja")]
        self.build_tools.append(str(Path(cc).resolve()))
        self.counter = self.build / "assertion-count.txt"
        self.junit = self.campaign.directory / "product-host.xml"
        self.command = [*qualification.ctest_command(self.root, self.build), "-R", "^(Bounded\\.Read)$", "-LE", "gpu",
                        "--no-tests=error", "--output-on-failure", "--output-junit", str(self.junit)]
        self.prepare()
        binary = self.build / ("BoundedEvidenceChild.exe" if os.name == "nt" else "BoundedEvidenceChild")
        self.tools = sorted(set([*self.build_tools, str(Path(shutil.which("ctest")).resolve()), str(binary),
                                  sys.executable, sys._base_executable,
                                  str(ROOT / "scripts/execution.py"), str(ROOT / "scripts/process_tree.py")]))

    def prepare(self):
        from build_evidence import prepare_build, check_scoped_preparation
        outputs = []
        def validate(path):
            proof = check_scoped_preparation(self.campaign, self.build, path, self.environment)["scoped_preparation"]
            outputs[:] = proof["outputs"]
        return qualification.run_stage(self.campaign, self.state_path, self.state, "prepare",
            ["cmake", "--build", str(self.build)], inputs=self.inputs, tools=self.build_tools,
            environment=self.environment, timeout=None,
            action=lambda: prepare_build(self.campaign, "prepare", source=self.root, build=self.build,
                inputs=self.build_inputs, timeout=None, targets=["all"], configure_args=["-DBUILD_TESTING=ON"],
                env=self.environment, checkpoint=True)[1], validate=validate, payloads=outputs)

    def execute(self, *, capture=True, action=None, repeat_reason=""):
        dependency = qualification.preparation_identity(self.build, self.state["prepare"])
        def run():
            return qualification.run_ctest_parts(self.campaign, "product", [self.command], {"host": ["Bounded.Read"]},
                build=self.build, inputs=self.inputs, tools=self.tools, environment=self.environment, timeout=None,
                repeat_reason=repeat_reason, state=self.state, state_path=self.state_path, payloads=[self.junit])
        return qualification.run_stage(self.campaign, self.state_path, self.state, "product", self.command,
            inputs=self.inputs, tools=self.tools, environment=self.environment, timeout=None,
            semantic_command=[qualification.ctest_identity(self.command)], action=action or run,
            validate=lambda paths: qualification.validate_ctest_stage(paths, {"host": ["Bounded.Read"]},
                ["Bounded.Read"], [qualification.ctest_identity(self.command)]),
            payloads=[self.junit, self.counter], dependencies=[dependency], ctest=True,
            repeat_reason=repeat_reason,
            bounded_capture="native-c11-miniature-v1" if capture else None)

    def current(self):
        return qualification.capture_execution_subject(self.campaign, self.state, self.command,
            inputs=self.inputs, tools=self.tools, environment=self.environment)

    def declaration(self, original, current):
        document = original["contract"]
        old, new = document["candidate"]["files"], current["candidate"]["files"]
        refs = []
        for name in ("tooling", "review", "owner"):
            path = self.campaign.directory / (name + ".json")
            if not path.exists():
                path.write_text(json.dumps({"scope": "fixture integrity reference, not authenticated approval", "name": name}))
            refs.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        changes = [{"path": name, "old_sha256": old[name], "new_sha256": new[name], "groups": ["product"]}
                   for name in sorted(old) if old[name] != new[name]]
        return {"protocol": "tooling-evidence-transition-bounded-v1", "original_candidate": document["candidate"],
                "current_candidate": current["candidate"], "changes": changes,
                "originals": [{"preset": self.preset, "group": "product", "contract": original["reference"]}],
                "compatibility": [{"path": row["path"], "group": "product", "role": "controller-only",
                                    "outcome_integrity": "preserved"} for row in changes],
                "fresh_evidence": {"tooling": [refs[0]], "quality": [], "quality_base": "fixture"},
                "review": refs[1], "owner_decision": refs[2]}

    def test_native_original_and_comparison_preserve_immutable_execution(self):
        result = self.execute()
        self.assertIn("execution_contract", result, result.get("bounded_ineligible"))
        original = qualification.read_execution_contract(result["execution_contract"])
        document = original["contract"]
        self.assertEqual(document["assurance"], "declared-dependencies-v1")
        self.assertEqual(document["stage"]["result"]["identity"]["dependencies"],
                         [document["preparation"]["identity"]])
        self.assertEqual(self.counter.read_bytes(), b"1\n")
        source = self.root / "input.txt"
        self.assertEqual(document["subject"]["dependencies"]["files"][str(source)]["sha256"],
                         hashlib.sha256(b"value=7\n").hexdigest())
        original_bytes = {path: path.read_bytes() for path in Path(result["receipt"]).parent.iterdir() if path.is_file()}
        source_stamp = source.stat()
        os.utime(source, ns=(source_stamp.st_atime_ns + 10_000_000, source_stamp.st_mtime_ns))
        self.controller.write_text("# reviewed controller change\n")
        current = self.current()
        declaration = self.declaration(original, current)
        declaration_path = self.campaign.directory / "transition.json"
        declaration_path.write_text(json.dumps(declaration))
        declaration_ref = {"path": str(declaration_path), "sha256": hashlib.sha256(declaration_path.read_bytes()).hexdigest()}
        self.assertEqual(qualification.read_reuse_declaration(declaration_ref), declaration)
        state_before = self.state_path.read_bytes()
        with patch.object(Campaign, "run", side_effect=AssertionError("comparison executed a command")), \
                patch.object(qualification, "write_json", side_effect=AssertionError("comparison changed state")), \
                patch.object(qualification, "digest", side_effect=AssertionError("comparison read a file")):
            compared = qualification.compare_execution_subject(original, current, declaration)
        self.assertEqual(compared["status"], "compatible")
        self.assertEqual(compared["scope"], "comparison-only")
        self.assertFalse(compared["review_authenticated"])
        with patch.dict(self.environment, {"LD_PRELOAD": str(self.root / "unqualified-project-library.so")}):
            with self.assertRaisesRegex(ValueError, "runtime injection"):
                self.current()
        presets = self.root / "CMakePresets.json"
        original_presets = presets.read_bytes()
        injected = json.loads(original_presets)
        injected["configurePresets"][0]["cacheVariables"]["CMAKE_C_FLAGS"] = "-include project-hook.h"
        presets.write_text(json.dumps(injected))
        try:
            with self.assertRaisesRegex(ValueError, "configuration injection"):
                self.current()
        finally:
            presets.write_bytes(original_presets)
        self.assertEqual(self.state_path.read_bytes(), state_before)
        self.assertEqual(self.counter.read_bytes(), b"1\n")
        self.assertTrue(all(path.read_bytes() == raw for path, raw in original_bytes.items()))
        for change in (lambda value: value["context"].update(runtime="all"),
                       lambda value: value["subject"]["dependencies"]["files"].pop(str(source)),
                       lambda value: value["stage_identity"].update(environment="0" * 64),
                       lambda value: value["policy"].update(complete=True)):
            changed = json.loads(json.dumps(current))
            change(changed)
            with self.subTest(changed=change), self.assertRaises(ValueError):
                qualification.compare_execution_subject(original, changed, self.declaration(original, changed))
        bad_review = json.loads(json.dumps(declaration))
        bad_review["compatibility"][0]["outcome_integrity"] = "false-historical-pass"
        with self.assertRaisesRegex(ValueError, "trustworthiness"):
            qualification.compare_execution_subject(original, current, bad_review)
        old_dependency = document["preparation"]["identity"]
        self.prepare()
        fresh = self.current()
        self.assertNotEqual(old_dependency, fresh["preparation"]["identity"])
        # Historical reader does not require the superseded producer to be current.
        self.assertEqual(qualification.read_execution_contract(original["reference"]), original)
        compared = qualification.compare_execution_subject(original, fresh, self.declaration(original, fresh))
        self.assertEqual(compared["original_preparation_identity"], old_dependency)
        self.assertEqual(compared["current_preparation_identity"], fresh["preparation"]["identity"])
        self.assertEqual(document["stage"]["result"]["identity"]["dependencies"], [old_dependency])
        contract_path = Path(original["reference"]["path"])
        contract_bytes = contract_path.read_bytes()
        forged = json.loads(contract_bytes)
        forged["preparation"]["identity"] = "0" * 64
        forged["stage"]["result"]["identity"]["dependencies"] = ["0" * 64]
        facts = {key: forged[key] for key in ("policy", "candidate", "context", "controller_paths", "subject", "preparation")}
        facts["stage_identity"] = forged["stage"]["result"]["identity"]
        seal = qualification._bounded_hash({"facts": facts, "producer": forged["producer"], "launch": forged["launch"]})
        forged["seals"] = {"entry": seal, "terminal": seal}
        contract_path.write_bytes(qualification._bounded_bytes(forged))
        try:
            with self.assertRaisesRegex(ValueError, "preparation dependency"):
                qualification.read_execution_contract(qualification._bounded_reference(contract_path))
        finally:
            contract_path.write_bytes(contract_bytes)
        for change in (lambda value: value["subject"]["dependencies"]["files"].pop(str(source)),
                       lambda value: value["subject"].update(registration=None)):
            forged = json.loads(contract_bytes)
            change(forged)
            facts = {key: forged[key] for key in ("policy", "candidate", "context", "controller_paths", "subject", "preparation")}
            facts["stage_identity"] = forged["stage"]["result"]["identity"]
            seal = qualification._bounded_hash({"facts": facts, "producer": forged["producer"], "launch": forged["launch"]})
            forged["seals"] = {"entry": seal, "terminal": seal}
            contract_path.write_bytes(qualification._bounded_bytes(forged))
            try:
                with self.subTest(malformed=change), self.assertRaises(ValueError):
                    qualification.read_execution_contract(qualification._bounded_reference(contract_path))
            finally:
                contract_path.write_bytes(contract_bytes)
        for key in ("process", "junit", "last_test"):
            reference = document["execution"][key]
            path = Path(reference["path"])
            raw = path.read_bytes()
            path.write_bytes(raw + b"tampered")
            try:
                with self.subTest(payload=key), self.assertRaises(ValueError):
                    qualification.read_execution_contract(original["reference"])
            finally:
                path.write_bytes(raw)

    def test_exact_legacy_reuse_cannot_backfill_an_original(self):
        first = self.execute(capture=False)
        raw = Path(first["receipt"]).read_bytes()
        with patch.object(Campaign, "run", side_effect=AssertionError("exact reuse reran CTest")):
            reused = self.execute()
        self.assertEqual(reused["reused_from"], first["receipt"])
        self.assertNotIn("execution_contract", reused)
        self.assertEqual(Path(first["receipt"]).read_bytes(), raw)
        self.assertEqual(self.counter.read_bytes(), b"1\n")
        self.controller.write_text("# another original controller\n")
        subprocess.run(["git", "-c", "core.autocrlf=false", "add", "scripts/qualification.py"],
                       cwd=self.root, check=True, capture_output=True)
        subprocess.run(["git", "-c", "user.name=Control", "-c", "user.email=control@example.invalid",
                        "-c", "commit.gpgsign=false", "commit", "-qm", "another fixture original"],
                       cwd=self.root, check=True, capture_output=True)
        self.counter.unlink()
        def stale():
            self.counter.write_bytes(b"1\n")
            return Path(first["receipt"])
        replayed = self.execute(action=stale)
        self.assertIn("fresh whole", replayed["bounded_ineligible"])
        self.assertNotIn("execution_contract", replayed)
        self.assertEqual(Path(first["receipt"]).read_bytes(), raw)

    def test_expected_stdout_with_failure_creates_no_contract(self):
        def fail():
            self.counter.mkdir()
            return qualification.run_ctest_parts(self.campaign, "product", [self.command], {"host": ["Bounded.Read"]},
                build=self.build, inputs=self.inputs, tools=self.tools, environment=self.environment, timeout=None,
                repeat_reason="", state=self.state, state_path=self.state_path, payloads=[self.junit])
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute(action=fail)
        self.assertEqual(self.state["product"]["status"], "failed")
        raw = self.campaign.attempts("product")[-1]
        self.assertEqual(raw["returncode"], 8)
        self.assertIn("result=14", (self.root / raw["output"]).read_text())
        self.assertFalse(list(self.campaign.directory.glob("attempts/*/execution-contract.json")))
        self.counter.rmdir()
        self.junit = self.campaign.directory / "retry-junit.xml"
        self.command[-1] = str(self.junit)
        completed = self.execute(repeat_reason="fixture counter destination repaired")
        self.assertEqual(completed["status"], "passed")
        self.assertIn("fresh whole", completed["bounded_ineligible"])
        self.assertNotIn("execution_contract", completed)
        self.assertEqual(self.counter.read_bytes(), b"1\n")

    def test_terminal_capture_change_preserves_only_the_ordinary_pass(self):
        capture = qualification.capture_execution_subject
        count = 0
        def changed(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                (self.root / "input.txt").write_bytes(b"value=8\n")
            return capture(*args, **kwargs)
        with patch.object(qualification, "capture_execution_subject", side_effect=changed):
            result = self.execute()
        self.assertEqual(result["status"], "passed")
        self.assertIn("terminal", result["bounded_ineligible"])
        self.assertNotIn("execution_contract", result)
        self.assertEqual(self.campaign.attempts("product")[-1]["status"], "passed")
        self.assertFalse(list(self.campaign.directory.glob("attempts/*/execution-contract.json")))

    def test_action_with_different_launch_environment_creates_no_contract(self):
        def changed():
            return qualification.run_ctest_parts(self.campaign, "product", [self.command], {"host": ["Bounded.Read"]},
                build=self.build, inputs=self.inputs, tools=self.tools,
                environment={**self.environment, "BOUNDED_ACTION_ENVIRONMENT": "different"}, timeout=None,
                repeat_reason="", state=self.state, state_path=self.state_path, payloads=[self.junit])
        result = self.execute(action=changed)
        self.assertEqual(result["status"], "passed")
        self.assertIn("provenance", result["bounded_ineligible"])
        self.assertNotIn("execution_contract", result)
        self.assertFalse(list(self.campaign.directory.glob("attempts/*/execution-contract.json")))


class BoundedAdmissionTests(unittest.TestCase):
    recipe = "native-c11-cli-miniature-v1"

    def setUp(self):
        import shutil
        self.preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        parent = Path(os.environ.get("JUICER_TEST_ARTIFACT_DIR", ROOT / "out/validation" / self.preset)) / "checkpoint-controls"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="admit-", dir=parent, delete=False)
        self.folder = Path(self.temporary.name).resolve()
        self.root = self.folder / "project"
        self.root.mkdir()
        self.build = self.root / "out/build" / self.preset
        self.build.mkdir(parents=True)
        self.state_path = self.build / "verification-checkpoint.json"
        self.environment = dict(os.environ)
        self.counter = self.build / "assertion-count.txt"
        self.controller = self.root / "scripts/qualification.py"
        record_fixture(self.folder, self, "running", owner_root=self.root)

        def finish():
            outcome = record_fixture(self.folder, self, owner_root=self.root)
            if outcome == "passed" and not os.environ.get("JUICER_TEST_RETAIN_FIXTURES"):
                try:
                    self.temporary.cleanup()
                except BaseException:
                    record_fixture(self.folder, self, "failed", owner_root=self.root)
                    raise
            else:
                print("Retained bounded admission fixture:", self.folder, file=sys.stderr, flush=True)
        self.addCleanup(finish)
        templates = ROOT / "tests/quality/evidence"
        files = {"CMakeLists.txt": templates / "cli/CMakeLists.txt",
                 "src/evidence_child.c": templates / "cli/evidence_child.c", "input.txt": templates / "input.txt",
                 "scripts/check-quality.py": templates / "cli/check_quality.py",
                 "tests/quality/test_miniature.py": templates / "cli/test_miniature.py"}
        for name, source in files.items():
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, self.root / name)
        self.controller.write_bytes(b"# original synthetic outer controller\n")
        (self.root / ".gitignore").write_text("out/\n__pycache__/\n")
        compiler = shutil.which("clang-cl.exe") if os.name == "nt" else "/usr/bin/gcc-13"
        self.assertIsNotNone(compiler, "qualified native C compiler is missing")
        write_json(self.root / "CMakePresets.json",
                   qualification._bounded_presets(self.root, self.preset, compiler, recipe=self.recipe))
        subprocess.run(["git", "init", "-q", str(self.root)], check=True, capture_output=True)
        subprocess.run(["git", "-c", "core.autocrlf=false", "add", "."], cwd=self.root, check=True, capture_output=True)
        subprocess.run(["git", "-c", "user.name=Control", "-c", "user.email=control@example.invalid",
                        "-c", "commit.gpgsign=false", "commit", "-qm", "synthetic CLI original"],
                       cwd=self.root, check=True, capture_output=True)
        self.base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.root, text=True).strip()
        self.original_campaign = Campaign(self.root, self.preset, "cli-original", ["prepare", "quality", "product"])
        state = {}
        with OutputLease(self.root, self.build):
            bootstrap = Campaign(self.root, self.preset, "cli-bootstrap", ["configure"])
            bootstrap.run("configure", ["cmake", "--preset", self.preset],
                inputs=["CMakeLists.txt", "CMakePresets.json", "src/evidence_child.c"],
                tools=[str(Path(shutil.which(name)).resolve()) for name in ("cmake", "ninja")] + [compiler],
                env=self.environment, timeout=None)
            prepared = qualification.prepare_checkpoint(self.original_campaign, self.build, self.state_path, state,
                self.environment, allow_build=True)
            dependency = qualification.preparation_identity(self.build, prepared)
            inputs = qualification.checkpoint_inputs(self.root, self.build, "product")
            qualification.run_stage(self.original_campaign, self.state_path, state, "quality",
                [sys.executable, "-B", "scripts/check-quality.py", "--preset", self.preset,
                 "--qualification", "--base", self.base],
                inputs=inputs, tools=qualification.checkpoint_tools(self.build, self.environment, "product"),
                environment=self.environment, timeout=None, dependencies=[dependency], synthetic_quality_base=self.base)
            self.junit = self.original_campaign.directory / "product-host.xml"
            self.command = [*qualification.ctest_command(self.root, self.build), "-R", "^(Bounded\\.Read)$",
                            "-LE", "gpu", "--no-tests=error", "--output-on-failure", "--output-junit", str(self.junit)]
            tools = qualification._bounded_cli_tools(self.build, self.environment)
            payloads = [self.junit]
            original = qualification.run_stage(self.original_campaign, self.state_path, state, "product", self.command,
                inputs=inputs, tools=tools, environment=self.environment, timeout=None,
                semantic_command=[qualification.ctest_identity(self.command)],
                action=lambda: qualification.run_ctest_parts(self.original_campaign, "product", [self.command],
                    {"host": ["Bounded.Read"]}, build=self.build, inputs=inputs, tools=tools,
                    environment=self.environment, timeout=None, repeat_reason="", state=state,
                    state_path=self.state_path, payloads=payloads),
                validate=lambda paths: qualification.validate_ctest_stage(paths, {"host": ["Bounded.Read"]},
                    ["Bounded.Read"], [qualification.ctest_identity(self.command)]),
                payloads=payloads, dependencies=[dependency], ctest=True, bounded_capture=self.recipe)
        self.assertIn("execution_contract", original, original.get("bounded_ineligible"))
        self.original = qualification.read_execution_contract(original["execution_contract"])
        self.original_stage = json.loads(json.dumps(original))
        self.original_bytes = {Path(reference["path"]): Path(reference["path"]).read_bytes()
                               for reference in self.original["contract"]["execution"].values()
                               if isinstance(reference, dict) and set(reference) == {"path", "sha256"}}
        self.original_bytes[Path(original["execution_contract"]["path"])] = Path(original["execution_contract"]["path"]).read_bytes()
        self.original_bytes[Path(prepared["receipt"])] = Path(prepared["receipt"]).read_bytes()
        self.assertEqual(self.counter.read_bytes(), b"1\n")

    @staticmethod
    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()

    @staticmethod
    def reference(path):
        path = Path(path).resolve(strict=True)
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def args(self, campaign, declaration=None, **changes):
        values = dict(reason="synthetic CLI admission control", selection=["product"], preset=self.preset,
                      runtime="host", campaign=campaign, repeat_reason="", base=self.base,
                      prepare=False, reuse_declaration=declaration)
        values.update(changes)
        return SimpleNamespace(**values)

    def cli(self, campaign, *, declaration=None, selection="product", runtime="host", success=True):
        code = ("import pathlib,sys,subprocess; sys.path.insert(0,sys.argv.pop(1)); import verify; "
                "verify.ROOT=pathlib.Path(sys.argv.pop(1)); sys.argv[0]='verify.py'; "
                "raise SystemExit(verify.main())")
        command = [sys.executable, "-B", "-c", code, str(ROOT / "scripts"), str(self.root), "qualification",
                   "--preset", self.preset, "--reason", "synthetic actual parser and dispatcher",
                   "--selection", selection, "--runtime", runtime, "--campaign", campaign]
        if selection == "product":
            command += ["--base", self.base]
        if declaration is not None:
            command += ["--reuse-declaration", str(declaration)]
        completed = subprocess.run(command, cwd=self.root, env=self.environment, capture_output=True, text=True)
        (self.folder / (campaign + ".stdout")).write_text(completed.stdout)
        (self.folder / (campaign + ".stderr")).write_text(completed.stderr)
        if success:
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            path = self.root / "out/validation" / self.preset / campaign / "qualification.json"
            return path, json.loads(path.read_text())
        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return completed

    def transition(self):
        self.controller.write_bytes(b"# reviewed synthetic controller change\n")
        tooling_path, _ = self.cli("current-tooling", selection="tooling")
        state = json.loads(self.state_path.read_text())
        preparation = Campaign(self.root, self.preset, "current-preparation", ["prepare"])
        with OutputLease(self.root, self.build):
            current_preparation = qualification.prepare_checkpoint(preparation, self.build, self.state_path, state,
                self.environment, allow_build=True)
            self.current_preparation_identity = qualification.preparation_identity(self.build, current_preparation)
        self.assertNotEqual(self.current_preparation_identity, self.original["contract"]["preparation"]["identity"])
        old = self.original["contract"]["candidate"]
        current = qualification._bounded_candidate(self.root, recipe=self.recipe)
        changes = [{"path": name, "old_sha256": old["files"][name], "new_sha256": current["files"][name],
                    "groups": ["product"]} for name in sorted(old["files"]) if old["files"][name] != current["files"][name]]
        declaration = {
            "protocol": "tooling-evidence-transition-bounded-v1", "original_candidate": old,
            "current_candidate": current, "changes": changes,
            "originals": [{"preset": self.preset, "group": "product", "contract": self.original["reference"]}],
            "compatibility": [{"path": row["path"], "group": "product", "role": "controller-only",
                               "outcome_integrity": "preserved"} for row in changes],
            "fresh_evidence": {"tooling": [self.reference(tooling_path)], "quality": [], "quality_base": self.base},
        }
        folder = self.root / "out/validation" / self.preset / "transition"
        folder.mkdir()
        core = hashlib.sha256(self.canonical(declaration)).hexdigest()
        for key, protocol, decision in (("review", "tooling-evidence-review-bounded-v1", "compatible"),
                                         ("owner_decision", "tooling-evidence-owner-bounded-v1", "approve")):
            proof = {"protocol": protocol, "transition_sha256": core,
                     "original_candidate_sha256": hashlib.sha256(self.canonical(old)).hexdigest(),
                     "current_candidate_sha256": hashlib.sha256(self.canonical(current)).hexdigest(),
                     "originals": declaration["originals"], "scope": {"preset": self.preset, "groups": ["product"]},
                     "decision": decision}
            path = folder / (key + ".json")
            path.write_bytes(self.canonical(proof))
            declaration[key] = self.reference(path)
        path = folder / "declaration.json"
        path.write_bytes(self.canonical(declaration))
        self.declaration = declaration
        self.declaration_path = path
        return path

    def seal_declaration(self, declaration, suffix):
        folder = self.declaration_path.parent / suffix
        folder.mkdir()
        core = {key: value for key, value in declaration.items() if key not in {"review", "owner_decision"}}
        checksum = hashlib.sha256(self.canonical(core)).hexdigest()
        for key in ("review", "owner_decision"):
            proof = json.loads(Path(self.declaration[key]["path"]).read_text())
            proof["transition_sha256"] = checksum
            proof["original_candidate_sha256"] = hashlib.sha256(self.canonical(declaration["original_candidate"])).hexdigest()
            proof["current_candidate_sha256"] = hashlib.sha256(self.canonical(declaration["current_candidate"])).hexdigest()
            proof["originals"] = declaration["originals"]
            path = folder / (key + ".json")
            path.write_bytes(self.canonical(proof))
            declaration[key] = self.reference(path)
        path = folder / "declaration.json"
        path.write_bytes(self.canonical(declaration))
        return path

    def assert_original_preserved(self):
        self.assertEqual(self.counter.read_bytes(), b"1\n")
        self.assertTrue(all(path.read_bytes() == raw for path, raw in self.original_bytes.items()))
        stage = json.loads(self.state_path.read_text())["product"]
        self.assertEqual({key: value for key, value in stage.items() if key != "reuse_admission"}, self.original_stage)

    def admission_records(self):
        return sorted((self.build / "verification-admissions").glob("*/admission.json"))

    def test_cli_admits_controller_change_and_recovers_without_execution(self):
        declaration = self.transition()
        path, record = self.cli("admitted", declaration=declaration)
        self.assertEqual(record["stage_provenance"]["product"], "admitted")
        self.assertEqual(record["assurance"], "synthetic-c11-cli-quality")
        self.assertEqual(record["stages"]["product"]["identity"]["dependencies"],
                         [self.original["contract"]["preparation"]["identity"]])
        self.assertNotEqual(record["stages"]["quality"]["identity"]["dependencies"],
                            record["stages"]["product"]["identity"]["dependencies"])
        first = record["stages"]["product"]["reuse_admission"]
        self.assertEqual(len(self.admission_records()), 1)
        admission_bytes = Path(first["path"]).read_bytes()
        for campaign in ("recovered", "recovered-again"):
            _, recovered = self.cli(campaign, declaration=declaration)
            self.assertEqual(recovered["stages"]["product"]["reuse_admission"], first)
            self.assertEqual(recovered["stage_provenance"]["product"], "admitted")
            self.assertEqual(len(self.admission_records()), 1)
            self.assertEqual(Path(first["path"]).read_bytes(), admission_bytes)
        self.assertTrue(path.is_file())
        self.assert_original_preserved()

        # Admission cannot become a new original, even with new integrity records.
        chained = json.loads(json.dumps(self.declaration))
        chained["originals"][0]["contract"] = first
        chained_path = self.seal_declaration(chained, "transitive")
        self.cli("transitive-original", declaration=chained_path, success=False)
        self.assertEqual(len(self.admission_records()), 1)
        self.assert_original_preserved()

        baseline = self.state_path.read_bytes()
        forged = json.loads(admission_bytes)
        forged["current_candidate"]["files"]["input.txt"] = "f" * 64
        forged_path = self.declaration_path.parent / "malformed-prior-admission.json"
        forged_path.write_bytes(self.canonical(forged))
        state = json.loads(baseline)
        state["product"]["reuse_admission"] = self.reference(forged_path)
        write_json(self.state_path, state)
        try:
            self.cli("malformed-committed-pointer", declaration=declaration, success=False)
            self.assertEqual(json.loads(self.state_path.read_text())["product"], state["product"])
            self.assertEqual(len(self.admission_records()), 1)
        finally:
            self.state_path.write_bytes(baseline)
        self.assert_original_preserved()

        completion = Path(first["path"]).with_name("completion.json")
        completion_bytes = completion.read_bytes()
        completed = json.loads(completion_bytes)
        wrong_wrapper = json.loads(json.dumps(record))
        wrong_wrapper["stages"]["product"].pop("reuse_admission")
        wrong_path = self.declaration_path.parent / "wrong-completion-wrapper.json"
        wrong_path.write_bytes(self.canonical(wrong_wrapper))
        status_only = json.loads(json.dumps(record))
        status_only["stages"] = {"prepare": {"status": "passed"}, "quality": {"status": "passed"},
                                 "product": {"status": "passed", "reuse_admission": first}}
        status_path = self.declaration_path.parent / "status-only-completion-wrapper.json"
        status_path.write_bytes(self.canonical(status_only))
        for name, wrong in (("extra-field", {**completed, "unknown": True}),
                            ("wrong-wrapper", {**completed, "wrapper": self.reference(wrong_path)}),
                            ("status-only-stages", {**completed, "wrapper": self.reference(status_path)})):
            with self.subTest(completion=name):
                completion.write_bytes(self.canonical(wrong))
                try:
                    self.cli("completion-" + name, declaration=declaration, success=False)
                    attempted = self.root / "out/validation" / self.preset / ("completion-" + name) / "qualification.json"
                    if attempted.exists():
                        self.assertNotEqual(json.loads(attempted.read_text())["status"], "passed")
                    self.assertEqual(json.loads(self.state_path.read_text())["product"], json.loads(baseline)["product"])
                    self.assertEqual(Path(first["path"]).read_bytes(), admission_bytes)
                    self.assertEqual(len(self.admission_records()), 1)
                finally:
                    completion.write_bytes(completion_bytes)
        if os.name == "posix":
            target = completion.with_name("identical-completion.json")
            completion.rename(target)
            completion.symlink_to(target.name)
            try:
                self.cli("completion-retarget", declaration=declaration, success=False)
                self.assertTrue(completion.is_symlink())
                attempted = self.root / "out/validation" / self.preset / "completion-retarget/qualification.json"
                if attempted.exists():
                    self.assertNotEqual(json.loads(attempted.read_text())["status"], "passed")
                self.assertEqual(json.loads(self.state_path.read_text())["product"], json.loads(baseline)["product"])
                self.assertEqual(Path(first["path"]).read_bytes(), admission_bytes)
                self.assertEqual(len(self.admission_records()), 1)
            finally:
                if completion.is_symlink():
                    completion.unlink()
                    target.rename(completion)
        self.assert_original_preserved()

        quality = json.loads(self.state_path.read_text())["quality"]
        reports = [Path(name) for name in quality["payloads"] if Path(name).name == "quality.json"]
        self.assertEqual(len(reports), 1, "quality completion must seal its report before admission")
        report_bytes = reports[0].read_bytes()
        reports[0].write_bytes(report_bytes + b"tampered")
        try:
            self.cli("tampered-quality-recovery", declaration=declaration, success=False)
            self.assertEqual(len(self.admission_records()), 1)
            self.assertNotEqual(json.loads(self.state_path.read_text())["quality"]["status"], "passed")
        finally:
            reports[0].write_bytes(report_bytes)
        self.assert_original_preserved()

    def test_changed_subject_artifact_and_context_reject_without_owner_replacement(self):
        declaration = self.transition()
        source = self.root / "input.txt"
        binary = self.build / ("BoundedEvidenceChild.exe" if os.name == "nt" else "BoundedEvidenceChild")
        for name, target, replacement in (("subject", source, b"value=8\n"),
                                           ("artifact", binary, binary.read_bytes() + b"changed")):
            with self.subTest(name=name):
                raw, stamp = target.read_bytes(), target.stat()
                try:
                    target.write_bytes(replacement)
                    os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                    self.cli("changed-" + name, declaration=declaration, success=False)
                finally:
                    target.write_bytes(raw)
                    os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                self.assert_original_preserved()
        self.cli("changed-runtime", declaration=declaration, runtime="all", success=False)
        extra = self.root / "unlisted.txt"
        extra.write_text("undeclared member\n")
        try:
            self.cli("added-source-member", declaration=declaration, success=False)
        finally:
            extra.unlink()
        moved = source.with_name("input-renamed.txt")
        source.rename(moved)
        try:
            self.cli("renamed-source-member", declaration=declaration, success=False)
        finally:
            moved.rename(source)
        shadow = self.root / "scripts/json.py"
        exclude = self.root / ".git/info/exclude"
        exclude_bytes = exclude.read_bytes()
        shadow.write_text("raise RuntimeError('undeclared project module must never execute')\n")
        exclude.write_bytes(exclude_bytes + b"\nscripts/json.py\n")
        try:
            self.assertEqual(subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--",
                             "scripts/json.py"], cwd=self.root), b"")
            with patch.object(Campaign, "run", side_effect=AssertionError("ignored project shadow executed work")):
                with self.assertRaises(ValueError):
                    qualification.qualify(self.root, self.args("ignored-project-shadow", declaration))
        finally:
            shadow.unlink()
            exclude.write_bytes(exclude_bytes)
        with patch.dict(os.environ, {"CPATH": str(self.root / "unsupported-project-include")}):
            with self.assertRaises(ValueError):
                qualification.qualify(self.root, self.args("changed-tool-context", declaration))
        import shutil
        native_ctest = Path(shutil.which("ctest", path=self.environment.get("PATH"))).resolve(strict=True)
        copied_tool = self.folder / "tool-copy" / native_ctest.name
        copied_tool.parent.mkdir()
        shutil.copyfile(native_ctest, copied_tool)
        if os.name == "posix":
            copied_tool.chmod(native_ctest.stat().st_mode & 0o777)
        self.assertEqual(hashlib.sha256(copied_tool.read_bytes()).hexdigest(),
                         hashlib.sha256(native_ctest.read_bytes()).hexdigest())
        import runpy
        checker = runpy.run_path(str(ROOT / "tests/quality/evidence/cli/check_quality.py"),
                                run_name="synthetic_executable_control")
        if os.name == "posix":
            mode = copied_tool.stat().st_mode & 0o777
            before = checker["executable_identity"](copied_tool)
            self.assertEqual(before["execute_mask"], mode & 0o111)
            self.assertTrue(before["launch_access"])
            self.assertEqual(before["sha256"], hashlib.sha256(copied_tool.read_bytes()).hexdigest())
            copied_tool.chmod(mode & ~0o111)
            try:
                self.assertEqual(hashlib.sha256(copied_tool.read_bytes()).hexdigest(), before["sha256"])
                with self.assertRaisesRegex(ValueError, "launch access"):
                    checker["executable_identity"](copied_tool)
            finally:
                copied_tool.chmod(mode)
        with patch.dict(os.environ, {"PATH": str(copied_tool.parent) + os.pathsep + os.environ["PATH"]}):
            with patch.object(Campaign, "run", side_effect=AssertionError("changed tool target executed work")):
                with self.assertRaises(ValueError):
                    qualification.qualify(self.root, self.args("changed-native-tool-target", declaration))
        self.assert_original_preserved()
        self.assertEqual(self.admission_records(), [])

    def test_malformed_and_stale_declarations_run_no_commands(self):
        declaration = self.transition()
        original = declaration.read_bytes()
        corruptions = [b'{"protocol":"x","protocol":"y"}',
                       self.canonical({**self.declaration, "unknown": True}),
                       self.canonical({**self.declaration, "current_candidate": {}})]
        stale = json.loads(original)
        stale["review"]["sha256"] = "0" * 64
        corruptions.append(self.canonical(stale))
        for number, raw in enumerate(corruptions):
            with self.subTest(number=number):
                declaration.write_bytes(raw)
                try:
                    with patch.object(Campaign, "run", side_effect=AssertionError("ineligible declaration executed work")):
                        with self.assertRaises((ValueError, OSError)):
                            qualification.qualify(self.root, self.args("malformed-" + str(number), declaration))
                finally:
                    declaration.write_bytes(original)
                self.assert_original_preserved()

        for name, mutate in (
                ("unlisted", lambda value: value["changes"].append({"path": "scripts/unlisted.py", "old_sha256": "a" * 64,
                     "new_sha256": "b" * 64, "groups": ["product"]})),
                ("wrong-pair", lambda value: value["original_candidate"]["files"].update({"input.txt": "f" * 64})),
                ("unsupported-production", lambda value: value["originals"][0].update(group="production")),
                ("missing-contract", lambda value: value["originals"][0].update(contract={"path": str(self.root / "missing.json"), "sha256": "f" * 64})),
                ("circular-support", lambda value: value["fresh_evidence"].update(tooling=[self.original["reference"]]))):
            with self.subTest(name=name):
                value = json.loads(original)
                mutate(value)
                path = self.seal_declaration(value, name)
                with patch.object(Campaign, "run", side_effect=AssertionError("invalid scope executed work")):
                    with self.assertRaises((ValueError, OSError)):
                        qualification.qualify(self.root, self.args(name, path))
                self.assert_original_preserved()

    def test_current_stage_and_preparation_owners_win(self):
        declaration = self.transition()
        baseline = self.state_path.read_bytes()
        for name in ("product", "prepare"):
            for status in ("pending", "failed", "blocked"):
                with self.subTest(name=name, status=status):
                    state = json.loads(baseline)
                    state[name]["status"] = status
                    write_json(self.state_path, state)
                    try:
                        with patch.object(Campaign, "run", side_effect=AssertionError("owner conflict executed work")):
                            with self.assertRaises(ValueError):
                                qualification.qualify(self.root, self.args(name + "-" + status, declaration))
                        self.assertEqual(json.loads(self.state_path.read_text())[name], state[name])
                        self.assertEqual(self.counter.read_bytes(), b"1\n")
                    finally:
                        self.state_path.write_bytes(baseline)
        state = json.loads(baseline)
        state["product"]["sha256"] = "f" * 64
        write_json(self.state_path, state)
        try:
            with self.assertRaises(ValueError):
                qualification.qualify(self.root, self.args("newer-product", declaration))
            self.assertEqual(json.loads(self.state_path.read_text())["product"], state["product"])
        finally:
            self.state_path.write_bytes(baseline)
        self.assert_original_preserved()

        from build_evidence import preparation_paths
        _, pending = preparation_paths(self.build, ["all"])
        self.assertFalse(pending.exists())
        pending.parent.mkdir(parents=True, exist_ok=True)
        pending.write_text('{"synthetic_pending_owner":true}')
        try:
            with patch.object(Campaign, "run", side_effect=AssertionError("pending preparation executed work")):
                with self.assertRaises(ValueError):
                    qualification.qualify(self.root, self.args("pending-preparation-owner", declaration))
            self.assertTrue(pending.exists())
        finally:
            pending.unlink()
        self.assert_original_preserved()

    def test_raw_original_and_support_tampering_reject(self):
        declaration = self.transition()
        original = self.original["contract"]
        paths = [Path(original["execution"][name]["path"]) for name in ("process", "junit", "last_test")]
        paths.append(Path(self.declaration["fresh_evidence"]["tooling"][0]["path"]))
        for number, path in enumerate(paths):
            with self.subTest(path=path):
                raw = path.read_bytes()
                path.write_bytes(raw + b"tampered")
                try:
                    with patch.object(Campaign, "run", side_effect=AssertionError("tampered evidence executed work")):
                        with self.assertRaises((ValueError, OSError)):
                            qualification.qualify(self.root, self.args("tampered-" + str(number), declaration))
                finally:
                    path.write_bytes(raw)
                self.assert_original_preserved()

        # Missing and partial raw proof cannot be repaired during explicit reuse.
        raw_reference = Path(original["execution"]["receipt"]["path"])
        raw = raw_reference.read_bytes()
        partial = json.loads(raw)
        partial.update(status="failed", returncode=8)
        raw_reference.write_bytes(self.canonical(partial))
        try:
            with patch.object(Campaign, "run", side_effect=AssertionError("partial original executed work")):
                with self.assertRaises(ValueError):
                    qualification.qualify(self.root, self.args("partial-original", declaration))
        finally:
            raw_reference.write_bytes(raw)
        self.assert_original_preserved()

        # A custom action must not claim the declared stage's launch environment.
        state = json.loads(self.state_path.read_text())
        campaign = Campaign(self.root, self.preset, "wrong-action-launch", ["tooling"])
        command = [sys.executable, "-B", "-c", "print('independent launch control')"]
        inputs = qualification.checkpoint_inputs(self.root, self.build, "tooling")
        with OutputLease(self.root, self.build):
            try:
                stage = qualification.run_stage(campaign, self.state_path, state, "tooling", command,
                    inputs=inputs, tools=[sys.executable], environment=self.environment, timeout=None,
                    action=lambda: campaign.run("tooling", command, inputs=inputs, tools=[sys.executable],
                        env={**self.environment, "SYNTHETIC_ACTION_ENVIRONMENT": "different"}, timeout=None))
            except ValueError:
                self.assertNotEqual(json.loads(self.state_path.read_text())["tooling"]["status"], "passed")
            else:
                with self.assertRaisesRegex(ValueError, "launch|environment|independent"):
                    qualification._check_stage_evidence(self.root, stage)
        self.assert_original_preserved()

    def test_terminal_mutations_do_not_publish(self):
        declaration = self.transition()
        original_write = qualification._write_admission_record
        source = self.root / "input.txt"
        raw = source.read_bytes()
        def changed(*args, **kwargs):
            result = original_write(*args, **kwargs)
            source.write_bytes(b"value=8\n")
            return result
        try:
            with patch.object(qualification, "_write_admission_record", side_effect=changed):
                with self.assertRaises(ValueError):
                    qualification.qualify(self.root, self.args("terminal-source", declaration))
            self.assertNotIn("reuse_admission", json.loads(self.state_path.read_text())["product"])
            wrapper = self.root / "out/validation" / self.preset / "terminal-source/qualification.json"
            self.assertNotEqual(json.loads(wrapper.read_text())["status"], "passed")
        finally:
            source.write_bytes(raw)
        self.assert_original_preserved()

        stage_before = json.loads(self.state_path.read_text())["product"]
        owner_declaration = self.seal_declaration(json.loads(declaration.read_bytes()), "terminal-owner")
        owner_mutations = []
        def changed_owner(*args, **kwargs):
            result = original_write(*args, **kwargs)
            state = json.loads(self.state_path.read_text())
            state["product"]["status"] = "pending"
            write_json(self.state_path, state)
            owner_mutations.append(str(args[0]))
            return result
        try:
            with patch.object(qualification, "_write_admission_record", side_effect=changed_owner):
                with self.assertRaises(ValueError):
                    qualification.qualify(self.root, self.args("terminal-owner", owner_declaration))
            self.assertEqual(len(owner_mutations), 1)
            self.assertEqual(json.loads(self.state_path.read_text())["product"]["status"], "pending")
        finally:
            state = json.loads(self.state_path.read_text())
            state["product"] = stage_before
            write_json(self.state_path, state)
        self.assert_original_preserved()

        record_declaration = self.seal_declaration(json.loads(declaration.read_bytes()), "terminal-record")
        record_mutations = []
        def changed_record(*args, **kwargs):
            result = original_write(*args, **kwargs)
            path = Path(args[0])
            path.write_bytes(path.read_bytes() + b"tampered")
            record_mutations.append(str(path))
            return result
        with patch.object(qualification, "_write_admission_record", side_effect=changed_record):
            with self.assertRaises(ValueError):
                qualification.qualify(self.root, self.args("terminal-admission", record_declaration))
        self.assertEqual(len(record_mutations), 1)
        self.assertNotIn("reuse_admission", json.loads(self.state_path.read_text())["product"])
        self.assert_original_preserved()

        if os.name == "posix":
            alias_declaration = self.seal_declaration(json.loads(declaration.read_bytes()), "terminal-retarget")
            retargets = []
            def retarget_record(*args, **kwargs):
                result = original_write(*args, **kwargs)
                path = Path(args[0])
                target = path.with_name("identical-record.json")
                path.rename(target)
                path.symlink_to(target.name)
                retargets.append(str(path))
                return result
            with patch.object(qualification, "_write_admission_record", side_effect=retarget_record):
                with self.assertRaises(ValueError):
                    qualification.qualify(self.root, self.args("terminal-retarget", alias_declaration))
            self.assertEqual(len(retargets), 1)
            self.assertNotIn("reuse_admission", json.loads(self.state_path.read_text())["product"])
            self.assert_original_preserved()

    def test_cancellation_and_publication_recovery(self):
        declaration = self.transition()
        baseline = self.state_path.read_bytes()
        with patch.object(qualification, "_write_admission_record", side_effect=KeyboardInterrupt("before record")):
            with self.assertRaises(KeyboardInterrupt):
                qualification.qualify(self.root, self.args("cancel-before-record", declaration))
        self.assertEqual(self.admission_records(), [])
        self.assertNotIn("reuse_admission", json.loads(self.state_path.read_text())["product"])
        original_write = qualification.write_json
        def before_pointer(path, record):
            if Path(path) == self.state_path and record.get("product", {}).get("reuse_admission"):
                raise KeyboardInterrupt("between record and pointer")
            return original_write(path, record)
        with patch.object(qualification, "write_json", side_effect=before_pointer):
            with self.assertRaises(KeyboardInterrupt):
                qualification.qualify(self.root, self.args("cancel-before-pointer", declaration))
        self.assertEqual(len(self.admission_records()), 1)
        self.assertNotIn("reuse_admission", json.loads(self.state_path.read_text())["product"])
        capture = qualification.capture_execution_subject
        owner_races = []
        def final_owner_race(*args, **kwargs):
            facts = capture(*args, **kwargs)
            current = json.loads(self.state_path.read_text())
            if current["product"].get("reuse_admission") and not owner_races:
                current["product"]["status"] = "pending"
                write_json(self.state_path, current)
                owner_races.append(current["product"]["reuse_admission"])
            return facts
        with patch.object(qualification, "capture_execution_subject", side_effect=final_owner_race):
            with self.assertRaises(ValueError):
                qualification.qualify(self.root, self.args("final-wrapper-owner-race", declaration))
        self.assertEqual(len(owner_races), 1)
        current = json.loads(self.state_path.read_text())
        self.assertEqual(current["product"]["status"], "pending")
        self.assertFalse(any(path.with_name("completion.json").exists() for path in self.admission_records()))
        current["product"]["status"] = "passed"
        write_json(self.state_path, current)
        quality_reports = [Path(name) for name in current["quality"]["payloads"] if Path(name).name == "quality.json"]
        self.assertEqual(len(quality_reports), 1)
        final_targets = [
            ("original", Path(self.original["contract"]["execution"]["process"]["path"])),
            ("tooling", Path(self.declaration["fresh_evidence"]["tooling"][0]["path"])),
            ("quality", quality_reports[0]),
            ("admission", self.admission_records()[0]),
            ("declaration", declaration),
        ]
        for name, target in final_targets:
            with self.subTest(final_capture=name):
                current = json.loads(self.state_path.read_text())
                current["product"] = json.loads(json.dumps(self.original_stage))
                write_json(self.state_path, current)
                original_bytes = target.read_bytes()
                mutations = []
                def final_file_race(*args, **kwargs):
                    facts = capture(*args, **kwargs)
                    if json.loads(self.state_path.read_text())["product"].get("reuse_admission") and not mutations:
                        target.write_bytes(original_bytes + b"tampered")
                        mutations.append(str(target))
                    return facts
                try:
                    with patch.object(qualification, "capture_execution_subject", side_effect=final_file_race):
                        with self.assertRaises(ValueError):
                            qualification.qualify(self.root, self.args("final-capture-" + name, declaration))
                    self.assertEqual(len(mutations), 1)
                    failed = self.root / "out/validation" / self.preset / ("final-capture-" + name) / "qualification.json"
                    self.assertNotEqual(json.loads(failed.read_text())["status"], "passed")
                    self.assertEqual(len(self.admission_records()), 1)
                    self.assertFalse(any(path.with_name("completion.json").exists() for path in self.admission_records()))
                finally:
                    target.write_bytes(original_bytes)
                self.assert_original_preserved()
        def before_wrapper(path, record):
            if Path(path).name == "qualification.json" and record.get("scope_status") == "passed":
                raise KeyboardInterrupt("after pointer before wrapper")
            return original_write(path, record)
        with patch.object(qualification, "write_json", side_effect=before_wrapper):
            with self.assertRaises(KeyboardInterrupt):
                qualification.qualify(self.root, self.args("cancel-before-wrapper", declaration))
        self.assertIn("reuse_admission", json.loads(self.state_path.read_text())["product"])
        self.assertFalse((self.root / "out/validation" / self.preset / "cancel-before-wrapper/qualification.json").exists())
        self.assertEqual(len(self.admission_records()), 1)
        self.assertFalse(any(path.with_name("completion.json").exists() for path in self.admission_records()))
        for campaign in ("recover-wrapper", "recover-wrapper-again"):
            _, record = self.cli(campaign, declaration=declaration)
            self.assertEqual(record["stage_provenance"]["product"], "admitted")
            self.assertEqual(len(self.admission_records()), 1)
        self.assert_original_preserved()
        self.assertNotEqual(self.state_path.read_bytes(), baseline)

    def test_no_option_preserves_exact_reuse_and_failure_precedence(self):
        _, executed = self.cli("ordinary-first")
        self.assertEqual(self.counter.read_bytes(), b"1\n1\n")
        self.assertNotIn("reuse_admission", executed["stages"]["product"])
        _, reused = self.cli("ordinary-exact")
        self.assertEqual(reused["stages"]["product"]["reused_from"], executed["stages"]["product"]["receipt"])
        self.assertEqual(self.counter.read_bytes(), b"1\n1\n")
        state = json.loads(self.state_path.read_text())
        state["product"]["status"] = "failed"
        write_json(self.state_path, state)
        self.cli("ordinary-failed-owner", success=False)
        self.assertEqual(json.loads(self.state_path.read_text())["product"]["status"], "failed")
        self.assertEqual(self.counter.read_bytes(), b"1\n1\n")

    def test_collector_validates_admission_but_requires_complete_aggregate(self):
        declaration = self.transition()
        path, record = self.cli("admitted-collection-guard", declaration=declaration)
        with self.assertRaisesRegex(ValueError, "full checkpoint membership"):
            qualification.collect_qualification(self.root, [path])
        promoted = json.loads(json.dumps(record))
        promoted["runtime"] = "all"
        target = path.with_name("host-promoted-to-all.json")
        write_json(target, promoted)
        with self.assertRaisesRegex(ValueError, "wrapper does not bind"):
            qualification.collect_qualification(self.root, [target])
        self.assert_original_preserved()


class CheckpointTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "foreign tool correspondence belongs to Linux collection")
    def test_foreign_tool_keys_preserve_ordinary_and_native_targets(self):
        lexical, canonical = r"C:\Tools\tool.EXE", r"C:\Tools\tool.exe"
        other, unsealed, absent = r"C:\Other\tool.exe", r"C:\Unsealed\tool.exe", r"C:\Tools\absent.toml"
        paths = {name: self.root / ("tool-binding-" + str(index))
                 for index, name in enumerate((lexical, canonical, other, unsealed, absent))}
        for name in (lexical, canonical, other, unsealed):
            paths[name].write_bytes(b"same tool bytes\n")
        bindings = {str(path): str(path.resolve()) for name, path in paths.items() if name != unsealed}
        tools = {target: digest(Path(target)) if Path(target).is_file() else None for target in bindings.values()}
        binding = {"resolved": canonical, "sha256": digest(paths[canonical])}
        native = {"lexical": {"files": {lexical: dict(binding)}},
                  "canonical": {"files": {canonical: dict(binding)}}}
        def key(path, maps=native, *, expected_target=None):
            return qualification._collection_tool_key(path, resolve=paths.__getitem__, native_maps=maps,
                bindings=bindings, tools=tools, expected_target=expected_target)
        # Metadata boundary controls; the actual F21 native phases independently
        # supply the unmocked Windows tool correspondence.
        self.assertEqual(key(lexical), bindings[str(paths[canonical])])
        self.assertEqual(key(canonical, expected_target=canonical), key(lexical))
        self.assertEqual(key(absent, {}, expected_target=absent), bindings[str(paths[absent])])
        self.assertEqual(key(canonical, {}, expected_target=canonical), bindings[str(paths[canonical])])
        self.assertNotEqual({key(lexical, {}): binding["sha256"]},
                            {key(canonical, {}, expected_target=canonical): binding["sha256"]})
        missing = {"canonical": native["canonical"]}
        self.assertNotEqual({key(lexical, missing): binding["sha256"]},
                            {key(canonical, missing, expected_target=canonical): binding["sha256"]})
        with self.assertRaisesRegex(ValueError, "lacks matching fresh native identity"):
            qualification._policy_live_file(lexical, binding, resolve=paths.__getitem__, native_maps=missing)
        for mutation in ("disagreeing", "hash", "unsealed-target", "already-sealed-target"):
            damaged = json.loads(json.dumps(native))
            if mutation == "disagreeing":
                damaged["other"] = {"files": {lexical: {**binding, "resolved": other}}}
            elif mutation == "hash":
                damaged["lexical"]["files"][lexical]["sha256"] = "0" * 64
            elif mutation == "unsealed-target":
                damaged["lexical"]["files"][lexical]["resolved"] = unsealed
            else:
                for groups in damaged.values():
                    for row in groups["files"].values():
                        row["resolved"] = other
            with self.subTest(native_tool_binding=mutation), self.assertRaisesRegex(ValueError, "Windows tool"):
                if mutation == "already-sealed-target":
                    key(canonical, damaged, expected_target=canonical)
                else:
                    key(lexical, damaged)

    def test_native_runtime_binding_rejects_missing_and_same_byte_other_target(self):
        name = r"C:\Python\python.exe"
        binding = {"resolved": name, "sha256": "a" * 64}
        groups = {"first": {"files": {name: binding}}, "second": {"files": {name: dict(binding)}}}
        self.assertEqual(qualification.native_runtime_binding(groups, name), binding)
        del groups["second"]["files"][name]
        with self.assertRaisesRegex(ValueError, "not bound"):
            qualification.native_runtime_binding(groups, name)
        groups["second"]["files"][name] = {**binding, "resolved": r"C:\Other\python.exe"}
        with self.assertRaisesRegex(ValueError, "not bound"):
            qualification.native_runtime_binding(groups, name)

    @unittest.skipUnless(os.name == "posix", "foreign interpreter admission belongs to Linux collection")
    def test_foreign_collection_rejects_unqualified_runtime_before_campaign(self):
        selected = r"C:\Qualified\python.exe"
        path = self.root / "native-preparation.json"
        write_json(path, {"scoped_preparation": {"source": r"C:\source", "context": {
            "files": {}, "rust_configuration": {}}}})
        record = {"selection": ["product"], "stages": {
            "prepare": {"receipt": str(path), "sha256": digest(path)},
            "quality": {"identity": {"tools": {selected: digest(self.source)}, "tool_paths": [selected]}}}}
        with patch.object(qualification, "Campaign", side_effect=AssertionError("unqualified native work")):
            with self.assertRaisesRegex(ValueError, "explicit --windows-python"):
                qualification.prepare_native_collection(self.root, [record], None, Path)
            with patch.object(qualification, "mapped_windows_path", return_value=self.source), \
                    patch.object(qualification, "windows_identity_interpreter", return_value={
                        "python": selected, "base_python": r"C:\Base\python.exe", "config": None}):
                with self.assertRaisesRegex(ValueError, "not bound by qualified native proof"):
                    qualification.prepare_native_collection(self.root, [record], selected, Path)

    @unittest.skipUnless(os.name == "posix", "foreign interpreter admission belongs to Linux collection")
    def test_foreign_collection_preserves_qualified_interpreter_alias_binding(self):
        selected, base = r"C:\Alias\python.exe", r"C:\BaseAlias\python.exe"
        bindings = {selected: {"resolved": r"C:\Real\python.exe", "sha256": digest(self.source)},
                    base: {"resolved": r"C:\RealBase\python.exe", "sha256": digest(self.source)}}
        path = self.root / "native-preparation.json"
        write_json(path, {"scoped_preparation": {"source": r"C:\source", "context": {
            "files": bindings, "rust_configuration": {}}}})
        record = {"selection": ["product"], "stages": {
            "prepare": {"receipt": str(path), "sha256": digest(path)}, "quality": {"identity": {
                "tools": {b["resolved"]: b["sha256"] for b in bindings.values()},
                "tool_paths": [selected, base]}}}}
        with patch.object(qualification, "Campaign"), \
                patch.object(qualification, "mapped_windows_path", return_value=self.source), \
                patch.object(qualification, "windows_identity_interpreter", return_value={
                    "python": selected, "base_python": base, "config": None}):
            native = qualification.prepare_native_collection(self.root, [record], selected, Path)
        self.assertEqual(native["python"], selected)
        self.assertEqual(native["expected"][digest(path)]["files"], bindings)
    def run(self, result=None):
        started = time.monotonic()
        print(f"Checkpoint begin {self.id()} suite_seconds={started - SUITE_STARTED:.3f}", file=sys.stderr, flush=True)
        try:
            return super().run(result)
        except KeyboardInterrupt:
            self.interrupted = True
            # unittest skips its cleanup callbacks when cancellation escapes.
            self.retain = True
            if hasattr(self, "lease") and self.lease.path in OutputLease.held:
                # Completed earlier commands cannot prove an interrupted fixture drained.
                self.lease.blocked = True
            self.doCleanups()
            raise
        finally:
            print(f"Checkpoint end {self.id()} case_seconds={time.monotonic() - started:.3f} "
                  f"suite_seconds={time.monotonic() - SUITE_STARTED:.3f}", file=sys.stderr, flush=True)

    def setUp(self):
        preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        artifact_root = os.environ.get("JUICER_TEST_ARTIFACT_DIR")
        parent = (Path(artifact_root) if artifact_root else ROOT / "out/validation" / preset) / "checkpoint-controls"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent, delete=False)
        self.directory = Path(self.temporary.name).resolve()
        self.root = self.directory / "project"
        self.retain = bool(os.environ.get("JUICER_TEST_RETAIN_FIXTURES"))
        if self.retain:
            print("Checkpoint fixture:", self.directory, file=sys.stderr, flush=True)
        def finish():
            if hasattr(self, "lease") and self.lease.path in OutputLease.held:
                self.lease.__exit__(None, None, None)
            outcome = record_fixture(self.directory, self, owner_root=self.root)
            failed = outcome != "passed"
            owners = list(self.root.glob("out/verification-locks/*/owner.json"))
            if owners or self.retain or failed:
                print("Retained checkpoint fixture:", self.directory, file=sys.stderr)
            else:
                try:
                    self.temporary.cleanup()
                except BaseException:
                    record_fixture(self.directory, self, "failed", owner_root=self.root)
                    raise
        self.addCleanup(finish)
        record_fixture(self.directory, self, "running", owner_root=self.root)

        self.root.mkdir()
        self.build = self.root / "out/build" / preset
        self.build.mkdir(parents=True)
        self.state_path = self.build / "verification-checkpoint.json"
        self.state = {}
        self.preset = preset
        self.environment = dict(os.environ)
        self.python = str(Path(sys.base_prefix) / "python.exe") if os.name == "nt" else sys.executable
        self.source = self.root / "input.txt"
        self.source.write_text("first")
        self.other = self.root / "other.txt"
        self.other.write_text("unrelated")
        self.lease = OutputLease(self.root, self.build)
        self.lease.__enter__()

    def campaign(self, name):
        return Campaign(self.root, self.preset, name, ["boundary", "tooling", "prepare", "corpus", "production"])

    def configure_fixture(self, preset, testing, name, *, timeout=30):
        from build_evidence import windows_vctip
        environment = dict(os.environ)
        campaign = Campaign(self.root, preset, name, ["prepare"])
        return campaign.run("prepare", ["cmake", "--preset", preset, "-DBUILD_TESTING=" + ("ON" if testing else "OFF")],
            inputs=[p for p in ("CMakeLists.txt", "CMakePresets.json", "binding.h.in") if (self.root / p).is_file()],
            env=environment, timeout=timeout, vctip_executable=windows_vctip(environment))

    def run_stage(self, name, *, campaign="first", inputs=None, command=None, tools=None, **kwargs):
        return qualification.run_stage(self.campaign(campaign), self.state_path, self.state, name,
            command or [self.python, "-c", "print('checked')"], inputs=inputs or ["input.txt"],
            tools=tools or [self.python], environment=self.environment, timeout=10, **kwargs)

    def test_new_campaign_reuses_original_success_without_execution(self):
        first = self.run_stage("boundary")
        result = self.run_stage("boundary", campaign="renamed")
        self.assertEqual(result["reused_from"], first["receipt"])
        self.assertEqual(result["sha256"], digest(Path(first["receipt"])))
        self.assertFalse((self.campaign("renamed").directory / "attempts").exists())

    def test_call_id_change_reuses_gate_but_unknown_environment_change_does_not(self):
        self.environment["CODEX_TOOL_CALL_ID"] = "first"
        first = self.run_stage("boundary")
        self.environment["CODEX_TOOL_CALL_ID"] = "second"
        with patch.object(Campaign, "run", side_effect=AssertionError("call ID reran a completed gate")):
            reused = self.run_stage("boundary", campaign="new-call")
        self.assertEqual(reused["reused_from"], first["receipt"])
        self.environment["UNKNOWN_BUILD_CONTROL"] = "changed"
        changed = self.run_stage("boundary", campaign="changed-control")
        self.assertNotIn("reused_from", changed)

    def test_checkpoint_presets_overlap_independent_tests_and_preserve_resource_locks(self):
        self.native_project()
        worker = self.root / "worker.py"
        scratch = self.root / "out/scheduling"
        scratch.mkdir(parents=True)
        worker.write_text(
            "import sys,time\nfrom pathlib import Path\n"
            "role=sys.argv[1];root=Path(sys.argv[2])\n"
            "if role in ('first','second'):\n"
            " (root/(role+'.ready')).touch()\n"
            " peer=root/(('second' if role=='first' else 'first')+'.ready')\n"
            " deadline=time.monotonic()+10\n"
            " while not peer.exists():\n"
            "  if time.monotonic()>deadline:raise RuntimeError('independent tests did not overlap')\n"
            "  time.sleep(.01)\n"
            "else:\n"
            " guard=root/'exclusive-owner';guard.mkdir()\n"
            " time.sleep(.15)\n guard.rmdir()\n"
        )
        with (self.root / "CMakeLists.txt").open("a") as stream:
            for role in ("first", "second", "locked-first", "locked-second"):
                stream.write(f'add_test(NAME {role} COMMAND "{Path(self.python).as_posix()}" '
                             f'"{worker.as_posix()}" {role} "{scratch.as_posix()}")\n'
                             f'set_tests_properties({role} PROPERTIES LABELS "host;boundary")\n')
            stream.write('set_tests_properties(locked-first locked-second PROPERTIES RESOURCE_LOCK fixture_owner)\n')
        self.configure_fixture(self.preset, True, "fixture-parallel")
        args = SimpleNamespace(reason="real checkpoint scheduling and lock witnesses", selection=["boundary"],
                               preset=self.preset, runtime="host", campaign="parallel", repeat_reason="", base=None, prepare=True)
        result = read_json(qualification.qualify(self.root, args))
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["membership"]["boundary"]), 5)

    def test_retry_reason_preserves_matching_successful_checks(self):
        first = self.run_stage("boundary")
        original = Path(first["receipt"]).read_bytes()
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_stage("tooling", campaign="failed-tooling",
                           command=[self.python, "-c", "raise SystemExit(7)"])
        reused = self.run_stage("boundary", campaign="retry-tooling",
                                repeat_reason="retry the failed tooling check")
        self.assertEqual(reused["reused_from"], first["receipt"])
        self.assertFalse((self.campaign("retry-tooling").directory / "attempts").exists())
        self.assertEqual(Path(first["receipt"]).read_bytes(), original)
        self.assertEqual(self.state["tooling"]["status"], "failed")

    def test_product_cli_requires_explicit_build_then_reuses_completed_checks(self):
        import shutil
        import verify
        self.compiled_project()
        if os.name == "nt":
            compiler = shutil.which("clang-cl.exe")
            self.assertIsNotNone(compiler, "qualified Windows compiler is missing")
            presets_path = self.root / "CMakePresets.json"
            presets = read_json(presets_path)
            for preset in presets["configurePresets"]:
                preset["cacheVariables"].update(CMAKE_C_COMPILER=compiler, CMAKE_CXX_COMPILER=compiler)
            write_json(presets_path, presets)
        self.configure_fixture(self.preset, True, "cli-configured")
        argv = ["verify.py", "qualification", "--preset", self.preset, "--reason", "admission regression",
                "--selection", "product", "--runtime", "host", "--base", "fixture", "--campaign", "cli-admit"]
        with patch.object(verify, "ROOT", self.root), patch.object(sys, "argv", argv), \
                patch.object(Campaign, "run", side_effect=AssertionError("default CLI built implicitly")):
            with self.assertRaisesRegex(ValueError, "admission failed without changing outputs"):
                verify.main()
        self.assertFalse(self.state_path.exists())
        self.assertFalse(list(self.build.glob("*.pending.json")))
        argv[-1] = "cli-explicit"
        with patch.object(verify, "ROOT", self.root), patch.object(sys, "argv", [*argv, "--prepare"]):
            self.assertEqual(verify.main(), 0)
        argv[-1] = "cli-reused"
        with patch.object(verify, "ROOT", self.root), \
                patch.object(sys, "argv", [*argv, "--repeat-reason", "finish another failed check"]), \
                patch.object(Campaign, "run", side_effect=AssertionError("completed CLI checks reran")):
            self.assertEqual(verify.main(), 0)
        self.assertEqual((self.root / "out" / (self.preset + "-quality-count.txt")).read_text(), "1")

    def test_readonly_admission_rejects_environment_source_and_artifact_changes(self):
        from build_dependencies import NinjaGraph
        first = self.prepare_compiled_project()
        state_bytes = self.state_path.read_bytes()
        outputs = {Path(name): Path(name).read_bytes() for name in first["payloads"]}
        receipt = Path(first["receipt"]).read_bytes()
        source = self.root / "src/native.cpp"
        original_source, stamp = source.read_bytes(), source.stat()
        cases = ["environment", "source", "artifact"]
        for fault in cases:
            with self.subTest(fault=fault):
                environment = dict(os.environ)
                if fault == "environment":
                    environment["RUNNER_DIAGNOSTIC_ONLY"] = "changed"
                elif fault == "source":
                    source.write_bytes(original_source.replace(b"6+BINDING", b"7+BINDING"))
                    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                else:
                    next(iter(outputs)).write_bytes(b"corrupt")
                before = {path: path.read_bytes() for path in outputs}
                with patch.object(Campaign, "run", side_effect=AssertionError("admission executed a command")), \
                        patch.object(NinjaGraph, "invalidate", side_effect=AssertionError("admission deleted outputs")):
                    with self.assertRaisesRegex(ValueError, "admission failed without changing outputs"):
                        qualification.prepare_checkpoint(self.campaign("readonly-" + fault), self.build,
                            self.state_path, self.state, environment)
                self.assertEqual(before, {path: path.read_bytes() for path in outputs})
                self.assertEqual(self.state_path.read_bytes(), state_bytes)
                self.assertEqual(Path(first["receipt"]).read_bytes(), receipt)
                self.assertFalse(list(self.build.glob("*.pending.json")))
                source.write_bytes(original_source)
                os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                for path, content in outputs.items():
                    path.write_bytes(content)
        reused = qualification.prepare_checkpoint(self.campaign("readonly-restored"), self.build,
            self.state_path, self.state, dict(os.environ))
        self.assertEqual(reused["receipt"], first["receipt"])

    def test_explicit_glob_preparation_is_admitted_without_rebuilding(self):
        from build_evidence import prepare_targets
        self.compiled_project()
        cmake = self.root / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() +
                         '\nfile(GLOB sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")\n')
        producer = prepare_targets(self.root, self.preset, ["all"])
        stamp = self.build / "CMakeFiles/cmake.verify_globs"
        stamp.touch()
        with patch.object(Campaign, "run", side_effect=AssertionError("qualification rebuilt explicit targets")):
            first = qualification.prepare_checkpoint(self.campaign("admit-explicit"), self.build,
                self.state_path, self.state, dict(os.environ))
        self.assertEqual(first["receipt"], str(producer))
        state_bytes = self.state_path.read_bytes()
        (self.root / "src/new.cpp").write_text("int new_source(){return 1;}\n")
        with patch.object(Campaign, "run", side_effect=AssertionError("new glob input rebuilt implicitly")):
            with self.assertRaisesRegex(ValueError, "admission failed without changing outputs"):
                qualification.prepare_checkpoint(self.campaign("changed-glob"), self.build,
                    self.state_path, self.state, dict(os.environ))
        self.assertEqual(self.state_path.read_bytes(), state_bytes)

    def test_gpu_retry_preserves_completed_host_partition(self):
        args = self.mixed_project()
        fault = self.root / "failure.txt"
        fault.write_text("gpu")
        with self.assertRaises(subprocess.CalledProcessError):
            qualification.qualify(self.root, args)
        directory = self.root / "out/validation" / self.preset / args.campaign
        original = {p: p.read_bytes() for p in directory.glob("attempts/*/result.json")}
        fault.unlink()
        args.campaign = "retry-gpu-only"
        args.repeat_reason = "diagnosed GPU fault removed"
        result = read_json(qualification.qualify(self.root, args))
        self.assertEqual((self.root / "out/order.txt").read_text().splitlines(), ["host", "gpu", "gpu"])
        self.assertEqual(result["scope_status"], "passed")
        self.assertTrue(all(p.read_bytes() == content for p, content in original.items()))

    def test_host_retry_runs_only_failed_case_and_composes_original_evidence(self):
        args = self.mixed_project()
        fault = self.root / "failure.txt"
        fault.write_text("host")
        with self.assertRaises(subprocess.CalledProcessError):
            qualification.qualify(self.root, args)
        fault.unlink()
        args.campaign = "retry-one-host-case"
        args.repeat_reason = "diagnosed host fault removed"
        path = qualification.qualify(self.root, args)
        record = read_json(path)
        stage = record["stages"]["boundary"]
        self.assertEqual(len(stage["additional_receipts"]), 2)
        paths = qualification.stage_receipts(stage)
        raw = [read_json(p) for p in paths]
        self.assertEqual([r["returncode"] for r in raw], [8, 0, 0])
        for receipt, chunk in zip(paths, stage["ctest_chunks"], strict=True):
            log = receipt.with_name("LastTest.log")
            self.assertEqual(chunk["last_test_log_sha256"], digest(log))
            self.assertEqual(stage["payloads"][str(log)], digest(log))
        self.assertEqual(raw[1]["identity"]["command"][3:5], ["-R", "^(host)$"])
        self.assertEqual(record["scope_status"], "passed")
        args.campaign = "reuse-composed"
        with patch.object(Campaign, "run", side_effect=AssertionError("completed cases reran")):
            reused = read_json(qualification.qualify(self.root, args))
        self.assertEqual(reused["stages"]["boundary"]["reused_from"], stage["receipt"])
        with self.assertRaisesRegex(ValueError, "full checkpoint membership"):
            qualification.collect_qualification(self.root, [path])

    def test_partial_result_tampering_is_rejected_before_retry_execution(self):
        args = self.mixed_project()
        (self.root / "failure.txt").write_text("gpu")
        with self.assertRaises(subprocess.CalledProcessError):
            qualification.qualify(self.root, args)
        state = read_json(self.state_path)
        chunk = state["boundary"]["ctest_chunks"][0]
        report = Path(chunk["report"])
        log = Path(chunk["receipt"]).with_name("LastTest.log")
        args.repeat_reason = "retry GPU"
        for name, path in (("report", report), ("log", log)):
            with self.subTest(payload=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                args.campaign = "tampered-partial-" + name
                with patch.object(Campaign, "run", side_effect=AssertionError("tampered partial result executed")):
                    with self.assertRaisesRegex(ValueError, "partial result payload changed"):
                        qualification.qualify(self.root, args)
                path.write_bytes(original)

    def test_missing_partition_log_stops_before_gpu(self):
        args = self.mixed_project()
        actual = Campaign.run
        def missing_log(campaign, stage, command, **kwargs):
            path = actual(campaign, stage, command, **kwargs)
            (self.build / "Testing/Temporary/LastTest.log").unlink()
            return path
        with patch.object(Campaign, "run", autospec=True, side_effect=missing_log):
            with self.assertRaises(FileNotFoundError):
                qualification.qualify(self.root, args)
        self.assertEqual((self.root / "out/order.txt").read_text().splitlines(), ["host"])
        self.assertEqual(read_json(self.state_path)["boundary"]["status"], "failed")

    def test_partition_logs_survive_overwrite_and_are_checked_on_reuse(self):
        args = self.mixed_project()
        record = read_json(qualification.qualify(self.root, args))
        stage = record["stages"]["boundary"]
        logs = [path.with_name("LastTest.log") for path in qualification.stage_receipts(stage)]
        self.assertEqual(len(logs), 2)
        for log in logs:
            self.assertTrue(log.is_file(), str(log))
            self.assertEqual(stage["payloads"][str(log)], digest(log))
        self.assertIn('"host"', logs[0].read_text())
        self.assertNotIn('"gpu"', logs[0].read_text())
        self.assertIn('"gpu"', logs[1].read_text())
        self.assertNotEqual(logs[0].read_bytes(), logs[1].read_bytes())
        self.assertEqual(logs[1].read_bytes(), (self.build / "Testing/Temporary/LastTest.log").read_bytes())
        args.campaign = "reused-partition-logs"
        with patch.object(Campaign, "run", side_effect=AssertionError("completed partition reran")):
            reused = read_json(qualification.qualify(self.root, args))
        self.assertEqual(reused["stages"]["boundary"]["payloads"], stage["payloads"])
        logs[0].write_bytes(logs[0].read_bytes() + b"changed")
        args.campaign = "changed-partition-log"
        with patch.object(Campaign, "run", side_effect=AssertionError("changed log launched a command")):
            with self.assertRaisesRegex(ValueError, "result payload changed"):
                qualification.qualify(self.root, args)

    def test_nonzero_ctest_without_reported_case_failure_cannot_be_composed(self):
        args = self.mixed_project()
        record = read_json(qualification.qualify(self.root, args))
        stage = record["stages"]["boundary"]
        raw = read_json(Path(stage["receipt"]))
        raw.update(status="failed", returncode=8)
        unproved = self.root / "nonzero-ctest.json"
        unproved.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "did not execute completely"):
            qualification.ctest_chunk(unproved, record["runtime_membership"]["boundary"],
                                      stage["identity"]["command"])

    def assert_runtime_artifact_change_invalidates_partial(self, *, directory):
        args = self.mixed_project()
        probe = self.root / "probe.py"
        generated = self.build / "case.py"
        resources = self.build / "Resources"
        resources.mkdir()
        mode = resources / "mode.txt"
        mode.write_text("gpu")
        source = probe.read_text().replace("root=Path(__file__).parent", "root=Path(__file__).parents[3]")
        original_mode = "mode=(root/'failure.txt').read_text() if (root/'failure.txt').exists() else ''"
        replacement = "mode=(Path(sys.argv[2])/'mode.txt').read_text()" if directory else "mode='gpu'"
        probe.write_text(source.replace(original_mode, replacement))
        cmake = self.root / "CMakeLists.txt"
        text = cmake.read_text().replace(f'"{probe.as_posix()}"', f'"{generated.as_posix()}"')
        if directory:
            for role in ("host", "gpu"):
                text = text.replace(f'" {role})', f'" {role} "{resources.as_posix()}")')
        cmake.write_text(text + '\nconfigure_file(probe.py "${CMAKE_BINARY_DIR}/case.py" COPYONLY)\n')
        self.configure_fixture(self.preset, True, "generated-consumers")
        with self.assertRaises(subprocess.CalledProcessError):
            qualification.qualify(self.root, args)
        state = read_json(self.state_path)
        before = snapshot(self.root, list(state["boundary"]["identity"]["inputs"]))
        if directory:
            mode.write_text("host")
        else:
            generated.write_text(generated.read_text().replace("mode='gpu'", "mode='host'"))
        self.assertEqual(snapshot(self.root, list(before)), before)
        args.campaign = "changed-runtime-consumer"
        args.repeat_reason = "retry after runtime artifact replacement"
        with self.assertRaises(subprocess.CalledProcessError):
            qualification.qualify(self.root, args)
        self.assertEqual((self.root / "out/order.txt").read_text().splitlines(), ["host", "gpu", "host"])
        self.assertEqual(read_json(self.state_path)["boundary"]["status"], "failed")

    def test_partial_retry_rechecks_changed_generated_runtime_file(self):
        self.assert_runtime_artifact_change_invalidates_partial(directory=False)

    def test_partial_retry_rechecks_changed_runtime_resource_directory(self):
        self.assert_runtime_artifact_change_invalidates_partial(directory=True)

    def test_runtime_artifact_binding_excludes_cargo_output_and_seals_membership(self):
        cargo = self.build / "cargo"
        cargo.mkdir()
        (cargo / "incremental-output").write_text("mutable")
        resources = self.build / "Resources"
        resources.mkdir()
        resource = resources / "input.json"
        resource.write_text("input")
        script = self.build / "case.py"
        script.write_text("runtime consumer")
        inventory = {"tests": [{"name": "consumer", "command": ["cargo", "test", "--target-dir", str(cargo),
                       str(script), str(resources)], "properties": [{"name": "WORKING_DIRECTORY", "value": str(self.build)}]}]}
        files, directories = qualification.ctest_artifacts(inventory, ["consumer"], self.build)
        self.assertEqual(set(files), {str(script), str(resource)})
        self.assertEqual(directories, [str(resources)])
        before = qualification.artifact_membership(directories)
        (cargo / "new-output").write_text("does not change runtime input membership")
        self.assertEqual(qualification.ctest_artifacts(inventory, ["consumer"], self.build), (files, directories))
        (resources / "new-input.json").write_text("new runtime member")
        self.assertNotEqual(qualification.artifact_membership(directories), before)

    def test_invalid_request_preserves_pass_before_reuse_or_input_change(self):
        first = self.run_stage("boundary")
        original_state = self.state_path.read_bytes()
        producer = Path(first["receipt"])
        evidence = {path: path.read_bytes() for path in producer.parent.iterdir() if path.is_file()}
        original_source = self.source.read_bytes()
        command = [self.python, "-c", "print('checked')"]
        cases = [
            ("undeclared", "boundary", command, ["input.txt"], 10, ["tooling"], "declared stage"),
            ("unsafe", "../boundary", command, ["input.txt"], 10, ["boundary"], "lowercase directory name"),
            ("command", "boundary", [], ["input.txt"], 10, ["boundary"], "command"),
            ("inputs", "boundary", command, [], 10, ["boundary"], "input files"),
            *[(f"timeout-{index}", "boundary", command, ["input.txt"], timeout, ["boundary"], "finite positive timeout")
              for index, timeout in enumerate((0, -1, float("nan"), float("inf"), -float("inf")))],
        ]
        for changed in (False, True):
            self.source.write_bytes(b"changed" if changed else original_source)
            for case, stage, selected_command, inputs, timeout, required, message in cases:
                with self.subTest(changed=changed, case=case):
                    campaign = Campaign(self.root, self.preset, f"rejected-{int(changed)}-{case}", required)
                    with patch.object(campaign, "run", side_effect=AssertionError("invalid request dispatched")) as run:
                        with self.assertRaisesRegex(ValueError, message):
                            qualification.run_stage(campaign, self.state_path, self.state, stage, selected_command,
                                inputs=inputs, tools=[self.python], environment=self.environment, timeout=timeout,
                                semantic_command=command, action=lambda: self.fail("invalid request invoked action"))
                        run.assert_not_called()
                    self.assertEqual(self.state_path.read_bytes(), original_state)
                    self.assertEqual(self.state, json.loads(original_state))
                    self.assertFalse((campaign.directory / "attempts").exists())
                    self.assertTrue(all(path.read_bytes() == content for path, content in evidence.items()))
        self.source.write_bytes(original_source)
        campaign = self.campaign("valid-after-rejection")
        with patch.object(campaign, "run", side_effect=AssertionError("reusable pass dispatched")):
            reused = qualification.run_stage(campaign, self.state_path, self.state, "boundary", command,
                inputs=["input.txt"], tools=[self.python], environment=self.environment, timeout=10,
                action=lambda: self.fail("reusable pass invoked action"))
        self.assertEqual(reused["reused_from"], first["receipt"])
        self.assertEqual(self.state_path.read_bytes(), original_state)
        self.assertTrue(all(path.read_bytes() == content for path, content in evidence.items()))

    def test_invalid_request_preserves_cold_and_failed_checkpoint(self):
        command = [self.python, "-c", "raise SystemExit(7)"]
        rejected = Campaign(self.root, self.preset, "undeclared", ["tooling"])
        with self.assertRaisesRegex(ValueError, "declared stage"):
            qualification.run_stage(rejected, self.state_path, self.state, "boundary", command,
                inputs=["input.txt"], tools=[self.python], environment=self.environment, timeout=10,
                action=lambda: self.fail("cold invalid request invoked action"))
        self.assertEqual(self.state, {})
        self.assertFalse(self.state_path.exists())
        self.assertFalse((rejected.directory / "attempts").exists())

        with self.assertRaises(subprocess.CalledProcessError):
            self.run_stage("boundary", command=command)
        original_state = self.state_path.read_bytes()
        self.assertEqual(self.state["boundary"]["status"], "failed")
        attempts = self.campaign("first").directory / "attempts"
        evidence = {path: path.read_bytes() for path in attempts.rglob("*") if path.is_file()}
        with self.assertRaisesRegex(ValueError, "declared stage"):
            qualification.run_stage(rejected, self.state_path, self.state, "boundary", command,
                inputs=["input.txt"], tools=[self.python], environment=self.environment, timeout=10,
                action=lambda: self.fail("invalid request replaced failed gate"))
        self.assertEqual(self.state_path.read_bytes(), original_state)
        self.assertEqual(self.state, json.loads(original_state))
        self.assertFalse((rejected.directory / "attempts").exists())
        self.assertTrue(all(path.read_bytes() == content for path, content in evidence.items()))
        with self.assertRaisesRegex(ValueError, "pending checkpoint"):
            self.run_stage("boundary", campaign="valid-still-pending", command=command)

    def test_known_output_destinations_do_not_change_gate_identity(self):
        first = self.run_stage("boundary", semantic_command=[self.python, "same", "<junit>"])
        result = self.run_stage("boundary", campaign="renamed", command=[self.python, "unused-output-path"],
                                semantic_command=[self.python, "same", "<junit>"])
        self.assertEqual(result["reused_from"], first["receipt"])

    def test_unrelated_repair_preserves_success_and_pending_failure(self):
        first = self.run_stage("boundary")
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_stage("tooling", inputs=["other.txt"], command=[self.python, "-c", "raise SystemExit(1)"])
        self.other.write_text("repair")
        result = self.run_stage("boundary", campaign="new-caller")
        self.assertEqual(result["reused_from"], first["receipt"])
        self.assertEqual(read_json(self.state_path)["tooling"]["status"], "failed")
        self.run_stage("prepare", campaign="preparation", inputs=["other.txt"])
        self.assertEqual(self.state["tooling"]["status"], "failed")

    def test_dependency_content_with_restored_timestamp_invalidates_reuse(self):
        first = self.run_stage("boundary")
        stamp = self.source.stat()
        self.source.write_text("other")
        os.utime(self.source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        result = self.run_stage("boundary", campaign="changed")
        self.assertNotIn("reused_from", result)
        self.assertNotEqual(result["receipt"], first["receipt"])

    def test_failed_gate_cannot_be_reset_by_changing_caller(self):
        command = [self.python, "-c", "raise SystemExit(1)"]
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_stage("boundary", command=command)
        with self.assertRaisesRegex(ValueError, "pending checkpoint"):
            self.run_stage("boundary", campaign="renamed", command=command)
        self.assertEqual(self.state["boundary"]["status"], "failed")

    def test_reuse_validation_failure_stays_pending_across_other_gates(self):
        first = self.run_stage("prepare")
        original = Path(first["receipt"]).read_bytes()
        def reject(path):
            raise ValueError("required output missing")
        with self.assertRaisesRegex(ValueError, "required output"):
            self.run_stage("prepare", campaign="missing-output", validate=reject)
        self.run_stage("tooling", campaign="different-caller", inputs=["other.txt"])
        self.assertEqual(self.state["prepare"]["status"], "failed")
        self.assertEqual(Path(first["receipt"]).read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "pending checkpoint"):
            self.run_stage("prepare", campaign="another-caller")

    def test_input_mutation_during_execution_cannot_publish_success(self):
        command = [self.python, "-c", "from pathlib import Path;Path('input.txt').write_text('mutated')"]
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_stage("boundary", command=command)
        self.assertEqual(self.state["boundary"]["status"], "failed")

    def test_old_pass_cannot_bypass_new_failure_during_collection(self):
        first = self.run_stage("boundary")
        record = {"mode": "checkpoint", "status": "passed", "scope_status": "passed", "reason": "retained",
                  "preset": self.preset, "selection": ["boundary"], "required": ["boundary"], "stages": {"boundary": first}}
        path = self.root / "old-checkpoint.json"
        path.write_text(json.dumps(record))
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_stage("boundary", campaign="failure", command=[self.python, "-c", "raise SystemExit(1)"])
        with self.assertRaisesRegex(ValueError, "pending"):
            qualification.collect_qualification(self.root, [path])

    def test_other_pending_gate_cannot_be_hidden_by_collecting_old_selection(self):
        first = self.run_stage("boundary")
        record = {"mode": "checkpoint", "status": "passed", "scope_status": "passed", "reason": "retained",
                  "preset": self.preset, "selection": ["boundary"], "required": ["boundary"], "stages": {"boundary": first}}
        path = self.root / "old-checkpoint.json"
        path.write_text(json.dumps(record))
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_stage("tooling", command=[self.python, "-c", "raise SystemExit(1)"])
        with self.assertRaisesRegex(ValueError, "pending checkpoint work"):
            qualification.collect_qualification(self.root, [path])

    def test_preparation_binds_its_owned_post_configuration_without_a_second_run(self):
        config = self.build / "CMakeCache.txt"
        config.write_text("before")
        inputs = ["input.txt", str(config.relative_to(self.root))]
        def prepare():
            path = self.campaign("first").run("prepare", [self.python, "-c", "print('configured')"], inputs=["input.txt"], timeout=10)
            config.write_text("after")
            return path
        first = self.run_stage("prepare", inputs=inputs, action=prepare)
        reused = self.run_stage("prepare", campaign="renamed", inputs=inputs, action=prepare)
        self.assertEqual(first["receipt"], reused["reused_from"])

    def test_boundary_inputs_bind_native_membership_and_foreign_fixture(self):
        for name in ("src/RenderValues.cpp", "src/JuicerState.cpp", "tests/ffi/color_preparation_test.cpp",
                     "tests/ffi/print_illuminant_test.inc", "tests/ffi/print_illuminant_fixture.h",
                     "tests/print/illuminant-fixtures/rust-manifest.json", "tests/ffi/print_preflash_test.inc",
                     "tests/print/preflash-fixtures/manifest.json", "tests/print/preflash-fixtures/linux-debug.json",
                     "tests/ffi/print_balance_test.inc", "tests/print/balance-fixtures/manifest.json",
                     "tests/print/balance-fixtures/linux-debug.json", "tests/ffi/scanner_correction_test.inc",
                     "tests/scanner/correction-fixtures/manifest.json", "tests/scanner/correction-fixtures/checked-expectations.json",
                     "tests/scanner/correction-fixtures/linux-debug.json", "tests/ffi/scanner_lut_test.inc", "tests/ffi/scanner_lut_descriptor_test.inc",
                     "tests/scanner/descriptor-fixtures/manifest.json", "tests/scanner/descriptor-fixtures/native.json",
                     "tests/scanner/descriptor-fixtures/context-linux-debug.json",
                     "tests/scanner/post-effects-fixtures/manifest.json", "tests/scanner/post-effects-fixtures/native.json",
                     "tests/scanner/post-effects-fixtures/context-linux-debug.json", "tests/ffi/scanner_post_effects_test.inc",
                     "tests/scanner/lut-fixtures/manifest.json", "tests/scanner/lut-fixtures/checked.json",
                     "tests/scanner/lut-fixtures/native.json", "tests/scanner/lut-fixtures/context-linux-debug.json",
                     "tests/ffi/exposure_fixture.h", "tests/ffi/scanner_color_test.inc", "tests/ffi/gaussian_kernel_test.inc", "tests/ffi/frame_geometry_test.inc",
                     "tests/frame/geometry-fixtures/manifest.json", "tests/frame/geometry-fixtures/native.json", "tests/frame/geometry-fixtures/context.json", "tests/frame/geometry-fixtures/inputs.json",
                     "tests/optics/gaussian-fixtures/manifest.json", "tests/optics/gaussian-fixtures/native.json", "tests/optics/gaussian-fixtures/context.json",
                     "tests/ffi/diffusion_frame_test.inc", "tests/diffusion/frame-fixtures/manifest.json",
                     "tests/diffusion/frame-fixtures/native.json", "tests/diffusion/frame-fixtures/context.json",
                     "tests/diffusion/frame-fixtures/checked.json", "tests/diffusion/frame-fixtures/inputs.json", "tests/diffusion/frame-fixtures/edge-inputs.json",
                     "tests/ffi/halation_frame_test.inc", "tests/scatter_halation/frame-fixtures/manifest.json",
                     "tests/scatter_halation/frame-fixtures/native.json", "tests/scatter_halation/frame-fixtures/context.json",
                     "tests/scatter_halation/frame-fixtures/checked.json", "tests/scatter_halation/frame-fixtures/inputs.json",
                     "tests/scanner/color-fixtures/manifest.json", "tests/scanner/color-fixtures/native.json",
                     "tests/scanner/color-fixtures/checked.json", "tests/ffi/film_reference_test.inc",
                     "tests/film/reference-fixtures/manifest.json", "tests/film/reference-fixtures/linux-debug.json"):
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("source")
        (self.build / "CMakeCache.txt").write_text("configured")
        (self.build / "CTestTestfile.cmake").write_text("")
        first = qualification.checkpoint_inputs(self.root, self.build, "boundary")
        self.assertIn("src/RenderValues.cpp", first)
        self.assertIn("tests/ffi/film_reference_test.inc", first)
        self.assertIn("tests/film/reference-fixtures/manifest.json", first)
        self.assertIn("tests/film/reference-fixtures/linux-debug.json", first)
        self.assertIn("tests/ffi/color_preparation_test.cpp", first)
        self.assertIn("tests/ffi/print_illuminant_test.inc", first)
        self.assertIn("tests/ffi/print_illuminant_fixture.h", first)
        self.assertIn("tests/print/illuminant-fixtures/rust-manifest.json", first)
        self.assertIn("tests/ffi/print_preflash_test.inc", first)
        self.assertIn("tests/print/preflash-fixtures/manifest.json", first)
        self.assertIn("tests/print/preflash-fixtures/linux-debug.json", first)
        self.assertIn("tests/ffi/print_balance_test.inc", first)
        self.assertIn("tests/print/balance-fixtures/manifest.json", first)
        self.assertIn("tests/print/balance-fixtures/linux-debug.json", first)
        self.assertIn("tests/ffi/scanner_correction_test.inc", first)
        self.assertIn("tests/scanner/correction-fixtures/manifest.json", first)
        self.assertIn("tests/scanner/correction-fixtures/checked-expectations.json", first)
        self.assertIn("tests/scanner/correction-fixtures/linux-debug.json", first)
        self.assertIn("tests/ffi/scanner_color_test.inc", first)
        self.assertIn("tests/scanner/color-fixtures/manifest.json", first)
        self.assertIn("tests/scanner/color-fixtures/native.json", first)
        self.assertIn("tests/scanner/color-fixtures/checked.json", first)
        self.assertIn("tests/ffi/scanner_lut_descriptor_test.inc", first)
        self.assertIn("tests/scanner/descriptor-fixtures/manifest.json", first)
        self.assertIn("tests/scanner/descriptor-fixtures/native.json", first)
        self.assertIn("tests/scanner/descriptor-fixtures/context-linux-debug.json", first)
        self.assertIn("tests/scanner/post-effects-fixtures/manifest.json", first)
        self.assertIn("tests/scanner/post-effects-fixtures/native.json", first)
        self.assertIn("tests/scanner/post-effects-fixtures/context-linux-debug.json", first)
        self.assertIn("tests/ffi/scanner_lut_test.inc", first)
        self.assertIn("tests/scanner/lut-fixtures/manifest.json", first)
        self.assertIn("tests/scanner/lut-fixtures/checked.json", first)
        self.assertIn("tests/scanner/lut-fixtures/native.json", first)
        self.assertIn("tests/scanner/lut-fixtures/context-linux-debug.json", first)
        self.assertIn("tests/ffi/exposure_fixture.h", first)
        added = self.root / "src/SpectralProcessing.h"
        added.write_text("new member")
        self.assertIn("src/SpectralProcessing.h", qualification.checkpoint_inputs(self.root, self.build, "boundary"))
        self.assertNotIn("src/SpectralProcessing.h", first)
        fixture = self.root / "tests/print/illuminant-fixtures/product-updates.json"
        fixture.write_text("new selected expectation member")
        self.assertIn(str(fixture.relative_to(self.root)).replace("\\", "/"),
                      qualification.checkpoint_inputs(self.root, self.build, "boundary"))
        self.assertNotIn("tests/print/illuminant-fixtures/product-updates.json", first)

    def test_cargo_configuration_content_and_membership_are_tool_inputs(self):
        (self.build / "CMakeCache.txt").write_text("")
        home = self.root / "cargo-home"
        home.mkdir()
        config = home / "config.toml"
        environment = {"PATH": "", "CARGO_HOME": str(home), "RUSTUP_HOME": str(self.root / "rustup")}
        before = qualification.checkpoint_tools(self.build, environment)
        absent = self.run_stage("boundary", tools=before)
        config.write_text('[build]\n')
        after = qualification.checkpoint_tools(self.build, environment)
        self.assertIn(str(config), before)
        self.assertIn(str(config), after)
        first = self.run_stage("boundary", tools=after)
        self.assertNotEqual(absent["receipt"], first["receipt"])
        config.write_text('[build]\nrustflags=["--cfg", "changed"]\n')
        changed = self.run_stage("boundary", campaign="config-changed", tools=qualification.checkpoint_tools(self.build, environment))
        self.assertNotEqual(first["receipt"], changed["receipt"])
        self.assertNotIn("reused_from", changed)

    def native_project(self):
        (self.root / ".gitignore").write_text("out/\n")
        subprocess.run(["git", "init", "-q", str(self.root)], check=True, capture_output=True, timeout=10)
        (self.root / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.24)\nproject(Checkpoint NONE)\ninclude(CTest)\n'
            'add_custom_target(material ALL COMMAND "${CMAKE_COMMAND}" -E copy_if_different '
            '"${CMAKE_SOURCE_DIR}/input.txt" "${CMAKE_BINARY_DIR}/material.txt" '
            'DEPENDS "${CMAKE_SOURCE_DIR}/input.txt" BYPRODUCTS "${CMAKE_BINARY_DIR}/material.txt")\n'
            'if(BUILD_TESTING)\nadd_test(NAME boundary COMMAND "${CMAKE_COMMAND}" -E true)\n'
            'set_tests_properties(boundary PROPERTIES LABELS "host;boundary")\nendif()\n')
        (self.root / "CMakePresets.json").write_text(json.dumps({"version": 6, "configurePresets": [{
            "name": self.preset, "generator": "Ninja", "binaryDir": str(self.build)}], "testPresets": [{
                "name": self.preset, "configurePreset": self.preset,
                "execution": {"jobs": 4, "noTestsAction": "error"}}]}))
        self.configure_fixture(self.preset, False, "fixture-base")

    def test_real_native_dispatcher_executes_and_reuses_exact_ctest_membership(self):
        self.native_project()
        self.configure_fixture(self.preset, True, "fixture-on")
        args = SimpleNamespace(reason="native caller regression", selection=["boundary"], preset=self.preset,
                               runtime="host", campaign="caller-one", repeat_reason="", base=None, prepare=True)
        import verify
        argv = ["verify.py", "qualification", "--preset", self.preset, "--reason", args.reason,
                "--selection", "boundary", "--runtime", "host", "--campaign", args.campaign]
        with patch.object(verify, "ROOT", self.root), patch.object(sys, "argv", argv):
            self.assertEqual(verify.main(), 0)
        first = read_json(self.root / "out/validation" / self.preset / "caller-one/qualification.json")
        args.campaign = "caller-two"
        second = read_json(qualification.qualify(self.root, args))
        self.assertEqual(first["membership"], {"boundary": ["boundary"]})
        self.assertEqual(second["status"], "passed")
        self.assertEqual(second["stages"]["boundary"]["reused_from"], first["stages"]["boundary"]["receipt"])
        self.assertFalse((self.root / "out/validation" / self.preset / "caller-two/attempts").exists())

    def test_documentation_changes_reuse_preparation_and_ctest_after_commit(self):
        from verify import source_snapshot
        self.native_project()
        cmake = self.root / "CMakeLists.txt"
        cmake.write_text(cmake.read_text().replace('"host;boundary"', '"host;corpus"'))
        documents = [self.root / name for name in (
            "README.md", "CONTRIBUTING.md", "docs/workflow.md", "tests/README.md",
            "tests/film/fixtures/README.md")]
        for path in documents:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Initial documentation.\n")

        def commit(message):
            subprocess.run(["git", "add", "--all"], cwd=self.root, check=True, capture_output=True, timeout=10)
            subprocess.run(["git", "-c", "user.name=Checkpoint fixture", "-c", "user.email=checkpoint@example.invalid",
                            "-c", "commit.gpgsign=false", "commit", "-qm", message],
                           cwd=self.root, check=True, capture_output=True, timeout=10)

        commit("Initial miniature source and documentation")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.root, timeout=10)
        prepared = qualification.prepare_checkpoint(self.campaign("documentation-preparation"), self.build,
            self.state_path, self.state, self.environment, timeout=60, allow_build=True)
        args = SimpleNamespace(reason="documentation discovery and retained native evidence", selection=["corpus"],
                               preset=self.preset, runtime="host", campaign="documentation-baseline", repeat_reason="", base=None, prepare=True)
        baseline = qualification.qualify(self.root, args)
        record = read_json(baseline)
        stage = record["stages"]["corpus"]
        self.assertEqual(record["membership"], {"corpus": ["boundary"]})
        protected = {path: digest(path) for path in (baseline, Path(prepared["receipt"]), Path(stage["receipt"]))}
        state_bytes = self.state_path.read_bytes()
        source = source_snapshot(self.root)

        for operation in ("edit", "add", "delete", "commit"):
            with self.subTest(operation=operation):
                if operation == "edit":
                    for path in documents:
                        path.write_text("Revised documentation.\n")
                elif operation == "add":
                    for name in ("docs/new-guide.md", "tests/new-domain/README.md"):
                        path = self.root / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text("New documentation.\n")
                elif operation == "delete":
                    for path in documents:
                        path.unlink()
                else:
                    commit("Documentation only")
                    self.assertNotEqual(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.root, timeout=10), head)
                self.assertEqual(source_snapshot(self.root), source)
                with patch.object(Campaign, "run", side_effect=AssertionError("documentation reran a gate")), \
                        patch.object(qualification, "prepare_build", side_effect=AssertionError("documentation rebuilt the fixture")):
                    current_state = read_json(self.state_path)
                    reused = qualification.prepare_checkpoint(self.campaign("documentation-prepare-" + operation),
                        self.build, self.state_path, current_state, self.environment, timeout=60, allow_build=True)
                    self.assertEqual(reused["reused_from"], prepared["receipt"])
                    args.campaign = "documentation-" + operation
                    current = read_json(qualification.qualify(self.root, args))
                    self.assertEqual(current["stages"]["corpus"]["reused_from"], stage["receipt"])
                    with self.assertRaisesRegex(ValueError, "full checkpoint membership is incomplete"):
                        qualification.collect_qualification(self.root, [baseline])
                self.assertEqual(self.state_path.read_bytes(), state_bytes)
                self.assertEqual(protected, {path: digest(path) for path in protected})
                self.assertEqual(prepared["payloads"], {name: digest(Path(name)) for name in prepared["payloads"]})
                self.assertEqual(stage["payloads"], {name: digest(Path(name)) for name in stage["payloads"]})

    def runtime_project(self):
        self.native_project()
        fixture = self.root / "tests/fixtures/runtime.json"
        fixture.parent.mkdir(parents=True)
        fixture.write_text('{"value": 1}\n')
        loader = self.root / "tests/runtime_loader.py"
        loader.write_text("import argparse, json, pathlib\n"
                          "parser = argparse.ArgumentParser()\n"
                          "parser.add_argument('fixture', type=pathlib.Path)\n"
                          "parser.add_argument('--output-dir', type=pathlib.Path, required=True)\n"
                          "args = parser.parse_args()\n"
                          "value = json.loads(args.fixture.read_text())['value']\n"
                          "(args.output_dir / 'observed.txt').write_text(str(value))\n"
                          "assert value > 0\n")
        cmake = self.root / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() +
            f'add_test(NAME runtime COMMAND "{Path(self.python).as_posix()}" '
            '"${CMAKE_SOURCE_DIR}/tests/runtime_loader.py" "${CMAKE_SOURCE_DIR}/tests/fixtures/runtime.json" '
            '--output-dir "${CMAKE_BINARY_DIR}")\n'
            'set_tests_properties(runtime PROPERTIES LABELS "host;corpus")\n')
        self.configure_fixture(self.preset, True, "runtime-on")
        return fixture, loader

    def test_runtime_edit_reuses_preparation_but_executes_changed_test_input(self):
        import build_evidence
        fixture, loader = self.runtime_project()
        first = qualification.prepare_checkpoint(self.campaign("runtime-prepare"), self.build,
            self.state_path, self.state, self.environment, timeout=60, allow_build=True)
        args = SimpleNamespace(reason="runtime preparation boundary", selection=["corpus"], preset=self.preset,
                               runtime="host", campaign="runtime-first", repeat_reason="", base=None, prepare=True)
        original = read_json(qualification.qualify(self.root, args))["stages"]["corpus"]
        observed = self.build / "observed.txt"
        self.assertEqual(observed.read_text(), "1")
        inventory = json.loads(subprocess.check_output(
            ["ctest", "--test-dir", str(self.build), "--show-only=json-v1"], text=True))
        self.assertEqual(qualification.ctest_artifacts(inventory, ["runtime"], self.build), ([], []))
        self.assertNotIn(str(observed), original["identity"]["tools"])
        protected = {p: p.read_bytes() for p in (
            Path(first["receipt"]), *build_evidence.preparation_paths(self.build, ["all"])[:1],
            *(Path(name) for name in first["payloads"]))}
        for index, path in enumerate((fixture, loader), start=2):
            with self.subTest(path=path.name):
                if path == fixture:
                    path.write_text('{"value": 2}\n')
                else:
                    path.write_text(path.read_text().replace("str(value)", "str(value + 1)"))
                with patch.object(qualification, "prepare_build", side_effect=AssertionError("preparation redispatched")), \
                        patch.object(Campaign, "run", side_effect=AssertionError("configure/build/clean dispatched")):
                    reused = qualification.prepare_checkpoint(self.campaign("runtime-reuse-" + str(index)), self.build,
                        self.state_path, read_json(self.state_path), self.environment, timeout=60, allow_build=True)
                self.assertEqual(reused["reused_from"], first["receipt"])
                self.assertTrue(all(p.read_bytes() == value for p, value in protected.items()))
                args.campaign = "runtime-changed-" + str(index)
                record_path = qualification.qualify(self.root, args)
                current = read_json(record_path)["stages"]["corpus"]
                self.assertNotIn("reused_from", current)
                self.assertNotEqual(current["receipt"], original["receipt"])
                self.assertNotIn(str(observed), current["identity"]["tools"])
                self.assertNotEqual(current["identity"]["inputs"], original["identity"]["inputs"])
                self.assertEqual(observed.read_text(), str(index))
                original = current

    def test_runtime_variable_configure_copy_requires_fresh_preparation(self):
        self.assert_configure_copy_reprepared(
            'set(stem runtime)\n'
            'execute_process(COMMAND "${CMAKE_COMMAND}" -E copy '
            '"${CMAKE_SOURCE_DIR}/tests/fixtures/${stem}.json" "${CMAKE_BINARY_DIR}/configured.json" '
            'COMMAND_ERROR_IS_FATAL ANY)\n')

    def test_runtime_variable_file_read_requires_fresh_preparation(self):
        self.assert_configure_copy_reprepared(
            'set(operation READ)\nset(stem runtime)\n'
            'file(${operation} "${CMAKE_SOURCE_DIR}/tests/fixtures/${stem}.json" payload)\n'
            'file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/configured.json" CONTENT "${payload}")\n')

    def test_runtime_literal_file_read_requires_fresh_preparation(self):
        self.assert_configure_copy_reprepared(
            'file(READ "${CMAKE_SOURCE_DIR}/tests/fixtures/runtime.json" payload)\n'
            'file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/configured.json" CONTENT "${payload}")\n')

    def test_runtime_build_tree_module_read_requires_fresh_preparation(self):
        self.assert_configure_copy_reprepared(
            'set(stem runtime)\n'
            'file(READ "${CMAKE_SOURCE_DIR}/tests/fixtures/${stem}.json" payload)\n'
            'file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/configured.json" CONTENT "${payload}")\n',
            module=self.build / "consumer.cmake")

    def test_runtime_external_module_copy_requires_fresh_preparation(self):
        self.assert_configure_copy_reprepared(
            'set(stem runtime)\n'
            'execute_process(COMMAND "${CMAKE_COMMAND}" -E copy '
            '"${CMAKE_SOURCE_DIR}/tests/fixtures/${stem}.json" "${CMAKE_BINARY_DIR}/configured.json" '
            'COMMAND_ERROR_IS_FATAL ANY)\n',
            module=self.directory / "external/consumer.cmake")

    def test_runtime_bundled_module_copy_requires_fresh_preparation(self):
        self.assert_configure_copy_reprepared(
            'include(CPackIFWConfigureFile)\nset(stem runtime)\n'
            'cpack_ifw_configure_file("${CMAKE_SOURCE_DIR}/tests/fixtures/${stem}.json" '
            '"${CMAKE_BINARY_DIR}/configured.json")\n',
            bundled_module="CPackIFWConfigureFile.cmake")

    def test_runtime_included_listfile_requires_fresh_preparation(self):
        self.assert_configure_copy_reprepared(
            'set(stem runtime)\n'
            'file(READ "${CMAKE_SOURCE_DIR}/tests/fixtures/${stem}.json" payload)\n'
            'file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/configured.json" CONTENT "${payload}")\n',
            module=self.build / "consumer.inc")

    def test_bootstrap_exceptions_require_exact_location_and_bytes(self):
        self.native_project()
        cache = self.build / "CMakeCache.txt"
        cmake_root = Path(next(line.partition("=")[2] for line in cache.read_text().splitlines()
                              if line.startswith("CMAKE_ROOT:INTERNAL=")))
        tool_root = self.root / "tool"
        bootstrap = tool_root / "Modules/CTestTargets.cmake"
        bootstrap.parent.mkdir(parents=True)
        content = (cmake_root / "Modules/CTestTargets.cmake").read_bytes()
        bootstrap.write_bytes(content)
        cache.write_text(f"CMAKE_ROOT:INTERNAL={tool_root.as_posix()}\n")

        def opaque(path):
            return qualification.has_opaque_configure_commands([], build=self.build,
                configuration={"regeneration_mtimes": {str(path): path.stat().st_mtime_ns}})

        self.assertFalse(opaque(bootstrap))
        self.assertTrue(qualification.has_opaque_configure_commands([bootstrap], build=self.build,
            configuration={"regeneration_mtimes": {str(bootstrap): bootstrap.stat().st_mtime_ns}}))
        # A known name/location alone never establishes the exception.
        bootstrap.write_bytes(content + b"\n# changed installation\n")
        self.assertTrue(opaque(bootstrap))
        shadow = self.build / "bootstrap.inc"
        shadow.write_bytes(content)
        self.assertTrue(opaque(shadow))
        shadow.write_bytes(b"\xff")
        self.assertTrue(opaque(shadow))

    def test_runtime_new_configure_membership_rejects_mid_preparation_edit(self):
        import build_evidence
        fixture, _ = self.runtime_project()
        module = self.build / "consumer.inc"
        module.write_text('set(stem runtime)\n'
            'file(READ "${CMAKE_SOURCE_DIR}/tests/fixtures/${stem}.json" payload)\n'
            'file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/configured.json" CONTENT "${payload}")\n')
        cmake = self.root / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() + f'include("{module.as_posix()}")\n'
            'add_custom_target(staged ALL COMMAND "${CMAKE_COMMAND}" -E copy_if_different '
            '"${CMAKE_BINARY_DIR}/configured.json" "${CMAKE_BINARY_DIR}/staged.json" '
            'DEPENDS "${CMAKE_BINARY_DIR}/configured.json" BYPRODUCTS "${CMAKE_BINARY_DIR}/staged.json")\n')
        self.assertNotIn("tests/fixtures/runtime.json", qualification.checkpoint_inputs(self.root, self.build, "prepare"))
        run = Campaign.run
        edited = False

        def edit_after_configure(campaign, stage, command, **kwargs):
            nonlocal edited
            result = run(campaign, stage, command, **kwargs)
            if not edited and ("--preset" in command or "-S" in command):
                fixture.write_text('{"value": 2}\n')
                edited = True
            return result

        campaign = self.campaign("new-membership-edit")
        with patch.object(Campaign, "run", new=edit_after_configure):
            with self.assertRaisesRegex(ValueError, "newly discovered preparation inputs changed"):
                qualification.prepare_checkpoint(campaign, self.build, self.state_path, self.state, self.environment, timeout=60, allow_build=True)
        self.assertTrue(edited)
        self.assertEqual(self.state["prepare"]["status"], "failed")
        write_json(campaign.directory / "failed-checkpoint.json", self.state)
        for name in ("configured.json", "staged.json"):
            self.assertEqual(read_json(self.build / name)["value"], 1)
        baseline = build_evidence.preparation_paths(self.build, ["all"])[0]
        original = Path(read_json(baseline)["producer"])
        original_receipt = original.read_bytes()
        renewed = qualification.prepare_checkpoint(self.campaign("new-membership-repaired"), self.build,
            self.state_path, self.state, self.environment, timeout=60, allow_build=True)
        self.assertNotIn("reused_from", renewed)
        for name in ("configured.json", "staged.json"):
            self.assertEqual(read_json(self.build / name)["value"], 2)
        self.assertEqual(original.read_bytes(), original_receipt)

    def assert_configure_copy_reprepared(self, configure_commands, *, module=None, bundled_module=None):
        started = time.monotonic()
        def progress(stage):
            print(f"Checkpoint {self._testMethodName} stage={stage} seconds={time.monotonic() - started:.3f}",
                  file=sys.stderr, flush=True)
        progress("runtime-project")
        fixture, _ = self.runtime_project()
        cmake = self.root / "CMakeLists.txt"
        if module:
            module.parent.mkdir(parents=True, exist_ok=True)
            module.write_text(configure_commands)
            configure_commands = f'include("{module.as_posix()}")\n'
        cmake.write_text(cmake.read_text() + configure_commands +
            'add_custom_target(staged ALL COMMAND "${CMAKE_COMMAND}" -E copy_if_different '
            '"${CMAKE_BINARY_DIR}/configured.json" "${CMAKE_BINARY_DIR}/staged.json" '
            'DEPENDS "${CMAKE_BINARY_DIR}/configured.json" BYPRODUCTS "${CMAKE_BINARY_DIR}/staged.json")\n')
        if module or bundled_module:
            from verify import source_snapshot
            self.assertFalse(qualification.has_opaque_configure_commands(self.root / name for name in source_snapshot(self.root)))
        progress("first-preparation")
        first = qualification.prepare_checkpoint(self.campaign("variable-copy-first"), self.build,
            self.state_path, self.state, self.environment, timeout=60, allow_build=True)
        progress("first-preparation-complete")
        producer = Path(first["receipt"])
        original_receipt = producer.read_bytes()
        if bundled_module:
            configuration = json.loads(original_receipt)["scoped_preparation"]["context"]["configuration"]
            module = next(Path(name) for name in configuration["regeneration_mtimes"] if Path(name).name == bundled_module)
        if module:
            configuration = json.loads(original_receipt)["scoped_preparation"]["context"]["configuration"]
            self.assertIn(str(module), configuration["regeneration_mtimes"])
            identities = configuration["inputs"] if module.is_relative_to(self.root) else configuration["tools"]
            name = module.relative_to(self.root).as_posix() if module.is_relative_to(self.root) else str(module)
            self.assertEqual(identities[name], digest(module))
            self.assertNotIn(str(fixture), configuration["regeneration_mtimes"])
            original_module = module.read_bytes()
        self.assertIn("tests/fixtures/runtime.json", first["identity"]["inputs"])
        configured, staged = (self.build / name for name in ("configured.json", "staged.json"))
        self.assertEqual(read_json(staged)["value"], 1)
        self.assertIn(str(staged), first["payloads"])
        (producer.parent / "original-staged.json").write_bytes(staged.read_bytes())
        progress("unchanged-admission")
        with patch.object(qualification, "prepare_build", side_effect=AssertionError("unchanged opaque scope redispatched")), \
                patch.object(Campaign, "run", side_effect=AssertionError("unchanged opaque scope executed")):
            unchanged = qualification.prepare_checkpoint(self.campaign("variable-copy-unchanged"), self.build,
                self.state_path, self.state, self.environment, timeout=60, allow_build=True)
        self.assertEqual(unchanged["reused_from"], first["receipt"])
        progress("unchanged-admission-complete")
        fixture.write_text('{"value": 2}\n')
        commands = []
        run = Campaign.run

        def observe(campaign, stage, command, **kwargs):
            commands.append(list(command))
            return run(campaign, stage, command, **kwargs)

        progress("changed-preparation")
        with patch.object(Campaign, "run", new=observe):
            second = qualification.prepare_checkpoint(self.campaign("variable-copy-changed"), self.build,
                self.state_path, self.state, self.environment, timeout=60, allow_build=True)
        progress("changed-preparation-complete")
        write_json(self.build / "variable-copy-control.json", {
            "first_receipt": first["receipt"], "second_receipt": second["receipt"],
            "commands": commands, "runtime": read_json(fixture),
            "configured": read_json(configured), "staged": read_json(staged)})
        self.assertEqual(read_json(configured)["value"], 2)
        self.assertEqual(read_json(staged)["value"], 2)
        if module:
            self.assertEqual(module.read_bytes(), original_module)
        self.assertNotIn("reused_from", second)
        self.assertNotEqual(second["receipt"], first["receipt"])
        self.assertIn("tests/fixtures/runtime.json", second["identity"]["inputs"])
        self.assertTrue(any("--preset" in command or "-S" in command for command in commands))
        self.assertEqual(producer.read_bytes(), original_receipt)

    def test_runtime_preparation_keeps_build_consumers_and_unknown_inputs(self):
        import build_evidence
        fixture, loader = self.runtime_project()
        cmake = self.root / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() +
            'configure_file(tests/fixtures/runtime.json generated.json COPYONLY)\n'
            'add_custom_target(staged ALL COMMAND "${CMAKE_COMMAND}" -E copy '
            '"${CMAKE_SOURCE_DIR}/tests/runtime_loader.py" "${CMAKE_BINARY_DIR}/staged.py" '
            'DEPENDS "${CMAKE_SOURCE_DIR}/tests/runtime_loader.py" BYPRODUCTS "${CMAKE_BINARY_DIR}/staged.py")\n')
        unknown = self.root / "tests/fixtures/unknown.json"
        unknown.write_text("unknown dependency")
        resources = self.root / "Resources/source.json"
        resources.parent.mkdir()
        resources.write_text("resource")
        tool = self.root / "generator.py"
        tool.write_text("# build generator\n")
        cmake.write_text(cmake.read_text() +
            'add_custom_target(generator ALL COMMAND "${CMAKE_COMMAND}" -E true '
            'DEPENDS "${CMAKE_SOURCE_DIR}/generator.py")\n')
        first = qualification.prepare_checkpoint(self.campaign("build-consumers"), self.build,
            self.state_path, self.state, self.environment, timeout=60, allow_build=True)
        preparation = qualification.checkpoint_inputs(self.root, self.build, "prepare")
        for path in (fixture, loader, unknown, resources, cmake, tool):
            self.assertIn(path.relative_to(self.root).as_posix(), preparation)
            before = path.read_bytes()
            path.write_bytes(before + b" ")
            self.assertNotEqual(snapshot(self.root, preparation), first["identity"]["inputs"])
            with self.assertRaises(ValueError):
                build_evidence.check_scoped_preparation(self.campaign("build-consumers"), self.build, first["receipt"])
            path.write_bytes(before)

    def test_runtime_preparation_retains_configure_copies_and_opaque_cargo_inputs(self):
        fixture, _ = self.runtime_project()
        cmake = self.root / "CMakeLists.txt"
        original = cmake.read_text()
        for operation in (
            'file(COPY "${CMAKE_SOURCE_DIR}/tests/fixtures/runtime.json" DESTINATION "${CMAKE_BINARY_DIR}/copied")\n',
            'file(READ "${CMAKE_SOURCE_DIR}/tests/fixtures/runtime.json" fixture_text)\n',
            'set(fixture "${CMAKE_SOURCE_DIR}/tests/fixtures/runtime.json")\nfile(READ "${fixture}" fixture_text)\n',
            'execute_process(COMMAND "${CMAKE_COMMAND}" -E cat "${CMAKE_SOURCE_DIR}/tests/fixtures/runtime.json")\n',
            'set(operation execute_process)\ncmake_language(CALL ${operation} COMMAND "${CMAKE_COMMAND}" -E true)\n',
        ):
            with self.subTest(operation=operation):
                cmake.write_text(original + operation)
                self.assertIn("tests/fixtures/runtime.json", qualification.checkpoint_inputs(self.root, self.build, "prepare"))
        cmake.write_text(original)
        cmake.write_text(original +
            'add_custom_target(copy_directory ALL COMMAND "${CMAKE_COMMAND}" -E copy_directory '
            '"${CMAKE_SOURCE_DIR}/tests/fixtures" "${CMAKE_BINARY_DIR}/staged-fixtures")\n')
        self.configure_fixture(self.preset, True, "runtime-directory-copy")
        self.assertIn("tests/fixtures/runtime.json", qualification.checkpoint_inputs(self.root, self.build, "prepare"))
        cmake.write_text(original)
        self.configure_fixture(self.preset, True, "runtime-directory-copy-restored")
        for name, content in (("rust/src/lib.rs", 'const DATA: &str = include_str!("data.json");\n'),
                              ("rust/src/lib.rs", 'const DATA: &[u8] = include_bytes!("data.json");\n'),
                              ("rust/src/lib.rs", 'include!("data.json");\n'),
                              ("Cargo.toml", '[package]\nbuild = "custom_build.rs"\n'),
                              ("rust/build.rs", "fn main() {}\n")):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            self.assertIn(fixture.relative_to(self.root).as_posix(),
                          qualification.checkpoint_inputs(self.root, self.build, "prepare"))
            path.unlink()

    def test_runtime_preparation_preserves_explicit_and_older_broad_requirements(self):
        import build_evidence
        fixture, _ = self.runtime_project()
        with patch.object(qualification, "preparation_inputs", side_effect=lambda root, build, paths, **kwargs: paths):
            first = qualification.prepare_checkpoint(self.campaign("broad"), self.build,
                self.state_path, self.state, self.environment, timeout=60, allow_build=True)
        receipt = Path(first["receipt"])
        original = receipt.read_bytes()
        fixture.write_text('{"value": 2}\n')
        with self.assertRaisesRegex(ValueError, "dependency contents or membership changed"):
            build_evidence.check_scoped_preparation(self.campaign("broad"), self.build, receipt)
        renewed = qualification.prepare_checkpoint(self.campaign("narrow"), self.build,
            self.state_path, self.state, self.environment, timeout=60, allow_build=True)
        self.assertNotEqual(renewed["receipt"], first["receipt"])
        self.assertEqual(receipt.read_bytes(), original)
        # Direct prepare_build inputs remain authoritative even for a file the
        # checkpoint can prove is runtime-only.
        explicit = build_evidence.prepare_build(self.campaign("explicit"), "prepare", source=self.root,
            build=self.build, inputs=["input.txt", "tests/fixtures/runtime.json"], timeout=60,
            targets=["all"], configure_args=["-DBUILD_TESTING=ON"], checkpoint=True)[1]
        original_explicit = explicit.read_bytes()
        fixture.write_text('{"value": 3}\n')
        with self.assertRaisesRegex(ValueError, "dependency contents or membership changed"):
            build_evidence.check_scoped_preparation(self.campaign("explicit"), self.build, explicit)
        self.assertEqual(explicit.read_bytes(), original_explicit)

    def test_runtime_preparation_does_not_admit_tampered_or_unproved_owners(self):
        import build_evidence
        fixture, _ = self.runtime_project()
        first = qualification.prepare_checkpoint(self.campaign("owned"), self.build,
            self.state_path, self.state, self.environment, timeout=60, allow_build=True)
        baseline, pending = build_evidence.preparation_paths(self.build, ["all"])
        original_owner = baseline.read_bytes()
        original_receipt = Path(first["receipt"]).read_bytes()
        artifact = next(Path(name) for name in first["payloads"] if Path(name).name == "material.txt")
        original_artifact = artifact.read_bytes()
        fixture.write_text('{"value": 2}\n')
        for fault in ("artifact", "pending", "unproved"):
            with self.subTest(fault=fault):
                if fault == "artifact":
                    artifact.write_bytes(b"tampered")
                elif fault == "pending":
                    pending.write_text('{"status": "unproved"}')
                else:
                    baseline.unlink()
                with patch.object(Campaign, "run", side_effect=AssertionError("unsafe owner dispatched")):
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        qualification.prepare_checkpoint(self.campaign("rejected-" + fault), self.build,
                            self.state_path, read_json(self.state_path), self.environment, timeout=60, allow_build=True)
                self.assertEqual(Path(first["receipt"]).read_bytes(), original_receipt)
                # Faults are isolated in this miniature, never shared owners.
                artifact.write_bytes(original_artifact)
                if pending.exists():
                    pending.rename(self.build / "unproved-pending-retained.json")
                if not baseline.exists():
                    baseline.write_bytes(original_owner)
                write_json(self.state_path, {"prepare": first})

    def test_runtime_preparation_collection_uses_its_own_membership(self):
        self.assert_runtime_preparation_collection()

    def test_runtime_module_preparation_collection_keeps_configure_inputs(self):
        self.assert_runtime_preparation_collection(module=self.build / "consumer.cmake")

    def assert_runtime_preparation_collection(self, *, module=None):
        fixture, _ = self.runtime_project()
        cmake = self.root / "CMakeLists.txt"
        cmake.write_text(cmake.read_text().replace('"host;corpus"', '"host;product"'))
        if module:
            module.write_text('set(stem runtime)\n'
                'file(READ "${CMAKE_SOURCE_DIR}/tests/fixtures/${stem}.json" payload)\n'
                'file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/configured.json" CONTENT "${payload}")\n')
            cmake.write_text(cmake.read_text() + f'include("{module.as_posix()}")\n')
        quality = self.root / "scripts/check-quality.py"
        quality.parent.mkdir()
        quality.write_text("print('miniature quality scheduling witness')\n")
        args = SimpleNamespace(reason="miniature preparation collection", selection=["product"], preset=self.preset,
                               runtime="all", campaign="collection-first", repeat_reason="", base="fixture", prepare=True)
        path = qualification.qualify(self.root, args)
        record = read_json(path)
        self.assertEqual("tests/fixtures/runtime.json" in record["stages"]["prepare"]["identity"]["inputs"], bool(module))
        self.assertIn("tests/fixtures/runtime.json", record["stages"]["product"]["identity"]["inputs"])
        original = path.read_bytes()
        with patch.object(Campaign, "run", side_effect=AssertionError("collector dispatched")):
            with self.assertRaisesRegex(ValueError, "full checkpoint membership is incomplete"):
                qualification.collect_qualification(self.root, [path])
        fixture.write_text('{"value": 2}\n')
        if module:
            with patch.object(Campaign, "run", side_effect=AssertionError("collector dispatched")):
                with self.assertRaisesRegex(ValueError, "checkpoint declared source/data inputs changed"):
                    qualification.collect_qualification(self.root, [path])
        args.campaign = "collection-changed"
        refreshed = qualification.qualify(self.root, args)
        result = read_json(refreshed)
        if module:
            self.assertNotIn("reused_from", result["stages"]["prepare"])
            self.assertEqual(read_json(self.build / "configured.json")["value"], 2)
        else:
            self.assertEqual(result["stages"]["prepare"]["reused_from"], record["stages"]["prepare"]["receipt"])
        self.assertNotIn("reused_from", result["stages"]["product"])
        with self.assertRaisesRegex(ValueError, "full checkpoint membership is incomplete"):
            qualification.collect_qualification(self.root, [refreshed])
        self.assertEqual(path.read_bytes(), original)

    def test_runtime_preparation_cold_discovery_preserves_producer_requirements(self):
        self.native_project()
        self.assertFalse((self.build / "CTestTestfile.cmake").exists())
        fixture = self.root / "tests/runtime.txt"
        fixture.parent.mkdir()
        fixture.write_text("runtime input")
        cmake = self.root / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() +
            'add_test(NAME runtime COMMAND "${CMAKE_COMMAND}" -E cat "${CMAKE_SOURCE_DIR}/tests/runtime.txt")\n')
        first = qualification.prepare_checkpoint(self.campaign("cold-discovery"), self.build,
            self.state_path, self.state, self.environment, timeout=60, allow_build=True)
        proof = read_json(Path(first["receipt"]))["scoped_preparation"]
        declared = {name for name in first["identity"]["inputs"] if not name.startswith("out/build/")}
        self.assertEqual(declared, set(proof["inputs"]))
        self.assertIn("tests/runtime.txt", declared)
        self.assertIn("tests/runtime.txt", qualification.checkpoint_inputs(self.root, self.build, "prepare",
            preparation=proof["context"]))
        self.assertNotIn("tests/runtime.txt", qualification.checkpoint_inputs(self.root, self.build, "prepare"))

    def test_explicitly_consumed_markdown_remains_a_freshness_input(self):
        name = "docs/runtime-contract.md"
        document = self.root / name
        document.parent.mkdir(parents=True)
        document.write_text("first")
        self.assertIn(name, qualification.checkpoint_inputs(self.root, self.build, "corpus", source_files=[name]))
        command = [self.python, "-c", "from pathlib import Path; assert Path('docs/runtime-contract.md').read_text() == 'first'"]
        first = self.run_stage("boundary", command=command, inputs=[name])
        receipt = Path(first["receipt"])
        original = receipt.read_bytes()
        document.write_text("changed consumed content")
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_stage("boundary", campaign="changed-markdown", command=command, inputs=[name])
        self.assertEqual(self.state["boundary"]["status"], "failed")
        self.assertEqual(receipt.read_bytes(), original)

    def production_fixture(self):
        """Native tiny probes exercise retained proof, not product mathematics."""
        prepared_source = self.root / "tests/ffi/production-prepared-render.cpp"
        prepared_source.parent.mkdir(parents=True, exist_ok=True)
        prepared_source.write_text("int main() { return 0; }\n", encoding="utf-8")
        exposure = self.root / "tests/ofx/fixtures/exposure"
        exposure.mkdir(parents=True, exist_ok=True)
        prepared_fixtures = {}
        for preset in qualification.PRESETS:
            fixture = exposure / f"processor-{preset}.dat"
            fixture.write_text("negative-direct 105 " + " ".join(["0"] * 105) + "\n"
                               "negative-print 140 " + " ".join(["0"] * 140) + "\n",
                               encoding="utf-8")
            prepared_fixtures[fixture.name] = digest(fixture)
        write_json(exposure / "manifest.json", {"files": prepared_fixtures})
        illuminant = self.root / "tests/print/illuminant-fixtures"
        illuminant.mkdir(parents=True, exist_ok=True)
        write_json(illuminant / "manifest.json", {"schema": 1})
        write_json(illuminant / "product-updates.json", {"schema": 1, "fixtures": {}})
        # The Windows PDB-only control supplies a synthetic graph. Give it the
        # exact output path too; real mini-project runs overwrite this with the
        # CMake-linked target during their explicit target preparation.
        render_executable = self.build / "tests/bin" / (
            "JuicerProductionPreparedRenderProbe.exe" if os.name == "nt"
            else "JuicerProductionPreparedRenderProbe")
        if not render_executable.exists():
            render_executable.parent.mkdir(parents=True, exist_ok=True)
            render_executable.write_bytes(b"miniature prepared-render executable")
        for name in ("grain_frame_abi_assertions.inc", "grain_probe_contract.py", "effects_frame_abi_assertions.inc", "effects_frame_probe_contract.py"):
            target = self.root / "tests/ffi" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / "tests/ffi" / name).read_bytes())
        directory = self.root / "tests/film/reference-fixtures"
        directory.mkdir(parents=True, exist_ok=True)
        payload = directory / "mini.json"
        write_json(payload, {"scope": "fixture membership protocol only", "products": [
            {"method": method, "input": {}, "reference": {"hash": 10}} for method in (0, 1)]})
        write_json(directory / "manifest.json", {"presets": {
            preset: payload.name for preset in qualification.PRESETS},
            "payload_sha256": {payload.name: digest(payload)}})
        directory = self.root / "tests/scanner/descriptor-fixtures"
        directory.mkdir(parents=True, exist_ok=True)
        payload = directory / "native.json"
        context = directory / "context.json"
        write_json(payload, {"scope": "fixture membership protocol only"})
        write_json(context, {"scope": "distinct native context protocol"})
        write_json(directory / "manifest.json", {"presets": {preset: {"native": "logical.json"} for preset in qualification.PRESETS},
            "payloads": {"logical.json": payload.name}, "contexts": {"logical.json": context.name},
            "sha256": {payload.name: digest(payload), context.name: digest(context)},
            "production_cases": [{"case": str(i), "envelope": [0, 0, 0, 0, 0], "hash": 10, "fields": "10-pass"} for i in range(24)],
            "production_malformed": [{"case": str(i), "envelope": [2, 0, 0, 1, 0], "hash": 0, "fields": "10-pass"} for i in range(17)]})
        post_directory = self.root / "tests/scanner/post-effects-fixtures"
        post_directory.mkdir(parents=True, exist_ok=True)
        post_payload = post_directory / "native.json"
        post_context = post_directory / "context.json"
        write_json(post_payload, {"scope": "post-effects retained membership protocol only"})
        write_json(post_context, {"scope": "distinct post-effects native context protocol"})
        write_json(post_directory / "manifest.json", {"presets": {preset: {"native": "logical.json"} for preset in qualification.PRESETS},
            "payloads": {"logical.json": post_payload.name}, "contexts": {"logical.json": post_context.name},
            "sha256": {post_payload.name: digest(post_payload), post_context.name: digest(post_context)},
            "production_cases": [{"case": str(i), "envelope": [0, 0, 0, 0, 0], "hash": 10, "fields": "13-pass"} for i in range(410)]})
        frame_directory = self.root / "tests/frame/geometry-fixtures"
        frame_directory.mkdir(parents=True, exist_ok=True)
        frame_payload = frame_directory / "native.json"
        frame_context = frame_directory / "context.json"
        frame_inputs = frame_directory / "inputs.json"
        write_json(frame_payload, {"scope": "frame full-field retained protocol only"})
        write_json(frame_context, {"scope": "distinct frame native context protocol"})
        write_json(frame_inputs, {"scope": "frame consumed specification protocol"})
        write_json(frame_directory / "manifest.json", {
            "presets": {preset: {"native": "logical.json"} for preset in qualification.PRESETS},
            "payloads": {"logical.json": frame_payload.name}, "contexts": {"logical.json": frame_context.name},
            "sha256": {p.name: digest(p) for p in (frame_payload, frame_context, frame_inputs)},
            "production_cases": [{"case": str(i), "envelope": [0, 0, 0, 0], "domain": [1, 2, 3, 4], "pixel_bits": 1, "diffusion_bits": 2, "diagnostic": ""} for i in range(69)]})
        halation_directory = self.root / "tests/scatter_halation/frame-fixtures"
        halation_directory.mkdir(parents=True, exist_ok=True)
        halation_files = {name: halation_directory / name for name in ("native.json", "context.json", "checked.json", "inputs.json")}
        for name, path in halation_files.items():
            write_json(path, {"scope": "halation retained full-field protocol only", "file": name})
        write_json(halation_directory / "manifest.json", {
            "presets": {preset: {"native": "logical.json"} for preset in qualification.PRESETS},
            "payloads": {"logical.json": "native.json"}, "contexts": {"logical.json": "context.json"}, "checked": "checked.json",
            "sha256": {name: digest(path) for name, path in halation_files.items()},
            "production_cases": [{"case": str(i), "envelope": [0, 0, 0], "failure": [0, 0, 3, 3], "sigma_bits": 0, "sigma_nan": False,
                                  "output": {"recipe_hash": i + 1, "words": [0] * 528 + [i]}, "diagnostic": ""} for i in range(86)]})
        diffusion_directory = self.root / "tests/diffusion/frame-fixtures"
        diffusion_directory.mkdir(parents=True, exist_ok=True)
        diffusion_files = {name: diffusion_directory / name for name in ("native.json", "context.json", "checked.json", "inputs.json", "edge-inputs.json")}
        for name, path in diffusion_files.items():
            write_json(path, {"scope": "diffusion retained full-field protocol only", "file": name})
        write_json(diffusion_directory / "manifest.json", {
            "presets": {preset: {"native": "logical.json", "edges": "logical.json"} for preset in qualification.PRESETS},
            "payloads": {"logical.json": "native.json"}, "contexts": {"logical.json": "context.json"}, "checked": "checked.json",
            "sha256": {name: digest(path) for name, path in diffusion_files.items()},
            "production_cases": [{"case": str(i), "envelope": [0, 0, 0], "failure": [0, 0, 0],
                                  "output": [0] * 35 + [i], "diagnostic": ""} for i in range(507)]})
        dir_directory = self.root / "tests/dir/frame-fixtures"
        dir_directory.mkdir(parents=True, exist_ok=True)
        dir_files = {name: dir_directory / name for name in ("native.json", "checked.json", "inputs.json")}
        for name, path in dir_files.items():
            write_json(path, {"scope": "DIR retained full-field/trace protocol only", "file": name})
        write_json(dir_directory / "manifest.json", {
            "presets": {preset: {"native": "logical.json"} for preset in qualification.PRESETS},
            "payloads": {"logical.json": "native.json"}, "checked": "checked.json",
            "sha256": {name: digest(path) for name, path in dir_files.items()},
            "production_cases": [{"case": str(i), "envelope": [0, 0, 0], "failure": [0, 0, 4] + [0] * 79,
                                  "output": [0] * 78 + [i], "diagnostic": ""} for i in range(121)]})
        grain_directory = self.root / "tests/grain/frame-fixtures"
        grain_directory.mkdir(parents=True, exist_ok=True)
        grain_files = {name: grain_directory / name for name in ("native.json", "checked.json", "inputs.json")}
        for name, path in grain_files.items():
            write_json(path, {"scope": "grain frame retained full-value/metadata protocol only", "file": name})
        write_json(grain_directory / "manifest.json", {
            "presets": {preset: {"native": "native.json"} for preset in qualification.PRESETS},
            "checked": "checked.json", "sha256": {name: digest(path) for name, path in grain_files.items()},
            "production_cases": [{"case": case, "envelope": [0, 0, 0], "failure": [0, 0, 3, 3],
                                  "output": [0] * 114 + [i], "diagnostic": "", "diagnostic_length": 0}
                                 for i, case in enumerate(qualification.GRAIN_FRAME_CASE_IDS)]})
        effects_directory = self.root / "tests/film/effects-frame-fixtures"
        effects_directory.mkdir(parents=True, exist_ok=True)
        effects_files = {name: effects_directory / name for name in ("native.json", "context.json", "checked.json", "inputs.json")}
        for name, path in effects_files.items():
            write_json(path, {"scope": "effects frame retained full-value/failure protocol only", "file": name})
        write_json(effects_directory / "manifest.json", {
            "presets": {preset: {"native": "logical.json"} for preset in qualification.PRESETS},
            "payloads": {"logical.json": "native.json"}, "contexts": {"logical.json": "context.json"},
            "checked": "checked.json", "sha256": {name: digest(path) for name, path in effects_files.items()},
            "production_cases": {preset: [{"case": case, "envelope": [0, 0, 0], "failure": [0],
                                          "output": [0] * 39 + [index], "diagnostic": "", "diagnostic_length": 0}
                                         for index, case in enumerate(qualification.EFFECTS_FRAME_CASE_IDS)]
                                 for preset in qualification.PRESETS}})
        boundary_directory = self.root / "tests/dir/boundary-fixtures"
        boundary_directory.mkdir(parents=True, exist_ok=True)
        boundary_files = {name: boundary_directory / name for name in ("native.json", "checked.json", "inputs.json")}
        for name, path in boundary_files.items():
            write_json(path, {"scope": "DIR boundary retained full-values protocol only", "file": name})
        write_json(boundary_directory / "manifest.json", {
            "presets": {preset: {"native": "native.json"} for preset in qualification.PRESETS},
            "checked": "checked.json", "sha256": {name: digest(path) for name, path in boundary_files.items()},
            "production_cases": [{"case": str(i), "envelope": [0, 0, 0], "key": i, "failure": [0, 0, 3, 0, 0, 0, 0],
                                  "shape": [1, 2, 3, 4, 1, 2], "weights": [i], "terminal": [0] * 17 + [i],
                                  "diagnostic": "", "diagnostic_length": 0, "owner": True} for i in range(171)]})
        return r'''
import json, sys, subprocess, hashlib, shutil, struct
preset=sys.argv[sys.argv.index("--preset")+1]
out=out.resolve()
def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
build=(Path('out/build')/preset).resolve()
build_sources=[build/name for name in ('CMakeCache.txt','build.ninja','compile_commands.json','cargo-features.txt',
                                     'libjuicer.a','juicer.lib','juicer.ofx','juicer.pdb',
                                     'tests/bin/JuicerProductionPreparedRenderProbe.exe' if sys.platform=='win32' else 'tests/bin/JuicerProductionPreparedRenderProbe') if (build/name).is_file()]
assert 'BUILD_TESTING:BOOL=OFF' in (build/'CMakeCache.txt').read_text().splitlines()
build_inputs={}
for source in build_sources:
    before=checksum(source)
    target=out/'build'/source.relative_to(build)
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    assert checksum(source)==checksum(target)==before
    build_inputs[str(source)]={'file':str(target),'sha256':before}
commands=[]
def execute(name, command):
    result=subprocess.run(command,capture_output=True,timeout=30,check=False)
    for extension,data in [('stdout',result.stdout),('stderr',result.stderr)]:
        (out/(name+'.'+extension)).write_bytes(data)
    commands.append({'name':name,'command':command,'returncode':result.returncode,
                     'stdout_sha256':checksum(out/(name+'.stdout')),
                     'stderr_sha256':checksum(out/(name+'.stderr'))})
    result.check_returncode()
header=out/'production-film-expected.h'
header.write_text('#define EXPECTED 10\n')
consumed={str(header):checksum(header)}
for name in ('grain_frame_abi_assertions.inc', 'grain_probe_contract.py', 'effects_frame_abi_assertions.inc', 'effects_frame_probe_contract.py'):
    helper=(Path('tests/ffi')/name).resolve()
    consumed[str(helper)]=checksum(helper)
prepared_source=Path('tests/ffi/production-prepared-render.cpp').resolve()
prepared_fixture_manifest=Path('tests/ofx/fixtures/exposure/manifest.json').resolve()
prepared_fixture=prepared_fixture_manifest.parent/f'processor-{preset}.dat'
prepared_illuminant_manifest=Path('tests/print/illuminant-fixtures/manifest.json').resolve()
prepared_overlay=Path('tests/print/illuminant-fixtures/product-updates.json').resolve()
consumed.update({str(path):checksum(path) for path in (Path('CMakeLists.txt').resolve(),prepared_source,
                    prepared_fixture_manifest,prepared_fixture,prepared_illuminant_manifest,prepared_overlay)})
reference_manifest=(Path('tests/film/reference-fixtures')/'manifest.json').resolve()
reference_spec=json.loads(reference_manifest.read_text())
reference_payload=reference_manifest.parent/reference_spec['presets'][preset]
consumed.update({str(path):checksum(path) for path in (reference_manifest,reference_payload)})
descriptor_manifest=(Path('tests/scanner/descriptor-fixtures')/'manifest.json').resolve()
descriptor_spec=json.loads(descriptor_manifest.read_text())
descriptor_payload=descriptor_manifest.parent/'native.json'
descriptor_context=descriptor_manifest.parent/'context.json'
consumed.update({str(path):checksum(path) for path in (descriptor_manifest,descriptor_payload,descriptor_context)})
descriptor_results=descriptor_spec['production_cases']+descriptor_spec['production_malformed']
post_manifest=(Path('tests/scanner/post-effects-fixtures')/'manifest.json').resolve()
post_spec=json.loads(post_manifest.read_text())
post_payload=post_manifest.parent/'native.json'
post_context=post_manifest.parent/'context.json'
consumed.update({str(path):checksum(path) for path in (post_manifest,post_payload,post_context)})
post_results=post_spec['production_cases']
frame_manifest=(Path('tests/frame/geometry-fixtures')/'manifest.json').resolve()
frame_spec=json.loads(frame_manifest.read_text())
frame_payload=frame_manifest.parent/'native.json'
frame_context=frame_manifest.parent/'context.json'
consumed.update({str(path):checksum(path) for path in (frame_manifest,frame_payload,frame_context,frame_manifest.parent/'inputs.json')})
frame_results=frame_spec['production_cases']
halation_manifest=(Path('tests/scatter_halation/frame-fixtures')/'manifest.json').resolve()
halation_spec=json.loads(halation_manifest.read_text())
halation_payload=halation_manifest.parent/'native.json'
halation_context=halation_manifest.parent/'context.json'
halation_checked=halation_manifest.parent/'checked.json'
halation_inputs=halation_manifest.parent/'inputs.json'
consumed.update({str(path):checksum(path) for path in (halation_manifest,halation_payload,halation_context,halation_checked,halation_inputs)})
halation_results=halation_spec['production_cases']
diffusion_manifest=(Path('tests/diffusion/frame-fixtures')/'manifest.json').resolve()
diffusion_spec=json.loads(diffusion_manifest.read_text())
diffusion_payload=diffusion_manifest.parent/'native.json'
diffusion_context=diffusion_manifest.parent/'context.json'
diffusion_checked=diffusion_manifest.parent/'checked.json'
diffusion_inputs=diffusion_manifest.parent/'inputs.json'
diffusion_edges_inputs=diffusion_manifest.parent/'edge-inputs.json'
consumed.update({str(path):checksum(path) for path in (diffusion_manifest,diffusion_payload,diffusion_context,diffusion_checked,diffusion_inputs,diffusion_edges_inputs)})
diffusion_results=diffusion_spec['production_cases']
dir_manifest=(Path('tests/dir/frame-fixtures')/'manifest.json').resolve()
dir_spec=json.loads(dir_manifest.read_text())
dir_payload=dir_manifest.parent/'native.json'
dir_checked=dir_manifest.parent/'checked.json'
dir_inputs=dir_manifest.parent/'inputs.json'
consumed.update({str(path):checksum(path) for path in (dir_manifest,dir_payload,dir_checked,dir_inputs)})
dir_results=dir_spec['production_cases']
grain_manifest=(Path('tests/grain/frame-fixtures')/'manifest.json').resolve()
grain_spec=json.loads(grain_manifest.read_text())
grain_payload=grain_manifest.parent/'native.json'
grain_checked=grain_manifest.parent/'checked.json'
grain_inputs=grain_manifest.parent/'inputs.json'
consumed.update({str(path):checksum(path) for path in (grain_manifest,grain_payload,grain_checked,grain_inputs)})
grain_results=grain_spec['production_cases']
effects_manifest=(Path('tests/film/effects-frame-fixtures')/'manifest.json').resolve()
effects_spec=json.loads(effects_manifest.read_text())
effects_payload=effects_manifest.parent/'native.json'
effects_context=effects_manifest.parent/'context.json'
effects_checked=effects_manifest.parent/'checked.json'
effects_inputs=effects_manifest.parent/'inputs.json'
consumed.update({str(path):checksum(path) for path in (effects_manifest,effects_payload,effects_context,effects_checked,effects_inputs)})
effects_results=effects_spec['production_cases'][preset]
boundary_manifest=(Path('tests/dir/boundary-fixtures')/'manifest.json').resolve()
boundary_spec=json.loads(boundary_manifest.read_text())
boundary_payload=boundary_manifest.parent/'native.json'
boundary_checked=boundary_manifest.parent/'checked.json'
boundary_inputs=boundary_manifest.parent/'inputs.json'
consumed.update({str(path):checksum(path) for path in (boundary_manifest,boundary_payload,boundary_checked,boundary_inputs)})
boundary_results=boundary_spec['production_cases']

reference_results=[]
for row in json.loads(reference_payload.read_text())['products']:
    for case in range(8 if row['method']==1 else 7):
        envelope=([0,0,0,0,0,0,0] if case in (0,5) else [2,0,0,1,0,0,0] if case in (1,4) else
                  [3,0,0,2,1,0,0] if case==2 else [3,0,0,2,2,1,0] if case==3 else
                  [3,0,0,3,0,7,0] if case==6 else [3,0,0,4,1,8,1])
        reference_results.append({'case':f"reference-{row['method']}-{case}",'envelope':envelope,
                                  'hash':row['reference']['hash'] if case in (0,5) else 0,'fields':'13-pass'})
probes=[]
for name in ['film','accepted-output-color']:
    source=out/(name+'.c')
    result_lines = ''.join('puts('+json.dumps('DIFFUSION_FRAME_RESULT '+json.dumps(row))+');' for row in diffusion_results) if name=='film' else ''
    if name=='film':result_lines += ''.join('puts('+json.dumps('GRAIN_FRAME_RESULT '+json.dumps(row))+');' for row in grain_results)
    if name=='film':result_lines += ''.join('puts('+json.dumps('EFFECTS_FRAME_RESULT '+json.dumps(row))+');' for row in effects_results)
    if name=='film':result_lines += ''.join('puts('+json.dumps('DIR_FRAME_RESULT '+json.dumps(row))+');' for row in dir_results)
    if name=='film':result_lines += ''.join('puts('+json.dumps('DIR_BOUNDARY_RESULT '+json.dumps(row))+');' for row in boundary_results)
    source.write_text('#include "production-film-expected.h"\n#include <stdio.h>\n'
                      'typedef struct Pair {int first; int second;} Pair;\n'
                      '_Static_assert(sizeof(Pair)==8,"ABI");\n'
                      'int main(void){printf("value=%d abi=%zu\\n",EXPECTED,sizeof(Pair));'+result_lines+'return EXPECTED!=10;}\n')
    executable=out/(name+('.exe' if sys.platform=='win32' else ''))
    if sys.platform=='win32':
        command=[shutil.which('clang-cl.exe'),'/nologo','/TC','/std:c11','/MD','/W4','/WX',
                 '/clang:-fuse-ld=lld',str(source),'/Fe:'+str(executable),'/Fo:'+str(out/(name+'.obj'))]
    else:
        command=['gcc-13','-std=c11','-Wall','-Wextra','-Werror',str(source),'-o',str(executable)]
    source_hash=checksum(source)
    execute(name+'-compile',command)
    linked=checksum(executable)
    before=checksum(executable)
    execute(name+'-values',[str(executable)])
    after=checksum(executable)
    assert linked==before==after and checksum(source)==source_hash
    probes.append({'name':name,'executable':str(executable),'linked_sha256':linked,
                   'before_run_sha256':before,'after_run_sha256':after,
                   'source':str(source),'source_sha256':source_hash})
render_target='JuicerProductionPreparedRenderProbe'+('.exe' if sys.platform=='win32' else '')
render_original=build/'tests/bin'/render_target
render_retained=out/'build/tests/bin'/render_target
render_sha=checksum(render_original)
def render_word_hash(words):
    return hashlib.sha256(b''.join(struct.pack('<I',word) for word in words)).hexdigest()
render_raw_rows=[]
render_rows=[]
for route,components in [('negative-direct',3),('negative-print',4)]:
    words=[0]*(5*7*components)
    for phase in ('cold','warm'):
        raw={'case_id':route+'/'+phase,'route':route,'pass':phase,'recipe_hash':1,
             'status':0,'api':0,'native_code':0,'diagnostic':'',
             'input_shape':[5,7,components],'output_shape':[5,7,components],
             'accepted_fixture_match':True,'submitted':True,'readback_complete':True,
             'state_released_after_submit':True,'views_released_after_submit':True,
             'stream_completed':True,'context_preserved':True,'context_restored':True,
             'input_f32_words':[0]*(5*7*components),'output_f32_words':words}
        render_raw_rows.append(raw)
        render_rows.append({key:value for key,value in raw.items()
                            if key not in {'input_f32_words','output_f32_words'}}|
                           {'input_sha256':render_word_hash(raw['input_f32_words']),
                            'output_f32_words_sha256':render_word_hash(words)})
render_stdout=out/'prepared-render.stdout'
render_stderr=out/'prepared-render.stderr'
render_stdout.write_text(json.dumps({'rows':render_raw_rows}))
render_stderr.write_bytes(b'')
prepared_render={'executable':str(render_retained),'executable_sha256_before':render_sha,
    'executable_sha256_after':checksum(render_original),
    'command':[str(render_original),str(Path('Resources').resolve()),str(prepared_fixture),preset,str(prepared_overlay)],
    'fixture':str(prepared_fixture),'fixture_sha256':checksum(prepared_fixture),
    'stdout_sha256':checksum(render_stdout),'stderr_sha256':checksum(render_stderr),'rows':render_rows}
commands.append({'name':'prepared-render','command':prepared_render['command'],'returncode':0,
                 'stdout_sha256':prepared_render['stdout_sha256'],
                 'stderr_sha256':prepared_render['stderr_sha256']})
for name,value in consumed.items():assert checksum(Path(name))==value
for name,row in build_inputs.items():assert checksum(Path(name))==checksum(Path(row['file']))==row['sha256']
record=out/'commands.json'
record.write_text(json.dumps(commands))
(out/'production.json').write_text(json.dumps({'scope':'native miniature retained-payload protocol only',
    'preset':preset,'build_testing':'OFF','build_inputs':build_inputs,'probes':probes,
    'prepared_render':prepared_render,'consumed_inputs':consumed,
    'reference_fixture':{'manifest':str(reference_manifest),'payload':str(reference_payload),'logical_preset':preset},
    'scanner_descriptor_fixture':{'manifest':str(descriptor_manifest),'payload':str(descriptor_payload),'context':str(descriptor_context),'logical_preset':preset},
    'scanner_descriptor_results':descriptor_results,
    'scanner_post_fixture':{'manifest':str(post_manifest),'payload':str(post_payload),'context':str(post_context),'logical_preset':preset},
    'scanner_post_results':post_results,
    'frame_fixture':{'manifest':str(frame_manifest),'payload':str(frame_payload),'context':str(frame_context),'logical_preset':preset},
    'frame_results':frame_results,
    'halation_frame_fixture':{'manifest':str(halation_manifest),'payload':str(halation_payload),'context':str(halation_context),
                              'normative':str(halation_checked),'input_spec':str(halation_inputs),'logical_preset':preset},
    'halation_frame_results':halation_results,
    'diffusion_frame_fixture':{'edges_payload':str(diffusion_payload),'edges_context':str(diffusion_context),'edges_input_spec':str(diffusion_edges_inputs),'manifest':str(diffusion_manifest),'payload':str(diffusion_payload),'context':str(diffusion_context),
                              'normative':str(diffusion_checked),'input_spec':str(diffusion_inputs),'logical_preset':preset},
    'diffusion_frame_results':diffusion_results,
    'dir_frame_fixture':{'manifest':str(dir_manifest),'payload':str(dir_payload),
                         'normative':str(dir_checked),'input_spec':str(dir_inputs),'logical_preset':preset},
    'dir_frame_results':dir_results,
    'grain_frame_fixture':{'manifest':str(grain_manifest),'payload':str(grain_payload),
                          'normative':str(grain_checked),'input_spec':str(grain_inputs),'logical_preset':preset},
    'grain_frame_results':grain_results,
    'effects_frame_fixture':{'manifest':str(effects_manifest),'payload':str(effects_payload),'context':str(effects_context),
                            'normative':str(effects_checked),'input_spec':str(effects_inputs),'logical_preset':preset},
    'effects_frame_results':effects_results,
    'dir_boundary_fixture':{'manifest':str(boundary_manifest),'payload':str(boundary_payload),
                            'normative':str(boundary_checked),'input_spec':str(boundary_inputs),'logical_preset':preset},
    'dir_boundary_results':boundary_results,
    'reference_results':reference_results,
    'commands':{'file':str(record),'sha256':checksum(record)},
    'command_outputs':{str(out/(row['name']+'.'+extension)):row[extension+'_sha256']
                       for row in commands for extension in ['stdout','stderr']}}))
print("PRODUCTION_MANIFEST_SHA256",checksum(out/"production.json"),flush=True)
'''

    def production_protected(self, stage):
        receipt = Path(stage["receipt"])
        command = read_json(receipt)
        manifest = qualification.production_manifest_identity(receipt, root=self.root)[0]
        paths = [receipt, self.root / command["output"], Path(command["process_tree"]["receipt"]), manifest]
        paths += [self.root / name for name in stage["identity"]["inputs"]
                  if Path(name).suffix.lower() in {".a", ".lib", ".ofx", ".pdb", ".exe"}]
        prepared = read_json(manifest)["prepared_render"]
        paths.extend((Path(prepared["executable"]), manifest.parent / "prepared-render.stdout",
                      manifest.parent / "prepared-render.stderr"))
        return {str(path): digest(path) for path in paths}

    def test_retained_grain_helpers_resolve_foreign_paths(self):
        """Exercise shared-input admission before the independent reference gate."""
        preset = "windows-clang-release" if os.name == "nt" else "linux-release"
        build = self.root / "out/build" / preset
        out = self.directory / "retained"
        (out / "build").mkdir(parents=True)
        build.mkdir(parents=True)
        inputs, copies, probes = {}, {}, []
        for name, content in (("CMakeCache.txt", b"BUILD_TESTING:BOOL=OFF\n"), ("libjuicer.a", b"archive")):
            source, retained = build / name, out / "build" / name
            source.write_bytes(content)
            retained.write_bytes(content)
            checksum = digest(source)
            inputs[source.relative_to(self.root).as_posix()] = checksum
            copies[str(source)] = {"file": str(retained), "sha256": checksum}
        render_name = "JuicerProductionPreparedRenderProbe" + (".exe" if os.name == "nt" else "")
        render_source = self.root / "tests/ffi/production-prepared-render.cpp"
        render_fixture = self.root / "tests/ofx/fixtures/exposure" / f"processor-{preset}.dat"
        render_fixture_manifest = self.root / "tests/ofx/fixtures/exposure/manifest.json"
        render_overlay = self.root / "tests/print/illuminant-fixtures/product-updates.json"
        render_illuminant_manifest = self.root / "tests/print/illuminant-fixtures/manifest.json"
        for path, content in ((render_source, b"prepared source"), (render_fixture, b"fixture"),
                              (render_overlay, b"{}\n"), (render_illuminant_manifest, b"{}\n"),
                              (self.root / "CMakeLists.txt", b"cmake")):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        render_fixture.write_text(
            "negative-direct 105 " + " ".join(["0"] * 105) + "\n"
            "negative-print 140 " + " ".join(["0"] * 140) + "\n",
            encoding="utf-8")
        fixture_sha = digest(render_fixture)
        write_json(render_fixture_manifest, {"files": {render_fixture.name: fixture_sha}})
        render_executable = build / "tests/bin" / render_name
        retained_render_build_executable = out / "build/tests/bin" / render_name
        retained_render_executable = retained_render_build_executable
        render_executable.parent.mkdir(parents=True)
        retained_render_build_executable.parent.mkdir(parents=True, exist_ok=True)
        render_executable.write_bytes(b"prepared executable")
        retained_render_build_executable.write_bytes(render_executable.read_bytes())
        self.assertEqual(retained_render_executable, retained_render_build_executable)
        render_sha = digest(render_executable)
        inputs[str(render_executable.relative_to(self.root).as_posix())] = render_sha
        copies[str(render_executable)] = {"file": str(retained_render_build_executable), "sha256": render_sha}
        render_consumed = [self.root / "CMakeLists.txt", render_source, render_fixture_manifest,
                           render_fixture, render_illuminant_manifest, render_overlay]
        for path in render_consumed:
            inputs[path.relative_to(self.root).as_posix()] = digest(path)
        def bits_hash(words):
            return hashlib.sha256(b"".join(word.to_bytes(4, "little") for word in words)).hexdigest()
        raw_rows, final_rows = [], []
        for case_id, route, phase, components in (
                ("negative-direct/cold", "negative-direct", "cold", 3),
                ("negative-direct/warm", "negative-direct", "warm", 3),
                ("negative-print/cold", "negative-print", "cold", 4),
                ("negative-print/warm", "negative-print", "warm", 4)):
            words = [0] * (5 * 7 * components)
            raw = {"case_id": case_id, "route": route, "pass": phase, "recipe_hash": 1,
                   "status": 0, "api": 0, "native_code": 0, "diagnostic": "",
                   "input_shape": [5, 7, components], "output_shape": [5, 7, components],
                   "accepted_fixture_match": True, "submitted": True, "readback_complete": True,
                   "state_released_after_submit": True, "views_released_after_submit": True,
                   "stream_completed": True, "context_preserved": True, "context_restored": True,
                   "input_f32_words": words, "output_f32_words": words}
            raw_rows.append(raw)
            final_rows.append({**{key: value for key, value in raw.items()
                                 if key not in {"input_f32_words", "output_f32_words"}},
                               "input_sha256": bits_hash(words),
                               "output_f32_words_sha256": bits_hash(words)})
        render_stdout = out / "prepared-render.stdout"
        render_stderr = out / "prepared-render.stderr"
        write_json(render_stdout, {"rows": raw_rows})
        render_stderr.write_bytes(b"")
        prepared_render = {"executable": str(retained_render_executable),
                           "executable_sha256_before": render_sha,
                           "executable_sha256_after": render_sha,
                           "command": [str(render_executable), str(self.root / "Resources"),
                                       str(render_fixture), preset, str(render_overlay)],
                           "fixture": str(render_fixture), "fixture_sha256": fixture_sha,
                           "stdout_sha256": digest(render_stdout), "stderr_sha256": digest(render_stderr),
                           "rows": final_rows}
        command_rows = [{"name": "prepared-render", "command": prepared_render["command"],
                         "returncode": 0, "stdout_sha256": prepared_render["stdout_sha256"],
                         "stderr_sha256": prepared_render["stderr_sha256"]}]
        commands_file = out / "commands.json"
        write_json(commands_file, command_rows)
        for name in ("film", "accepted-output-color"):
            source, executable = out / (name + ".c"), out / (name + (".exe" if os.name == "nt" else ""))
            source.write_text("/* retained source identity */\n")
            executable.write_bytes(b"retained executable identity")
            probes.append({"name": name, "source": str(source), "source_sha256": digest(source),
                           "executable": str(executable), "linked_sha256": digest(executable),
                           "before_run_sha256": digest(executable), "after_run_sha256": digest(executable)})
        helpers = [self.root / "tests/ffi" / name for name in
                   ("grain_frame_abi_assertions.inc", "grain_probe_contract.py", "effects_frame_abi_assertions.inc", "effects_frame_probe_contract.py")]
        for helper in helpers:
            helper.parent.mkdir(parents=True, exist_ok=True)
            helper.write_bytes((ROOT / helper.relative_to(self.root)).read_bytes())
            inputs[helper.relative_to(self.root).as_posix()] = digest(helper)
        reference = self.root / "tests/film/reference-fixtures/manifest.json"
        reference.parent.mkdir(parents=True)
        write_json(reference, {"presets": {preset: "native.json"}})
        manifest = out / "production.json"

        for prefix, separator in (("Z:\\Film-Juicer\\", "\\"), ("/foreign/Film-Juicer/", "/")):
            def foreign(path):
                return prefix + path.relative_to(self.root).as_posix().replace("/", separator)
            def resolve(name):
                if name.startswith(prefix):
                    return self.root / name[len(prefix):].replace(separator, "/")
                return Path(name)
            document = {"preset": preset, "build_testing": "OFF", "build_inputs": copies,
                        "probes": probes, "prepared_render": prepared_render,
                        "consumed_inputs": {foreign(p): digest(p) for p in [*helpers, *render_consumed]},
                        "commands": {"file": str(commands_file), "sha256": digest(commands_file)},
                        "command_outputs": {str(render_stdout): digest(render_stdout),
                                            str(render_stderr): digest(render_stderr)}}
            write_json(manifest, document)
            # This intentionally incomplete tiny record must reach the next gate;
            # it does not claim real archive execution or full payload admission.
            with self.assertRaisesRegex(ValueError, "film-reference fixture membership"):
                qualification.production_payloads(manifest, digest(manifest), root=self.root, inputs=inputs, resolve=resolve)
            for path in (render_source, render_fixture):
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"changed")
                    with self.assertRaisesRegex(ValueError, "prepared-render consumed input identity changed"):
                        qualification.production_payloads(manifest, digest(manifest), root=self.root,
                                                          inputs=inputs, resolve=resolve)
                finally:
                    path.write_bytes(original)
            for path, message in ((retained_render_executable, "production execution payload missing or changed"),
                                  (render_stdout, "prepared-render retained output missing or changed"),
                                  (render_stderr, "prepared-render retained output missing or changed")):
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"changed")
                    with self.assertRaisesRegex(ValueError, message):
                        qualification.production_payloads(manifest, digest(manifest), root=self.root,
                                                          inputs=inputs, resolve=resolve)
                finally:
                    path.write_bytes(original)
            for mutation in ("unknown", "missing", "reordered", "missing-command", "changed-command",
                             "unbound-retention-path"):
                damaged = json.loads(json.dumps(document))
                if mutation == "unknown":
                    damaged["prepared_render"]["rows"][0]["unexpected"] = 1
                elif mutation == "missing":
                    del damaged["prepared_render"]["rows"][0]["api"]
                elif mutation == "reordered":
                    damaged["prepared_render"]["rows"].reverse()
                elif mutation == "unbound-retention-path":
                    damaged["prepared_render"]["executable"] = str(out / render_name)
                elif mutation == "missing-command":
                    write_json(commands_file, [])
                    damaged["commands"]["sha256"] = digest(commands_file)
                else:
                    changed_commands = json.loads(json.dumps(command_rows))
                    changed_commands[0]["command"][1] += "-changed"
                    write_json(commands_file, changed_commands)
                    damaged["commands"]["sha256"] = digest(commands_file)
                write_json(manifest, damaged)
                message = ("prepared-render row schema or order changed" if mutation in {"unknown", "missing"}
                           else "prepared-render row membership" if mutation == "reordered"
                           else "production retained command membership is incomplete or duplicated"
                           if mutation == "missing-command"
                           else "production prepared-render execution identities disagree"
                           if mutation == "unbound-retention-path"
                           else "production prepared-render command execution identity changed")
                try:
                    with self.assertRaisesRegex(ValueError, message):
                        qualification.production_payloads(manifest, digest(manifest), root=self.root,
                                                          inputs=inputs, resolve=resolve)
                finally:
                    write_json(commands_file, command_rows)
            write_json(manifest, document)
            for helper in helpers:
                for mode in ("missing", "hash", "content"):
                    with self.subTest(prefix=prefix, helper=helper.name, mode=mode):
                        damaged = json.loads(json.dumps(document))
                        original = helper.read_bytes()
                        try:
                            if mode == "missing":
                                del damaged["consumed_inputs"][foreign(helper)]
                            elif mode == "hash":
                                damaged["consumed_inputs"][foreign(helper)] = "0" * 64
                            else:
                                helper.write_bytes(original + b"\n")
                            write_json(manifest, damaged)
                            with self.assertRaisesRegex(ValueError, "grain C11 assertion input identity changed"):
                                qualification.production_payloads(manifest, digest(manifest), root=self.root, inputs=inputs, resolve=resolve)
                        finally:
                            helper.write_bytes(original)

    def production_mutations(self, stage):
        manifest = qualification.production_manifest_identity(stage["receipt"], root=self.root)[0]
        document = read_json(manifest)
        artifact = next(Path(row["file"]) for name, row in document["build_inputs"].items()
                        if Path(name).suffix.lower() in {".a", ".lib", ".ofx"})
        prepared = document["prepared_render"]
        return [Path(row[key]) for row in document["probes"] for key in ("executable", "source")] + [
            Path(prepared["executable"]), manifest.parent / "prepared-render.stdout",
            manifest.parent / "prepared-render.stderr", manifest.parent / "production-film-expected.h",
            manifest.parent / "film-values.stdout", artifact]

    def assert_production_reuse_controls(self, path, checkpoint, calls):
        record = read_json(path)
        stage = record["stages"]["production"]
        owner = self.root / "out/build" / record["preset"] / "verification-checkpoint.json"
        owner_bytes = owner.read_bytes()
        protected = self.production_protected(stage)
        before = len(calls)
        reused = read_json(checkpoint("retained-unchanged"))["stages"]["production"]
        self.assertEqual(reused["reused_from"], stage["receipt"])
        self.assertEqual(len(calls), before, "unchanged retained evidence executed probes again")
        outcomes = [{"operation": "reuse", "case": "unchanged", "status": "passed", "commands": 0}]
        reused = read_json(checkpoint("retained-retry-reason",
            "retry another failed check"))["stages"]["production"]
        self.assertEqual(reused["reused_from"], stage["receipt"])
        self.assertEqual(len(calls), before, "retry reason reran completed production probes")
        outcomes.append({"operation": "reuse", "case": "retry-reason", "status": "passed", "commands": 0})
        for target in self.production_mutations(stage):
            data, stamp = target.read_bytes(), target.stat()
            message = ("prepared-render retained output missing or changed"
                       if target.name in {"prepared-render.stdout", "prepared-render.stderr"}
                       else "production execution payload missing or changed")
            for mutation in ("replace", "remove"):
                with self.subTest(operation="reuse", file=target.name, mutation=mutation):
                    try:
                        target.write_bytes(b"changed after execution") if mutation == "replace" else target.unlink()
                        with self.assertRaisesRegex(ValueError, message):
                            checkpoint("retained-" + target.name.replace(".", "-").lower() + "-" + mutation)
                        self.assertEqual(read_json(owner)["production"]["status"], "failed")
                        self.assertEqual(len(calls), before, "rejected reuse executed probes")
                        untouched = {name: checksum for name, checksum in protected.items() if Path(name) != target}
                        self.assertEqual(untouched, {name: digest(Path(name)) for name in untouched})
                        if mutation == "remove":
                            self.assertFalse(target.exists(), "rejected reuse repaired a missing payload")
                        else:
                            self.assertEqual(target.read_bytes(), b"changed after execution")
                        outcomes.append({"operation": "reuse", "case": target.name, "mutation": mutation,
                                         "rejected": message})
                    finally:
                        target.write_bytes(data)
                        os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                        owner.write_bytes(owner_bytes)
        try:
            state = json.loads(owner_bytes)
            header = next(name for name in stage["payloads"] if Path(name).name == "production-film-expected.h")
            state["production"]["payloads"].pop(header)
            write_json(owner, state)
            with self.assertRaisesRegex(ValueError, "production retained payload membership"):
                checkpoint("retained-omitted-member")
            self.assertEqual(len(calls), before)
            outcomes.append({"operation": "reuse", "case": "omitted-member", "rejected": "production retained payload membership"})
        finally:
            owner.write_bytes(owner_bytes)
        for target_kind, cause in [("probe", "production execution payload missing or changed"),
                                   ("manifest", "production manifest execution identity changed")]:
            prior = Campaign.run
            def drift(campaign, name, command, **kwargs):
                receipt = prior(campaign, name, command, **kwargs)
                if name == "production":
                    raw = read_json(receipt)
                    self.assertTrue(raw["process_tree"]["drained"])
                    manifest = qualification.production_manifest_identity(receipt, root=self.root)[0]
                    document = read_json(manifest)
                    if target_kind == "probe":
                        Path(document["probes"][0]["executable"]).write_bytes(b"drift after drained execution")
                    else:
                        document["scope"] = "manifest replaced after drained execution"
                        manifest.write_text(json.dumps(document))
                return receipt
            try:
                # Initial registration has no completed stage to reuse. A retry
                # reason must preserve a matching pass rather than force this run.
                state = json.loads(owner_bytes)
                del state["production"]
                write_json(owner, state)
                registration_start = len(calls)
                with patch.object(Campaign, "run", new=drift):
                    with self.assertRaisesRegex(ValueError, cause):
                        checkpoint("retained-registration-drift-" + target_kind)
                self.assertEqual([call["stage"] for call in calls[registration_start:]], ["production"],
                                 "registration control must execute one probe run without rebuilding")
                self.assertEqual(read_json(owner)["production"]["status"], "failed")
                self.assertEqual(protected, {name: digest(Path(name)) for name in protected})
                outcomes.append({"operation": "registration", "case": "post-drain-drift-" + target_kind,
                                 "rejected": cause})
            finally:
                owner.write_bytes(owner_bytes)
        return outcomes

    def assert_production_collection_controls(self, paths):
        preset = "windows-clang-release" if os.name == "nt" else "linux-release"
        path = next(path for path in paths if read_json(path)["preset"] == preset
                    and read_json(path)["selection"] == ["production"])
        record = read_json(path)
        stage = record["stages"]["production"]
        owner = self.root / "out/build" / preset / "verification-checkpoint.json"
        owner_bytes = owner.read_bytes()
        protected = self.production_protected(stage)
        protected[str(path)] = digest(path)
        outcomes = [{"operation": "collection", "case": "unchanged", "status": "passed"}]
        for target in self.production_mutations(stage):
            data, stamp = target.read_bytes(), target.stat()
            message = ("prepared-render retained output missing or changed"
                       if target.name in {"prepared-render.stdout", "prepared-render.stderr"}
                       else "production execution payload missing or changed")
            for mutation in ("replace", "remove"):
                with self.subTest(operation="collection", file=target.name, mutation=mutation):
                    try:
                        target.write_bytes(b"changed after execution") if mutation == "replace" else target.unlink()
                        with self.assertRaisesRegex(ValueError, message):
                            qualification.collect_qualification(self.root, paths)
                        untouched = {name: checksum for name, checksum in protected.items() if Path(name) != target}
                        self.assertEqual(untouched, {name: digest(Path(name)) for name in untouched})
                        if mutation == "remove":
                            self.assertFalse(target.exists(), "collection repaired a missing payload")
                        else:
                            self.assertEqual(target.read_bytes(), b"changed after execution")
                        outcomes.append({"operation": "collection", "case": target.name, "mutation": mutation,
                                         "rejected": message})
                    finally:
                        target.write_bytes(data)
                        os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        try:
            header = next(name for name in stage["payloads"] if Path(name).name == "production-film-expected.h")
            state = json.loads(owner_bytes)
            state["production"]["payloads"].pop(header)
            record["stages"]["production"]["payloads"].pop(header)
            write_json(owner, state)
            omitted = path.parent / "omitted-membership.json"
            write_json(omitted, record)
            with self.assertRaisesRegex(ValueError, "production retained payload membership"):
                qualification.collect_qualification(self.root, [omitted if other == path else other for other in paths])
            self.assertEqual(protected, {name: digest(Path(name)) for name in protected})
            outcomes.append({"operation": "collection", "case": "omitted-member",
                             "rejected": "production retained payload membership"})
        finally:
            owner.write_bytes(owner_bytes)
        self.assertEqual(qualification.collect_qualification(self.root, paths)["status"], "passed")
        return outcomes

    def test_collection_rejects_prepare_without_scoped_proof(self):
        self.native_project()
        from verify import source_snapshot
        required = ["prepare", "quality", "product"]
        campaign = Campaign(self.root, self.preset, "unproved-preparation", required)
        stages = {name: qualification.run_stage(campaign, self.state_path, self.state, name,
            [self.python, "-c", "print('ordinary command without preparation proof')"],
            inputs=["input.txt"], tools=[self.python], environment=self.environment, timeout=10)
            for name in required}
        record = {"mode": "checkpoint", "status": "passed", "scope_status": "passed",
                  "reason": "ordinary command cannot supply scoped preparation", "preset": self.preset,
                  "runtime": "all", "selection": ["product"], "required": required,
                  "stages": stages, "source": source_snapshot(self.root)}
        path = campaign.directory / "qualification.json"
        write_json(path, record)
        protected = {p: p.read_bytes() for p in
                     [path, self.state_path, *(Path(stage["receipt"]) for stage in stages.values())]}
        with patch.object(Campaign, "run", side_effect=AssertionError("collection dispatched")):
            with self.assertRaisesRegex(ValueError, "legacy sealed-build preparation"):
                qualification.collect_qualification(self.root, [path])
        self.assertEqual(protected, {p: p.read_bytes() for p in protected})

    def test_production_retained_execution_payloads(self):
        """Real OFF owners and preparations; unrelated execution uses unit scaffolds."""
        started = time.monotonic()
        files = self.compiled_project()
        from verify import source_snapshot
        paths, outcomes, commands = [], [], []
        actual_run = Campaign.run
        def counted(campaign, stage, command, **kwargs):
            commands.append({"stage": stage, "command": list(command)})
            return actual_run(campaign, stage, command, **kwargs)
        def checkpoint(preset, name, repeat_reason=""):
            args = SimpleNamespace(reason="native retained OFF payload controls", selection=["production"],
                                   preset=preset, runtime="all", campaign=name,
                                   repeat_reason=repeat_reason, base=None, prepare=True)
            if preset == ("windows-clang-release" if os.name == "nt" else "linux-release"):
                return qualification.qualify(self.root, args)
            with OutputLease(self.root, self.root / "out/build" / preset):
                return qualification._qualify_owned(self.root, self.root / "out/build" / preset, args)
        with patch.object(Campaign, "run", autospec=True, side_effect=counted):
            for preset in ("linux-release", "windows-clang-release"):
                build = self.root / "out/build" / preset
                campaign = Campaign(self.root, preset, "retained-off-prepare", ["prepare"])
                _, prepared = qualification.prepare_build(campaign, "prepare", source=self.root, build=build,
                    inputs=sorted(set(files) | {
                        "CMakeLists.txt", "tests/ffi/production-prepared-render.cpp",
                        "tests/ofx/fixtures/exposure/manifest.json",
                        f"tests/ofx/fixtures/exposure/processor-{preset}.dat",
                        "tests/print/illuminant-fixtures/manifest.json",
                        "tests/print/illuminant-fixtures/product-updates.json"}), timeout=60,
                    targets=["all", "JuicerProductionPreparedRenderProbe"],
                    configure_args=["-DBUILD_TESTING=OFF"], checkpoint=True)
                self.assertEqual(read_json(prepared)["scoped_preparation"]["targets"],
                                 sorted(["all", "JuicerProductionPreparedRenderProbe"]))
                path = checkpoint(preset, "retained-off")
                paths.append(path)
                if preset == ("windows-clang-release" if os.name == "nt" else "linux-release"):
                    outcomes += self.assert_production_reuse_controls(path,
                        lambda name, repeat_reason="": checkpoint(preset, name, repeat_reason), commands)
            # Real miniature preparations supply the collector's context and
            # artifact identity. Only unrelated relevance execution is scaffolded;
            # no preparation or production validation is mocked.
            for preset in qualification.PRESETS:
                build = self.root / "out/build" / preset
                self.configure_fixture(preset, True, "retained-prerequisite-on", timeout=60)
                state_path = build / "verification-checkpoint.json"
                state = read_json(state_path) if state_path.exists() else {}
                groups = ["boundary", "product"]
                if preset.endswith("debug"):
                    groups.append("tooling")
                if preset == "linux-debug":
                    groups.append("corpus")
                required = ["prepare", "quality", *groups]
                campaign = Campaign(self.root, preset, "retained-unrelated-scaffold", required)
                stages = {"prepare": qualification.prepare_checkpoint(campaign, build, state_path, state,
                    self.environment, timeout=60, allow_build=True)}
                dependency = qualification.preparation_identity(build, stages["prepare"])
                for name in required[1:]:
                    stages[name] = qualification.run_stage(campaign, state_path, state, name,
                        [self.python, "-c", "print('unrelated collection prerequisite scaffold')"],
                        inputs=qualification.checkpoint_inputs(self.root, build, name if name in qualification.GROUPS else "product"),
                        tools=[self.python], environment=self.environment, timeout=10,
                        dependencies=[dependency] if name in {"quality", "product"} else [])
                record = {"mode": "checkpoint", "status": "passed", "scope_status": "passed",
                          "reason": "synthetic unrelated prerequisites for isolated production collector controls",
                          "preset": preset, "runtime": "all", "selection": groups, "required": required,
                          "stages": stages, "source": source_snapshot(self.root)}
                path = campaign.directory / "qualification.json"
                write_json(path, record)
                paths.append(path)
            self.assertEqual(qualification.collect_qualification(self.root, paths)["status"], "passed")
            outcomes += self.assert_production_collection_controls(paths)
        destination = self.root / "out/validation" / self.preset / "retained-production-controls.json"
        write_json(destination, {"scope": "real production checkpoint/reuse/collector controls with explicit unrelated product scaffolds; no product qualification",
            "native": os.name, "controls": outcomes, "commands": commands,
            "seconds": time.monotonic() - started})

    def test_production_evidence_survives_on_restoration_and_unrelated_failure(self):
        files = self.compiled_project()
        cmake = self.root / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() +
                         "target_compile_definitions(juicer PRIVATE MINI_TESTING=$<BOOL:${BUILD_TESTING}>)\n")
        native = self.root / "src/native.cpp"
        native.write_text(native.read_text() + "int testing_feature(){return MINI_TESTING;}\n")
        preset = "windows-clang-release" if os.name == "nt" else "linux-release"
        build = self.root / "out/build" / preset
        state_path = build / "verification-checkpoint.json"
        campaign = Campaign(self.root, preset, "off-preparation", ["prepare"])
        qualification.prepare_build(campaign, "prepare", source=self.root, build=build,
            inputs=list(files), timeout=60,
            targets=["all", "JuicerProductionPreparedRenderProbe"],
            configure_args=["-DBUILD_TESTING=OFF"], checkpoint=True)
        state = {}
        with OutputLease(self.root, build):
            with self.assertRaises(subprocess.CalledProcessError):
                qualification.run_stage(Campaign(self.root, preset, "unrelated-failure", ["boundary"]),
                    state_path, state, "boundary", [self.python, "-c", "raise SystemExit(7)"],
                    inputs=["input.txt"], tools=[self.python], environment=self.environment, timeout=10)
        args = SimpleNamespace(reason="retained OFF evidence across ON restoration", selection=["production"],
                               preset=preset, runtime="all", campaign="original-off", repeat_reason="", base=None, prepare=True)
        path = qualification.qualify(self.root, args)
        original_record = path.read_bytes()
        record = json.loads(original_record)
        self.assertEqual(record["status"], "incomplete")
        self.assertEqual(record["scope_status"], "passed")
        self.assertEqual(record["pending"], ["boundary"])
        stage = record["stages"]["production"]
        manifest = qualification.production_manifest_identity(stage["receipt"], root=self.root)[0]
        document = read_json(manifest)
        self.assertEqual(len(document["reference_results"]), 15)
        self.assertEqual(len(document["scanner_post_results"]), 410)
        for field in (
            "scanner_lut_results", "scanner_lut_fixture", "gaussian_results", "gaussian_fixture",
            "optics_results", "optics_fixture", "print_illuminant_results",
            "print_illuminant_fixture", "preflash_results", "preflash_fixture",
        ):
            self.assertNotIn(field, document)
        self.assertEqual(len(document["frame_results"]), 69)
        self.assertEqual(len(document["halation_frame_results"]), 86)
        self.assertEqual(len(document["diffusion_frame_results"]), 507)
        self.assertEqual(len(document["dir_frame_results"]), 121)
        self.assertEqual(len(document["grain_frame_results"]), 188)
        self.assertEqual(len(document["effects_frame_results"]), 453)
        self.assertEqual(len(document["dir_boundary_results"]), 171)
        stdout = manifest.parent / "film-values.stdout"
        commands_path = manifest.parent / "commands.json"
        original_stdout, original_commands = stdout.read_bytes(), commands_path.read_bytes()
        try:
            line = next(line for line in original_stdout.decode().splitlines() if line.startswith("DIR_FRAME_RESULT "))
            original_row = json.loads(line.removeprefix("DIR_FRAME_RESULT "))
            changed_row = json.loads(json.dumps(original_row))
            changed_row["failure"][-1] ^= 1
            changed_stdout = original_stdout.replace(json.dumps(original_row).encode(), json.dumps(changed_row).encode(), 1)
            self.assertNotEqual(changed_stdout, original_stdout)
            stdout.write_bytes(changed_stdout)
            commands = json.loads(original_commands)
            next(row for row in commands if row["name"] == "film-values")["stdout_sha256"] = digest(stdout)
            write_json(commands_path, commands)
            damaged = json.loads(json.dumps(document))
            damaged["commands"]["sha256"] = digest(commands_path)
            damaged["command_outputs"][str(stdout)] = digest(stdout)
            damaged_path = manifest.with_name("dir-executed-trace.json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "DIR frame executed stdout"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])
        finally:
            stdout.write_bytes(original_stdout)
            commands_path.write_bytes(original_commands)

        try:
            line = next(line for line in original_stdout.decode().splitlines() if line.startswith("DIR_BOUNDARY_RESULT "))
            original_row = json.loads(line.removeprefix("DIR_BOUNDARY_RESULT "))
            changed_row = json.loads(json.dumps(original_row))
            changed_row["weights"][-1] ^= 1
            changed_stdout = original_stdout.replace(json.dumps(original_row).encode(), json.dumps(changed_row).encode(), 1)
            self.assertNotEqual(changed_stdout, original_stdout)
            stdout.write_bytes(changed_stdout)
            commands = json.loads(original_commands)
            next(row for row in commands if row["name"] == "film-values")["stdout_sha256"] = digest(stdout)
            write_json(commands_path, commands)
            damaged = json.loads(json.dumps(document))
            damaged["commands"]["sha256"] = digest(commands_path)
            damaged["command_outputs"][str(stdout)] = digest(stdout)
            damaged_path = manifest.with_name("dir-boundary-executed-word.json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "DIR boundary executed stdout"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])
        finally:
            stdout.write_bytes(original_stdout)
            commands_path.write_bytes(original_commands)

        for mode in ("missing", "duplicate", "word", "key", "envelope", "failure", "terminal", "shape", "diagnostic_length", "owner"):
            damaged = json.loads(json.dumps(document))
            rows = damaged["dir_boundary_results"]
            if mode == "missing":
                rows.pop()
            elif mode == "duplicate":
                rows[-1] = rows[0]
            elif mode == "word":
                rows[0]["weights"][-1] ^= 1
            elif mode == "key":
                rows[0]["key"] ^= 1
            elif mode == "envelope":
                rows[0]["envelope"][0] = 3
            elif mode == "failure":
                rows[0]["failure"][2] = 0
            elif mode == "terminal":
                rows[0]["terminal"][-1] ^= 1
            elif mode == "shape":
                rows[0]["shape"][0] ^= 1
            elif mode == "diagnostic_length":
                rows[0]["diagnostic_length"] += 1
            else:
                rows[0]["owner"] = False
            damaged_path = manifest.with_name("dir-boundary-" + mode + ".json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "DIR boundary"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])

        for mode in ("missing", "word", "failure", "diagnostic"):
            damaged = json.loads(json.dumps(document))
            rows = damaged["effects_frame_results"]
            if mode == "missing":
                rows.pop()
            elif mode == "word":
                rows[0]["output"][-1] ^= 1
            elif mode == "failure":
                rows[0]["failure"][0] = 1
            else:
                rows[0]["diagnostic"] = "wrong"
            damaged_path = manifest.with_name("effects-" + mode + ".json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "effects frame"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])
        try:
            line = next(line for line in original_stdout.decode().splitlines() if line.startswith("EFFECTS_FRAME_RESULT "))
            row = json.loads(line.removeprefix("EFFECTS_FRAME_RESULT "))
            changed = json.loads(json.dumps(row))
            changed["output"][-1] ^= 1
            stdout.write_bytes(original_stdout.replace(json.dumps(row).encode(), json.dumps(changed).encode(), 1))
            commands = json.loads(original_commands)
            next(command for command in commands if command["name"] == "film-values")["stdout_sha256"] = digest(stdout)
            write_json(commands_path, commands)
            damaged = json.loads(json.dumps(document))
            damaged["commands"]["sha256"] = digest(commands_path)
            damaged["command_outputs"][str(stdout)] = digest(stdout)
            damaged_path = manifest.with_name("effects-rehashed-executed-output.json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "effects frame executed stdout"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])
        finally:
            stdout.write_bytes(original_stdout)
            commands_path.write_bytes(original_commands)

        for mode in ("missing", "duplicate", "case", "word", "envelope", "failure", "diagnostic", "length"):
            damaged = json.loads(json.dumps(document))
            rows = damaged["grain_frame_results"]
            if mode == "missing":
                rows.pop()
            elif mode == "duplicate":
                rows[-1] = rows[0]
            elif mode == "case":
                rows[0]["case"] = "unknown-grain-case"
            elif mode == "word":
                rows[0]["output"][-1] ^= 1
            elif mode == "envelope":
                rows[0]["envelope"][0] = 3
            elif mode == "failure":
                rows[0]["failure"][2] = 0
            elif mode == "diagnostic":
                rows[0]["diagnostic"] = "wrong"
            else:
                rows[0]["diagnostic_length"] += 1
            damaged_path = manifest.with_name("grain-" + mode + ".json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "grain frame"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])
        for name in ("grain_frame_abi_assertions.inc", "grain_probe_contract.py", "effects_frame_abi_assertions.inc", "effects_frame_probe_contract.py"):
            target = self.root / "tests/ffi" / name
            original = target.read_bytes()
            try:
                target.write_bytes(original + b"\n")
                with self.assertRaisesRegex(ValueError, "grain C11 assertion input identity changed"):
                    qualification.production_payloads(manifest, digest(manifest), root=self.root, inputs=stage["identity"]["inputs"])
            finally:
                target.write_bytes(original)

        # Even an aligned, rehashed expected/result substitution cannot redefine
        # the independently reviewed complete case set.
        target = Path(document["grain_frame_fixture"]["manifest"])
        original = target.read_bytes()
        try:
            spec = json.loads(original)
            spec["production_cases"][0]["case"] = "unknown-grain-case"
            write_json(target, spec)
            damaged = json.loads(json.dumps(document))
            damaged["grain_frame_results"][0]["case"] = "unknown-grain-case"
            damaged["consumed_inputs"][str(target)] = digest(target)
            inputs = dict(stage["identity"]["inputs"])
            inputs[target.relative_to(self.root).as_posix()] = digest(target)
            damaged_path = manifest.with_name("grain-aligned-case-substitution.json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "grain frame full values"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=inputs)
        finally:
            target.write_bytes(original)
        for member in ("manifest", "payload", "normative", "input_spec"):
            target = Path(document["grain_frame_fixture"][member])
            original = target.read_bytes()
            try:
                target.write_bytes(original + b" ")
                with self.assertRaisesRegex(ValueError, "grain frame consumed fixture"):
                    qualification.production_payloads(manifest, digest(manifest), root=self.root, inputs=stage["identity"]["inputs"])
            finally:
                target.write_bytes(original)
        try:
            line = next(line for line in original_stdout.decode().splitlines() if line.startswith("GRAIN_FRAME_RESULT "))
            original_row = json.loads(line.removeprefix("GRAIN_FRAME_RESULT "))
            changed_row = json.loads(json.dumps(original_row))
            changed_row["output"][-1] ^= 1
            changed_stdout = original_stdout.replace(json.dumps(original_row).encode(), json.dumps(changed_row).encode(), 1)
            self.assertNotEqual(changed_stdout, original_stdout)
            stdout.write_bytes(changed_stdout)
            commands = json.loads(original_commands)
            next(row for row in commands if row["name"] == "film-values")["stdout_sha256"] = digest(stdout)
            write_json(commands_path, commands)
            damaged = json.loads(json.dumps(document))
            damaged["commands"]["sha256"] = digest(commands_path)
            damaged["command_outputs"][str(stdout)] = digest(stdout)
            damaged_path = manifest.with_name("grain-executed-values.json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "grain frame executed stdout"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])
        finally:
            stdout.write_bytes(original_stdout)
            commands_path.write_bytes(original_commands)

        for mode in ("missing", "duplicate", "word", "envelope", "failure", "trace", "diagnostic"):
            damaged = json.loads(json.dumps(document))
            rows = damaged["dir_frame_results"]
            if mode == "missing":
                rows.pop()
            elif mode == "duplicate":
                rows[-1] = rows[0]
            elif mode == "word":
                rows[0]["output"][-1] ^= 1
            elif mode == "envelope":
                rows[0]["envelope"][0] = 3
            elif mode == "failure":
                rows[0]["failure"][2] = 1
            elif mode == "trace":
                rows[0]["failure"][-1] ^= 1
            else:
                rows[0]["diagnostic"] = "wrong"
            damaged_path = manifest.with_name("dir-" + mode + ".json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "DIR frame"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])

        try:
            line = next(line for line in original_stdout.decode().splitlines() if line.startswith("DIFFUSION_FRAME_RESULT "))
            original_row = json.loads(line.removeprefix("DIFFUSION_FRAME_RESULT "))
            changed_row = json.loads(json.dumps(original_row))
            changed_row["output"][-1] ^= 1
            changed_stdout = original_stdout.replace(json.dumps(original_row).encode(), json.dumps(changed_row).encode(), 1)
            self.assertNotEqual(changed_stdout, original_stdout)
            stdout.write_bytes(changed_stdout)
            commands = json.loads(original_commands)
            next(row for row in commands if row["name"] == "film-values")["stdout_sha256"] = digest(stdout)
            write_json(commands_path, commands)
            damaged = json.loads(json.dumps(document))
            damaged["commands"]["sha256"] = digest(commands_path)
            damaged["command_outputs"][str(stdout)] = digest(stdout)
            damaged_path = manifest.with_name("diffusion-executed-word.json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "diffusion frame executed stdout"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])
        finally:
            stdout.write_bytes(original_stdout)
            commands_path.write_bytes(original_commands)

        for mode in ("missing", "duplicate", "word", "envelope", "failure", "diagnostic"):
            damaged = json.loads(json.dumps(document))
            rows = damaged["diffusion_frame_results"]
            if mode == "missing":
                rows.pop()
            elif mode == "duplicate":
                rows[-1] = rows[0]
            elif mode == "word":
                rows[0]["output"][-1] ^= 1
            elif mode == "envelope":
                rows[0]["envelope"][0] = 3
            elif mode == "failure":
                rows[0]["failure"][2] = 1
            else:
                rows[0]["diagnostic"] = "wrong"
            damaged_path = manifest.with_name("diffusion-" + mode + ".json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "diffusion frame"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])

        for mode in ("missing", "duplicate", "word", "hash", "envelope", "failure", "sigma", "diagnostic"):
            damaged = json.loads(json.dumps(document))
            rows = damaged["halation_frame_results"]
            if mode == "missing":
                rows.pop()
            elif mode == "duplicate":
                rows[-1] = rows[0]
            elif mode == "word":
                rows[0]["output"]["words"][-1] ^= 1
            elif mode == "hash":
                rows[0]["output"]["recipe_hash"] ^= 1
            elif mode == "envelope":
                rows[0]["envelope"][0] = 3
            elif mode == "failure":
                rows[0]["failure"][3] = 0
            elif mode == "sigma":
                rows[0]["sigma_bits"] ^= 1
            else:
                rows[0]["diagnostic"] = "wrong"
            damaged_path = manifest.with_name("halation-" + mode + ".json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "halation frame"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])
        for mode in ("missing", "duplicate", "changed-domain", "changed-pixel", "changed-diffusion", "changed-envelope", "changed-diagnostic"):
            damaged = json.loads(json.dumps(document))
            if mode == "missing":
                damaged["frame_results"].pop()
            elif mode == "duplicate":
                damaged["frame_results"][-1] = damaged["frame_results"][0]
            elif mode == "changed-domain":
                damaged["frame_results"][0]["domain"][3] ^= 1
            elif mode == "changed-pixel":
                damaged["frame_results"][0]["pixel_bits"] ^= 1
            elif mode == "changed-diffusion":
                damaged["frame_results"][0]["diffusion_bits"] ^= 1
            elif mode == "changed-envelope":
                damaged["frame_results"][0]["envelope"][0] = 3
            else:
                damaged["frame_results"][0]["diagnostic"] = "wrong"
            damaged_path = manifest.with_name("frame-" + mode + ".json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "frame geometry"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])

        for field in (
            "scanner_lut_results", "scanner_lut_fixture", "gaussian_results", "gaussian_fixture",
            "optics_results", "optics_fixture", "print_illuminant_results",
            "print_illuminant_fixture", "preflash_results", "preflash_fixture",
        ):
            damaged = json.loads(json.dumps(document))
            damaged[field] = None
            damaged_path = manifest.with_name("retired-" + field + ".json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "retired mathematical owner field"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root,
                    inputs=stage["identity"]["inputs"])

        for mode in ("missing", "duplicate", "changed"):
            damaged = json.loads(json.dumps(document))
            if mode == "missing":
                damaged["scanner_post_results"].pop()
            elif mode == "duplicate":
                damaged["scanner_post_results"][-1] = damaged["scanner_post_results"][0]
            else:
                damaged["scanner_post_results"][0]["hash"] ^= 1
            damaged_path = manifest.with_name("post-" + mode + ".json")
            write_json(damaged_path, damaged)
            with self.assertRaisesRegex(ValueError, "post-effects value/failure"):
                qualification.production_payloads(damaged_path, digest(damaged_path), root=self.root, inputs=stage["identity"]["inputs"])
        for field in ("effects_frame_results", "effects_frame_fixture", "grain_frame_results", "grain_frame_fixture", "dir_boundary_results", "dir_boundary_fixture", "dir_frame_results", "dir_frame_fixture", "diffusion_frame_results", "diffusion_frame_fixture", "halation_frame_results", "halation_frame_fixture", "frame_results", "frame_fixture", "reference_results", "reference_fixture", "scanner_descriptor_results", "scanner_descriptor_fixture", "scanner_post_results", "scanner_post_fixture"):
            missing = manifest.with_name("without-" + field + ".json")
            write_json(missing, {key: value for key, value in document.items() if key != field})
            expected = "effects frame" if field.startswith("effects_frame") else "grain frame" if field.startswith("grain_frame") else "DIR boundary" if field.startswith("dir_boundary") else "DIR frame" if field.startswith("dir_frame") else "diffusion frame" if field.startswith("diffusion") else "halation frame" if field.startswith("halation") else "frame geometry" if field.startswith("frame") else "scanner post-effects" if field.startswith("scanner_post") else "scanner descriptor" if field.startswith("scanner_descriptor") else "film-reference"
            with self.assertRaisesRegex(ValueError, expected):
                qualification.production_payloads(missing, digest(missing), root=self.root,
                    inputs=stage["identity"]["inputs"])
        artifact = build / ("juicer.lib" if os.name == "nt" else "libjuicer.a")
        retained = Path(document["build_inputs"][str(artifact)]["file"])
        off_hash = stage["identity"]["inputs"][artifact.relative_to(self.root).as_posix()]
        self.assertEqual(digest(artifact), off_hash)
        self.assertEqual(digest(retained), off_hash)
        evidence = {name: checksum for name, checksum in self.production_protected(stage).items()
                    if not Path(name).is_relative_to(build)}
        payloads = {name: digest(Path(name)) for name in stage["payloads"]}
        legacy = manifest.with_name("without-retained-build-inputs.json")
        write_json(legacy, {key: value for key, value in document.items() if key != "build_inputs"})
        with self.assertRaisesRegex(ValueError, "fresh OFF execution required"):
            qualification.production_payloads(legacy, digest(legacy), root=self.root,
                inputs=stage["identity"]["inputs"])

        qualification.prepare_build(Campaign(self.root, preset, "restore-on", ["prepare"]), "prepare",
            source=self.root, build=build, inputs=list(files), timeout=60, targets=["all"],
            configure_args=["-DBUILD_TESTING=ON"], checkpoint=True)
        self.assertIn("BUILD_TESTING:BOOL=ON", (build / "CMakeCache.txt").read_text())
        self.assertNotEqual(digest(artifact), off_hash)
        self.assertEqual(payloads, {name: digest(Path(name)) for name in payloads})

        def rejected_collection(message):
            with patch.object(Campaign, "run", side_effect=AssertionError("collection executed a command")):
                with self.assertRaisesRegex(ValueError, message):
                    qualification.collect_qualification(self.root, [path])

        rejected_collection("pending checkpoint work")
        state = read_json(state_path)
        with OutputLease(self.root, build):
            qualification.run_stage(Campaign(self.root, preset, "unrelated-repair", ["boundary"]),
                state_path, state, "boundary", [self.python, "-c", "print('repaired')"],
                inputs=["input.txt"], tools=[self.python], environment=self.environment, timeout=10)
        rejected_collection("full checkpoint membership is incomplete")
        data, stamp = retained.read_bytes(), retained.stat()
        on_hash = digest(artifact)
        for mutation in ("replace", "remove"):
            with self.subTest(retained_artifact=mutation):
                try:
                    retained.write_bytes(b"changed retained OFF archive") if mutation == "replace" else retained.unlink()
                    rejected_collection("production execution payload missing or changed")
                    self.assertEqual(digest(artifact), on_hash)
                finally:
                    retained.write_bytes(data)
                    os.utime(retained, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        rejected_collection("full checkpoint membership is incomplete")
        self.assertEqual(path.read_bytes(), original_record)
        self.assertEqual(evidence, {name: digest(Path(name)) for name in evidence})
        self.assertEqual(payloads, {name: digest(Path(name)) for name in payloads})

        with OutputLease(self.root, build):
            with self.assertRaises(subprocess.CalledProcessError):
                qualification.run_stage(Campaign(self.root, preset, "later-production-failure", ["production"]),
                    state_path, state, "production", [self.python, "-c", "raise SystemExit(9)"],
                    inputs=["input.txt"], tools=[self.python], environment=self.environment, timeout=10)
        rejected_collection("pending checkpoint work")
        self.assertEqual(read_json(state_path)["production"]["status"], "failed")
        self.assertEqual(path.read_bytes(), original_record)
        self.assertEqual(evidence, {name: digest(Path(name)) for name in evidence})

    def compiled_project(self):
        self.native_project()
        files = {
            "CMakeLists.txt": 'cmake_minimum_required(VERSION 3.24)\nproject(Checkpoint LANGUAGES C CXX)\n'
                'set(CMAKE_C_STANDARD 11)\nset(CMAKE_CXX_STANDARD 20)\n'
                'set(CMAKE_EXPORT_COMPILE_COMMANDS ON)\ninclude(CTest)\n'
                'configure_file(binding.h.in generated/binding.h COPYONLY)\n'
                'add_library(juicer STATIC src/native.cpp src/abi.c)\n'
                'target_include_directories(juicer PUBLIC "${CMAKE_BINARY_DIR}/generated")\n'
                'add_executable(JuicerProductionPreparedRenderProbe EXCLUDE_FROM_ALL tests/ffi/production-prepared-render.cpp)\n'
                'target_link_libraries(JuicerProductionPreparedRenderProbe PRIVATE juicer)\n'
                'set_target_properties(JuicerProductionPreparedRenderProbe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/bin")\n'
                'add_executable(behavior src/main.cpp)\ntarget_link_libraries(behavior PRIVATE juicer)\n'
                'if(BUILD_TESTING)\nforeach(group product boundary tooling corpus)\n'
                'add_test(NAME mini.${group} COMMAND behavior 10)\n'
                'set_tests_properties(mini.${group} PROPERTIES LABELS "host;${group}")\nendforeach()\nendif()\n',
            "binding.h.in": "#define BINDING 4\n",
            "tests/ffi/production-prepared-render.cpp": "int main() { return 0; }\n",
            "src/abi.h": 'typedef struct Witness { int first; int second; } Witness;\n',
            "src/abi.c": '#include <stddef.h>\n#include "abi.h"\n'
                '_Static_assert(sizeof(Witness)==8,"C11 ABI");\n'
                'int c_size(void){return sizeof(Witness);}\nint c_offset(void){return offsetof(Witness,second);}\n',
            "src/native.cpp": '#include "binding.h"\nint native(){return 6+BINDING;}\n',
            "src/main.cpp": '#include <cstdio>\n#include <cstdlib>\n#include <cstddef>\n#include "abi.h"\n'
                'static_assert(sizeof(Witness)==8);\nint native();\nextern "C" int c_size(void);\nextern "C" int c_offset(void);\n'
                'int main(int argc,char** argv){int v=native();const char* offset=std::getenv("JUICER_P0_A_RUNTIME_OFFSET");'
                'if(offset)v+=std::atoi(offset);std::printf("value=%d abi=%d offset=%d\\n",v,c_size(),c_offset());'
                'return argc!=2 || v!=std::atoi(argv[1]) || c_size()!=sizeof(Witness) || c_offset()!=offsetof(Witness,second);}\n',
            "scripts/check-quality.py": 'import sys\nfrom pathlib import Path\n'
                'preset=sys.argv[sys.argv.index("--preset")+1]\np=Path("out")/(preset+"-quality-count.txt")\n'
                'p.write_text(str(int(p.read_text())+1) if p.exists() else "1")\nprint("miniature quality scheduling witness")\n',
            "tests/ffi/production_contract.py": 'import json,sys,subprocess\nfrom pathlib import Path\n'
                'preset=sys.argv[sys.argv.index("--preset")+1]\nout=Path(sys.argv[sys.argv.index("--artifacts")+1])\n'
                'build=Path("out/build")/preset\nexe=build/("behavior.exe" if sys.platform=="win32" else "behavior")\n'
                'value=subprocess.check_output([str(exe.resolve()),"10"],text=True,timeout=10)\n'
                'assert "value=10 abi=8 offset=4" in value\nout.mkdir(parents=True)\n'
                + self.production_fixture(),
        }
        for name, content in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        for name in ("grain_frame_abi_assertions.inc", "grain_probe_contract.py", "effects_frame_abi_assertions.inc", "effects_frame_probe_contract.py"):
            relative = "tests/ffi/" + name
            content = (ROOT / relative).read_text()
            (self.root / relative).write_text(content)
            files[relative] = content
        compiler = ({"CMAKE_C_COMPILER": "clang-cl.exe", "CMAKE_CXX_COMPILER": "clang-cl.exe", "CMAKE_LINKER_TYPE": "LLD",
                     "CMAKE_C_FLAGS": "/Brepro", "CMAKE_CXX_FLAGS": "/Brepro"}
                    if os.name == "nt" else {"CMAKE_C_COMPILER": "/usr/bin/gcc-13", "CMAKE_CXX_COMPILER": "/usr/bin/g++-13"})
        (self.root / "CMakePresets.json").write_text(json.dumps({"version": 6, "configurePresets": [
            {"name": p, "generator": "Ninja", "binaryDir": str(self.root / "out/build" / p),
             "cacheVariables": {**compiler, "CMAKE_BUILD_TYPE": "Release" if p.endswith("release") else "Debug"}}
            for p in qualification.PRESETS], "testPresets": [
                {"name": p, "configurePreset": p, "execution": {"jobs": 4, "noTestsAction": "error"}}
                for p in qualification.PRESETS]}))
        return files

    def replay_preparation_renewal(self):
        """Four protocol lanes use this OS's tiny binaries, never product qualification."""
        calls, deletions, artifact_reads, input_checks, tool_checks = [], [], [], [], []
        started = time.monotonic()
        progress = {"scope": "native miniature protocol only", "complete": False, "phases": []}
        progress_path = self.root / "out/validation" / self.preset / "replay-progress.json"
        progress_path.parent.mkdir(parents=True)
        def record_phase(name, action, *args, **kwargs):
            row = {"name": name, "status": "running"}
            progress["phases"].append(row)
            write_json(progress_path, progress)
            start = time.monotonic()
            try:
                result = action(*args, **kwargs)
                row["status"] = "passed"
                return result
            except BaseException as error:
                row.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "raised",
                           error=type(error).__name__)
                raise
            finally:
                row["seconds"] = time.monotonic() - start
                progress.update(seconds=time.monotonic() - started, commands=calls, deleted=deletions,
                                preparation_input_checks=len(input_checks),
                                preparation_input_check_seconds=sum(input_checks),
                                tool_checks=len(tool_checks), tool_check_seconds=sum(tool_checks),
                                artifact_hash_reads=len(artifact_reads),
                                artifact_hash_seconds=sum(value[0] for value in artifact_reads))
                write_json(progress_path, progress)
        files = record_phase("fixture.configure", self.compiled_project)
        artifact_paths = set()
        actual_digest = qualification.digest
        def measured_digest(path):
            path = Path(path).resolve()
            started = time.monotonic()
            result = actual_digest(path)
            if path in artifact_paths:
                artifact_reads.append((time.monotonic() - started, path.stat().st_size))
            return result
        actual_run = Campaign.run
        from build_dependencies import NinjaGraph
        actual_invalidate = NinjaGraph.invalidate
        def counted(campaign, stage, command, **kwargs):
            started = time.monotonic()
            try:
                return actual_run(campaign, stage, command, **kwargs)
            finally:
                calls.append({"stage": stage, "command": list(command), "seconds": time.monotonic() - started})
        def invalidate(graph, names):
            removed = actual_invalidate(graph, names)
            deletions.extend(removed)
            return removed
        actual_check = qualification.check_preparation_inputs
        def measured_inputs(*args, **kwargs):
            started = time.monotonic()
            try:
                return actual_check(*args, **kwargs)
            finally:
                input_checks.append(time.monotonic() - started)
        actual_tools = qualification.tool_snapshot
        def measured_tools(*args, **kwargs):
            started = time.monotonic()
            try:
                return actual_tools(*args, **kwargs)
            finally:
                tool_checks.append(time.monotonic() - started)
        def checkpoint(preset, campaign, selection):
            args = SimpleNamespace(reason="miniature complete protocol replay", selection=selection, preset=preset,
                                   runtime="all", campaign=campaign, repeat_reason="", base="fixture-baseline", prepare=True)
            # Other names model collector membership with native binaries on this OS.
            # The public native-platform admission remains exercised for the native lane.
            if preset == self.preset:
                return record_phase(f"checkpoint.{preset}.{campaign}", qualification.qualify, self.root, args)
            with OutputLease(self.root, self.root / "out/build" / preset):
                return record_phase(f"checkpoint.{preset}.{campaign}", qualification._qualify_owned,
                                    self.root, self.root / "out/build" / preset, args)
        paths = []
        with patch.object(Campaign, "run", autospec=True, side_effect=counted), \
                patch.object(NinjaGraph, "invalidate", autospec=True, side_effect=invalidate), \
                patch.object(qualification, "digest", side_effect=measured_digest), \
                patch.object(qualification, "check_preparation_inputs", side_effect=measured_inputs), \
                patch.object(qualification, "tool_snapshot", side_effect=measured_tools), \
                patch("execution.tool_snapshot", side_effect=measured_tools):
            for preset in qualification.PRESETS:
                build = self.root / "out/build" / preset
                campaign = Campaign(self.root, preset, "cold", ["prepare"])
                if preset.endswith("release"):
                    record_phase(f"prepare.{preset}.cold", qualification.prepare_build, campaign, "prepare", source=self.root, build=build,
                        inputs=list(files), timeout=60,
                        targets=["all", "JuicerProductionPreparedRenderProbe"],
                        configure_args=["-DBUILD_TESTING=OFF"], checkpoint=True)
                    paths.append(checkpoint(preset, "off", ["production"]))
                else:
                    record_phase(f"configure.{preset}.cold", self.configure_fixture, preset, True, "fixture-on", timeout=60)
                groups = ["product", "boundary"]
                if preset.endswith("debug"):
                    groups += ["tooling"]
                if preset == "linux-debug":
                    groups += ["corpus"]
                paths.append(checkpoint(preset, "initial", groups))
            initial = self.root / "out/validation" / self.preset / "initial/qualification.json"
            selection = read_json(initial)["selection"]
            # CMake may resolve preset compiler spellings during cold setup.
            # Admit the resulting tool inventory before the proof-only comparison.
            stable = checkpoint(self.preset, "stable", selection)
            paths = [stable if p == initial else p for p in paths]
            initial = stable
            # Assembly uses simulated foreign facts; real WindowsJob/identity
            # controls live in test_execution.WindowsIdentityInteropTests.
            foreign = {r["stages"]["prepare"]["sha256"]: {"files": {}, "rust_configuration": {}}
                       for p in paths for r in [read_json(p)]
                       if r["preset"].startswith("windows-") and "product" in r["selection"]}
            native = {"expected": foreign, "stages": []}
            phases, native_consumers = [], []
            from copy import deepcopy
            phase_maps = {phase: deepcopy(foreign) for phase in ("entry", "terminal")}
            original_identity = qualification.preparation_identity
            def assembled_identity(build, stage, **kwargs):
                facts = kwargs.pop("native_file_identities", None)
                if facts is not None:
                    self.assertEqual(facts, foreign[stage["sha256"]])
                    self.assertIs(facts, phase_maps[phases[-1]][stage["sha256"]])
                    native_consumers.append(stage["sha256"])
                return original_identity(build, stage, **kwargs)
            def native_phase(root, state, phase):
                if phase == "entry":
                    self.assertFalse(native_consumers)
                else:
                    self.assertCountEqual(native_consumers, foreign)
                phases.append(phase)
                state["stages"].append({"phase": phase})
                return phase_maps[phase]
            with patch.object(qualification, "prepare_native_collection", return_value=native), \
                    patch.object(qualification, "run_native_identity_phase", side_effect=native_phase), \
                    patch.object(qualification, "preparation_identity", side_effect=assembled_identity), \
                    patch.object(qualification, "check_native_identity_evidence") as native_seal:
                initial_collection = record_phase("collect.initial", qualification.collect_qualification, self.root, paths)
                self.assertEqual(phases, ["entry", "terminal"])
                self.assertCountEqual(native_consumers, [*foreign, *foreign])
                native_seal.assert_called_once_with(self.root, native["stages"], foreign)
                self.assertEqual(initial_collection["native_identities"], native["stages"])
            self.assertEqual(initial_collection["status"], "passed")
            first = read_json(initial)
            proof = read_json(Path(first["stages"]["prepare"]["receipt"]))["scoped_preparation"]
            artifacts = {p: digest(Path(p)) for p in proof["outputs"]}
            before_source = snapshot(self.root, list(files))
            originals = {str(p): digest(p) for p in (self.root / "out/validation").rglob("*")
                         if p.is_file() and p != progress_path}
            for path in paths:
                artifact_paths.update(Path(p).resolve() for p in read_json(path)["stages"].get("prepare", {}).get("payloads", {}))
            warm = []
            for trial in range(3):
                start = time.monotonic()
                call_start, delete_start, hash_start = len(calls), len(deletions), len(artifact_reads)
                check_start = len(input_checks)
                tool_start = len(tool_checks)
                with patch.object(qualification, "check_scoped_preparation", side_effect=ValueError("injected checkpoint proof validation failure")):
                    with self.assertRaisesRegex(ValueError, "injected checkpoint"):
                        checkpoint(self.preset, f"failed-{trial}", first["selection"])
                with self.assertRaisesRegex(ValueError, "pending checkpoint work"):
                    qualification.collect_qualification(self.root, paths)
                self.assertEqual(artifacts, {p: digest(Path(p)) for p in artifacts})
                state = read_json(self.state_path)
                repair = self.campaign(f"repair-{trial}")
                repair_start = time.monotonic()
                renewed = record_phase(f"prepare.repair-{trial}", qualification.prepare_checkpoint,
                    repair, self.build, self.state_path, state, dict(os.environ),
                    timeout=60, repeat_reason="diagnosed checkpoint validation failure; completed artifact producer intact")
                repair_seconds = time.monotonic() - repair_start
                self.assertEqual(renewed["sha256"], first["stages"]["prepare"]["sha256"])
                current = checkpoint(self.preset, f"continued-{trial}", first["selection"])
                paths = [current if p == initial or p.parent.name.startswith("continued-") and p.parent.parent.name == self.preset else p for p in paths]
                collection_start = time.monotonic()
                collected = record_phase(f"collect.continued-{trial}", qualification.collect_qualification, self.root, paths)
                collection_seconds = time.monotonic() - collection_start
                self.assertEqual(collected["status"], "passed")
                self.assertEqual(before_source, snapshot(self.root, list(files)))
                self.assertEqual(artifacts, {p: digest(Path(p)) for p in artifacts})
                value = subprocess.check_output([str(self.build / ("behavior.exe" if os.name == "nt" else "behavior")), "10"], text=True, timeout=10)
                self.assertIn("value=10 abi=8 offset=4", value)
                logs = [self.root / read_json(p)["output"] for p in repair.directory.glob("attempts/*/result.json")]
                self.assertFalse(logs, "proof-only renewal executed build commands")
                warm.append({"seconds": time.monotonic() - start, "commands": calls[call_start:], "deleted": deletions[delete_start:],
                             "repair_seconds": repair_seconds, "collection_seconds": collection_seconds,
                             "artifact_hash_reads": len(artifact_reads) - hash_start,
                             "artifact_hash_seconds": sum(row[0] for row in artifact_reads[hash_start:]),
                             "artifact_bytes_hashed": sum(row[1] for row in artifact_reads[hash_start:]),
                             "preparation_input_checks": len(input_checks) - check_start,
                             "preparation_input_check_seconds": sum(input_checks[check_start:]),
                             "tool_checks": len(tool_checks) - tool_start,
                             "tool_check_seconds": sum(tool_checks[tool_start:]),
                             "renewed": renewed, "checkpoint": str(current), "collection": collected})
            self.assertEqual(originals, {p: digest(Path(p)) for p in originals})
            summary = {"scope": "native miniature protocol; four synthetic lane names, no Film-Juicer/GPU/OFX qualification",
                       "fixture": str(self.root), "native": os.name, "artifacts": artifacts, "inputs": before_source,
                       "initial_collection": initial_collection, "warm": warm, "seconds": time.monotonic() - started,
                       "tool_checks": len(tool_checks), "tool_check_seconds": sum(tool_checks),
                       "rust_recovery": "not applicable; no Rust fixture", "cold_and_warm_commands": calls}
            for trial in warm:
                self.assertFalse(trial["deleted"])
                self.assertEqual(trial["commands"], [], "proof-only renewal repeated completed work")
            write_json(self.root / "out/validation" / self.preset / "replay.json", summary)
            progress.update(complete=True, seconds=time.monotonic() - started)
            write_json(progress_path, progress)

    def prepare_compiled_project(self):
        self.compiled_project()
        self.configure_fixture(self.preset, True, "fixture-on", timeout=60)
        return qualification.prepare_checkpoint(self.campaign("compiled"), self.build, self.state_path, self.state,
                                                dict(os.environ), timeout=60, allow_build=True)

    def test_preparation_output_bytes_are_required_for_reuse(self):
        self.assert_output_mutation("behavior.exe" if os.name == "nt" else "behavior")

    def test_collection_rechecks_current_preparation_inputs(self):
        self.assert_collection_input_mutations([
            ("generated/binding.h", lambda b: b.replace(b"BINDING 4", b"BINDING 5"), "dependency contents"),
            ("CTestTestfile.cmake", lambda b: b + b'\nadd_test(unrun "missing")\n', "test registration"),
            ("generated/new-shadow.h", lambda b: b"#define SHADOW 1\n", "include membership"),
        ])

    def test_terminal_preparation_callback_mutation_is_read_freshly(self):
        self.prepare_compiled_project()
        args = SimpleNamespace(reason="terminal preparation freshness witness", selection=["product"],
                               preset=self.preset, runtime="all", campaign="admitted", repeat_reason="",
                               base="fixture-baseline", prepare=True)
        path = qualification.qualify(self.root, args)
        record = read_json(path)
        stage = record["stages"]["prepare"]
        target = self.root / "src/native.cpp"
        proof = read_json(Path(stage["receipt"]))["scoped_preparation"]
        self.assertIn("src/native.cpp", stage["identity"]["inputs"])
        self.assertIn(str(target.resolve()), proof["context"]["files"])
        original_bytes, original_stat = target.read_bytes(), target.stat()
        original_identity = qualification.preparation_identity
        calls = 0

        def mutate_before_validation(build, candidate, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                target.write_bytes(original_bytes + b"\n# terminal freshness witness\n")
                os.utime(target, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            return original_identity(build, candidate, **kwargs)

        try:
            with patch.object(qualification, "preparation_identity", side_effect=mutate_before_validation), \
                    self.assertRaisesRegex(ValueError, "preparation dependency contents or paths changed"):
                qualification.collect_qualification(self.root, [path])
            self.assertEqual(calls, 2, "the mutation must occur inside the terminal validation call")
        finally:
            target.write_bytes(original_bytes)
            os.utime(target, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    def test_collection_rechecks_changed_cache(self):
        # Retain this mutation in its own fixture. DrvFS can round timestamps
        # when restoring them, so it cannot be another case's fresh baseline.
        self.assert_collection_input_mutations([
            ("CMakeCache.txt", lambda b: b.replace(b"CMAKE_CXX_FLAGS:STRING=", b"CMAKE_CXX_FLAGS:STRING=-DABI_CHANGED=1 "), "configuration inputs"),
        ], retain_mutation=True)

    def assert_collection_input_mutations(self, mutations, *, retain_mutation=False):
        self.prepare_compiled_project()
        args = SimpleNamespace(reason="collection input drift control", selection=["product"], preset=self.preset,
                               runtime="all", campaign="admitted", repeat_reason="", base="fixture-baseline", prepare=True)
        path = qualification.qualify(self.root, args)
        record = read_json(path)
        stage = record["stages"]["prepare"]
        identity = qualification.preparation_identity(self.build, stage)
        # Other fixture lanes are deliberately absent; the complete replay owns
        # positive aggregation. This control must reach that completeness check.
        with self.assertRaisesRegex(ValueError, "full checkpoint membership"):
            qualification.collect_qualification(self.root, [path])
        executable = self.build / ("behavior.exe" if os.name == "nt" else "behavior")
        outcomes = []
        for name, mutate, diagnostic in mutations:
            with self.subTest(input=name):
                target = self.build / name
                content = target.read_bytes() if target.exists() else None
                stamp = target.stat() if target.exists() else None
                try:
                    changed = mutate(content or b"")
                    self.assertNotEqual(content, changed)
                    target.write_bytes(changed)
                    if stamp:
                        os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                    with self.assertRaisesRegex(ValueError, diagnostic):
                        qualification.preparation_identity(self.build, stage)
                    with self.assertRaisesRegex(ValueError, diagnostic):
                        qualification.collect_qualification(self.root, [path])
                    value = subprocess.check_output([str(executable), "10"], text=True, timeout=10)
                    self.assertIn("value=10 abi=8 offset=4", value)
                    outcomes.append({"input": name, "rejected": diagnostic,
                                     "mtime_restoration_exact": bool(stamp) and target.stat().st_mtime_ns == stamp.st_mtime_ns,
                                     "behavior": value.strip(), "mutation_retained": retain_mutation})
                finally:
                    if retain_mutation:
                        pass
                    elif content is None:
                        target.unlink()
                    else:
                        target.write_bytes(content)
                        os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            if not retain_mutation:
                self.assertEqual(identity, qualification.preparation_identity(self.build, stage))
        write_json(path.parent / "collection-controls.json", outcomes)

    def test_consumed_environment_change_cannot_reuse_product(self):
        files = self.compiled_project()
        self.configure_fixture(self.preset, True, "fixture-on", timeout=60)
        args = SimpleNamespace(reason="consumed runtime environment control", selection=["product"], preset=self.preset,
                               runtime="all", campaign="environment-baseline", repeat_reason="", base="fixture-baseline", prepare=True)
        outcomes = {}
        with patch.dict(os.environ, {"JUICER_P0_A_RUNTIME_OFFSET": "0"}):
            first = qualification.qualify(self.root, args)
            original = first.read_bytes()
            source = snapshot(self.root, list(files))
            args.campaign = "environment-changed"
            with patch.dict(os.environ, {"JUICER_P0_A_RUNTIME_OFFSET": "1"}):
                executable = self.build / ("behavior.exe" if os.name == "nt" else "behavior")
                stale = subprocess.run([str(executable), "10"], capture_output=True, text=True, timeout=10)
                self.assertNotEqual(stale.returncode, 0)
                self.assertIn("value=11 abi=8 offset=4", stale.stdout)
                with self.assertRaises(subprocess.CalledProcessError):
                    qualification.qualify(self.root, args)
                with self.assertRaisesRegex(ValueError, "pending checkpoint work"):
                    qualification.collect_qualification(self.root, [first])
                self.assertEqual((self.root / "out" / (self.preset + "-quality-count.txt")).read_text(), "2")
                failed = self.root / "out/validation" / self.preset / args.campaign / "qualification.json"
                outcomes.update(changed_checkpoint=str(failed), witness=stale.stdout.strip(), variable="JUICER_P0_A_RUNTIME_OFFSET")
            self.assertEqual(original, first.read_bytes())
            self.assertEqual(source, snapshot(self.root, list(source)))
            args.campaign = "environment-restored"
            current = qualification.qualify(self.root, args)
            self.assertEqual(read_json(current)["status"], "passed")
            outcomes.update(baseline=str(first), restored=str(current), status="changed environment executed quality and failed real product; restored environment passed")
            write_json(current.parent / "environment-control.json", outcomes)

    def test_preparation_archive_bytes_are_required_for_reuse(self):
        self.assert_output_mutation("juicer.lib" if os.name == "nt" else "libjuicer.a")

    def assert_output_mutation(self, name):
        first = self.prepare_compiled_project()
        original = Path(first["receipt"]).read_bytes()
        artifact = self.build / name
        content, stamp = artifact.read_bytes(), artifact.stat()
        artifact.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
        os.utime(artifact, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "payload changed|artifacts changed"):
            qualification.prepare_checkpoint(self.campaign("mutated"), self.build, self.state_path, self.state,
                                             dict(os.environ), timeout=60, allow_build=True)
        self.assertEqual(self.state["prepare"]["status"], "failed")
        self.assertEqual(original, Path(first["receipt"]).read_bytes())
        for attempt in range(2):
            with self.assertRaisesRegex(ValueError, "artifacts changed"):
                qualification.prepare_checkpoint(self.campaign(f"repair-mutated-{attempt}"), self.build, self.state_path,
                    self.state, dict(os.environ), timeout=60, repeat_reason="diagnosed artifact admission failure", allow_build=True)
            self.assertEqual(self.state["prepare"]["status"], "failed")
            self.assertEqual(first["payloads"], self.state["prepare"]["payloads"])

    def test_preparation_receipt_and_owner_permutations_preserve_one_baseline(self):
        first = self.prepare_compiled_project()
        record = read_json(Path(first["receipt"]))
        process = Path(record["process_tree"]["receipt"])
        original_process = process.read_bytes()
        document = json.loads(original_process)
        document["drained"] = False
        process.write_text(json.dumps(document))
        with patch.object(Campaign, "run", side_effect=AssertionError("dispatched with unproved drainage")):
            for trial in range(2):
                with self.assertRaisesRegex(ValueError, "process evidence"):
                    qualification.prepare_checkpoint(self.campaign(f"undrained-{trial}"), self.build, self.state_path,
                        self.state, dict(os.environ), timeout=60, repeat_reason="diagnosed process proof failure", allow_build=True)
                self.assertEqual(self.state["prepare"]["status"], "failed")
                self.assertEqual(first["payloads"], self.state["prepare"]["payloads"])
        process.write_bytes(original_process)

        receipt = Path(first["receipt"]).read_bytes()
        expected = qualification.preparation_identity(self.build, first)
        candidate = {**first, "payloads": dict(first["payloads"])}
        candidate["payloads"].pop(next(iter(candidate["payloads"])))
        with self.assertRaisesRegex(ValueError, "exact artifact identities"):
            qualification.preparation_identity(self.build, candidate)
        from build_evidence import preparation_paths
        baseline, pending = preparation_paths(self.build, ["all"])
        pending.write_text('{"test": "unproved artifact writer"}')
        with self.assertRaisesRegex(ValueError, "pending preparation"):
            qualification.preparation_identity(self.build, first)
        # Only this fixture's injected pending marker is removed; no writer ran.
        pending.unlink()
        self.assertEqual(expected, qualification.preparation_identity(self.build, first))
        original = baseline.read_bytes()
        changed = json.loads(original)
        changed["producer"] = "superseded"
        baseline.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "latest completed preparation"):
            qualification.preparation_identity(self.build, first)
        baseline.write_bytes(original)
        self.assertEqual(Path(first["receipt"]).read_bytes(), receipt)

    def test_restored_time_edit_exposes_stale_behavior_before_preparation(self):
        first = self.prepare_compiled_project()
        executable = self.build / ("behavior.exe" if os.name == "nt" else "behavior")
        subprocess.run([str(executable), "10"], check=True, capture_output=True, timeout=10)
        source = self.root / "src/native.cpp"
        content, stamp = source.read_bytes(), source.stat()
        source.write_bytes(content.replace(b"6+BINDING", b"7+BINDING"))
        os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        old = subprocess.run([str(executable), "11"], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(old.returncode, 0)
        self.assertIn("value=10 abi=8 offset=4", old.stdout)
        with self.assertRaisesRegex(ValueError, "dependency contents"):
            qualification.check_scoped_preparation(self.campaign("stale"), self.build, Path(first["receipt"]))
        qualification.prepare_checkpoint(self.campaign("rebuilt"), self.build, self.state_path, self.state,
                                         dict(os.environ), timeout=60, allow_build=True)
        new = subprocess.run([str(executable), "11"], check=True, capture_output=True, text=True, timeout=10)
        self.assertIn("value=11 abi=8 offset=4", new.stdout)

    def mixed_project(self):
        self.native_project()
        probe = self.root / "probe.py"
        probe.write_text(
            "import json,sys,time\nfrom pathlib import Path\n"
            "root=Path(__file__).parent\nrole=sys.argv[1]\n"
            f"state=json.loads((root/'out/build/{self.preset}/verification-checkpoint.json').read_text())\n"
            "assert state['boundary']['status']=='pending'\n"
            "leases=list((root/'out/verification-locks').glob('*/owner.json'))\n"
            "assert any(p.parent.name.startswith('build-') for p in leases)\n"
            "assert (root/'out/verification-locks/gpu/owner.json').is_file()==(role=='gpu')\n"
            "with (root/'out/order.txt').open('a') as f:f.write(role+'\\n')\n"
            "mode=(root/'failure.txt').read_text() if (root/'failure.txt').exists() else ''\n"
            "if mode=='timeout' and role=='gpu':time.sleep(30)\n"
            "raise SystemExit(1 if mode==role else 0)\n")
        cmake = self.root / "CMakeLists.txt"
        text = cmake.read_text()
        text += '\n'
        for role in ("host", "gpu"):
            text += (f'add_test(NAME {role} COMMAND "{Path(self.python).as_posix()}" "{probe.as_posix()}" {role})\n'
                     f'set_tests_properties({role} PROPERTIES LABELS "{role};boundary")\n')
        cmake.write_text(text)
        self.configure_fixture(self.preset, True, "fixture-on")
        return SimpleNamespace(reason="mixed runtime regression", selection=["boundary"], preset=self.preset,
                               runtime="all", campaign="mixed", repeat_reason="", base=None, prepare=True)

    def test_mixed_ctest_owns_partitions_and_reuses_without_gpu_lease(self):
        args = self.mixed_project()
        actual_lease = qualification.OutputLease
        def bounded_lease(root, build=None, **kwargs):
            if build is None:
                self.assertIsNone(kwargs["timeout"], "ordinary GPU queueing has no elapsed deadline")
                self.assertEqual(read_json(self.state_path)["boundary"]["status"], "pending")
            return actual_lease(root, build, **kwargs)
        with patch.object(qualification, "OutputLease", side_effect=bounded_lease):
            first = read_json(qualification.qualify(self.root, args))
        self.assertEqual((self.root / "out/order.txt").read_text().splitlines(), ["host", "gpu"])
        stage = first["stages"]["boundary"]
        self.assertEqual(len(stage["additional_receipts"]), 1)
        self.assertEqual(first["runtime_membership"]["boundary"], {"host": ["boundary", "host"], "gpu": ["gpu"]})
        self.assertFalse((OutputLease(self.root).path / "owner.json").exists())
        args.campaign = "reused-mixed"
        actual_lease = qualification.OutputLease
        def build_lease(root, build=None, **kwargs):
            self.assertIsNotNone(build, "unchanged reuse attempted to acquire the GPU lease")
            return actual_lease(root, build, **kwargs)
        with patch.object(qualification, "run_ctest_parts", side_effect=AssertionError("executed reused stage")), \
                patch.object(qualification, "OutputLease", side_effect=build_lease):
            reused = read_json(qualification.qualify(self.root, args))
        self.assertEqual(reused["stages"]["boundary"]["reused_from"], stage["receipt"])
        self.assertEqual((self.root / "out/order.txt").read_text().splitlines(), ["host", "gpu"])
        secondary = Path(next(iter(stage["additional_receipts"])))
        original = secondary.read_bytes()
        secondary.write_bytes(original + b" ")
        args.campaign = "changed-secondary"
        with self.assertRaisesRegex(ValueError, "receipt changed"):
            qualification.qualify(self.root, args)
        self.assertEqual(read_json(self.state_path)["boundary"]["status"], "failed")
        secondary.write_bytes(original)
        args.campaign = "cannot-hide-failure"
        with self.assertRaisesRegex(ValueError, "pending checkpoint"):
            qualification.qualify(self.root, args)

    def test_mixed_ctest_host_failure_stops_gpu_and_gpu_failure_stays_pending(self):
        args = self.mixed_project()
        for role in ("host", "gpu"):
            (self.root / "failure.txt").write_text(role)
            args.campaign = "fail-" + role
            args.repeat_reason = "exercise distinct partition failure"
            (self.root / "out/order.txt").unlink(missing_ok=True)
            with self.assertRaises(subprocess.CalledProcessError):
                qualification.qualify(self.root, args)
            self.assertEqual((self.root / "out/order.txt").read_text().splitlines(),
                             ["host"] if role == "host" else ["host", "gpu"])
            self.assertEqual(read_json(self.state_path)["boundary"]["status"], "failed")
            self.assertFalse((OutputLease(self.root).path / "owner.json").exists())
            args.campaign = "new-caller-" + role
            args.repeat_reason = ""
            with self.assertRaisesRegex(ValueError, "pending checkpoint"):
                qualification.qualify(self.root, args)

    def test_zero_exit_host_with_unrun_case_never_schedules_gpu(self):
        args = self.mixed_project()
        actual = Campaign.run
        def incomplete(campaign, stage, command, **kwargs):
            path = actual(campaign, stage, command, **kwargs)
            if "-LE" in command:
                import xml.etree.ElementTree as ET
                report = Path(command[command.index("--output-junit") + 1])
                document = ET.parse(report)
                next(document.getroot().iter("testcase")).set("status", "notrun")
                document.write(report)
            return path
        with patch.object(Campaign, "run", autospec=True, side_effect=incomplete):
            with self.assertRaisesRegex(ValueError, "did not execute"):
                qualification.qualify(self.root, args)
        self.assertEqual((self.root / "out/order.txt").read_text().splitlines(), ["host"])
        self.assertEqual(read_json(self.state_path)["boundary"]["status"], "failed")
        self.assertEqual(len(Campaign(self.root, self.preset, "mixed", ["boundary"]).attempts("boundary")), 1)

    def test_mixed_ctest_timeout_drains_gpu_and_keeps_whole_stage_failed(self):
        args = self.mixed_project()
        (self.root / "failure.txt").write_text("timeout")
        actual = qualification.run_ctest_parts
        def bounded(*args, **kwargs):
            kwargs["timeout"] = 15
            return actual(*args, **kwargs)
        with patch.object(qualification, "run_ctest_parts", side_effect=bounded):
            with self.assertRaises(subprocess.TimeoutExpired):
                qualification.qualify(self.root, args)
        self.assertEqual(read_json(self.state_path)["boundary"]["status"], "failed")
        raw = Campaign(self.root, self.preset, "mixed", ["boundary"]).attempts("boundary")
        self.assertEqual([r["status"] for r in raw], ["passed", "failed"])
        self.assertTrue(all(r["process_tree"]["drained"] for r in raw))
        self.assertFalse((OutputLease(self.root).path / "owner.json").exists())

    def test_collection_binds_all_partition_receipts_and_aggregate_membership(self):
        args = self.mixed_project()
        path = qualification.qualify(self.root, args)
        record = read_json(path)
        # The valid partial checkpoint reaches only the final full-matrix gate.
        with self.assertRaisesRegex(ValueError, "membership is incomplete"):
            qualification.collect_qualification(self.root, [path])
        for change in ("secondary", "overlap", "missing", "swapped"):
            candidate = json.loads(json.dumps(record))
            if change == "secondary":
                candidate["stages"]["boundary"]["additional_receipts"] = {}
            elif change == "overlap":
                candidate["runtime_membership"]["boundary"]["host"].append("gpu")
            elif change == "missing":
                del candidate["runtime_membership"]["boundary"]["gpu"]
            else:
                parts = candidate["runtime_membership"]["boundary"]
                parts["host"], parts["gpu"] = parts["gpu"], parts["host"]
            path.write_text(json.dumps(candidate))
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "ownership|partition"):
                qualification.collect_qualification(self.root, [path])

    def test_real_off_to_on_preparation_owns_generated_configuration(self):
        self.native_project()
        inputs = qualification.checkpoint_inputs(self.root, self.build, "product")
        source_inputs = [p for p in inputs if not p.startswith("out/build/")]
        def prepare():
            return qualification.prepare_build(self.campaign("first"), "prepare", source=self.root,
                build=self.build, inputs=source_inputs, timeout=30, targets=["all"],
                configure_args=["-DBUILD_TESTING=ON"], checkpoint=True)[1]
        first = self.run_stage("prepare", inputs=inputs, action=prepare,
            refresh_inputs=lambda: qualification.checkpoint_inputs(self.root, self.build, "product"),
            validate=lambda p: qualification.check_scoped_preparation(self.campaign("first"), self.build, p))
        self.assertIn("BUILD_TESTING:BOOL=ON", (self.build / "CMakeCache.txt").read_text())
        self.assertTrue((self.build / "CTestTestfile.cmake").is_file())
        refreshed = qualification.checkpoint_inputs(self.root, self.build, "product")
        second = self.run_stage("prepare", campaign="next-caller", inputs=refreshed, action=prepare,
            validate=lambda p: qualification.check_scoped_preparation(self.campaign("next-caller"), self.build, p))
        self.assertEqual(second["reused_from"], first["receipt"])

    def test_production_inputs_do_not_require_test_registration(self):
        self.native_project()
        self.production_fixture()
        helpers = ("tests/ffi/grain_frame_abi_assertions.inc", "tests/ffi/grain_probe_contract.py", "tests/ffi/effects_frame_abi_assertions.inc", "tests/ffi/effects_frame_probe_contract.py")
        for name in helpers:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        self.assertFalse((self.build / "CTestTestfile.cmake").exists())
        inputs = qualification.checkpoint_inputs(self.root, self.build, "production")
        self.assertIn("CMakeLists.txt", inputs)
        self.assertIn((self.build / "CMakeCache.txt").relative_to(self.root).as_posix(), inputs)
        for name in helpers:
            self.assertIn(name, inputs)
        for name in ("tests/ffi/production-prepared-render.cpp",
                     "tests/ofx/fixtures/exposure/manifest.json",
                     f"tests/ofx/fixtures/exposure/processor-{self.preset}.dat",
                     "tests/print/illuminant-fixtures/manifest.json",
                     "tests/print/illuminant-fixtures/product-updates.json"):
            self.assertIn(name, inputs)

    @unittest.skipUnless(os.name == "nt", "Windows production PDB")
    def test_production_pdb_changes_and_missing_repair_cannot_reuse_old_pass(self):
        self.native_project()
        self.preset = "windows-clang-release"
        self.build = self.root / "out/build" / self.preset
        self.build.mkdir()
        (self.build / "CMakeCache.txt").write_text(
            f"CMAKE_HOME_DIRECTORY:INTERNAL={self.root.as_posix()}\nBUILD_TESTING:BOOL=OFF\n")
        module = self.build / "juicer.ofx"
        module.write_text("module")
        pdb = self.build / "juicer.pdb"
        pdb.write_text("first")
        driver = self.root / "tests/ffi/production_contract.py"
        driver.parent.mkdir(parents=True)
        driver.write_text(
            "import argparse\nfrom pathlib import Path\n"
            "p=argparse.ArgumentParser();p.add_argument('--preset');p.add_argument('--artifacts')\n"
            "a=p.parse_args();root=Path.cwd();out=Path(a.artifacts);out.mkdir(parents=True,exist_ok=True)\n"
            "count=root/'out/production-count.txt'\n"
            "count.write_text(str(int(count.read_text())+1) if count.exists() else '1')\n"
            "assert (root/'out/build'/a.preset/'juicer.pdb').is_file()\n"
            + self.production_fixture())
        render_executable = self.build / "tests/bin/JuicerProductionPreparedRenderProbe.exe"
        args = SimpleNamespace(reason="production PDB identity regression", selection=["production"],
                               preset=self.preset, runtime="host", campaign="first", repeat_reason="", base=None, prepare=True)
        count = self.root / "out/production-count.txt"
        with patch("build_dependencies.NinjaGraph", return_value=SimpleNamespace(
                outputs={str(module.resolve()), str(render_executable.resolve())})), \
                patch.object(qualification, "checkpoint_tools", return_value=[self.python]):
            first = read_json(qualification.qualify(self.root, args))["stages"]["production"]
            self.assertEqual(first["identity"]["inputs"][pdb.relative_to(self.root).as_posix()], digest(pdb))
            args.campaign = "unchanged"
            reused = read_json(qualification.qualify(self.root, args))["stages"]["production"]
            self.assertEqual(reused["reused_from"], first["receipt"])
            self.assertEqual(count.read_text(), "1")
            stamp = pdb.stat()
            pdb.write_text("other")
            os.utime(pdb, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            args.campaign = "pdb-changed"
            changed = read_json(qualification.qualify(self.root, args))["stages"]["production"]
            self.assertNotIn("reused_from", changed)
            self.assertEqual(count.read_text(), "2")
            pdb.unlink()
            args.campaign = "pdb-missing"
            with self.assertRaises(subprocess.CalledProcessError):
                qualification.qualify(self.root, args)
            state = self.build / "verification-checkpoint.json"
            self.assertEqual(read_json(state)["production"]["status"], "failed")
            self.assertEqual(count.read_text(), "3")
            args.campaign = "missing-still-pending"
            with self.assertRaisesRegex(ValueError, "pending checkpoint"):
                qualification.qualify(self.root, args)
            self.assertEqual(count.read_text(), "3")
            pdb.write_text("other")
            args.campaign = "pdb-repaired"
            repaired = read_json(qualification.qualify(self.root, args))["stages"]["production"]
            self.assertNotIn("reused_from", repaired)
            self.assertEqual(count.read_text(), "4")

    @unittest.skipUnless(os.name == "nt", "Windows production symbol tools")
    def test_production_symbol_tool_changes_invalidate_reuse(self):
        (self.build / "CMakeCache.txt").write_text("")
        llvm = self.root / "llvm"
        llvm.mkdir()
        for name in ("llvm-nm.exe", "clang-cl.exe", "llvm-readobj.exe", "llvm-pdbutil.exe"):
            (llvm / name).write_text("first")
        environment = {"PATH": "", "CARGO_HOME": str(self.root / "cargo-home"),
                       "RUSTUP_HOME": str(self.root / "rustup")}
        def local_path(value, *parts):
            if value == "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/bin":
                return llvm
            return Path(value, *parts)
        with patch.object(qualification, "Path", side_effect=local_path) as factory:
            factory.home.return_value = Path.home()
            tools = qualification.checkpoint_tools(self.build, environment, "production")
        for name in ("llvm-readobj.exe", "llvm-pdbutil.exe"):
            self.assertIn(str(llvm / name), tools)
        previous = self.run_stage("production", tools=tools)
        for name in ("llvm-readobj.exe", "llvm-pdbutil.exe"):
            tool = llvm / name
            stamp = tool.stat()
            tool.write_text("other")
            os.utime(tool, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            changed = self.run_stage("production", campaign=name.removesuffix(".exe"), tools=tools)
            self.assertNotIn("reused_from", changed)
            self.assertNotEqual(changed["receipt"], previous["receipt"])
            previous = changed

    def retained_record(self, stage):
        record = {"mode": "checkpoint", "status": "passed", "scope_status": "passed", "reason": "retained",
                  "preset": self.preset, "selection": ["corpus"], "required": ["corpus"], "stages": {"corpus": stage}}
        path = self.root / "checkpoint.json"
        path.write_text(json.dumps(record))
        return path

    def test_collection_rechecks_authored_data_and_input_membership(self):
        data = self.root / "Resources/data.txt"
        data.parent.mkdir()
        data.write_text("authored")
        inputs = ["Resources/data.txt"]
        stage = self.run_stage("corpus", inputs=inputs)
        path = self.retained_record(stage)
        with patch.object(qualification, "checkpoint_inputs", return_value=inputs):
            with self.assertRaisesRegex(ValueError, "membership is incomplete"):
                qualification.collect_qualification(self.root, [path])
            data.write_text("changed")
            with self.assertRaisesRegex(ValueError, "source/data inputs changed"):
                qualification.collect_qualification(self.root, [path])
            data.write_text("authored")
        with patch.object(qualification, "checkpoint_inputs", return_value=inputs + ["Resources/new.txt"]):
            with self.assertRaisesRegex(ValueError, "source/data membership changed"):
                qualification.collect_qualification(self.root, [path])

    def test_collection_rejects_previously_absent_tool_configuration(self):
        config = self.root / "cargo-config.toml"
        stage = self.run_stage("corpus", tools=[self.python, str(config)])
        path = self.retained_record(stage)
        config.write_text('[build]\n')
        with patch.object(qualification, "checkpoint_inputs", return_value=["input.txt"]):
            with self.assertRaisesRegex(ValueError, "toolchain/configuration changed"):
                qualification.collect_qualification(self.root, [path])

    def test_collection_rechecks_tool_bytes_at_completion(self):
        self.assert_collection_tool_seal("bytes")

    def test_collection_rejects_unbacked_stage_provenance(self):
        stage = self.run_stage("corpus")
        path = self.retained_record(stage)
        record = json.loads(path.read_text())
        for provenance in ({"corpus": "admitted"}, {"corpus": "unsupported"},
                           {"corpus": "exact-reused", "product": "admitted"}):
            write_json(path, {**record, "stage_provenance": provenance})
            with self.subTest(provenance=provenance), \
                    patch.object(qualification, "checkpoint_inputs", return_value=["input.txt"]), \
                    self.assertRaisesRegex(ValueError, "provenance"):
                qualification.collect_qualification(self.root, [path])

    def test_native_collection_phases_seal_ordinary_evidence_and_owners(self):
        # Inject at the native-reader boundary; actual native supervision and
        # path-reader controls remain in WindowsIdentityInteropTests.
        resource = self.root / "Resources/ignored.txt"
        resource.parent.mkdir()
        resource.write_bytes(b"authored\n")
        tool = self.root / "out/owned-tool"
        tool.write_bytes(b"tool\n")
        runtime = self.root / "out/runtime"
        runtime.mkdir()
        (runtime / "member").write_bytes(b"runtime\n")
        inputs = ["input.txt", "Resources/ignored.txt"]
        stage = self.run_stage("corpus", inputs=inputs, tools=[self.python, str(tool)],
                               artifact_directories=[str(runtime)])
        path = self.retained_record(stage)
        original = {target: target.read_bytes() for target in (resource, tool, self.state_path, path)}
        raw = Path(stage["receipt"])
        original[raw] = raw.read_bytes()
        for fault in ("entry-wrapper", "terminal-tool", "terminal-input", "terminal-artifact", "terminal-owner", "terminal-raw"):
            phases = []

            def phase(root, native, name):
                phases.append(name)
                if fault == "entry-wrapper" and name == "entry":
                    changed = json.loads(original[path])
                    changed["reason"] = "changed after the wrapper was parsed"
                    path.write_text(json.dumps(changed))
                if name == "terminal":
                    if fault == "terminal-tool":
                        tool.write_bytes(b"other\n")
                    elif fault == "terminal-input":
                        resource.write_bytes(b"modified\n")
                    elif fault == "terminal-artifact":
                        (runtime / "added").write_bytes(b"new member\n")
                    elif fault == "terminal-owner":
                        changed = json.loads(original[self.state_path])
                        changed["corpus"]["status"] = "failed"
                        write_json(self.state_path, changed)
                    elif fault == "terminal-raw":
                        raw.write_bytes(original[raw] + b"changed\n")
                return {}

            messages = {"entry-wrapper": "immutable evidence", "terminal-tool": "toolchain/configuration changed during",
                "terminal-input": "source/data inputs changed during", "terminal-artifact": "artifact membership changed during",
                "terminal-owner": "ownership prevents terminal", "terminal-raw": "immutable evidence"}
            try:
                with self.subTest(fault=fault), patch.object(qualification, "checkpoint_inputs", return_value=inputs), \
                        patch.object(qualification, "prepare_native_collection", return_value={"stages": [], "expected": {}}), \
                        patch.object(qualification, "run_native_identity_phase", side_effect=phase), \
                        patch.object(qualification, "check_native_identity_evidence"), \
                        self.assertRaisesRegex(ValueError, messages[fault]):
                    qualification.collect_qualification(self.root, [path])
                self.assertEqual(phases, ["entry", "terminal"])
            finally:
                for target, content in original.items():
                    target.write_bytes(content)
                (runtime / "added").unlink(missing_ok=True)

    @unittest.skipUnless(os.name == "posix", "owned owner-file symlink control uses POSIX links")
    def test_collection_rejects_same_bytes_at_a_retargeted_owner(self):
        stage = self.run_stage("corpus")
        path = self.retained_record(stage)
        original = self.state_path.read_bytes()
        alternate = self.state_path.with_name("identical-owner.json")
        phases = []

        def phase(root, native, name):
            phases.append(name)
            if name == "terminal":
                alternate.write_bytes(original)
                self.state_path.unlink()
                self.state_path.symlink_to(alternate.name)
            return {}

        try:
            with patch.object(qualification, "checkpoint_inputs", return_value=["input.txt"]), \
                    patch.object(qualification, "prepare_native_collection", return_value={"stages": [], "expected": {}}), \
                    patch.object(qualification, "run_native_identity_phase", side_effect=phase), \
                    patch.object(qualification, "check_native_identity_evidence"), \
                    self.assertRaisesRegex(ValueError, "immutable evidence"):
                qualification.collect_qualification(self.root, [path])
            self.assertEqual(phases, ["entry", "terminal"])
        finally:
            if self.state_path.is_symlink():
                self.state_path.unlink()
            self.state_path.write_bytes(original)

    def test_collection_rechecks_source_data_at_completion(self):
        data = self.root / "Resources/data.txt"
        data.parent.mkdir()
        data.write_text("authored")
        inputs = [data.relative_to(self.root).as_posix()]
        stage = self.run_stage("corpus", inputs=inputs)
        path = self.retained_record(stage)
        actual_digest = qualification.digest
        def mutate_during_read(target):
            value = actual_digest(target)
            if str(target) == stage["receipt"]:
                stamp = data.stat()
                data.write_text("modified")
                os.utime(data, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            return value
        with patch.object(qualification, "checkpoint_inputs", return_value=inputs), \
                patch.object(qualification, "digest", side_effect=mutate_during_read):
            with self.assertRaisesRegex(ValueError, "source/data inputs changed during collection"):
                qualification.collect_qualification(self.root, [path])

    def test_collection_rechecks_source_membership_at_completion(self):
        import verify
        self.native_project()
        inputs = qualification.checkpoint_inputs(self.root, self.build, "corpus")
        stage = self.run_stage("corpus", inputs=inputs)
        path = self.retained_record(stage)
        record = read_json(path)
        target = self.root / "out/validation" / self.preset / "corpus-record.json"
        path.rename(target)
        record["source"] = verify.source_snapshot(self.root)
        target.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "full checkpoint membership"):
            qualification.collect_qualification(self.root, [target])
        actual_digest = qualification.digest
        def mutate_during_read(path):
            value = actual_digest(path)
            if str(path) == stage["receipt"]:
                (self.root / "new-consumed.cpp").write_text("int added();\n")
            return value
        with patch.object(qualification, "digest", side_effect=mutate_during_read):
            with self.assertRaisesRegex(ValueError, "source/data membership changed during collection"):
                qualification.collect_qualification(self.root, [target])
        self.assertEqual(digest(Path(stage["receipt"])), stage["sha256"])

    def test_collection_rechecks_ignored_input_membership_at_completion(self):
        import verify
        self.native_project()
        added = self.root / "tests/ffi/out/added.bin"
        added.parent.mkdir(parents=True)
        inputs = qualification.checkpoint_inputs(self.root, self.build, "corpus")
        stage = self.run_stage("corpus", inputs=inputs)
        path = self.retained_record(stage)
        record = read_json(path)
        target = self.root / "out/validation" / self.preset / "ignored-membership-record.json"
        path.rename(target)
        record["source"] = verify.source_snapshot(self.root)
        target.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "full checkpoint membership"):
            qualification.collect_qualification(self.root, [target])
        actual_inputs = qualification.checkpoint_inputs
        changed = []
        def mutate_after_discovery(*args, **kwargs):
            result = actual_inputs(*args, **kwargs)
            if not changed:
                added.write_bytes(b"new declared test input")
                changed.append(added.relative_to(self.root).as_posix())
            return result
        with patch.object(qualification, "checkpoint_inputs", side_effect=mutate_after_discovery):
            with self.assertRaisesRegex(ValueError, "source/data membership changed during collection"):
                qualification.collect_qualification(self.root, [target])
        self.assertEqual(changed, [added.relative_to(self.root).as_posix()])
        subprocess.run(["git", "check-ignore", "--quiet", "--", changed[0]], cwd=self.root,
                       check=True, capture_output=True, timeout=10)
        self.assertEqual(record["source"], verify.source_snapshot(self.root))
        self.assertIn(changed[0], actual_inputs(self.root, self.build, "corpus"))
        self.assertEqual(digest(Path(stage["receipt"])), stage["sha256"])
        write_json(target.parent / "ignored-membership-control.json",
                   {"added": changed[0], "git_ignored": True, "candidate_unchanged": True,
                    "rejected": "source/data membership changed during collection",
                    "receipt": stage["receipt"], "sha256": stage["sha256"]})

    def test_collection_rechecks_absent_tool_configuration_at_completion(self):
        self.assert_collection_tool_seal("absent")

    def test_collection_rechecks_tool_path_binding_at_completion(self):
        self.assert_collection_tool_seal("binding")

    def assert_collection_tool_seal(self, change):
        directory = self.root / "out/validation" / self.preset / "tool-seal"
        first, second, alias = (directory / name for name in ("first", "second", "alias"))
        for target in (first, second):
            target.mkdir(parents=True)
            (target / "tool").write_text("first")
        def bind(target):
            if os.name == "nt":
                subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(alias), str(target)],
                               check=True, capture_output=True, timeout=10)
            else:
                alias.symlink_to(target, target_is_directory=True)
        def unbind():
            if os.name == "nt":
                alias.rmdir()
            else:
                alias.unlink()
        bind(first)
        config = directory / "absent.toml"
        tools = [self.python, str(first / "tool"), str(second / "tool"), str(alias / "tool"), str(config)]
        stages = {name: self.run_stage(name, tools=tools) for name in ("boundary", "corpus", "tooling")}
        path = self.retained_record(stages["corpus"])
        record = read_json(path)
        record.update(selection=list(stages), required=list(stages), stages=stages)
        path.write_text(json.dumps(record))
        originals = {stage["receipt"]: digest(Path(stage["receipt"])) for stage in stages.values()}
        actual_digest = qualification.digest
        changed = []
        def mutate_during_read(path):
            value = actual_digest(path)
            if not changed and str(path) in originals:
                if change == "bytes":
                    tool = first / "tool"
                    stamp = tool.stat()
                    tool.write_text("other")
                    os.utime(tool, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                elif change == "absent":
                    config.write_text("[build]\n")
                else:
                    unbind()
                    bind(second)
                changed.append(change)
            return value
        with patch.object(qualification, "checkpoint_inputs", return_value=["input.txt"]):
            with patch.object(qualification, "tool_snapshot", wraps=qualification.tool_snapshot) as reads:
                with self.assertRaisesRegex(ValueError, "full checkpoint membership"):
                    qualification.collect_qualification(self.root, [path])
                self.assertEqual(reads.call_count, 2, "one opening and one closing tool scan for all stages")
            with patch.object(qualification, "digest", side_effect=mutate_during_read):
                with self.assertRaisesRegex(ValueError, "toolchain/configuration changed during collection"):
                    qualification.collect_qualification(self.root, [path])
            self.assertEqual(changed, [change])
            if change != "binding":
                # A later collection must observe the retained mutation afresh.
                with self.assertRaisesRegex(ValueError, "toolchain/configuration changed"):
                    qualification.collect_qualification(self.root, [path])
        self.assertEqual(originals, {p: digest(Path(p)) for p in originals})
        write_json(directory / "control.json", {"change": change, "rejected": True,
                   "scope": "real collector; incomplete lane membership remains rejected",
                   "original_receipts": originals, "tool_reads_per_collection": 2})

    def test_collection_rejects_production_without_execution_bound_manifest(self):
        artifact = self.build / "plugin.a"
        artifact.write_bytes(b"inspected final artifact")
        inputs = ["input.txt", artifact.relative_to(self.root).as_posix()]
        stage = self.run_stage("production", inputs=inputs)
        record = {"mode": "checkpoint", "status": "passed", "scope_status": "passed", "reason": "production",
                  "preset": self.preset, "selection": ["production"], "required": ["production"],
                  "stages": {"production": stage}}
        path = self.root / "production-checkpoint.json"
        path.write_text(json.dumps(record))
        with patch.object(qualification, "checkpoint_inputs", return_value=inputs):
            with self.assertRaisesRegex(ValueError, "production command does not declare retained execution evidence"):
                qualification.collect_qualification(self.root, [path])

    def test_probe_artifact_ancestor_configuration_invalidates_checkpoint(self):
        (self.build / "CMakeCache.txt").write_text("")
        environment = {"PATH": "", "CARGO_HOME": str(self.root / "cargo-home")}
        tools = qualification.checkpoint_tools(self.build, environment)
        config = self.root / "out/.cargo/config.toml"
        first = self.run_stage("boundary", tools=tools)
        self.assertIn(str(config), tools)
        config.parent.mkdir(parents=True)
        config.write_text('[build]\n')
        changed = self.run_stage("boundary", campaign="artifact-config", tools=tools)
        self.assertNotEqual(first["receipt"], changed["receipt"])
        import importlib.util
        spec = importlib.util.spec_from_file_location("checkpoint_quality", ROOT / "scripts/check-quality.py")
        quality = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = quality
        spec.loader.exec_module(quality)
        with self.assertRaisesRegex(quality.QualityError, "different Cargo configuration"):
            quality.check_probe_configuration(self.root, self.root / "out/probe")
        config.unlink()
        config.mkdir()
        with self.assertRaisesRegex(quality.QualityError, "not a file"):
            quality.check_probe_configuration(self.root, self.root / "out/probe")

    def test_tool_snapshot_distinguishes_absent_files_from_invalid_directories(self):
        config = self.root / "config.toml"
        with self.assertRaises(FileNotFoundError):
            tool_snapshot([str(config)])
        self.assertEqual(tool_snapshot([str(config)], allow_missing=True), {str(config): None})
        config.mkdir()
        for allow_missing in (False, True):
            with self.subTest(allow_missing=allow_missing), self.assertRaisesRegex(ValueError, "not a file"):
                tool_snapshot([str(config)], allow_missing=allow_missing)


class FixtureLifetimeTests(unittest.TestCase):
    def run_checkpoint_fixture(self, environment, body):
        case = CheckpointTests("test_new_campaign_reuses_original_success_without_execution")
        case.test_new_campaign_reuses_original_success_without_execution = lambda: body(case)
        with patch.dict(os.environ, environment):
            result = case.run(unittest.TestResult())
        self.addCleanup(case.temporary.cleanup)
        return case, result

    def test_artifact_routing_cleans_successful_checkpoint_fixture(self):
        with tempfile.TemporaryDirectory() as artifact_root:
            artifact_root = Path(artifact_root).resolve()
            case, result = self.run_checkpoint_fixture({
                "JUICER_TEST_ARTIFACT_DIR": str(artifact_root),
                "JUICER_TEST_RETAIN_FIXTURES": "",
            }, lambda case: None)
        self.assertTrue(result.wasSuccessful())
        self.assertFalse(case.directory.exists())
        self.assertTrue(case.directory.is_relative_to(artifact_root))

    def test_named_downstream_hold_survives_outer_pass_until_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact_root = Path(temporary).resolve()
            case = PolicyCollectionTests("test_policy_reader_source_is_one_immutable_blob_per_owner_hash")
            case.test_policy_reader_source_is_one_immutable_blob_per_owner_hash = lambda: case.fixture_holds.add(
                "policy-f21-downstream")
            with patch.dict(os.environ, {"JUICER_TEST_ARTIFACT_DIR": str(artifact_root),
                                         "JUICER_TEST_RETAIN_FIXTURES": ""}):
                result = case.run(unittest.TestResult())
            self.assertTrue(result.wasSuccessful())
            marker = json.loads((case.folder / MARKER).read_text())
            self.assertEqual(marker["outcome"], "passed")
            self.assertEqual(marker["holds"], ["policy-f21-downstream"])
            self.assertEqual(selected_paths(artifact_root), [case.folder])
            self.assertTrue(release_fixture_hold(case.folder, "policy-f21-downstream", "final-consumer"))
            self.assertFalse(case.folder.exists())
            self.assertEqual(selected_paths(artifact_root), [])

    def test_review_hold_preserves_fixture_after_downstream_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact_root = Path(temporary).resolve()
            case = PolicyCollectionTests("test_policy_reader_source_is_one_immutable_blob_per_owner_hash")
            case.test_policy_reader_source_is_one_immutable_blob_per_owner_hash = lambda: case.fixture_holds.update(
                {"policy-f21-downstream", "policy-f21-review"})
            with patch.dict(os.environ, {"JUICER_TEST_ARTIFACT_DIR": str(artifact_root),
                                         "JUICER_TEST_RETAIN_FIXTURES": ""}):
                result = case.run(unittest.TestResult())
            self.assertTrue(result.wasSuccessful())
            marker = read_fixture_marker(case.folder / MARKER)
            self.assertEqual(marker["holds"], ["policy-f21-downstream", "policy-f21-review"])
            self.assertFalse(release_fixture_hold(case.folder, "policy-f21-downstream", "foreign-consumer"))
            self.assertEqual(read_fixture_marker(case.folder / MARKER)["holds"], ["policy-f21-review"])
            self.assertTrue(case.folder.exists())
            self.assertIn(case.folder, selected_paths(artifact_root))
            self.assertTrue(release_fixture_hold(case.folder, "policy-f21-review",
                                                 "r4-independent-review-owner-acceptance"))
            self.assertFalse(case.folder.exists())
            self.assertEqual(selected_paths(artifact_root), [])

    def test_hold_release_rechecks_live_owner_and_running_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact_root = Path(temporary).resolve()
            for blocker in ("owner-directory", "campaign-running-lock"):
                with self.subTest(blocker=blocker):
                    folder = artifact_root / "checkpoint-controls" / blocker
                    owner_root = folder / "project"
                    owner_root.mkdir(parents=True)
                    case = unittest.FunctionTestCase(lambda: None)
                    case.fixture_holds = {"consumer-proof"}
                    with patch.dict(os.environ, {"JUICER_TEST_RETAIN_FIXTURES": ""}):
                        record_fixture(folder, case, "passed", owner_root=owner_root)
                    if blocker == "owner-directory":
                        (owner_root / "out/verification-locks/owner-created-after-pass").mkdir(parents=True)
                    else:
                        running = owner_root / "out/validation/linux-debug/late-campaign/running.lock"
                        running.parent.mkdir(parents=True)
                        running.write_text("still running\n", encoding="utf-8")
                    before = read_json(folder / MARKER)
                    with patch.dict(os.environ, {"JUICER_TEST_RETAIN_FIXTURES": ""}), self.assertRaisesRegex(
                            ValueError, "current owner records or campaign work remain"):
                        release_fixture_hold(folder, "consumer-proof", "final-consumer")
                    after = read_json(folder / MARKER)
                    self.assertEqual(after["holds"], before["holds"])
                    self.assertEqual(after.get("hold_releases", []), [])
                    self.assertIn(folder, selected_paths(artifact_root))

    def test_producer_opt_in_survives_empty_consumer_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact_root = Path(temporary).resolve()
            folder = artifact_root / "checkpoint-controls/opted-in producer"
            owner_root = folder / "project"
            owner_root.mkdir(parents=True)
            case = unittest.FunctionTestCase(lambda: None)
            case.fixture_holds = {"consumer-proof"}
            with patch.dict(os.environ, {"JUICER_TEST_RETAIN_FIXTURES": "1"}):
                record_fixture(folder, case, "passed", owner_root=owner_root)
            with patch.dict(os.environ, {"JUICER_TEST_RETAIN_FIXTURES": ""}):
                self.assertFalse(release_fixture_hold(folder, "consumer-proof", "final-consumer"))
            record = read_json(folder / MARKER)
            self.assertTrue(record["diagnostic_retain"])
            self.assertEqual(record["holds"], [])
            self.assertTrue(folder.exists())
            # Passed opt-in output remains local-only rather than CI failure evidence.
            self.assertEqual(selected_paths(artifact_root), [])

    def test_release_rejects_duplicate_keys_and_empty_release_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact_root = Path(temporary).resolve()
            folder = artifact_root / "checkpoint-controls/strict marker"
            folder.mkdir(parents=True)
            marker = folder / MARKER
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text('{"outcome":"passed","outcome":"failed"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate fixture marker key"):
                release_fixture_hold(folder, "hold", "consumer")

            owner_root = folder / "project"
            owner_root.mkdir()
            case = unittest.FunctionTestCase(lambda: None)
            case.fixture_holds = {"hold"}
            record_fixture(folder, case, "passed", owner_root=owner_root)
            for name, consumer in (("", "consumer"), ("hold", " ")):
                with self.subTest(name=name, consumer=consumer), self.assertRaisesRegex(ValueError, "fixed policy-control owner"):
                    release_fixture_hold(folder, name, consumer)
            self.assertEqual(read_json(marker)["holds"], ["hold"])

    def test_foreign_owner_root_uses_explicit_release_resolver(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary).resolve() / "checkpoint-controls/foreign spelling"
            owner_root = folder / "project"
            owner_root.mkdir(parents=True)
            case = unittest.FunctionTestCase(lambda: None)
            case.fixture_holds = {"consumer-proof"}
            record_fixture(folder, case, "passed", owner_root=owner_root)
            marker = read_json(folder / MARKER)
            marker["fixture"] = r"C:\fixture\foreign spelling"
            marker["owner_root"] = r"C:\fixture\foreign spelling\project"
            write_json(folder / MARKER, marker)
            mapped = {
                r"C:\fixture\foreign spelling": folder,
                r"C:\fixture\foreign spelling\project": owner_root,
            }
            late_owner = owner_root / "out/verification-locks/late-owner"
            late_owner.mkdir(parents=True)
            def resolver(value):
                return mapped.get(str(value), value)

            with self.assertRaisesRegex(ValueError, "current owner records or campaign work remain"):
                release_fixture_hold(folder, "consumer-proof", "final-consumer", resolve=resolver)
            self.assertEqual(read_json(folder / MARKER)["holds"], ["consumer-proof"])
            late_owner.rmdir()
            self.assertTrue(release_fixture_hold(folder, "consumer-proof", "final-consumer",
                                                 resolve=resolver))
            self.assertFalse(folder.exists())

    def test_legacy_hold_missing_release_metadata_fails_closed(self):
        for field in ("diagnostic_retain", "owner_root"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary).resolve() / "checkpoint-controls/legacy hold"
                owner_root = folder / "project"
                owner_root.mkdir(parents=True)
                case = unittest.FunctionTestCase(lambda: None)
                case.fixture_holds = {"downstream"}
                record_fixture(folder, case, "passed", owner_root=owner_root)
                marker = read_json(folder / MARKER)
                del marker[field]
                write_json(folder / MARKER, marker)
                with self.assertRaisesRegex(ValueError, "explicit owner-root or diagnostic-retention"):
                    release_fixture_hold(folder, "downstream", "final-consumer")
                self.assertEqual(read_json(folder / MARKER)["holds"], ["downstream"])
                self.assertTrue(folder.exists())

    def test_release_rejects_redirected_owner_root_with_live_project_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact_root = Path(temporary).resolve()
            folder = artifact_root / "checkpoint-controls/redirected owner"
            owner_root = folder / "project"
            owner_root.mkdir(parents=True)
            ignored_root = folder / "ignore-me"
            ignored_root.mkdir()
            case = unittest.FunctionTestCase(lambda: None)
            case.fixture_holds = {"downstream"}
            record_fixture(folder, case, "passed", owner_root=owner_root)
            lock = owner_root / "out/verification-locks/live-owner/owner.json"
            lock.parent.mkdir(parents=True)
            lock.write_text('{"status":"running"}', encoding="utf-8")
            marker = read_json(folder / MARKER)
            marker["owner_root"] = str(ignored_root)
            write_json(folder / MARKER, marker)
            with self.assertRaisesRegex(ValueError, "owner root is malformed or outside its fixture"):
                release_fixture_hold(folder, "downstream", "final-consumer")
            self.assertEqual(read_json(folder / MARKER)["holds"], ["downstream"])
            self.assertTrue(lock.exists())
            self.assertTrue(folder.exists())
            self.assertEqual(selected_paths(artifact_root), [folder])

    def test_policy_fixture_cleanup_retains_new_live_owner_or_running_lock(self):
        for blocker in ("owner-directory", "campaign-running-lock"):
            with self.subTest(blocker=blocker), tempfile.TemporaryDirectory() as temporary:
                artifact_root = Path(temporary).resolve()
                case = PolicyCollectionTests("test_policy_reader_source_is_one_immutable_blob_per_owner_hash")
                def leave_work():
                    if blocker == "owner-directory":
                        (case.root / "out/verification-locks/new-owner").mkdir(parents=True)
                    else:
                        running = case.root / "out/validation/linux-debug/late-campaign/running.lock"
                        running.parent.mkdir(parents=True)
                        running.write_text("still running\n", encoding="utf-8")
                case.test_policy_reader_source_is_one_immutable_blob_per_owner_hash = leave_work
                with patch.dict(os.environ, {"JUICER_TEST_ARTIFACT_DIR": str(artifact_root),
                                             "JUICER_TEST_RETAIN_FIXTURES": ""}):
                    result = case.run(unittest.TestResult())
                self.assertTrue(result.wasSuccessful())
                self.addCleanup(case.temporary.cleanup)
                self.assertTrue(case.folder.exists())
                self.assertTrue(read_json(case.folder / MARKER)["blocked_owners"])
                self.assertEqual(selected_paths(artifact_root), [case.folder])

    def test_external_inputs_follow_checkpoint_fixture_outcome(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact_root = Path(temporary).resolve()
            for outcome in ("passed", "failed", "retained"):
                with self.subTest(outcome=outcome):
                    def body(case):
                        module = case.directory / "external/consumer.cmake"
                        module.parent.mkdir()
                        module.write_bytes(b"# original external input\n")
                        self.assertFalse(module.is_relative_to(case.root))
                        if outcome == "failed":
                            case.fail("external input retention control")
                    case, result = self.run_checkpoint_fixture({
                        "JUICER_TEST_ARTIFACT_DIR": str(artifact_root),
                        "JUICER_TEST_RETAIN_FIXTURES": "1" if outcome == "retained" else "",
                    }, body)
                    self.assertEqual(result.wasSuccessful(), outcome != "failed")
                    self.assertEqual(case.directory.exists(), outcome != "passed")
                    self.assertEqual(selected_paths(artifact_root), [case.directory] if outcome == "failed" else [])
                    if outcome != "passed":
                        self.assertEqual((case.directory / "external/consumer.cmake").read_bytes(), b"# original external input\n")
                        record = read_json(case.directory / MARKER)
                        self.assertEqual(record["outcome"], "failed" if outcome == "failed" else "passed")
                        self.assertEqual(record["blocked_owners"], [])
                    case.temporary.cleanup()

    def test_diagnostic_opt_in_and_test_failures_retain_checkpoint_fixture(self):
        with tempfile.TemporaryDirectory() as artifact_root:
            case, result = self.run_checkpoint_fixture({
                "JUICER_TEST_ARTIFACT_DIR": artifact_root,
                "JUICER_TEST_RETAIN_FIXTURES": "1",
            }, lambda case: None)
            self.assertTrue(result.wasSuccessful())
            self.assertTrue(case.root.exists())
            self.assertEqual(read_json(case.directory / MARKER)["outcome"], "passed")
            self.assertFalse((case.lease.path / "owner.json").exists())
            case.temporary.cleanup()

            def fail_test(case):
                case.fail("fixture lifecycle control")

            failed_case, failed_result = self.run_checkpoint_fixture({
                "JUICER_TEST_ARTIFACT_DIR": artifact_root,
                "JUICER_TEST_RETAIN_FIXTURES": "",
            }, fail_test)
            self.assertEqual(len(failed_result.failures), 1)
            self.assertTrue(failed_case.root.exists())
            self.assertEqual(read_json(failed_case.directory / MARKER)["outcome"], "failed")
            failed_case.temporary.cleanup()

            def fail_subtest(case):
                with case.subTest(control="failure"):
                    case.fail("fixture lifecycle control")

            failed_case, failed_result = self.run_checkpoint_fixture({
                "JUICER_TEST_ARTIFACT_DIR": artifact_root,
                "JUICER_TEST_RETAIN_FIXTURES": "",
            }, fail_subtest)
            self.assertEqual(len(failed_result.failures), 1)
            self.assertTrue(failed_case.root.exists())
            self.assertFalse((failed_case.lease.path / "owner.json").exists())
            failed_case.temporary.cleanup()

    def test_missing_result_retains_checkpoint_fixture(self):
        case = CheckpointTests("test_new_campaign_reuses_original_success_without_execution")
        case.setUp()
        self.addCleanup(case.temporary.cleanup)
        case.doCleanups()
        self.assertTrue(case.root.exists())

    def test_cancellation_keeps_fixture_owner_after_completed_or_unproved_command(self):
        for drained in (True, False):
            with self.subTest(drained=drained):
                case = CheckpointTests("test_new_campaign_reuses_original_success_without_execution")
                previous = os.environ.get("JUICER_OUTPUT_LEASES")
                def cancel():
                    case.retain = True
                    module = case.directory / "external/consumer.cmake"
                    module.parent.mkdir()
                    module.write_bytes(b"# interrupted external input\n")
                    if drained:
                        case.run_stage("boundary")
                        raise KeyboardInterrupt
                    # No process is launched: cancellation has no terminal proof.
                    def unproved_launch(*args, **kwargs):
                        args[7].update(drained=False, status="running")
                        raise KeyboardInterrupt
                    with patch("execution.run_process_tree", side_effect=unproved_launch):
                        case.run_stage("boundary")
                setattr(case, case._testMethodName, cancel)
                with self.assertRaises(KeyboardInterrupt):
                    case.run(unittest.TestResult())
                self.addCleanup(case.temporary.cleanup)
                self.assertNotIn(case.lease.path, OutputLease.held)
                self.assertEqual(os.environ.get("JUICER_OUTPUT_LEASES"), previous)
                receipt = next((case.root / "out/validation").glob("*/first/attempts/*/result.json"))
                record = read_json(receipt)
                owner = case.lease.path / "owner.json"
                running_lock = case.root / "out/validation" / case.preset / "first/running.lock"
                self.assertEqual(record["process_tree"]["drained"], drained)
                self.assertTrue(owner.exists())
                self.assertEqual(read_json(owner)["status"], "blocked")
                self.assertEqual((case.directory / "external/consumer.cmake").read_bytes(), b"# interrupted external input\n")
                marker = read_json(case.directory / MARKER)
                self.assertEqual(marker["outcome"], "interrupted")
                expected_blockers = {str(owner)}
                if not drained:
                    self.assertTrue(running_lock.exists())
                    expected_blockers.add(str(running_lock))
                self.assertEqual(set(marker["blocked_owners"]), expected_blockers)
                self.assertEqual(len(marker["blocked_owners"]), len(expected_blockers))
                self.assertIn(case.directory, selected_paths(case.directory.parent.parent))
                if drained:
                    self.assertEqual(record["status"], "passed")
                    process = read_json(Path(record["process_tree"]["receipt"]))
                    self.assertTrue(process["drained"])
                    self.assertEqual(process["status"], "completed")
                else:
                    self.assertEqual(record["status"], "blocked")
                    self.assertTrue(running_lock.exists())
                write_json(case.root / "out/validation" / case.preset / "cancellation.json",
                           {"scope": "cancelled fixture remains blocked; unproved branch launches no child",
                            "command_drained": drained, "receipt": str(receipt), "sha256": digest(receipt),
                            "owner_retained": owner.exists()})


class MembershipTests(unittest.TestCase):
    def test_complete_policy_requires_thirteen_roles_in_six_contexts(self):
        # Policy-only controls; native execution validity belongs to collector
        # fixtures, not these deliberately minimal role records.
        records = [
            {"preset": "linux-debug", "runtime": "all", "selection": ["product", "boundary", "tooling", "corpus"]},
            {"preset": "windows-clang-debug", "runtime": "all", "selection": ["product", "boundary", "tooling"]},
            {"preset": "linux-release", "runtime": "all", "selection": ["product", "boundary"]},
            {"preset": "windows-clang-release", "runtime": "all", "selection": ["product", "boundary"]},
            {"preset": "linux-release", "runtime": "all", "selection": ["production"]},
            {"preset": "windows-clang-release", "runtime": "all", "selection": ["production"]},
        ]
        qualification.validate_qualification_membership(records)
        for index, record in enumerate(records):
            for group in record["selection"]:
                for mutation in ("remove", "duplicate"):
                    changed = json.loads(json.dumps(records))
                    selection = changed[index]["selection"]
                    selection.remove(group) if mutation == "remove" else selection.append(group)
                    with self.subTest(preset=record["preset"], group=group, mutation=mutation), self.assertRaises(ValueError):
                        qualification.validate_qualification_membership(changed)
            for mutation in ("remove-context", "duplicate-context"):
                changed = records[:index] + records[index + 1:] if mutation == "remove-context" else [*records, record]
                with self.subTest(context=index, mutation=mutation), self.assertRaises(ValueError):
                    qualification.validate_qualification_membership(changed)
        for index in (1, 2, 3):
            changed = json.loads(json.dumps(records))
            changed[0]["selection"].remove("corpus")
            changed[index]["selection"].append("corpus")
            self.assertEqual(sum("corpus" in record["selection"] for record in changed), 1)
            with self.subTest(wrong_corpus=changed[index]["preset"]), self.assertRaisesRegex(ValueError, "corpus"):
                qualification.validate_qualification_membership(changed)
        for index in range(4):
            changed = json.loads(json.dumps(records))
            changed[index]["runtime"] = "host"
            with self.subTest(host_only=index), self.assertRaisesRegex(ValueError, "host-only"):
                qualification.validate_qualification_membership(changed)
        changed = json.loads(json.dumps(records))
        changed[2]["selection"].append("tooling")
        with self.assertRaises(ValueError):
            qualification.validate_qualification_membership(changed)
        # These finite path examples check metadata partitioning only. Native
        # reads and their consumers are exercised by the focused F21 fixture.
        from pathlib import PureWindowsPath
        runtime = {r"C:\Runtime\python.exe": {"resolved": r"C:\Runtime\python.exe", "sha256": "a" * 64},
                   r"C:\Base\python.exe": {"resolved": r"C:\Base\python.exe", "sha256": "b" * 64}}
        lexical, target = r"C:\SDK\include\header.h", r"C:\SDK\Include\header.h"
        binding = {"resolved": target, "sha256": "c" * 64}
        files = json.loads(json.dumps(runtime))
        files.update({lexical: dict(binding), target: dict(binding)})
        partitions = qualification._policy_native_partitions(files, runtime)
        self.assertEqual(len(partitions), 2)
        self.assertEqual({path: row for partition in partitions for path, row in partition.items()}, files)
        self.assertEqual(partitions, qualification._policy_native_partitions(dict(reversed(list(files.items()))), runtime))
        for partition in partitions:
            self.assertEqual(len(partition), len({PureWindowsPath(path) for path in partition}))
            self.assertEqual({path: partition[path] for path in runtime}, runtime)
            self.assertEqual(sum(path in partition for path in (lexical, target)), 1)
        for mutation in ("hash", "target", "runtime-alias"):
            damaged = json.loads(json.dumps(files))
            if mutation == "hash":
                damaged[lexical]["sha256"] = "0" * 64
            elif mutation == "target":
                damaged[lexical]["resolved"] = r"C:\Other\header.h"
            else:
                damaged[r"C:\Runtime\PYTHON.exe"] = dict(runtime[r"C:\Runtime\python.exe"])
            with self.subTest(native_partition_metadata=mutation), \
                    self.assertRaisesRegex(ValueError, "conflicting bindings|runtime paths have conflicting aliases"):
                qualification._policy_native_partitions(damaged, runtime)

    @staticmethod
    def inventory(*rows):
        return {"tests": [{"name": name, "properties": [{"name": "LABELS", "value": labels}]} for name, labels in rows]}

    def test_relevance_is_disjoint_and_runtime_is_orthogonal(self):
        inventory = self.inventory(("behavior", ["host", "product"]), ("boundary", ["host", "boundary"]),
                                   ("runner", ["host", "tooling"]), ("data", ["host", "corpus"]),
                                   ("device", ["gpu", "product"]))
        result = qualification.membership(inventory, ["product", "boundary", "tooling", "corpus"], "all")
        self.assertEqual(sorted(n for names in result.values() for n in names), ["behavior", "boundary", "data", "device", "runner"])
        self.assertNotIn("device", qualification.membership(inventory, ["product"], "host")["product"])

    def test_missing_duplicate_unclassified_and_empty_membership_fail(self):
        for inventory in (self.inventory(), self.inventory(("a", ["host"])),
                          self.inventory(("a", ["host", "product", "tooling"])),
                          self.inventory(("a", ["host", "product"]), ("a", ["host", "product"]))):
            with self.subTest(inventory=inventory), self.assertRaises(ValueError):
                qualification.membership(inventory, ["product"], "host")

    def test_disabled_required_case_is_rejected(self):
        inventory = self.inventory(("a", ["host", "boundary"]))
        inventory["tests"][0]["properties"].append({"name": "DISABLED", "value": True})
        with self.assertRaisesRegex(ValueError, "disabled"):
            qualification.membership(inventory, ["boundary"], "host")

    def test_global_serialization_requires_a_named_owner(self):
        inventory = self.inventory(("a", ["host", "product"]))
        inventory["tests"][0]["properties"].append({"name": "RUN_SERIAL", "value": True})
        with self.assertRaisesRegex(ValueError, "global serialization"):
            qualification.membership(inventory, ["product"], "host")
        inventory["tests"][0]["properties"].pop()
        inventory["tests"][0]["properties"].append({"name": "RESOURCE_LOCK", "value": ["owned_resource"]})
        self.assertEqual(qualification.membership(inventory, ["product"], "host"), {"product": ["a"]})

    def test_explicit_scratch_roots_cannot_collide_across_selected_groups(self):
        inventory = self.inventory(("a", ["host", "product"]), ("b", ["host", "boundary"]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inventory["tests"][0]["command"] = ["case", "--scratch-root", str(root / "scratch/../shared")]
            inventory["tests"][1]["command"] = ["case", "--scratch-root", str(root / "shared")]
            with self.assertRaisesRegex(ValueError, "colliding scratch"):
                qualification.membership(inventory, ["product", "boundary"], "all")
            inventory["tests"][1]["command"][-1] = str(root / "other")
            result = qualification.membership(inventory, ["product", "boundary"], "all")
            self.assertEqual(result, {"product": ["a"], "boundary": ["b"]})
            for arguments in (["--scratch-root", "shared"], ["--scratch-root"],
                              ["--scratch-root", str(root / "other"), "--scratch-root", str(root / "shared")]):
                inventory["tests"][1]["command"] = ["case", *arguments]
                with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "scratch destination"):
                    qualification.membership(inventory, ["product", "boundary"], "all")

    def test_ctest_exit_zero_cannot_hide_missing_skipped_or_unrun_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "tests.xml"
            for body in ('', '<testcase name="required" status="notrun"/>',
                         '<testcase name="required" status="run"><skipped/></testcase>',
                         '<testcase name="required" status="run"><failure/></testcase>'):
                report.write_text('<testsuite>' + body + '</testsuite>')
                with self.subTest(body=body), self.assertRaisesRegex(ValueError, "did not execute"):
                    qualification.validate_ctest_report(report, ["required"])
            report.write_text('<testsuite><testcase name="required" status="run"/></testsuite>')
            qualification.validate_ctest_report(report, ["required"])

    def ci_jobs(self, native):
        text = (ROOT / f".github/workflows/tests-{native}-host.yml").read_text()
        body = text.split("\njobs:\n", 1)[1]
        starts = list(re.finditer(r"^  ([a-z][a-z0-9-]*):$", body, re.M))
        jobs = {match[1]: body[match.end():starts[index + 1].start() if index + 1 < len(starts) else len(body)]
                for index, match in enumerate(starts)}
        return text, jobs

    def test_native_ci_preserves_independent_product_and_tooling_coverage(self):
        for native, family in (("linux", "linux"), ("windows", "windows-clang")):
            text, jobs = self.ci_jobs(native)
            children = {f"{native}-{lane}" for lane in ("product-debug", "product-release", "analysis", "tooling")}
            self.assertEqual(set(jobs), children | {f"{native}-debug-host"})
            self.assertIn("cancel-in-progress: true", text)
            for trigger in ("workflow_dispatch:", "pull_request:", "push:"):
                self.assertIn(trigger, text)
            self.assertNotRegex(text, re.compile(r"^    paths(?:-ignore)?:", re.M))
            for name in children:
                self.assertNotRegex(jobs[name], re.compile(r"^    (?:needs|if):", re.M))
                if name != "windows-tooling":
                    self.assertNotRegex(jobs[name], re.compile(r"^    strategy:", re.M))
                self.assertIn("fetch-depth: 0", jobs[name])
                self.assertIn("tests/ctest/control_artifacts.py --preset", jobs[name])
                self.assertIn("include-hidden-files: true", jobs[name])
            for profile in ("debug", "release"):
                block = jobs[f"{native}-product-{profile}"]
                preset = f"{family}-{profile}"
                self.assertRegex(block, re.compile(rf"^          cmake --build --preset {preset}$", re.M))
                labels = "product|boundary|corpus" if (native, profile) == ("linux", "debug") else "product|boundary"
                selections = re.findall(rf'ctest --preset {preset}(?: -N)? -L "([^"\n]+)" -LE gpu', block)
                self.assertEqual(selections, [labels, labels])
                self.assertIn("--no-tests=error", block)
                self.assertIn(f"out/validation/{preset}/quality/rust-naming/**", block)
                self.assertIn(f"out/validation/{preset}/quality/rust-boundaries/**", block)
                if native == "linux":
                    # NativeBoundary inspects real ordinary objects in both product lanes.
                    self.assertIn("llvm-toolchain-noble-22 main", block)
                    self.assertRegex(block, r"sudo apt-get install[^\n]*\bllvm-22\b")
                    self.assertIn("llvm-nm-22 --version", block)
                if native == "windows":
                    self.assertIn(f"out/build/{preset}/juicer.ofx", block)
                    for operation in ("NativeLibrary]::Load", "TryGetExport", "OfxGetNumberOfPlugins", "OfxGetPlugin", "NativeLibrary]::Free"):
                        self.assertIn(operation, block)
            tooling = jobs[f"{native}-tooling"]
            self.assertIn(f"cmake -S tests/tooling -B out/build/tooling-{native}", tooling)
            label = "tooling"
            if native == "windows":
                strategy = re.search(r"^    strategy:\n((?:      [^\n]*\n)+)", tooling, re.M)
                self.assertIsNotNone(strategy)
                matrix = re.fullmatch(r"      fail-fast: false\n      matrix:\n        shard: \[([^\]\n]+)\]\n",
                                      strategy[1])
                self.assertIsNotNone(matrix, "only the complete tooling shard matrix may add a strategy")
                self.assertCountEqual([value.strip() for value in matrix[1].split(",")],
                                      ["core", "BuildEvidence", "Checkpoint", "PreparationRenewal"])
                label = '"^tooling-${{ matrix.shard }}$"'
            selections = re.findall(rf"^          ctest --test-dir out/build/tooling-{native}( -N)? -L (\"[^\"\n]+\"|\S+)",
                                    tooling, re.M)
            self.assertEqual(selections, [(" -N", label), ("", label)])
            # Include arguments on Bash and PowerShell continuation lines.
            commands = re.sub(r"(?:\\|`)\n[ \t]*", " ", tooling)
            self.assertNotRegex(commands, r"ctest[^\n]* (?:-R|-E|-LE|--tests-regex|--exclude-regex|--label-exclude)\b")
            self.assertIn("--no-tests=error", tooling)
            self.assertIn(f"out/validation/{family}-debug", tooling)
            self.assertNotIn("Install CUDA", tooling)
            self.assertNotIn("cmake --preset", tooling)
            self.assertIn("Report cumulative verifier scope (skips disabled)", tooling)
            self.assertIn("github.event.pull_request.base.sha || github.event.before", tooling)
            self.assertIn("SCOPE_CANDIDATE: ${{ github.sha }}", tooling)
            self.assertIn("--verifier-scope-output", tooling)
            self.assertIn("subprocess.run(command, check=True)", tooling)
            self.assertIn("verifier-scope.json", tooling)
            for obsolete in ("--probe-manifest", "JUICER_RUST_PROBE_MANIFEST", "probe_evidence.py"):
                self.assertNotIn(obsolete, text)

    def test_native_ci_analysis_owns_cumulative_debug_and_release_quality(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("ci_checkpoint_quality", ROOT / "scripts/check-quality.py")
        quality = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = quality
        self.addCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(quality)
        for native, family in (("linux", "linux"), ("windows", "windows-clang")):
            _, jobs = self.ci_jobs(native)
            block = jobs[f"{native}-analysis"]
            self.assertRegex(block, re.compile(
                rf"^        run: cmake --build --preset {family}-debug --target juicer_analysis_inputs$", re.M))
            self.assertIn(f"cmake --preset {family}-release -DBUILD_TESTING=ON", block)
            self.assertIn(f"cmake --build --preset {family}-release --target juicer_rust_build", block)
            self.assertIn("github.event.pull_request.base.sha", block)
            self.assertIn("github.event.before", block)
            self.assertIn("git cat-file -e", block)
            self.assertNotIn("HEAD^", block)
            commands = [shlex.split(arguments) for arguments in re.findall(
                r"^          (?:  )?python -u scripts/check-quality.py ([^\n]+)$", block, re.M)]
            base = "$QUALITY_BASE" if native == "linux" else "$qualityBase"
            self.assertEqual(commands, [
                ["--base", base, "--preset", f"{family}-debug"],
                ["--all-owned", "--preset", f"{family}-debug"],
                ["--files", "Cargo.toml", "--preset", f"{family}-release"],
            ])
            # Exercise the real parser with the workflow arguments: every gate
            # inherits the bounded CPU-aware default instead of fixing two workers.
            for available, expected in ((None, 1), (1, 1), (2, 2), (4, 4), (16, 4)):
                for command in commands:
                    with self.subTest(native=native, available=available, command=command), \
                            patch.object(sys, "argv", [str(spec.origin), *command]), \
                            patch.object(quality.os, "process_cpu_count", return_value=available):
                        self.assertEqual(quality.parse_arguments().jobs, expected)
            for profile in ("debug", "release"):
                self.assertIn(f"out/validation/{family}-{profile}/quality/**/*.log", block)
                self.assertIn(f"out/build/{family}-{profile}/compile_commands.json", block)

    def test_required_ci_aggregates_reject_every_non_success_and_missing_child(self):
        for native, display in (("linux", "Linux"), ("windows", "Windows")):
            _, jobs = self.ci_jobs(native)
            aggregate = jobs[f"{native}-debug-host"]
            expected = {f"{native}-{lane}" for lane in ("product-debug", "product-release", "analysis", "tooling")}
            self.assertIn(f"name: {display} Debug host and Python checks", aggregate)
            self.assertIn("    if: always()", aggregate)
            needs = set(re.findall(r"^      - ([a-z][a-z0-9-]*)$", aggregate, re.M))
            self.assertEqual(needs, expected)
            self.assertIn("REQUIRED_RESULTS: ${{ toJSON(needs) }}", aggregate)
            self.assertIn("shell: python", aggregate)
            # Execute the actual inline gate, not a separately implemented truth table.
            code = textwrap.dedent(aggregate.split("        run: |\n", 1)[1])
            success = {name: {"result": "success"} for name in expected}
            def run(document):
                return subprocess.run([sys.executable, "-c", code],
                    env={**os.environ, "REQUIRED_RESULTS": json.dumps(document)},
                    capture_output=True, text=True, timeout=10).returncode
            self.assertEqual(run(success), 0)
            # Report-only decisions never authorize a skipped lane, even when
            # the complete same-candidate classification says it is unnecessary.
            for decision in ({"complete": True, "required": False, "candidate": "current"},
                             {"complete": True, "required": True, "candidate": "current"},
                             {"complete": False}, {}, None):
                tooling = {"result": "skipped", "outputs": {"classification": decision}}
                self.assertNotEqual(run({**success, f"{native}-tooling": tooling}), 0)
            for name in sorted(expected):
                for status in ("failure", "cancelled", "timed_out", "skipped", "", None):
                    with self.subTest(native=native, child=name, status=status):
                        self.assertNotEqual(run({**success, name: {"result": status}}), 0)
                self.assertNotEqual(run({key: value for key, value in success.items() if key != name}), 0)
                self.assertNotEqual(run({**success, name: {}}), 0)
            for malformed in ({}, {**success, "unexpected-child": {"result": "success"}}, [], None):
                self.assertNotEqual(run(malformed), 0)

    def test_malformed_checkpoint_selection_and_empty_collection_schedule_no_work(self):
        import verify
        preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        with patch.object(qualification, "OutputLease") as lease:
            for changes in ({"reason": ""}, {"selection": []}, {"selection": ["unknown"]},
                            {"preset": "unsupported-debug"}, {"runtime": "unsupported"}):
                args = SimpleNamespace(**({"reason": "requested checkpoint", "selection": ["boundary"],
                                          "preset": preset, "runtime": "host"} | changes))
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    qualification.qualify(ROOT, args)
            with self.assertRaisesRegex(ValueError, "checkpoint receipts are required"):
                qualification.collect_qualification(ROOT, [])
            argv = ["verify.py", "qualification", "--preset", preset]
            with patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as stopped:
                verify.main()
            self.assertEqual(stopped.exception.code, 2)
            lease.assert_not_called()

    def test_partial_development_and_failed_results_cannot_supply_checkpoint(self):
        for record in ({"mode": "development", "status": "passed"}, {"mode": "checkpoint", "status": "failed"},
                       {"mode": "checkpoint", "status": "passed", "scope_status": "passed", "reason": "partial", "required": ["boundary"], "stages": {}}):
            with self.assertRaises(ValueError):
                qualification.require_checkpoint(record)


class ProductionTranscriptTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name).resolve()

    def streams_row(self, directory, stdout):
        import execution

        stderr = b""
        stdout_ref = execution.write_transcript(directory / "command.stdout", stdout)
        stderr_ref = execution.write_transcript(directory / "command.stderr", stderr)
        return {
            "name": "command",
            "stdout_sha256": stdout_ref["decoded_sha256"],
            "stderr_sha256": stderr_ref["decoded_sha256"],
            "streams": {"stdout": stdout_ref, "stderr": stderr_ref},
        }

    def test_legacy_raw_and_gzip_decode_to_same_complete_command_bytes(self):
        import execution

        payload = (b"production command transcript\n"
                   * (execution.COMMAND_TRANSCRIPT_GZIP_THRESHOLD // 29 + 1))
        payload += b"\xffFORBIDDEN_SYMBOL_AT_FINAL_BYTE"
        legacy_directory = self.root / "legacy"
        legacy_directory.mkdir()
        legacy_path = legacy_directory / "command.stdout"
        legacy_path.write_bytes(payload)
        legacy_row = {"name": "command", "stdout_sha256": hashlib.sha256(payload).hexdigest()}
        legacy = qualification.production_transcript(legacy_row, "stdout", legacy_directory)

        current_directory = self.root / "current"
        current_directory.mkdir()
        current_row = self.streams_row(current_directory, payload)
        current = qualification.production_transcript(
            current_row, "stdout", current_directory,
            protocol=execution.COMMAND_TRANSCRIPT_PROTOCOL,
        )
        self.assertEqual(legacy[0], current[0])
        self.assertEqual(current[0], payload)
        self.assertIn(b"\xff", current[0])
        self.assertTrue(current[0].endswith(b"FORBIDDEN_SYMBOL_AT_FINAL_BYTE"))
        self.assertEqual(legacy[1], str(legacy_path))
        self.assertEqual(current[1], str(current_directory / "command.stdout"))
        self.assertEqual(current[2], current_row["streams"]["stdout"]["encoded_sha256"])

    def test_new_reference_path_identity_protocol_and_stream_membership_are_strict(self):
        import execution

        directory = self.root / "current"
        directory.mkdir()
        row = self.streams_row(directory, b"output")
        protocol = execution.COMMAND_TRANSCRIPT_PROTOCOL
        reference = row["streams"]["stdout"]

        for replacement, message in (
            ({**reference, "path": str(self.root / "outside.stdout")}, "path or decoded identity"),
            ({**reference, "path": str(directory / "other.stdout")}, "path or decoded identity"),
            ({**reference, "decoded_sha256": "0" * 64}, "path or decoded identity"),
        ):
            malformed = {**row, "streams": {**row["streams"], "stdout": replacement}}
            with self.subTest(reference=replacement), self.assertRaisesRegex(ValueError, message):
                qualification.production_transcript(malformed, "stdout", directory, protocol=protocol)

        with self.assertRaisesRegex(ValueError, "unsupported production transcript protocol"):
            qualification.production_transcript(row, "stdout", directory, protocol="future-transcript-v9")
        missing = {**row, "streams": {"stdout": reference}}
        with self.assertRaisesRegex(ValueError, "stream membership"):
            qualification.production_transcript(missing, "stdout", directory, protocol=protocol)

    def test_transcript_reader_rejects_truncated_and_bitflipped_gzip(self):
        directory = self.root / "current"
        directory.mkdir()
        payload = b"compressed production output\n" * 10000
        row = self.streams_row(directory, payload)
        reference = row["streams"]["stdout"]
        path = Path(reference["path"])
        encoded = path.read_bytes()

        path.write_bytes(encoded[:-4])
        with self.assertRaisesRegex(ValueError, "truncated"):
            qualification.read_transcript(reference)

        changed = bytearray(encoded)
        changed[4] ^= 1  # Gzip timestamp metadata changes without altering decoded bytes.
        path.write_bytes(changed)
        with self.assertRaisesRegex(ValueError, "encoded digest mismatch"):
            qualification.read_transcript(reference)

if __name__ == "__main__":
    unittest.main()
