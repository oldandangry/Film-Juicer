#!/usr/bin/env python3
"""Explicit OFX performance campaign; never an ordinary performance test."""

from __future__ import annotations

import argparse
import importlib
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import statistics
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
execution = importlib.import_module("execution")
verify = importlib.import_module("verify")

ROWS = ("negative-direct", "negative-print", "negative-direct-grain", "negative-print-grain")
HALATION_EXTENTS = {
    f"halation-{size}-{route}{spatial}": extent
    for size, extent in (("1080p", (1920, 1080)), ("uhd", (3840, 2160)))
    for spatial in ("", "-spatial", "-boost", "-composed-boost")
    for route in ("direct", "print")
}
HALATION_EXTENTS.update(
    {f"halation-{size}-direct-boost-zero": extent for size, extent in (("1080p", (1920, 1080)), ("uhd", (3840, 2160)))}
)
MATCHED_HIGHLIGHT_WORKLOADS = (
    "halation-1080p-direct-boost",
    "halation-1080p-print-boost",
    "halation-1080p-direct-composed-boost",
    "halation-1080p-print-composed-boost",
    "halation-uhd-direct-boost",
    "halation-uhd-print-boost",
    "halation-uhd-direct-composed-boost",
    "halation-uhd-print-composed-boost",
    "halation-1080p-direct-boost-zero",
    "halation-uhd-direct-boost-zero",
)
COMPARISONS = ("standard", "matched-highlight")
HALATION_REFERENCE = "31a4d5d94b1576a6b2dadc7f6eb8e918c62224e5"
REFERENCE_PROOF = (
    ("reference-production-result.json", "result.json"),
    ("reference-production-contract.json", "production-contract.json"),
    ("reference-production-source.json", "candidate.json"),
    ("reference-production-source.patch", "candidate.patch"),
)
LANES = ("shipping", "observed")
SIDES = ("reference", "candidate")
REVISION = "19ede96179d5b27b3f65c2deaf9be8822d407d02"
SHIPPING = {
    "linux-release": {
        "reference": "3630b7dffb6e2d5e2e9969a67be8331d86b07b90d65c29ff088fdd689e7de61f",
        "candidate": "40a11a625bd1ef3c8adea03b42ed7334eccb72a4fcff2b0a6612fd08adbcca56",
    },
    "windows-clang-release": {
        "reference": "fe8e01b2178311ea2bc4793f94570d519f7d04741f8810e83e302a60e7b0e181",
        "candidate": "2d1841c0d6a9f29ae79588304484907bbcebae445b308bd305fc6851d36b37d5",
    },
}


def inventory(path):
    path = Path(path).resolve(strict=True)
    if path.is_file():
        return execution.digest(path)
    if not path.is_dir():
        raise ValueError(f"not a regular file or directory: {path}")
    files = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError(f"linked resource: {item}")
        if item.is_file():
            files[item.relative_to(path).as_posix()] = execution.digest(item)
    if not files:
        raise ValueError(f"empty resource directory: {path}")
    return files


def identities(inputs):
    result = {}
    for key in ("driver", "overlay"):
        entry = inputs[key]
        entry["path"] = str(Path(entry["path"]).resolve(strict=True))
        actual = inventory(entry["path"])
        if actual != entry["sha256"]:
            raise ValueError(f"{key} identity mismatch")
        result[key] = actual
    if "matched_highlight_baseline" in inputs:
        entry = inputs["matched_highlight_baseline"]
        entry["path"] = str(Path(entry["path"]).resolve(strict=True))
        actual = inventory(entry["path"])
        if actual != entry.get("sha256"):
            raise ValueError("matched highlight baseline manifest identity mismatch")
        result["matched_highlight_baseline"] = actual
    for side in SIDES:
        result[side] = {}
        for key in ("shipping", "observed", "resources", "observed_resources"):
            entry = inputs[side][key]
            entry["path"] = str(Path(entry["path"]).resolve(strict=True))
            actual = inventory(entry["path"])
            expected = entry["files"] if key in ("resources", "observed_resources") else entry["sha256"]
            if actual != expected:
                raise ValueError(f"{side} {key} identity mismatch")
            result[side][key] = actual
        for lane, key in (("shipping", "resources"), ("observed", "observed_resources")):
            expected = Path(inputs[side][lane]["path"]).resolve(strict=True).parent.parent / "Resources"
            if Path(inputs[side][key]["path"]).resolve(strict=True) != expected.resolve(strict=True):
                raise ValueError(f"{side} {lane} module-relative Resources mismatch")
        if result[side]["resources"] != result[side]["observed_resources"]:
            raise ValueError(f"{side} shipping and observed Resources differ")
    return result


def candidate_qualification(
    result_bytes, contract_bytes, snapshot_bytes, patch_bytes, preset, module_hash, *, check_inputs
):
    result = json.loads(result_bytes)
    contract = json.loads(contract_bytes)
    commands = result.get("commands")
    if (
        result.get("status") != "passed"
        or result.get("preset") != preset
        or result.get("production_contract") is not True
        or result.get("candidate_unchanged") is not True
        or not commands
        or any(
            c.get("drained") is not True or c.get("returncode") != 0 or c.get("process_status") != "completed"
            for c in commands
        )
    ):
        raise ValueError("candidate production closure incomplete")
    member = result.get("members", {}).get("production-contract.json", {})
    contract_hash = hashlib.sha256(contract_bytes).hexdigest()
    if member.get("sha256") != contract_hash or member.get("bytes") != len(contract_bytes):
        raise ValueError("candidate production contract identity mismatch")
    for name, data in (("candidate.json", snapshot_bytes), ("candidate.patch", patch_bytes)):
        member = result.get("members", {}).get(name, {})
        if member.get("sha256") != hashlib.sha256(data).hexdigest() or member.get("bytes") != len(data):
            raise ValueError("candidate source snapshot identity mismatch")
    if (
        contract.get("preset") != preset
        or contract.get("build_testing") != "OFF"
        or contract.get("test_support") != "OFF"
        or sorted(contract.get("exports", [])) != ["OfxGetNumberOfPlugins", "OfxGetPlugin"]
        or contract.get("loaded_module", {}).get("sha256") != module_hash
    ):
        raise ValueError("candidate module or production isolation mismatch")
    consumed = contract.get("consumed_inputs")
    if not isinstance(consumed, dict) or not consumed:
        raise ValueError("candidate consumed inputs missing")
    for path, hashes in consumed.items():
        before, after = hashes.get("sha256_before"), hashes.get("sha256_after")
        if not before or before != after:
            raise ValueError("candidate consumed input changed during closure")
        if check_inputs and execution.digest(Path(path).resolve(strict=True)) != after:
            raise ValueError(f"candidate consumed input changed: {path}")
    return {
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "contract_sha256": contract_hash,
        "module_sha256": module_hash,
        "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        "patch_sha256": hashlib.sha256(patch_bytes).hexdigest(),
    }


def reference_qualification(proof, preset, module_hash, *, check_inputs, comparison="standard"):
    snapshot = json.loads(proof[2])
    if comparison == "standard" and (
        snapshot.get("head") != HALATION_REFERENCE or snapshot.get("new_files_base64") or proof[3]
    ):
        raise ValueError("halation reference must be the clean pinned revision")
    if comparison not in COMPARISONS:
        raise ValueError("unknown OFX comparison mode")
    # A matched comparison intentionally binds the retained prior candidate.
    # Its consumed source paths may now refer to candidate files changed by the
    # implementation, so validate the copied closure and source proof instead.
    return candidate_qualification(
        *proof, preset, module_hash, check_inputs=check_inputs if comparison == "standard" else False
    )


def validate_matched_highlight_baseline(
    inputs,
    preset,
    identities_result,
    *,
    manifest_bytes=None,
    build_result_bytes=None,
    build_snapshot_bytes=None,
    build_patch_bytes=None,
):
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    baseline_input = inputs.get("matched_highlight_baseline")
    if not isinstance(baseline_input, dict) or not isinstance(baseline_input.get("path"), str):
        raise ValueError("matched-highlight comparison requires the retained baseline artifacts manifest")
    manifest_path = Path(baseline_input["path"]).resolve(strict=manifest_bytes is None)
    data = manifest_path.read_bytes() if manifest_bytes is None else manifest_bytes
    if hashlib.sha256(data).hexdigest() != baseline_input.get("sha256"):
        raise ValueError("matched-highlight baseline manifest hash mismatch")
    manifest_hash = hashlib.sha256(data).hexdigest()
    manifest = json.loads(data)
    baseline = manifest.get(preset)
    if not isinstance(baseline, dict):
        raise ValueError("matched-highlight baseline manifest lacks this preset")
    module_dir = "Win64" if preset.startswith("windows") else "Linux-x86-64"
    shipping_relative = f"Contents/{module_dir}/juicer.ofx"
    observed_relative = f"Contents/{module_dir}/JuicerOfxMeasurementModule.ofx"
    resource_prefix = "Contents/Resources/"
    members = baseline.get("members")
    if not isinstance(members, dict):
        raise ValueError("matched-highlight baseline member inventory missing")
    resource_members = {
        name[len(resource_prefix) :]: checksum for name, checksum in members.items() if name.startswith(resource_prefix)
    }
    require(resource_members, "matched-highlight baseline resources missing")
    reference = inputs["reference"]
    shipping_bundle = Path(baseline["shipping"]).resolve(strict=manifest_bytes is None)
    observed_bundle = Path(baseline["observed"]).resolve(strict=manifest_bytes is None)
    require(
        Path(reference["shipping"]["path"]).resolve(strict=manifest_bytes is None)
        == shipping_bundle / shipping_relative,
        "matched-highlight reference shipping module path mismatch",
    )
    require(
        Path(reference["observed"]["path"]).resolve(strict=manifest_bytes is None)
        == observed_bundle / observed_relative,
        "matched-highlight reference observed module path mismatch",
    )
    require(
        identities_result["reference"]["shipping"] == members.get(shipping_relative),
        "matched-highlight reference shipping module hash mismatch",
    )
    require(
        identities_result["reference"]["observed"]
        == baseline.get("observed_module_sha256")
        == members.get(observed_relative),
        "matched-highlight reference observed module hash mismatch",
    )
    require(
        identities_result["reference"]["shipping"] != identities_result["candidate"]["shipping"]
        and identities_result["reference"]["observed"] != identities_result["candidate"]["observed"],
        "matched-highlight baseline and candidate modules must be distinct",
    )
    require(
        identities_result["reference"]["resources"]
        == identities_result["reference"]["observed_resources"]
        == resource_members,
        "matched-highlight reference resource identity mismatch",
    )
    require(
        identities_result["candidate"]["resources"]
        == identities_result["candidate"]["observed_resources"]
        == resource_members,
        "matched-highlight candidate resource identity mismatch",
    )
    for side in SIDES:
        for module_key, resource_key in (("shipping", "resources"), ("observed", "observed_resources")):
            module_path = Path(inputs[side][module_key]["path"]).resolve(strict=manifest_bytes is None)
            resource_path = Path(inputs[side][resource_key]["path"]).resolve(strict=manifest_bytes is None)
            require(
                resource_path == module_path.parent.parent / "Resources",
                "matched-highlight module-relative Resources path mismatch",
            )
    if manifest_bytes is None:
        require(inventory(observed_bundle) == members, "matched-highlight observed bundle inventory mismatch")
    build_result_path = Path(baseline.get("build_result", "")).resolve(strict=build_result_bytes is None)
    build_directory = build_result_path.parent
    build_data = build_result_path.read_bytes() if build_result_bytes is None else build_result_bytes
    build_result = json.loads(build_data)
    require(
        build_result.get("schema") == "native-build-v1"
        and build_result.get("status") == "passed"
        and build_result.get("preset") == preset
        and "JuicerOfxMeasurementModule" in build_result.get("targets", [])
        and bool(build_result.get("commands"))
        and all(
            command.get("returncode") == 0
            and command.get("drained") is True
            and command.get("process_status") == "completed"
            for command in build_result["commands"]
        ),
        "matched-highlight observed module build receipt is incomplete",
    )
    if build_result_bytes is None:
        qualification = importlib.import_module("qualification")
        require(
            qualification.read_native_result(build_directory).get("status") == "passed",
            "matched-highlight observed module build result is invalid",
        )
    snapshot_path = build_directory / "candidate.json"
    patch_path = build_directory / "candidate.patch"
    if build_snapshot_bytes is None:
        require(snapshot_path.is_file(), "matched-highlight observed source snapshot missing")
        snapshot_bytes = snapshot_path.read_bytes()
    else:
        snapshot_bytes = build_snapshot_bytes
    if build_patch_bytes is None:
        require(patch_path.is_file(), "matched-highlight observed source patch missing")
        patch_bytes = patch_path.read_bytes()
    else:
        patch_bytes = build_patch_bytes
    for name, content in (("candidate.json", snapshot_bytes), ("candidate.patch", patch_bytes)):
        member = build_result.get("members", {}).get(name, {})
        require(
            member.get("sha256") == hashlib.sha256(content).hexdigest() and member.get("bytes") == len(content),
            "matched-highlight observed source proof identity mismatch",
        )
    binding = {
        "manifest_sha256": manifest_hash,
        "shipping_module_sha256": identities_result["reference"]["shipping"],
        "observed_module_sha256": identities_result["reference"]["observed"],
        "observed_build_result_sha256": hashlib.sha256(build_data).hexdigest(),
        "observed_build_source_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        "observed_build_patch_sha256": hashlib.sha256(patch_bytes).hexdigest(),
    }
    return {
        "binding": binding,
        "manifest_bytes": data,
        "build_result_bytes": build_data,
        "build_snapshot_bytes": snapshot_bytes,
        "build_patch_bytes": patch_bytes,
        "build_result_sha256": hashlib.sha256(build_data).hexdigest(),
        "build_snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        "build_patch_sha256": hashlib.sha256(patch_bytes).hexdigest(),
    }


def nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_sample(sample, mode, lane, row):
    if (
        sample.get("schema") != 1
        or sample.get("status") != "passed"
        or (sample.get("mode"), sample.get("lane"), sample.get("workload")) != (mode, lane, row)
    ):
        raise ValueError("failed or mismatched process sample")
    extent = (7, 5) if mode == "check" else HALATION_EXTENTS.get(row, (6048, 4032))
    if (sample.get("width"), sample.get("height")) != extent:
        raise ValueError("incorrect geometry")
    if not sample.get("controls") or not sample.get("control_types") or not sample.get("device"):
        raise ValueError("missing controls or device")
    renders = sample.get("renders", [])
    if len(renders) != (15 if mode == "check" else 71):
        raise ValueError("incomplete render membership")
    warmups = 10 if mode == "measure" else 1
    samples = 30 if mode == "measure" else 1
    expected = [("cold", 0, 0)]
    expected += [("warmup", 0, index) for index in range(warmups)]
    expected += [("warm", 0, index) for index in range(samples)]
    expected += [(phase, 0, index) for index in range(3) for phase in ("changed", "changed-reuse")]
    for cycle in (1, 2):
        expected.append(("cycle-cold", cycle, 0))
        expected += [("cycle-warmup", cycle, index) for index in range(warmups)]
        expected.append(("cycle-end", cycle, 0))
    if [(r.get("phase"), r.get("cycle"), r.get("index")) for r in renders] != expected:
        raise ValueError("incorrect render order or membership")
    for render in renders:
        if (
            render.get("callback_status") != 0
            or not render.get("releases_valid")
            or not render.get("context_preserved")
            or not nonnegative(render.get("callback_ns"))
            or not nonnegative(render.get("stream_interval_ms"))
        ):
            raise ValueError("failed render or missing timing")
        if render.get("output_checked") and not render.get("output_valid"):
            raise ValueError("failed output validation")
        if lane == "observed":
            construction = render.get("construction") or {}
            cold = render.get("phase") in ("cold", "changed", "cycle-cold")
            if (
                construction.get("starts") != int(cold)
                or construction.get("pairs") != int(cold)
                or construction.get("incomplete") != 0
                or construction.get("outcome") != ("success" if cold else "reused")
            ):
                raise ValueError("invalid construction span")
            if cold and not nonnegative(construction.get("duration_ns")):
                raise ValueError("missing construction duration")
    required_checked = [
        r for r in renders if r["phase"] in ("cold", "changed", "changed-reuse", "cycle-cold", "cycle-end")
    ]
    required_checked += [
        r for r in renders if r["phase"] == "warm" and r.get("index") == (29 if mode == "measure" else 0)
    ]
    if any(not r.get("output_checked") or not r.get("output_valid") for r in required_checked):
        raise ValueError("missing required output validation")
    edits = sample.get("edits") or []
    if (
        len(edits) != 3
        or [e.get("gamma") for e in edits] != [1.1, 1.0, 1.1]
        or any(e.get("status") not in (0, 14) or not nonnegative(e.get("edit_ns")) for e in edits)
    ):
        raise ValueError("invalid edit membership")
    timings = sample.get("timings") or {}
    if any(not nonnegative(timings.get(k)) for k in ("module_open_ns", "load_ns", "describe_ns", "unload_ns")):
        raise ValueError("incomplete lifecycle timings")
    if any(
        not isinstance(timings.get(k), list) or len(timings[k]) != 3 or not all(nonnegative(v) for v in timings[k])
        for k in ("create_ns", "author_ns", "destroy_ns")
    ):
        raise ValueError("incomplete per-instance timings")
    owned = sample.get("driver_owned") or {}
    if any(not nonnegative(owned.get(k)) for k in ("device_buffer_bytes", "host_buffer_bytes")):
        raise ValueError("incomplete driver buffer measurements")
    life = sample.get("lifecycle") or {}
    if (
        life.get("module_load_status") != 0
        or life.get("module_unload_status") != 0
        or not life.get("context_restored")
        or not life.get("module_closed")
        or life.get("callback_error")
        or life.get("effects_created") != 3
        or life.get("effects_destroyed") != 3
    ):
        raise ValueError("invalid lifecycle")
    memory = sample.get("memory") or []
    endpoints = {("before-load", -1, 0), ("after-sample", 0, 30 if mode == "measure" else 1), ("after-unload", -1, 0)}
    endpoints.update(
        (name, cycle, 0)
        for name in ("after-create", "after-cold", "after-warmups", "after-destroy")
        for cycle in range(3)
    )
    endpoints.update(("after-edit", 0, index) for index in range(3))
    if len(memory) != len(endpoints) or {(m.get("name"), m.get("cycle"), m.get("index")) for m in memory} != endpoints:
        raise ValueError("incomplete memory endpoint membership")
    if any(
        not nonnegative(m.get("rss_bytes"))
        or not nonnegative(m.get("peak_rss_bytes"))
        or m["peak_rss_bytes"] < m["rss_bytes"]
        or (m.get("private_bytes") is not None and not nonnegative(m["private_bytes"]))
        for m in memory
    ):
        raise ValueError("incomplete process memory")
    if lane == "observed":
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
            if endpoint["name"] == "before-load":
                continue
            observation = endpoint.get("observation") or {}
            if any(not nonnegative(observation.get(key)) for key in keys[:-2]):
                raise ValueError("incomplete owned-memory observation")
            if any(
                not isinstance(observation.get(key), list)
                or len(observation[key]) != 4
                or not all(nonnegative(v) for v in observation[key])
                for key in keys[-2:]
            ):
                raise ValueError("incomplete upload-pool state")
            for current, peak in (
                ("device_reserved_bytes", "device_peak_reserved_bytes"),
                ("device_committed_bytes", "device_peak_committed_bytes"),
                ("device_retiring_bytes", "device_peak_retiring_bytes"),
                ("device_charged_bytes", "device_peak_charged_bytes"),
                ("device_records", "device_peak_records"),
                ("pinned_upload_pool_physical_bytes", "pinned_upload_pool_peak_bytes"),
            ):
                if observation[peak] < observation[current]:
                    raise ValueError("high-water below current ownership")
        for endpoint in memory:
            if endpoint["name"] == "after-cold":
                observation = endpoint["observation"]
                if (
                    observation["device_peak_charged_bytes"] <= 0
                    or observation["device_peak_records"] <= 0
                    or observation["pinned_upload_pool_peak_bytes"] <= 0
                    or observation.get("device_id") != sample["device"].get("ordinal")
                    or observation.get("context_opaque") != sample["device"].get("context_opaque")
                ):
                    raise ValueError("missing accounting producer or selected context evidence")
        drained = sample.get("drained") or {}
        if any(
            drained.get(key) != 0
            for key in ("device_charged_bytes", "device_records", "pinned_upload_pool_physical_bytes")
        ):
            raise ValueError("undrained or unknown owned memory")
        final = next(m["observation"] for m in memory if m["name"] == "after-unload")
        if any(
            final[k] != 0
            for k in (
                "device_reserved_bytes",
                "device_committed_bytes",
                "device_retiring_bytes",
                "device_charged_bytes",
                "device_records",
                "pinned_upload_pool_physical_bytes",
            )
        ) or any(
            final[k] != [0, 0, 0, 0] for k in ("pinned_upload_pool_state_bytes", "pinned_upload_pool_state_counts")
        ):
            raise ValueError("undrained final owned-memory families")


def workloads(value):
    available = (*ROWS, *HALATION_EXTENTS)
    if not isinstance(value, list) or not value or value != [row for row in available if row in value]:
        raise ValueError("workloads must be a nonempty ordered subset without duplicates")
    return value


def check_reference_workloads(selection, has_reference_production_result):
    if any((row in HALATION_EXTENTS) != has_reference_production_result for row in selection):
        raise ValueError("halation workloads require a qualified reference production closure")


def validate_comparison(comparison, selected, has_reference_production_result):
    if comparison not in COMPARISONS:
        raise ValueError("unknown OFX comparison mode")
    if comparison == "matched-highlight":
        if not has_reference_production_result:
            raise ValueError("matched-highlight comparison requires a reference production closure")
        if tuple(selected) != MATCHED_HIGHLIGHT_WORKLOADS:
            raise ValueError("matched-highlight comparison requires all eight boosted and two zero-boost workloads")


def check_halation_side(sample, row, side, comparison="standard"):
    if row not in HALATION_EXTENTS:
        return
    if comparison == "matched-highlight":
        if sample.get("advanced_halation_available") is not True:
            raise ValueError("matched highlight comparison requires advanced controls on both modules")
        expected_boost = 4.0 if row.endswith("-boost") else 0.0
        if (
            sample.get("controls", {}).get("Exposure") != [6.0]
            or sample.get("halation_boost_ev") != expected_boost
            or sample.get("halation_protection_ev") != 0.0
        ):
            raise ValueError("matched highlight comparison requires exposure 6, matching boost, and zero protection")
        return
    candidate = side == "candidate"
    if sample.get("advanced_halation_available") is not candidate:
        raise ValueError("unexpected advanced halation control surface")
    expected_boost = 4.0 if candidate and row.endswith("-boost") else 0.0
    if sample.get("halation_boost_ev") != expected_boost:
        raise ValueError("unexpected highlight boost for the paired workload")


def evidence_file(directory, name, expected_hash):
    if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
        raise ValueError("unsafe evidence path")
    path = directory / name
    if path.is_symlink() or not path.is_file() or execution.digest(path) != expected_hash:
        raise ValueError("missing or changed evidence file: " + name)
    return path


def read_run(directory, mode):
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    if (
        result.get("status") != "passed"
        or result.get("operation") != mode
        or result.get("source_unchanged") is not True
        or result.get("identities_unchanged") is not True
    ):
        raise ValueError("run incomplete")
    selected = workloads(result.get("workloads", list(ROWS)))
    evidence = result["evidence"]
    for name in ("candidate.patch", "candidate.json", "inputs.json"):
        evidence_file(directory, name, evidence[name])
    if json.loads((directory / "candidate.json").read_text()) != result["source"]:
        raise ValueError("source evidence mismatch")
    inputs = json.loads((directory / "inputs.json").read_text())
    declared = {key: inputs[key]["sha256"] for key in ("driver", "overlay")}
    for side in SIDES:
        declared[side] = {
            key: inputs[side][key]["files" if key in ("resources", "observed_resources") else "sha256"]
            for key in ("shipping", "observed", "resources", "observed_resources")
        }
    comparison = result.get("comparison", "standard")
    reference_binding = result.get("reference_qualification")
    validate_comparison(comparison, selected, reference_binding is not None)
    if comparison == "matched-highlight":
        declared["matched_highlight_baseline"] = inputs.get("matched_highlight_baseline", {}).get("sha256")
    if declared != result["identities"]:
        raise ValueError("input identity evidence mismatch")
    preset = result.get("preset")
    check_reference_workloads(selected, reference_binding is not None)
    if preset not in SHIPPING:
        raise ValueError("unknown native Release preset")
    if reference_binding is None:
        if declared["reference"]["shipping"] != SHIPPING[preset]["reference"]:
            raise ValueError("offline reference differs from the approved fixed bundle")
    else:
        proof = tuple(evidence_file(directory, name, evidence[name]).read_bytes() for name, _ in REFERENCE_PROOF)
        if (
            reference_qualification(
                proof,
                preset,
                declared["reference"]["shipping"],
                check_inputs=False,
                comparison=comparison,
            )
            != reference_binding
        ):
            raise ValueError("offline halation reference differs from its production closure")
    if comparison == "matched-highlight":
        baseline_manifest = evidence_file(
            directory,
            "matched-highlight-baseline-artifacts.json",
            result["matched_highlight_baseline_binding"]["manifest_sha256"],
        ).read_bytes()
        baseline_build_result = evidence_file(
            directory,
            "matched-highlight-observed-build-result.json",
            result["matched_highlight_baseline_binding"]["observed_build_result_sha256"],
        ).read_bytes()
        baseline_build_snapshot = evidence_file(
            directory,
            "matched-highlight-observed-candidate.json",
            result["matched_highlight_baseline_binding"]["observed_build_source_sha256"],
        ).read_bytes()
        baseline_build_patch = evidence_file(
            directory,
            "matched-highlight-observed-candidate.patch",
            result["matched_highlight_baseline_binding"]["observed_build_patch_sha256"],
        ).read_bytes()
        actual_baseline_binding = validate_matched_highlight_baseline(
            inputs,
            preset,
            declared,
            manifest_bytes=baseline_manifest,
            build_result_bytes=baseline_build_result,
            build_snapshot_bytes=baseline_build_snapshot,
            build_patch_bytes=baseline_build_patch,
        )["binding"]
        if actual_baseline_binding != result.get("matched_highlight_baseline_binding"):
            raise ValueError("matched-highlight baseline artifact binding mismatch")
    binding = result.get("candidate_qualification")
    if binding is None and (
        comparison == "matched-highlight" or declared["candidate"]["shipping"] != SHIPPING[preset]["candidate"]
    ):
        raise ValueError("offline candidate production qualification missing")
    if binding is not None:
        production_result = evidence_file(
            directory, "candidate-production-result.json", evidence["candidate-production-result.json"]
        ).read_bytes()
        production_contract = evidence_file(
            directory, "candidate-production-contract.json", evidence["candidate-production-contract.json"]
        ).read_bytes()
        production_snapshot = evidence_file(
            directory, "candidate-production-source.json", evidence["candidate-production-source.json"]
        ).read_bytes()
        production_patch = evidence_file(
            directory, "candidate-production-source.patch", evidence["candidate-production-source.patch"]
        ).read_bytes()
        actual_binding = candidate_qualification(
            production_result,
            production_contract,
            production_snapshot,
            production_patch,
            result["preset"],
            declared["candidate"]["shipping"],
            check_inputs=False,
        )
        if binding != actual_binding:
            raise ValueError("candidate qualification binding mismatch")
        source_snapshot = json.loads(production_snapshot)
        if (
            any(
                source_snapshot.get(key) != result["source"].get(key)
                for key in ("head", "inputs", "new_files_base64", "git_filemode", "git_version", "source_identity")
            )
            or production_patch != (directory / "candidate.patch").read_bytes()
        ):
            raise ValueError("candidate production source mismatch")
    platform_path = PureWindowsPath if result["preset"].startswith("windows") else PurePosixPath
    original_directory = platform_path(result["run_directory"])
    if not original_directory.is_absolute():
        raise ValueError("missing original run directory")
    members = {}
    controls = {}
    command_records = result.get("commands") or []
    if len(command_records) != len(result["samples"]):
        raise ValueError("command membership incomplete")
    for member, command in zip(result["samples"], command_records):
        key = (member["lane"], member["row"], member["pair"], member["side"])
        name = f"{key[0]}-{key[1]}-{key[2]}-{key[3]}"
        if (
            key in members
            or member.get("drained") is not True
            or member.get("returncode") != 0
            or command.get("name") != name
            or command.get("drained") is not True
            or command.get("returncode") != 0
        ):
            raise ValueError("duplicate, failed or undrained child")
        receipt_name = name + ".process.json"
        receipt = json.loads(evidence_file(directory, receipt_name, evidence[receipt_name]).read_text())
        if receipt.get("drained") is not True or receipt.get("returncode") != 0 or receipt.get("status") != "completed":
            raise ValueError("child process completion unproved")
        expected_command = [
            inputs["driver"]["path"],
            "--mode",
            mode,
            "--module",
            inputs[key[3]][key[0]]["path"],
            "--resources",
            inputs[key[3]]["resources" if key[0] == "shipping" else "observed_resources"]["path"],
            "--lane",
            key[0],
            "--workload",
            key[1],
            "--output",
            str(original_directory / (name + ".json")),
        ]
        if command.get("command") != expected_command:
            raise ValueError("child command identity mismatch")
        if member["file"] != name + ".json":
            raise ValueError("incorrect sample file")
        sample = json.loads(evidence_file(directory, member["file"], evidence[member["file"]]).read_text())
        validate_sample(sample, mode, key[0], key[1])
        check_halation_side(sample, key[1], key[3], comparison)
        if platform_path(sample.get("resources", "")) != platform_path(expected_command[6]):
            raise ValueError("driver resource path mismatch")
        if key[1] in controls and controls[key[1]] != (sample["controls"], sample["control_types"]):
            raise ValueError("control mismatch")
        controls[key[1]] = (sample["controls"], sample["control_types"])
        members[key] = sample
    order = [
        (lane, row, pair, side)
        for lane in LANES
        for row in selected
        for pair in range(3 if mode == "measure" else 1)
        for side in (SIDES if pair != 1 else SIDES[::-1])
    ]
    if list(members) != order:
        raise ValueError("incorrect matched-pair membership or order")
    return result, members


def report_run(directory):
    return read_run(directory, "measure")


def distribution(values):
    values = sorted(values)
    if not values:
        raise ValueError("empty timing distribution")
    return {
        "samples": values,
        "median": statistics.median(values),
        "min": values[0],
        "max": values[-1],
        "p95": values[min(len(values) - 1, int((len(values) - 1) * 0.95))],
    }


def offline_report(directories):
    runs = [report_run(directory) for directory in directories]
    presets = {result["preset"] for result, _ in runs}
    if presets != {"linux-release", "windows-clang-release"}:
        raise ValueError("one native Release run per OS required")
    if len({tuple(result.get("workloads", ROWS)) for result, _ in runs}) != 1:
        raise ValueError("native reports require identical workloads")
    comparisons = {result.get("comparison", "standard") for result, _ in runs}
    if len(comparisons) != 1:
        raise ValueError("native reports require identical comparison modes")
    summary = {}
    for directory, (result, samples) in zip(directories, runs):
        os_summary = {}
        for lane in LANES:
            for row in workloads(result.get("workloads", list(ROWS))):
                pairs = []
                processes = []
                for pair in range(3):
                    pair_values = {}
                    for side in SIDES:
                        sample = samples[lane, row, pair, side]
                        warm = [r for r in sample["renders"] if r["phase"] == "warm"]
                        callback = distribution([r["callback_ns"] for r in warm])
                        stream = distribution([r["stream_interval_ms"] for r in warm])
                        constructions = (
                            [
                                r["construction"]["duration_ns"]
                                for r in sample["renders"]
                                if r["phase"] in ("cold", "changed", "cycle-cold")
                            ]
                            if lane == "observed"
                            else []
                        )
                        measured = {
                            "warm_callback_ns": callback["median"],
                            "warm_callback_p95_ns": callback["p95"],
                            "warm_stream_interval_ms": stream["median"],
                            "module_open_ns": sample["timings"]["module_open_ns"],
                            "load_ns": sample["timings"]["load_ns"],
                            "create_ns": statistics.median(sample["timings"]["create_ns"]),
                            "edit_ns": statistics.median(e["edit_ns"] for e in sample["edits"]),
                        }
                        measured["cold_callback_ns"] = next(
                            r["callback_ns"] for r in sample["renders"] if r["phase"] == "cold"
                        )
                        for index in range(3):
                            measured[f"edit_{index}_ns"] = sample["edits"][index]["edit_ns"]
                            measured[f"changed_{index}_callback_ns"] = next(
                                r["callback_ns"]
                                for r in sample["renders"]
                                if r["phase"] == "changed" and r["index"] == index
                            )
                        if lane == "observed":
                            measured["construction_ns"] = statistics.median(constructions)
                            measured["cold_construction_ns"] = constructions[0]
                            for index in range(3):
                                measured[f"changed_{index}_construction_ns"] = constructions[index + 1]
                            steady = next(m["observation"] for m in sample["memory"] if m["name"] == "after-sample")
                            measured["steady_device_charged_bytes"] = steady["device_charged_bytes"]
                            measured["steady_pinned_upload_pool_bytes"] = steady["pinned_upload_pool_physical_bytes"]
                        pair_values[side] = measured
                        processes.append(
                            {
                                "pair": pair,
                                "side": side,
                                "warm_callback_ns": callback,
                                "warm_stream_interval_ms": stream,
                                "construction_ns": constructions,
                                "renders": sample["renders"],
                                "edits": sample["edits"],
                                "timings": sample["timings"],
                                "memory": sample["memory"],
                                "drained": sample["drained"],
                                "driver_owned": sample["driver_owned"],
                                "controls": sample["controls"],
                                "control_types": sample["control_types"],
                                "device": sample["device"],
                            }
                        )
                    reference, candidate = pair_values["reference"], pair_values["candidate"]
                    metrics = {
                        key: {
                            "reference": reference[key],
                            "candidate": candidate[key],
                            "delta": candidate[key] - reference[key],
                            "delta_percent": (candidate[key] / reference[key] - 1) * 100 if reference[key] else None,
                        }
                        for key in reference
                    }
                    pairs.append(metrics)
                reference = [p["warm_callback_ns"]["reference"] for p in pairs]
                spread = max(reference) - min(reference)
                os_summary[f"{lane}/{row}"] = {
                    "pairs": pairs,
                    "processes": processes,
                    "reference_spread_ns": spread,
                    "above_five_percent_outside_spread": sum(
                        p["warm_callback_ns"]["delta"] > spread
                        and p["warm_callback_ns"]["delta_percent"] is not None
                        and p["warm_callback_ns"]["delta_percent"] > 5
                        for p in pairs
                    )
                    >= 2,
                    "repeatable_warm_slowdown": sum(p["warm_callback_ns"]["delta"] > 0 for p in pairs) >= 2,
                    "increased_steady_owned_memory": lane == "observed"
                    and any(
                        p[key]["delta"] > 0
                        for p in pairs
                        for key in ("steady_device_charged_bytes", "steady_pinned_upload_pool_bytes")
                    ),
                    "candidate_post_destroy_rss_bytes": [
                        [m["rss_bytes"] for m in process["memory"] if m["name"] == "after-destroy"]
                        for process in processes
                        if process["side"] == "candidate"
                    ],
                    "raw_samples": [
                        str(Path(directory) / m["file"])
                        for m in result["samples"]
                        if m["lane"] == lane and m["row"] == row
                    ],
                }
        summary[result["preset"]] = {"identities": result["identities"], "rows": os_summary}
    return {
        "schema": 1,
        "comparison": next(iter(comparisons)),
        "comparisons": summary,
        "decision": "requires investigation and owner review",
        "limitations": "Stream intervals include host gaps, not pure kernel time. Process peaks include fixture and driver. Pinned values cover upload-pool staging only; scan-error pinned bytes and complete pinned drainage are not quantified. Cold means process/owner cold, not cold storage. Grain randomness remains production-owned. Observed timings include instrumentation overhead; inspect smaller repeatable regressions and retention separately.",
    }


def campaign(args):
    if args.preset.startswith("windows") != (os.name == "nt"):
        raise ValueError("run on the native platform")
    if not args.preset.endswith("release"):
        raise ValueError("campaign requires Release")
    selected = workloads(args.workloads)
    comparison = args.comparison
    validate_comparison(comparison, selected, args.reference_production_result is not None)
    check_reference_workloads(selected, args.reference_production_result is not None)
    directory = (
        ROOT / "out/validation" / args.preset / "s7-b-performance-memory" / execution.name_component(args.run_id)
    )
    with execution.OutputLease(ROOT, ROOT / "out/build" / args.preset), execution.OutputLease(ROOT):
        inputs = json.loads(Path(args.inputs).read_text())
        entry = identities(inputs)
        matched_baseline = None
        if comparison == "matched-highlight":
            if not args.candidate_production_result:
                raise ValueError("matched-highlight comparison requires both production closures")
            matched_baseline = validate_matched_highlight_baseline(inputs, args.preset, entry)
        reference_binding = None
        reference_proof = ()
        if args.reference_production_result:
            result_path = Path(args.reference_production_result).resolve(strict=True)
            reference_proof = (result_path.read_bytes(),) + tuple(
                (result_path.parent / source).read_bytes() for _, source in REFERENCE_PROOF[1:]
            )
            reference_binding = reference_qualification(
                reference_proof,
                args.preset,
                entry["reference"]["shipping"],
                check_inputs=True,
                comparison=comparison,
            )
            if not args.candidate_production_result:
                raise ValueError("halation comparison requires the candidate production closure")
        elif entry["reference"]["shipping"] != SHIPPING[args.preset]["reference"]:
            raise ValueError("reference module differs from the approved fixed bundle")
        production_result = production_contract = production_snapshot = production_patch = None
        binding = None
        if args.candidate_production_result:
            result_path = Path(args.candidate_production_result).resolve(strict=True)
            production_result = result_path.read_bytes()
            production_contract = (result_path.parent / "production-contract.json").read_bytes()
            production_snapshot = (result_path.parent / "candidate.json").read_bytes()
            production_patch = (result_path.parent / "candidate.patch").read_bytes()
            binding = candidate_qualification(
                production_result,
                production_contract,
                production_snapshot,
                production_patch,
                args.preset,
                entry["candidate"]["shipping"],
                check_inputs=True,
            )
        elif entry["candidate"]["shipping"] != SHIPPING[args.preset]["candidate"]:
            raise ValueError("candidate differs from the approved fixed bundle")
        directory.mkdir(parents=True, exist_ok=False)
        run = execution.NativeRun(ROOT, directory)
        source, patch, _ = verify.capture_native_source(ROOT, "tests/benchmarks/ofx.py", domain_tests=True)
        (directory / "candidate.patch").write_bytes(patch)
        execution.write_json(directory / "candidate.json", source)
        execution.write_json(directory / "inputs.json", inputs)
        for (name, _), content in zip(REFERENCE_PROOF, reference_proof):
            (directory / name).write_bytes(content)
        if matched_baseline is not None:
            (directory / "matched-highlight-baseline-artifacts.json").write_bytes(matched_baseline["manifest_bytes"])
            (directory / "matched-highlight-observed-build-result.json").write_bytes(
                matched_baseline["build_result_bytes"]
            )
            (directory / "matched-highlight-observed-candidate.json").write_bytes(
                matched_baseline["build_snapshot_bytes"]
            )
            (directory / "matched-highlight-observed-candidate.patch").write_bytes(
                matched_baseline["build_patch_bytes"]
            )
        if binding is not None:
            (directory / "candidate-production-result.json").write_bytes(production_result)
            (directory / "candidate-production-contract.json").write_bytes(production_contract)
            (directory / "candidate-production-source.json").write_bytes(production_snapshot)
            (directory / "candidate-production-source.patch").write_bytes(production_patch)
        result = {
            "schema": 1,
            "status": "failed",
            "operation": args.operation,
            "preset": args.preset,
            "identities": entry,
            "source": source,
            "run_directory": str(directory),
            "samples": [],
            "commands": run.commands,
            "candidate_qualification": binding,
            "reference_qualification": reference_binding,
            "comparison": comparison,
            "matched_highlight_baseline_binding": (
                matched_baseline["binding"] if matched_baseline is not None else None
            ),
            "workloads": selected,
        }
        try:
            if binding is not None:
                source_snapshot = json.loads(production_snapshot)
                if (
                    any(
                        source_snapshot.get(key) != source.get(key)
                        for key in (
                            "head",
                            "inputs",
                            "new_files_base64",
                            "git_filemode",
                            "git_version",
                            "source_identity",
                        )
                    )
                    or production_patch != patch
                ):
                    raise ValueError("candidate production source mismatch")
            if args.operation == "measure":
                check_dir = Path(inputs["compatibility_check"])
                check, _ = read_run(check_dir, "check")
                if (
                    check.get("identities") != entry
                    or check.get("preset") != args.preset
                    or check.get("source") != source
                    or check.get("candidate_qualification") != binding
                    or check.get("reference_qualification") != reference_binding
                    or check.get("comparison", "standard") != comparison
                    or check.get("workloads", list(ROWS)) != selected
                ):
                    raise ValueError("matching completed compatibility check required")
            controls = {}
            for lane in LANES:
                for row in selected:
                    for pair in range(3 if args.operation == "measure" else 1):
                        for side in SIDES if pair != 1 else SIDES[::-1]:
                            name = f"{lane}-{row}-{pair}-{side}"
                            target = directory / (name + ".json")
                            command = [
                                inputs["driver"]["path"],
                                "--mode",
                                args.operation,
                                "--module",
                                inputs[side][lane]["path"],
                                "--resources",
                                inputs[side]["resources" if lane == "shipping" else "observed_resources"]["path"],
                                "--lane",
                                lane,
                                "--workload",
                                row,
                                "--output",
                                str(target),
                            ]
                            try:
                                run.command(name, command)
                            finally:
                                record = run.commands[-1]
                                result["samples"].append(
                                    {
                                        "lane": lane,
                                        "row": row,
                                        "pair": pair,
                                        "side": side,
                                        "file": target.name,
                                        "drained": record["drained"],
                                        "returncode": record["returncode"],
                                    }
                                )
                            sample = json.loads(target.read_text())
                            validate_sample(sample, args.operation, lane, row)
                            check_halation_side(sample, row, side, comparison)
                            if Path(sample.get("resources", "")).resolve() != Path(command[6]).resolve():
                                raise ValueError("driver resource path mismatch")
                            if row in controls and controls[row] != (sample["controls"], sample["control_types"]):
                                raise ValueError(f"control mismatch for {row}")
                            controls[row] = (sample["controls"], sample["control_types"])
            result["status"] = "passed"
        except Exception as error:
            result["error"] = str(error)
        try:
            result["identities_unchanged"] = identities(inputs) == entry
            if reference_binding is not None:
                result["identities_unchanged"] = (
                    result["identities_unchanged"]
                    and reference_qualification(
                        reference_proof,
                        args.preset,
                        entry["reference"]["shipping"],
                        check_inputs=True,
                        comparison=comparison,
                    )
                    == reference_binding
                )
            if matched_baseline is not None:
                final_baseline = validate_matched_highlight_baseline(inputs, args.preset, entry)
                result["identities_unchanged"] = (
                    result["identities_unchanged"] and final_baseline["binding"] == matched_baseline["binding"]
                )
            if binding is not None:
                result["identities_unchanged"] = (
                    result["identities_unchanged"]
                    and candidate_qualification(
                        production_result,
                        production_contract,
                        production_snapshot,
                        production_patch,
                        args.preset,
                        entry["candidate"]["shipping"],
                        check_inputs=True,
                    )
                    == binding
                )
            final, final_patch, _ = verify.capture_native_source(ROOT, "tests/benchmarks/ofx.py", domain_tests=True)
            result["source_unchanged"] = source == final and patch == final_patch
            if not result["identities_unchanged"] or not result["source_unchanged"]:
                raise ValueError("input identity or source changed during campaign")
        except Exception as error:
            result["status"] = "failed"
            result["completion_error"] = str(error)
        evidence_names = ["candidate.patch", "candidate.json", "inputs.json"]
        if reference_binding is not None:
            evidence_names += [name for name, _ in REFERENCE_PROOF]
        if matched_baseline is not None:
            evidence_names += [
                "matched-highlight-baseline-artifacts.json",
                "matched-highlight-observed-build-result.json",
                "matched-highlight-observed-candidate.json",
                "matched-highlight-observed-candidate.patch",
            ]
        if binding is not None:
            evidence_names += [
                "candidate-production-result.json",
                "candidate-production-contract.json",
                "candidate-production-source.json",
                "candidate-production-source.patch",
            ]
        evidence_names += [
            name
            for member in result["samples"]
            for name in (
                member["file"],
                f"{member['lane']}-{member['row']}-{member['pair']}-{member['side']}.process.json",
            )
        ]
        result["evidence"] = {
            name: execution.digest(directory / name) for name in evidence_names if (directory / name).is_file()
        }
        if all(command["drained"] for command in run.commands):
            run.publish(result)
        else:
            execution.write_json(directory / "incomplete.json", result)
        if result["status"] != "passed":
            raise ValueError(result.get("error", "campaign failed"))


def prepare_reference(args):
    if (
        args.revision != REVISION
        or args.preset not in ("linux-release", "windows-clang-release")
        or args.preset.startswith("windows") != (os.name == "nt")
    ):
        raise ValueError("reference requires the pinned revision and native Release preset")
    source = (ROOT / "out/validation" / args.preset / "s7-b-performance-memory/reference-source").resolve()
    build = (ROOT / "out/build" / (args.preset + "-s7-b-reference-observed")).resolve()
    if Path(args.source_dir).resolve() != source or Path(args.build_dir).resolve() != build:
        raise ValueError("reference paths must use the prescribed isolated locations")
    overlay = Path(args.overlay).resolve(strict=True)
    directory = (
        ROOT / "out/validation" / args.preset / "s7-b-performance-memory" / execution.name_component(args.run_id)
    )
    with execution.OutputLease(ROOT, build):
        directory.mkdir(parents=True, exist_ok=False)
        run = execution.NativeRun(ROOT, directory)
        initial, patch, _ = verify.capture_native_source(ROOT, "tests/benchmarks/ofx.py", domain_tests=True)
        (directory / "candidate.patch").write_bytes(patch)
        result = {
            "schema": 1,
            "status": "failed",
            "operation": "prepare-reference",
            "revision": REVISION,
            "overlay_sha256": execution.digest(overlay),
            "commands": run.commands,
        }
        manifest_path = source.parent / "reference-source-manifest.json"
        source_manifest = None
        try:
            tree = run.command("tree", ["git", "rev-parse", REVISION + "^{tree}"], stdout_only=True)
            result["tree"] = tree.read_text().strip()
            if args.resume_from:
                previous = json.loads((Path(args.resume_from) / "result.json").read_text())
                if (
                    previous.get("status") != "failed"
                    or previous.get("operation") != "prepare-reference"
                    or previous.get("revision") != REVISION
                    or previous.get("tree") != result["tree"]
                    or previous.get("overlay_sha256") != result["overlay_sha256"]
                    or not previous.get("source_unchanged")
                    or not previous.get("commands")
                    or any(not command.get("drained") for command in previous["commands"])
                    or not any(
                        command.get("name") in ("configure", "build") and command.get("returncode") != 0
                        for command in previous["commands"]
                    )
                ):
                    raise ValueError(
                        "resume requires a drained failed configure/build with matching source and overlay"
                    )
                source_manifest = json.loads(manifest_path.read_text())
                if previous.get("source_manifest") != source_manifest:
                    raise ValueError("reference source manifest differs from prior attempt")
                result["resume_from"] = str(Path(args.resume_from).resolve())
            else:
                if source.exists() or build.exists() or manifest_path.exists():
                    raise ValueError(
                        "reference output already exists; use explicit resume after a failed configure/build"
                    )
                archive = run.command("archive", ["git", "archive", "--format=tar", REVISION], stdout_only=True)
                source.mkdir(parents=True)
                with tarfile.open(archive) as tar:
                    tar.extractall(source, filter="data")
                environment = dict(os.environ, GIT_CEILING_DIRECTORIES=str(source.parent))
                run.command("apply-overlay", ["git", "apply", "--check", str(overlay)], cwd=source, env=environment)
                run.command("apply", ["git", "apply", str(overlay)], cwd=source, env=environment)
                source_manifest = {
                    "revision": REVISION,
                    "tree": result["tree"],
                    "overlay_sha256": result["overlay_sha256"],
                    "files": inventory(source),
                }
                execution.write_json(manifest_path, source_manifest)
            result["source_manifest"] = source_manifest
            if (
                source_manifest["revision"] != REVISION
                or source_manifest["tree"] != result["tree"]
                or source_manifest["overlay_sha256"] != result["overlay_sha256"]
                or inventory(source) != source_manifest["files"]
            ):
                raise ValueError("reference export identity mismatch")
            if (build / "CMakeCache.txt").exists():
                cache = verify.cache_values(build)
                if not Path(cache["CMAKE_HOME_DIRECTORY"]).samefile(source):
                    raise ValueError("reference cache belongs to another source")
            (directory / "reference-observation.patch").write_bytes(overlay.read_bytes())
            execution.write_json(directory / "reference-source-manifest.json", source_manifest)
            run.command(
                "configure",
                [
                    "cmake",
                    "--preset",
                    args.preset,
                    "-S",
                    str(source),
                    "-B",
                    str(build),
                    "-DBUILD_TESTING=OFF",
                    "-DPython3_EXECUTABLE=" + sys.executable,
                ],
                cwd=source,
                vctip_executable=execution.windows_vctip(os.environ),
            )
            run.command(
                "build",
                ["cmake", "--build", str(build), "--target", "JuicerOfxMeasurementModule", "--", "-k0"],
                cwd=source,
                vctip_executable=execution.windows_vctip(os.environ),
            )
            result["status"] = "passed"
        except Exception as error:
            result["error"] = str(error)
        try:
            final, final_patch, _ = verify.capture_native_source(ROOT, "tests/benchmarks/ofx.py", domain_tests=True)
            result["source_unchanged"] = (
                source_manifest is not None
                and inventory(source) == source_manifest["files"]
                and initial == final
                and patch == final_patch
                and execution.digest(overlay) == result["overlay_sha256"]
            )
            if not result["source_unchanged"]:
                raise ValueError("source or overlay changed during preparation")
        except Exception as error:
            result["status"] = "failed"
            result["completion_error"] = str(error)
        if all(command["drained"] for command in run.commands):
            run.publish(result)
        else:
            execution.write_json(directory / "incomplete.json", result)
        if result["status"] != "passed":
            raise ValueError(result.get("error", "reference preparation failed"))


def check_row(driver, shipping, observed, resources, directory, row):
    controls = None
    for lane, module in (("shipping", shipping), ("observed", observed)):
        output = directory / f"{row}-{lane}.json"
        if output.exists():
            raise ValueError(f"raw compatibility output already exists: {output}")
        module_resources = str(Path(module).resolve().parent.parent / "Resources")
        subprocess.run(
            [
                driver,
                "--mode",
                "check",
                "--module",
                module,
                "--resources",
                module_resources,
                "--lane",
                lane,
                "--workload",
                row,
                "--output",
                str(output),
            ],
            check=True,
        )
        sample = json.loads(output.read_text())
        validate_sample(sample, "check", lane, row)
        if Path(sample.get("resources", "")).resolve() != Path(module_resources).resolve():
            raise ValueError("driver resource path mismatch")
        if inventory(module_resources) != inventory(resources):
            raise ValueError("test module Resources mismatch")
        if controls is not None and controls != (sample["controls"], sample["control_types"]):
            raise ValueError("control mismatch")
        controls = (sample["controls"], sample["control_types"])


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="operation", required=True)
    for operation in ("check", "measure"):
        command = sub.add_parser(operation)
        command.add_argument("--preset", required=True)
        command.add_argument("--inputs", required=True)
        command.add_argument("--run-id", required=True)
        command.add_argument("--candidate-production-result")
        command.add_argument("--reference-production-result")
        command.add_argument("--comparison", choices=COMPARISONS, default="standard")
        command.add_argument("--workloads", nargs="+", choices=(*ROWS, *HALATION_EXTENTS), default=list(ROWS))
    command = sub.add_parser("prepare-reference")
    for option in ("preset", "revision", "source-dir", "build-dir", "overlay", "run-id"):
        command.add_argument("--" + option, required=True)
    sub.choices["prepare-reference"].add_argument("--resume-from")
    command = sub.add_parser("report")
    command.add_argument("--runs", nargs=2, required=True)
    command.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.operation == "report":
        execution.write_json(Path(args.output), offline_report(args.runs))
    elif args.operation == "prepare-reference":
        prepare_reference(args)
    else:
        campaign(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
