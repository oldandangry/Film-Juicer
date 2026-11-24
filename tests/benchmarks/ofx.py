#!/usr/bin/env python3
"""Explicit OFX performance campaign; never an ordinary performance test."""

from __future__ import annotations

import argparse
import base64
import importlib
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import statistics
import struct
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
execution = importlib.import_module("execution")
verify = importlib.import_module("verify")
qualification = importlib.import_module("qualification")

ROWS = ("negative-direct", "negative-print", "negative-direct-grain", "negative-print-grain")
SCANNER_PAIRED_ROWS = {
    **{
        f"ordinary-negative-{route}-current-{size}": ("ordinary", route, "materialized", geometry)
        for route in ("direct", "print")
        for size, geometry in (("1080p", (1920, 1080)), ("4k", (3840, 2160)))
    },
    **{
        f"ordinary-negative-{route}-{schedule}-4k": ("ordinary", route, schedule, (3840, 2160))
        for route in ("direct", "print")
        for schedule in ("monolithic", "fused-dir")
    },
    **{
        f"preview-negative-{route}-film-v1-full-36": ("preview", route, "materialized", (4832, 2016))
        for route in ("direct", "print")
    },
    "preview-negative-direct-film-v1-full-8-high": ("preview", "direct", "materialized", (4832, 2016)),
    "preview-negative-direct-film-v1-full-11-66-high": ("preview", "direct", "materialized", (4832, 2016)),
    "preview-negative-direct-film-v1-quarter-8-high": ("preview", "direct", "materialized", (1208, 504)),
}
SCANNER_SCHEDULE_ENTRIES = {
    ("direct", "monolithic"): "juicer_cuda_negative_direct_pipeline",
    ("print", "monolithic"): "juicer_cuda_print_focused_pipeline",
    ("direct", "fused-dir"): "juicer_cuda_direct_focused_scan_linear_rgb",
    ("print", "fused-dir"): "juicer_cuda_print_focused_scan_linear_rgb",
    ("direct", "materialized"): "juicer_cuda_direct_focused_scan_linear_density_rgb",
    ("print", "materialized"): "juicer_cuda_print_focused_scan_linear_density_rgb",
}
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
    extent = (
        (7, 5)
        if mode == "check"
        else ((1920, 1080) if row.startswith("g3-hd-") else (3840, 2160))
        if row.startswith("g3-")
        else HALATION_EXTENTS.get(row, (6048, 4032))
    )
    if (sample.get("width"), sample.get("height")) != extent:
        raise ValueError("incorrect geometry")
    if not sample.get("controls") or not sample.get("control_types") or not sample.get("device"):
        raise ValueError("missing controls or device")
    renders = sample.get("renders", [])
    g3 = row.startswith("g3-")
    cycles = 1 if g3 else 3
    if len(renders) != ((3 if mode == "check" else 41) if g3 else (15 if mode == "check" else 71)):
        raise ValueError("incomplete render membership")
    warmups = 10 if mode == "measure" else 1
    samples = 30 if mode == "measure" else 1
    expected = [("cold", 0, 0)]
    expected += [("warmup", 0, index) for index in range(warmups)]
    expected += [("warm", 0, index) for index in range(samples)]
    if not g3:
        expected += [(phase, 0, index) for index in range(3) for phase in ("changed", "changed-reuse")]
        for cycle in (1, 2):
            expected.append(("cycle-cold", cycle, 0))
            expected += [("cycle-warmup", cycle, index) for index in range(warmups)]
            expected.append(("cycle-end", cycle, 0))
    if [(r.get("phase"), r.get("cycle"), r.get("index")) for r in renders] != expected:
        raise ValueError("incorrect render order or membership")
    if row.startswith("g3-"):
        parts = row.split("-")
        offset = 0.5 if parts[7] == "half" else 0.0
        for render in renders:
            phase, index = render["phase"], render["index"]
            frame = offset
            if parts[6] == "varied":
                if phase in ("warmup", "cycle-warmup"):
                    frame += index - 10
                elif phase == "warm":
                    frame += index
            if render.get("frame_time") != frame:
                raise ValueError("incorrect G3 frame sequence")
        controls = sample["controls"]
        expected_stock = {"portra": "kodak_portra_400", "gold": "kodak_gold_200", "provia": "fujifilm_provia_100f"}[
            parts[2]
        ]
        expected_route = (
            "NegativePrintScan"
            if parts[3] == "print"
            else ("PositiveDirectScan" if parts[2] == "provia" else "NegativeDirectScan")
        )
        for key, expected_value in (
            ("FilmProfileKey", expected_stock),
            ("ScanRoute", expected_route),
            ("GrainActive", int(parts[5] == "on")),
            ("FilmGammaFactor", 1.0),
            ("CouplersActive", 0),
            ("CameraFilmFormatMm", 36.0),
        ):
            if controls.get(key) != [expected_value]:
                raise ValueError(f"incorrect G3 control: {key}")
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
    if (g3 and edits) or (
        not g3
        and (
            len(edits) != 3
            or [e.get("gamma") for e in edits] != [1.1, 1.0, 1.1]
            or any(e.get("status") not in (0, 14) or not nonnegative(e.get("edit_ns")) for e in edits)
        )
    ):
        raise ValueError("invalid edit membership")
    timings = sample.get("timings") or {}
    if any(not nonnegative(timings.get(k)) for k in ("module_open_ns", "load_ns", "describe_ns", "unload_ns")):
        raise ValueError("incomplete lifecycle timings")
    if any(
        not isinstance(timings.get(k), list) or len(timings[k]) != cycles or not all(nonnegative(v) for v in timings[k])
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
        or life.get("effects_created") != cycles
        or life.get("effects_destroyed") != cycles
    ):
        raise ValueError("invalid lifecycle")
    memory = sample.get("memory") or []
    endpoints = {("before-load", -1, 0), ("after-sample", 0, 30 if mode == "measure" else 1), ("after-unload", -1, 0)}
    endpoints.update(
        (name, cycle, 0)
        for name in ("after-create", "after-cold", "after-warmups", "after-destroy")
        for cycle in range(cycles)
    )
    if not g3:
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


G3_PRODUCT_DIRS = ("src", "rust", "native", "cmake", "Resources", "third_party", ".cargo")
G3_PRODUCT_FILES = (
    "CMakeLists.txt",
    "CMakePresets.json",
    "Cargo.lock",
    "Cargo.toml",
    "rust-toolchain.toml",
    ".gitattributes",
)


def g3_product_source(source, patch):
    prefixes = tuple(f"{name}/" for name in G3_PRODUCT_DIRS)

    def product(name):
        return name.startswith(prefixes) or name in G3_PRODUCT_FILES

    # The verifier emits binary Git patches with no renames. Quoted paths
    # retain the ASCII repository directory prefix.
    starts = tuple(f"diff --git {quote}a/{prefix}".encode() for prefix in prefixes for quote in ("", '"'))
    files = {f"diff --git a/{name} b/{name}".encode() for name in G3_PRODUCT_FILES}
    sections = re.split(rb"(?m)(?=^diff --git )", patch)
    return (
        {name: value for name, value in source["inputs"].items() if product(name)},
        {name: value for name, value in source["new_files_base64"].items() if product(name)},
        [section for section in sections if section.startswith(starts) or section.partition(b"\n")[0] in files],
    )


def g3_production_source(snapshot, production_patch, source, patch, *, root):
    """Bind the index, tracked worktree changes and added product inputs."""
    production = json.loads(snapshot)
    if production["head"] != source["head"]:
        if not all(re.fullmatch(r"[0-9a-f]{40,64}", value) for value in (production["head"], source["head"])):
            raise ValueError("invalid production source revision")
        difference = subprocess.run(
            ["git", "diff", "--quiet", production["head"], source["head"], "--", *G3_PRODUCT_DIRS, *G3_PRODUCT_FILES],
            cwd=root,
            check=False,
        )
        if difference.returncode != 0:
            raise ValueError("production witness product base revision differs")
    if g3_product_source(production, production_patch) != g3_product_source(source, patch):
        raise ValueError("production witness product source or membership differs")


def g3_production_binding(proof, preset, selected, source, patch, *, root):
    binding = candidate_qualification(*proof, preset, selected["shipping"], check_inputs=True)
    g3_production_source(proof[2], proof[3], source, patch, root=root)
    contract = json.loads(proof[1])
    module = contract["loaded_module"]["file"]
    path_type = PureWindowsPath if PureWindowsPath(module).drive else PurePosixPath
    resources = path_type(module).parent.parent / "Resources"
    qualified = {
        path_type(name).relative_to(resources).as_posix(): hashes["sha256_after"]
        for name, hashes in contract["consumed_inputs"].items()
        if path_type(name).is_relative_to(resources)
    }
    if not qualified or selected["resources"] != qualified or inventory(Path(str(resources))) != qualified:
        raise ValueError("shipping module-relative Resources differ from qualified bundle")
    return binding


def g3_identities(inputs):
    result = {}
    for key in ("driver",):
        entry = inputs[key]
        entry["path"] = str(Path(entry["path"]).resolve(strict=True))
        actual = inventory(entry["path"])
        if actual != entry["sha256"]:
            raise ValueError(f"{key} identity mismatch")
        result[key] = actual
    result["candidate"] = {}
    for key in ("shipping", "resources"):
        entry = inputs["candidate"][key]
        entry["path"] = str(Path(entry["path"]).resolve(strict=True))
        actual = inventory(entry["path"])
        expected = entry["files" if key == "resources" else "sha256"]
        if actual != expected:
            raise ValueError(f"candidate {key} identity mismatch")
        result["candidate"][key] = actual
    module = Path(inputs["candidate"]["shipping"]["path"])
    if Path(inputs["candidate"]["resources"]["path"]) != (module.parent.parent / "Resources").resolve(strict=True):
        raise ValueError("shipping module-relative Resources mismatch")
    return result


def g3_route(args):
    """Run a current production route witness under the native owner."""
    if args.inputs and (args.module or args.resources):
        raise ValueError("--inputs cannot be combined with --module or --resources")
    if args.mode not in ("check", "measure"):
        raise ValueError("invalid G3 route mode")
    if args.preset not in SHIPPING or args.preset.startswith("windows") != (os.name == "nt"):
        raise ValueError("G3 route requires a native Release preset")
    if args.g3_manifest:
        grain_g3 = importlib.import_module("grain_g3")
        manifest_path = Path(args.g3_manifest).resolve(strict=True)
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest.get("schema") != "grain-g3-workloads-v1" or manifest.get("rows") != grain_g3.expand_rows(
            manifest["workloads"], manifest["carriers"]
        ):
            raise ValueError("G3 manifest membership mismatch")
        selections = []
        stock_names = {"kodak_portra_400": "portra", "kodak_gold_200": "gold", "fujifilm_provia_100f": "provia"}
        for row in manifest["rows"]:
            if row["preset"] != args.preset or row["role"] != "decision":
                continue
            current = manifest["workloads"][row["workload"]]["current"]
            size = {(1920, 1080): "hd", (3840, 2160): "uhd"}[(current["width"], current["height"])]
            stock = stock_names[current["stock"]]
            route = {0: "direct", 1: "print", 2: "direct"}[current["route"]]
            if (current["route"] == 2) != (stock == "provia") or current["long_edge_mm"] != 36.0:
                raise ValueError("unsupported G3 route facts")
            offset = {0.0: "integer", 0.5: "half"}[row["offset"]]
            for enabled in ("off", "on"):
                workload = f"g3-{size}-{stock}-{route}-{current['look']}-{enabled}-{row['time_mode']}-{offset}"
                selections.append((row["id"], workload))
        if len(selections) != 88 or len(set(selections)) != 88:
            raise ValueError("incorrect G3 route membership")
    else:
        manifest_path = None
        manifest_bytes = None
        selections = [("single", args.workload)]
    for _, workload in selections:
        parts = workload.split("-")
        if (
            len(parts) != 8
            or parts[0] != "g3"
            or parts[1] not in ("hd", "uhd")
            or parts[2] not in ("portra", "gold", "provia")
            or parts[3] not in ("direct", "print")
            or parts[4] not in ("default", "expensive", "simple")
            or parts[5] not in ("on", "off")
            or parts[6] not in ("varied", "unchanged")
            or parts[7] not in ("integer", "half")
            or (parts[2] != "portra" and parts[3] != "direct")
        ):
            raise ValueError("invalid G3 route workload")
    directory = ROOT / "out/validation" / args.preset / "results" / execution.name_component(args.run_id)
    with execution.OutputLease(ROOT, ROOT / "out/build" / args.preset), execution.OutputLease(ROOT):
        if args.inputs:
            inputs = json.loads(Path(args.inputs).read_text())
        else:
            if not args.module or not args.resources:
                raise ValueError("provide both --module and --resources")
            driver = (
                ROOT
                / "out/build"
                / args.preset
                / "tests/bin"
                / ("JuicerOfxMeasurement.exe" if os.name == "nt" else "JuicerOfxMeasurement")
            )
            inputs = {
                "driver": {"path": str(driver), "sha256": inventory(driver)},
                "candidate": {
                    "shipping": {"path": args.module, "sha256": inventory(args.module)},
                    "resources": {"path": args.resources, "files": inventory(args.resources)},
                },
            }
        entry = g3_identities(inputs)
        result_path = Path(args.candidate_production_result).resolve(strict=True)
        proof = [result_path.read_bytes()] + [
            (result_path.parent / name).read_bytes()
            for name in ("production-contract.json", "candidate.json", "candidate.patch")
        ]
        source, patch, _ = verify.capture_native_source(ROOT, "tests/benchmarks/ofx.py", domain_tests=True)
        binding = g3_production_binding(proof, args.preset, entry["candidate"], source, patch, root=ROOT)
        directory.mkdir(parents=True, exist_ok=False)
        run = execution.NativeRun(ROOT, directory)
        execution.write_json(directory / "candidate.json", source)
        (directory / "candidate.patch").write_bytes(patch)
        execution.write_json(directory / "inputs.json", inputs)
        if manifest_bytes is not None:
            (directory / "g3-manifest.json").write_bytes(manifest_bytes)
        for name, data in zip(
            ("production-result.json", "production-contract.json", "production-source.json", "production-source.patch"),
            proof,
        ):
            (directory / name).write_bytes(data)
        result = {
            "schema": 1,
            "operation": "g3-route",
            "status": "failed",
            "preset": args.preset,
            "selections": selections,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest() if manifest_bytes else None,
            "mode": args.mode,
            "production_qualification": binding,
            "identities": entry,
            "source": source,
            "commands": run.commands,
            "samples": [],
            "intervals": "callback stream events include host submission gaps; active-kernel time is unmeasured",
        }
        try:
            for index, (row_id, workload) in enumerate(selections):
                name = f"shipping-{index:03d}"
                target = directory / f"{name}.json"
                command = [
                    inputs["driver"]["path"],
                    "--mode",
                    args.mode,
                    "--module",
                    inputs["candidate"]["shipping"]["path"],
                    "--resources",
                    inputs["candidate"]["resources"]["path"],
                    "--lane",
                    "shipping",
                    "--workload",
                    workload,
                    "--output",
                    str(target),
                ]
                member = {"row": row_id, "workload": workload, "file": target.name, "status": "started"}
                result["samples"].append(member)
                environment = {key: value for key, value in os.environ.items() if key.upper() != "JUICER_DIAGNOSTICS"}
                run.command(name, command, env=environment)
                sample = json.loads(target.read_text())
                validate_sample(sample, args.mode, "shipping", workload)
                if Path(sample["resources"]).resolve() != Path(command[6]).resolve():
                    raise ValueError("driver resource path mismatch")
                member.update(status="passed", sha256=execution.digest(target))
            final, final_patch, _ = verify.capture_native_source(ROOT, "tests/benchmarks/ofx.py", domain_tests=True)
            if (
                final != source
                or final_patch != patch
                or g3_identities(inputs) != entry
                or (manifest_path is not None and manifest_path.read_bytes() != manifest_bytes)
            ):
                raise ValueError("source or inputs changed during G3 route witness")
            if g3_production_binding(proof, args.preset, entry["candidate"], final, final_patch, root=ROOT) != binding:
                raise ValueError("production closure changed during G3 route witness")
            result["status"] = "passed"
        except Exception as error:
            result["error"] = str(error)
        if all(command["drained"] for command in run.commands):
            run.publish(result)
        else:
            execution.write_json(directory / "incomplete.json", result)
        if result["status"] != "passed":
            raise ValueError(result.get("error", "G3 route witness failed"))


def scanner_capacity_receipt(output, observer, module, resources):
    driver = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    observed = json.loads(observer.read_text(encoding="utf-8"))
    if (
        driver["schema"] != "juicer-scanner-capacity-driver-v1"
        or Path(driver["module"]) != module
        or Path(driver["resources"]) != resources
        or driver["bounds"] != [0, 0, 12800, 96]
        or driver["row_bytes"] != (12800 * 3 + 5) * 4
        or driver["frame_rate"] != 24.0
        or driver["context_restored"] is not True
        or driver["unloaded"] is not True
    ):
        raise ValueError("capacity driver identity, geometry or lifecycle differs")
    controls = {
        "CameraFilmFormatPreset": [0],
        "CameraFilmFormatMm": [8.0],
        "FilmGrainAmountEv": [3.0],
        "FilmGrainChromaFraction": [1.0],
        "FilmGrainTexelPitchUm": [0.5],
        "GrainModel": ["film_v1"],
        "GrainActive": [1],
        "InputColorSpace": [0],
        "InputCctfDecoding": [0],
        "OutputCctfEncoding": [1],
        "OutputGamutCompression": [0],
        "CouplersActive": [0],
        "HalationActive": [0],
        "GlareActive": [0],
    }
    if driver["controls"] != controls:
        raise ValueError("capacity control readbacks differ")
    frames = observed["frames"]
    if (
        observed["schema"] != "juicer-scanner-capacity-observation-v1"
        or observed["exited"] is not True
        or observed["inferior_exit"] != 0
        or observed["errors"]
        or observed["pending_returns"] != 0
        or len(frames) != 2
        or len(driver["rows"]) != 2
    ):
        raise ValueError("capacity child did not complete both native renders")
    guards = (
        "finite_output",
        "alpha_preserved",
        "padding_preserved",
        "source_preserved",
        "stream_completed",
        "readback_complete",
        "context_preserved",
    )
    source = output / "source-pitched.f32"
    if source.stat().st_size != driver["row_bytes"] * 96:
        raise ValueError("capacity pitched source size differs")
    rasters = {source.name: execution.digest(source)}
    owner_fields = ("resources", "device", "context", "epoch")
    for index, (row, draw, phase, time_bits) in enumerate(
        zip(driver["rows"], frames, ("measured", "reporting"), (0, 0x3FF0000000000000), strict=True)
    ):
        retained, poll = draw["retained"], draw["poll"]
        if (
            row["pass"] != phase
            or row["output_file"] != phase + ".f32"
            or row["ofx_status"] != 0
            or any(row[key] is not True for key in guards)
            or row["source_image_releases"] != 1
            or row["output_image_releases"] != 1
            or row["time_bits"] != time_bits
            or draw["time_bits"] != time_bits
            or draw["index"] != index
            or draw["token"] != index
            or draw["route"] != 0
            or draw["outcome"] != [0, 0, 0, 0]
            or draw["launches"] != ["spectral_launch"]
            or not all(draw[key] for key in ("instance", "submission", "context", "stream", "thread"))
            or not all(retained[key] for key in ("host", "event", "resources", "epoch", "success", "handoff"))
            or any(retained[key] != poll[key] for key in owner_fields)
            or retained["epoch"] != retained["epoch_at_production"]
            or retained["device"] != draw["device"]
            or retained["context"] != draw["context"]
            or poll["stream"] != draw["stream"]
            or poll["success"] is not True
            or poll["status"] != 0
        ):
            raise ValueError(f"capacity {phase} guards, spectral launch or readback ownership differs")
        for operation in ("allocations", "uploads"):
            if any(
                not isinstance(item["label"], str)
                or item["bytes"] <= 0
                or re.search(r"pchip|scanner lut|scanner cube", item["label"], re.IGNORECASE)
                for item in draw[operation]
            ):
                raise ValueError(f"capacity {phase} {operation} includes retired scanner traffic")
        raster = output / row["output_file"]
        if raster.stat().st_size != 12800 * 96 * 3 * 4:
            raise ValueError(f"capacity {phase} raster size differs")
        rasters[raster.name] = execution.digest(raster)
    measured, reporter = frames
    surviving = {"film TC LUT", "focused film tablesIllum", "film Mallett basis"}
    allocations = {(item["label"], item["bytes"]) for item in measured["allocations"] if item["success"]}
    uploads = {(item["label"], item["bytes"]) for item in measured["uploads"] if item["success"]}
    if not any(label in surviving for label, _ in allocations & uploads):
        raise ValueError("capacity lacks a matching successful surviving labeled allocation and upload")
    retained = measured["retained"]
    terminal = reporter["poll"]
    if (
        any(measured[key] != reporter[key] for key in ("instance", "device", "context", "stream", "thread"))
        or reporter["submission"] != measured["submission"] + 1
        or any(retained[key] != terminal[key] for key in owner_fields)
        or measured["poll"]["queries"] != []
        or measured["poll"]["destroyed"] != []
        or terminal["queries"] != [{"event": retained["event"], "status": 0, "completed_flag": 0}]
        or terminal["destroyed"] != [retained["event"]]
        or rasters["measured.f32"] == rasters["reporting.f32"]
    ):
        raise ValueError("capacity terminal reporter did not consume the exact predecessor's completed zero flag")
    return {"rasters": rasters, "frames": frames, "controls": controls}


def scanner_capacity(args):
    if args.preset.startswith("windows-") != (os.name == "nt"):
        raise ValueError("run scanner capacity on its native platform")
    build = ROOT / "out/build" / args.preset
    directory = ROOT / "out/validation" / args.preset / "results" / execution.name_component(args.run_id)
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    observer = Path(__file__).with_name("observe_seed.py").resolve(strict=True)
    run = execution.NativeRun(ROOT, directory)
    with execution.OutputLease(ROOT, build), execution.OutputLease(ROOT) as gpu:
        configuration = verify.cache_values(build)
        if (
            not qualification.read_preview_and_replay_configuration(configuration)[0]
            or configuration["BUILD_TESTING"] != "ON"
        ):
            raise ValueError("capacity requires the native testing-ON Preview configuration")
        _, built, staged = qualification.production_module_paths(args.preset, configuration)
        module = Path(staged).resolve(strict=True)
        resources = (module.parent.parent / "Resources").resolve(strict=True)
        driver = Path(qualification.prepared_render_probe_path(args.preset, built)).resolve(strict=True)
        debugger = Path(inputs["debugger"]).resolve(strict=True)
        symbols = Path(inputs["symbols"]).resolve(strict=True) if os.name == "nt" else module
        files = {"module": module, "symbols": symbols, "driver": driver, "debugger": debugger, "observer": observer}
        hashes = {name: execution.digest(path) for name, path in files.items()}
        resource_hashes = inventory(resources)
        if (
            hashes["module"] != execution.digest(Path(built))
            or hashes["module"] != inputs["module_sha256"]
            or hashes["symbols"] != inputs["symbols_sha256"]
        ):
            raise ValueError("capacity stage, build or symbols differ from the admitted instrumentation layout")
        report = {
            "schema": "juicer-scanner-capacity-v1",
            "status": "failed",
            "preset": args.preset,
            "inputs": {name: {"path": str(path), "sha256": hashes[name]} for name, path in files.items()},
            "resources": resource_hashes,
            "gpu_lease": str(gpu.path),
            "commands": run.commands,
        }
        directory.mkdir(parents=False, exist_ok=False)
        try:
            output = directory / "product"
            output.mkdir()
            settings = directory / "observer.json"
            receipt = directory / "observations.json"
            execution.write_json(
                settings,
                {
                    "module": str(module),
                    "symbols": str(symbols) if os.name == "nt" else None,
                    "uuid": inputs["uuid"],
                    "points": inputs["points"],
                    "layout": inputs.get("layout", {}),
                    "trace": str(directory / "observations.jsonl"),
                    "receipt": str(receipt),
                },
            )
            command = [debugger, "--batch"]
            if os.name != "nt":
                command.extend(("-o", "settings set target.disable-aslr false"))
            command.extend(
                (
                    "-o",
                    f'command script import "{observer.as_posix()}"',
                    "-o",
                    "script observe_seed.install(lldb.debugger)",
                    "-o",
                    "run",
                    "-o",
                    "script observe_seed.finish(lldb.debugger)",
                    "--",
                    driver,
                    "--scanner-capacity",
                    module,
                    output,
                )
            )
            environment = {**os.environ, "FJ_SCANNER_OBSERVER": str(settings)}
            command, environment = scanner_debugger_environment(command, environment, inputs)
            run.command("capacity", command, env=environment)
            report["capacity"] = scanner_capacity_receipt(output, receipt, module, resources)
            if {name: execution.digest(path) for name, path in files.items()} != hashes or inventory(
                resources
            ) != resource_hashes:
                raise ValueError("scanner capacity inputs changed during the run")
            report["status"] = "passed"
        except Exception as error:
            report["error"] = str(error)
            raise
        finally:
            execution.write_json(directory / "result.json", report)


def scanner_debugger_environment(command, environment, inputs):
    debugger_environment = inputs.get("debugger_environment", {})
    if set(debugger_environment) - {"PATH", "PYTHONHOME"}:
        raise ValueError("unsupported scanner debugger environment")
    if os.name != "nt" or not debugger_environment:
        return command, {**environment, **debugger_environment}

    # The Python version embedded in LLDB does not own the process supervisor.
    def quote(value):
        return "'" + str(value).replace("'", "''") + "'"

    script = [f"$env:{key} = {quote(value)}" for key, value in debugger_environment.items()]
    script.extend(("& " + " ".join(map(quote, command)), "exit $LASTEXITCODE"))
    encoded = base64.b64encode("\n".join(script).encode("utf-16-le")).decode("ascii")
    shell = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    return [shell, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded], environment


def spectral_replay_poll(predecessor, reporter):
    held = predecessor["retained"]
    poll = reporter["poll"]
    query = {"event": held["event"], "status": 0, "completed_flag": 0}
    if (
        not held["success"]
        or not held["handoff"]
        or not held["event"]
        or not held["host"]
        or held["producer"][:2] != [predecessor["submission"], predecessor["token"]]
        or any(held[key] != poll[key] for key in ("resources", "context", "device", "epoch"))
        or predecessor["context"] != held["context"]
        or reporter["context"] != held["context"]
        or held["epoch_at_production"] != held["epoch"]
        or reporter["stream"] != predecessor["stream"]
        or reporter["stream"] != poll["stream"]
        or reporter["prepare"] != {"stream": reporter["stream"], "outcome": 0}
        or reporter["submission"] != predecessor["submission"] + 1
        or reporter["token"] == predecessor["token"]
        or poll["before"][0] == poll["before"][1]
        or poll["after"][0] != poll["after"][1]
        or poll["entry"] != [held["host"], held["event"]]
        or poll["queries"] != [query]
        or held["event"] not in poll["destroyed"]
        or not poll["success"]
        or poll["status"] != 0
    ):
        raise ValueError("spectral replay did not consume the exact preceding owner event and zero flag")
    return {"predecessor": held, "reporter": reporter["submission"], "poll": poll}


def read_spectral_pilot_child(directory, settings, companion):
    host = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    observation = json.loads(Path(settings["receipt"]).read_text(encoding="utf-8"))
    frames = observation["frames"]
    module = Path(settings["module"])
    if (
        host["schema"] != "juicer-ofx-spectral-seed-replay-v1"
        or observation["schema"] != "juicer-scanner-replay-observation-v2"
        or Path(host["module"]).resolve() != module
        or Path(host["resources"]).resolve() != module.parent.parent / "Resources"
        or any(host[key] != settings[key] or observation[key] != settings[key] for key in ("variant", "grain", "route"))
        or host["replay_export"] is not companion
        or (companion and host["replay_tokens"] != observation["tokens"])
        or not all(host[key] is True for key in ("context_restored", "grain_changes_output", "fraction_changes_output"))
        or observation["errors"]
        or not observation["exited"]
        or observation["inferior_exit"] != 0
        or observation["pending_returns"]
        or len(frames) != 4
        or len(host["rows"]) != 4
    ):
        raise ValueError("spectral replay child has a different identity or incomplete outcome")

    width, height = 64, 32
    pitch = (width * 4 + 5) * 4
    if host["bounds"] != [0, 0, width, height] or host["canonical_rod"] != host["bounds"] or host["row_bytes"] != pitch:
        raise ValueError("spectral replay host geometry differs")
    source = (directory / "source-pitched.f32").read_bytes()
    if len(source) != height * pitch:
        raise ValueError("spectral replay source has the wrong pitched extent")
    alpha = tuple(
        struct.unpack_from("<I", source, y * pitch + x * 16 + 12)[0] for y in range(height) for x in range(width)
    )
    rasters = {"source": hashlib.sha256(source).hexdigest()}
    words = []
    densities = {}
    names = ("grain-off", "integral", "fractional", "reporting")
    expected_route = int(settings["route"] == "negative-print")
    active_model = "film_grain" if settings["grain"] == "film-v1" else "current_grain"
    inactive_model = "current_grain" if active_model == "film_grain" else "film_grain"
    for index, (phase, frame_time, host_row, frame) in enumerate(
        zip(names, (0.0, 0.0, 0.5, 1.0), host["rows"], frames)
    ):
        time_bits = struct.unpack("<Q", struct.pack("<d", frame_time))[0]
        controls = host_row["controls"]
        if (
            frame["tokens"] != observation["tokens"]
            or frame["thread"] != frames[0]["thread"]
            or frame["context"] != frames[0]["context"]
            or frame["stream"] != frames[0]["stream"]
            or frame["outcome"] != [0, 0, 0, 0]
            or frame["index"] != index
            or frame["route"] != expected_route
            or frame["token"] != int(frame_time)
            or frame["time_bits"] != time_bits
            or host_row["time_bits"] != time_bits
            or host_row["pass"] != phase
            or host_row["ofx_status"] != 0
            or host_row["source_image_releases"] != 1
            or host_row["output_image_releases"] != 1
            or not all(
                host_row[key] is True
                for key in (
                    "alpha_preserved",
                    "padding_preserved",
                    "source_preserved",
                    "finite_output",
                    "readback_complete",
                    "stream_completed",
                    "context_preserved",
                )
            )
            or controls["GrainActive"] != [int(index > 0)]
            or controls["GlareActive"] != [0]
            or (
                settings["variant"] == "preview"
                and controls["GrainModel"] != ["film_v1" if active_model == "film_grain" else "current"]
            )
            or frame[inactive_model] is not None
            or bool(frame[active_model]) != bool(index)
            or (index and frame["launches"] != ["spectral_launch"])
        ):
            raise ValueError(f"spectral replay {phase} controls, native draw or guards differ")
        filename = phase + ".f32"
        if host_row["output_file"] != filename:
            raise ValueError(f"spectral replay {phase} raster is missing")
        pixels = (directory / filename).read_bytes()
        packed = struct.unpack(f"<{width * height * 4}I", pixels)
        if packed[3::4] != alpha:
            raise ValueError(f"spectral replay {phase} changed alpha")
        rasters[phase] = hashlib.sha256(pixels).hexdigest()
        words.append(frame[active_model])
        if index:
            declaration = frame["density"]
            receipt = Path(settings["receipt"])
            expected = f"{receipt.stem}-density-{index}.f32"
            if declaration["file"] != expected or (declaration["width"], declaration["height"]) != (width, height):
                raise ValueError(f"spectral replay {phase} did not bind the actual scanner density image")
            density = (receipt.parent / expected).read_bytes()
            digest = hashlib.sha256(density).hexdigest()
            if (
                len(density) != width * height * 3 * 4
                or digest != declaration["sha256"]
                or any(not math.isfinite(lane) for (lane,) in struct.iter_unpack("<f", density))
            ):
                raise ValueError(f"spectral replay {phase} density words missing or altered")
            densities[phase] = digest
    if words[1][-1] != 0 or words[2][-1] != 0x3F000000 or words[1][0] != words[2][0] or words[2][0] == words[2][1]:
        raise ValueError("spectral replay temporal grain words differ")
    return {
        "controls": [row["controls"] for row in host["rows"]],
        "rasters": rasters,
        "tokens": observation["tokens"],
        "grain_words": words,
        "density_words": densities,
        "terminal_poll": spectral_replay_poll(*frames[-2:]),
    }


def spectral_seed_pilot(args):
    if args.preset.startswith("windows-") != (os.name == "nt"):
        raise ValueError("run spectral replay on its native platform")
    build = ROOT / "out/build" / args.preset
    directory = ROOT / "out/validation" / args.preset / "results" / execution.name_component(args.run_id)
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    observer = Path(__file__).with_name("observe_seed.py").resolve(strict=True)
    run = execution.NativeRun(ROOT, directory)
    with execution.OutputLease(ROOT, build), execution.OutputLease(ROOT) as gpu:
        configuration = verify.cache_values(build)
        preview, _ = qualification.read_preview_and_replay_configuration(configuration)
        variant = "preview" if preview else "ordinary"
        if configuration["BUILD_TESTING"] != "ON" or (not preview and args.grain != "current"):
            raise ValueError("spectral replay requires testing ON and matching configured grain model")
        _, built, staged = qualification.production_module_paths(args.preset, configuration)
        product = Path(staged).resolve(strict=True)
        manifest = build / "tests/spectral-replay/artifacts.json"
        companion = json.loads(manifest.read_text(encoding="utf-8"))
        archive = Path(companion["rust_archive"]).resolve(strict=True)
        expected_features = ["test-support", "grain-preview"] if preview else ["test-support"]
        rust_cache = build / "cargo/test-support"
        if preview:
            rust_cache /= "preview-replay"
        if (
            companion["variant"] != variant
            or companion["rust_features"] != expected_features
            or not archive.is_relative_to(rust_cache)
        ):
            raise ValueError("spectral replay archive variant differs from configured product")
        modules = {"product": product, "companion": Path(companion["module"]).resolve(strict=True)}
        resources = {side: module.parent.parent / "Resources" for side, module in modules.items()}
        if (
            modules["companion"] == product
            or resources["companion"] != Path(companion["resources"])
            or inventory(resources["product"]) != inventory(resources["companion"])
        ):
            raise ValueError("spectral replay requires distinct modules with matching Resources")
        suffix = ".exe" if os.name == "nt" else ""
        driver = (build / "tests/bin" / ("JuicerHalationComparisonProbe" + suffix)).resolve(strict=True)
        debugger = Path(inputs["debugger"]).resolve(strict=True)
        products = {side: inputs[side] for side in modules}
        symbols = {
            side: Path(products[side]["symbols"]).resolve(strict=True) if os.name == "nt" else module
            for side, module in modules.items()
        }
        files = {
            "driver": driver,
            "debugger": debugger,
            "observer": observer,
            "companion_manifest": manifest,
            "companion_rust_archive": archive,
            "native_archive": Path(companion["native_archive"]),
            **modules,
            **{side + "_symbols": path for side, path in symbols.items()},
        }
        hashes = {name: execution.digest(path) for name, path in files.items()}
        resource_hashes = inventory(resources["product"])
        if hashes["product"] != execution.digest(Path(built)) or any(
            hashes[side] != products[side]["module_sha256"]
            or hashes[side + "_symbols"] != products[side]["symbols_sha256"]
            for side in modules
        ):
            raise ValueError("spectral replay staged bytes or symbols differ from observer inputs")
        report = {
            "schema": "juicer-ofx-spectral-seed-pilot-v2",
            "status": "failed",
            "preset": args.preset,
            "variant": variant,
            "route": args.route,
            "grain": args.grain,
            "inputs": {name: {"path": str(path), "sha256": hashes[name]} for name, path in files.items()},
            "resources": resource_hashes,
            "gpu_lease": str(gpu.path),
            "commands": run.commands,
        }
        directory.mkdir(parents=False, exist_ok=False)
        try:
            captures = {}
            for side, module in modules.items():
                output = directory / side
                output.mkdir()
                settings_path = directory / f"{side}-observer.json"
                settings = {
                    "mode": "replay",
                    "variant": variant,
                    "grain": args.grain,
                    "route": args.route,
                    "module": str(module),
                    "symbols": str(symbols[side]) if os.name == "nt" else None,
                    "uuid": products[side]["uuid"],
                    "points": products[side]["points"],
                    "layout": products[side]["layout"],
                    "trace": str(directory / f"{side}-observations.jsonl"),
                    "receipt": str(directory / f"{side}-draw.json"),
                }
                execution.write_json(settings_path, settings)
                command = [debugger, "--batch"]
                if os.name != "nt":
                    command.extend(("-o", "settings set target.disable-aslr false"))
                command.extend(
                    (
                        "-o",
                        f'command script import "{observer.as_posix()}"',
                        "-o",
                        "script observe_seed.install(lldb.debugger)",
                        "-o",
                        "run",
                        "-o",
                        "script observe_seed.finish(lldb.debugger)",
                        "--",
                        driver,
                        "--spectral-seed-pilot",
                        module,
                        output,
                        variant,
                        args.grain,
                        args.route,
                    )
                )
                if side == "companion":
                    command.append(directory / "product-draw.json")
                environment = {**os.environ, "FJ_SCANNER_OBSERVER": str(settings_path)}
                command, environment = scanner_debugger_environment(command, environment, inputs)
                try:
                    run.command(side, command, env=environment)
                except subprocess.CalledProcessError as error:
                    raise ValueError(
                        f"spectral replay {side} exited {error.returncode}. See {directory / (side + '.log')}"
                    ) from error
                captures[side] = read_spectral_pilot_child(output, settings, side == "companion")
            report["captures"] = captures
            if any(
                captures["companion"][key] != captures["product"][key]
                for key in ("controls", "rasters", "tokens", "grain_words", "density_words")
            ):
                raise ValueError("spectral replay did not reproduce exact source, control, seed and output words")
            if {name: execution.digest(path) for name, path in files.items()} != hashes or any(
                inventory(path) != resource_hashes for path in resources.values()
            ):
                raise ValueError("spectral replay inputs changed during the pair")
            report["status"] = "passed"
        except Exception as error:
            report["error"] = str(error)
            raise
        finally:
            execution.write_json(directory / "result.json", report)


def read_scanner_timing_arm(host, tail, workload, mode, arm, scanner_entry):
    ready = tail["ready"]
    predecessor = tail["predecessor"]
    [reporter] = tail["frames"]
    variant, route, schedule, geometry = SCANNER_PAIRED_ROWS[workload]
    row = host["scanner_paired"]
    expected_id = "com.juicer.Juicer" + (".GrainPreview" if variant == "preview" else "")
    witnessed_route = "NegativePrintScan" if route == "print" else "NegativeDirectScan"
    last = 2 if mode == "check" else 40
    phases = ["cold", *(["warmup"] * (1 if mode == "check" else 10)), *(["warm"] * (1 if mode == "check" else 30))]
    controls = host["controls"]
    life = host["lifecycle"]
    if (
        tail["schema"] != "juicer-scanner-measured-tail-v1"
        or tail["errors"]
        or tail["pending_returns"]
        or tail["exit"] != 0
        or tail["child_exit"] != 0
        or host["status"] != "passed"
        or host["mode"] != mode
        or host["lane"] != "shipping"
        or host["campaign"] != "scanner-paired"
        or host["workload"] != workload
        or row["variant"] != variant
        or row["product_id"] != expected_id
        or row["route"] != witnessed_route
        or row["expected_schedule"] != schedule
        or row["geometry"] != list(geometry)
        or row["natural_grain"] != (schedule == "materialized")
        or row.get("authored_grid") != (17 if arm == "cubic" else None)
        or controls["ScanRoute"] != [witnessed_route]
        or controls["GrainActive"] != [int(row["natural_grain"])]
        or (route == "print" and controls["GlareActive"] != [0])
        or host["tail_ready"] != ready
        or ready["ordinal"] != last
        or life["effects_created"] != 1
        or life["effects_destroyed"] != 1
        or not life["context_restored"]
        or not life["module_closed"]
        or life["callback_error"]
        or host["edits"]
        or [sample["phase"] for sample in host["renders"]] != phases
    ):
        raise ValueError(f"{arm} {workload} timing identity, lifecycle or row membership differs")
    result = host["tail_reporter"]
    if (
        result["token"] != last + 1
        or result["status"] != 0
        or not result["output_guard_valid"]
        or not result["releases_valid"]
        or any(ready[k] != result[k] for k in ("context", "stream", "effect", "instance_data"))
        or predecessor["route"] != int(route == "print")
        or reporter["instance"] != predecessor["instance"]
        or reporter["native_handle"] != predecessor["handle"]
        or reporter["submission"] != predecessor["next_id"]
        or reporter["token"] != result["token"]
        or reporter["route"] != predecessor["route"]
        or reporter["outcome"] != [0, 0, 0, 0]
        or reporter["launches"] != ([] if scanner_entry is None else [scanner_entry])
    ):
        raise ValueError(f"{arm} {workload} reporter does not match measured owner and latch")
    held = {**predecessor, "success": True, "handoff": True}
    if not held["producer"]:
        # The cubic entry identifies the preceding request by its sole retained event and live Rust latch.
        held["producer"] = [predecessor["latch_id"], predecessor["frame_token"]]
    terminal = spectral_replay_poll(
        {
            "retained": held,
            "submission": predecessor["latch_id"],
            "token": predecessor["frame_token"],
            "context": ready["context"],
            "stream": ready["stream"],
        },
        reporter,
    )
    for index, sample in enumerate(host["renders"]):
        if (
            sample["frame_time"] != index
            or sample["callback_status"] != 0
            or not sample["releases_valid"]
            or not sample["context_preserved"]
            or not nonnegative(sample["stream_interval_ms"])
            or not isinstance(sample["completed_render_ns"], int)
            or sample["completed_render_ns"] <= 0
            or (sample["output_checked"] and sample["output_valid"] is not True)
        ):
            raise ValueError(f"{arm} {workload} incomplete measured render {index}")
    return {
        "terminal": terminal,
        "scanner_entry": scanner_entry,
        "observed_schedule": schedule if scanner_entry is not None else None,
        "source_digest": host["source_digest"],
        "device": {k: host["device"][k] for k in ("name", "ordinal", "driver_version")},
        "controls": controls,
        "cold": host["renders"][0],
        "memory": host["memory"],
        "frame_samples": [sample for sample in host["renders"] if sample["phase"] == "warm"],
    }


def scanner_paired(args):
    if args.preset.startswith("windows-") != (os.name == "nt"):
        raise ValueError("run scanner paired timing on its native platform")
    if len(set(args.workloads)) != len(args.workloads):
        raise ValueError("scanner paired workload membership contains duplicates")
    if (args.mode == "measure") != (args.check_result is not None):
        raise ValueError("scanner paired timing requires prior check-mode evidence")
    if args.check_arm and args.mode != "check":
        raise ValueError("single-arm observation only applies in check mode")
    scope = "single-arm-tail" if args.check_arm else "paired"
    build = ROOT / "out/build" / args.preset
    directory = ROOT / "out/validation" / args.preset / "results" / execution.name_component(args.run_id)
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    inputs_sha = execution.digest(args.inputs)
    driver = (
        build / "tests/bin" / ("JuicerOfxMeasurement.exe" if os.name == "nt" else "JuicerOfxMeasurement")
    ).resolve(strict=True)
    observer = Path(__file__).with_name("observe_seed.py").resolve(strict=True)
    debugger = Path(inputs["debugger"]).resolve(strict=True)
    run = execution.NativeRun(ROOT, directory)
    if args.check_result:
        checked = json.loads(args.check_result.read_text(encoding="utf-8"))
        if (
            checked["schema"] != "juicer-scanner-paired-v1"
            or checked["mode"] != "check"
            or checked["scope"] != "paired"
            or checked["preset"] != args.preset
            or checked["inputs_sha256"] != inputs_sha
            or checked["status"] != "passed"
            or not set(args.workloads) <= checked["workloads"].keys()
        ):
            raise ValueError("check-mode evidence is outside this campaign scope")
    report = {
        "schema": "juicer-scanner-paired-v1",
        "status": "failed",
        "preset": args.preset,
        "mode": args.mode,
        "scope": scope,
        "inputs_sha256": inputs_sha,
        "workloads": {},
        "commands": run.commands,
    }
    with execution.OutputLease(ROOT, build), execution.OutputLease(ROOT) as gpu:
        if (
            execution.digest(driver) != inputs["driver_sha256"]
            or execution.digest(debugger) != inputs["debugger_sha256"]
        ):
            raise ValueError("scanner timing driver or debugger identity differs")
        report["gpu_lease"] = str(gpu.path)
        report["observer_sha256"] = execution.digest(observer)
        if args.check_result and checked["observer_sha256"] != report["observer_sha256"]:
            raise ValueError("scanner timing observer changed since check mode")
        directory.mkdir(parents=False, exist_ok=False)
        try:
            all_gates = True
            for workload in args.workloads:
                variant, route, schedule, _ = SCANNER_PAIRED_ROWS[workload]
                arms = inputs["variants"][variant]
                if not args.check_arm:
                    qualification_row = inputs["row_bindings"][workload]
                    if (
                        any(
                            qualification_row[arm]["module_sha256"] != arms[arm]["module_sha256"]
                            or qualification_row[arm]["schedule"] != schedule
                            for arm in ("cubic", "spectral")
                        )
                        or qualification_row["cubic"]["grid"] != 17
                    ):
                        raise ValueError(f"{workload} is missing the actual schedule and cubic grid qualification")
                row_dir = directory / workload
                row_dir.mkdir()
                pairs = (
                    [(args.check_arm,) if args.check_arm else ("cubic", "spectral")]
                    if args.mode == "check"
                    else [
                        ("cubic", "cubic"),
                        ("cubic", "cubic"),
                        ("cubic", "cubic"),
                        ("cubic", "spectral"),
                        ("spectral", "cubic"),
                        ("cubic", "spectral"),
                    ]
                )
                results = []
                binding = None
                for pair_index, pair in enumerate(pairs):
                    runs = []
                    for ordinal, arm in enumerate(pair):
                        declaration = arms[arm]
                        points = declaration["points"].copy()
                        scanner_entry = None
                        if "schedule_points" in declaration or not args.check_arm:
                            scanner_entry = SCANNER_SCHEDULE_ENTRIES[route, schedule]
                            point = declaration.get("schedule_points", {}).get(scanner_entry)
                            if not point or point["symbol"] != scanner_entry:
                                raise ValueError(f"{arm} {workload} lacks the compiled scanner entry for {schedule}")
                            points["scanner_schedule"] = point
                        module = Path(declaration["module"]).resolve(strict=True)
                        resources = (module.parent.parent / "Resources").resolve(strict=True)
                        symbols = Path(declaration["symbols"]).resolve(strict=True) if os.name == "nt" else module
                        if (
                            execution.digest(module) != declaration["module_sha256"]
                            or execution.digest(symbols) != declaration["symbols_sha256"]
                            or inventory(resources) != inputs["variants"][variant]["resources"]
                        ):
                            raise ValueError(f"{arm} {workload} module, symbols or Resources differ")
                        name = f"{pair_index}-{ordinal}-{arm}"
                        arm_dir = row_dir / name
                        arm_dir.mkdir()
                        settings_path = arm_dir / "observer.json"
                        native_receipt = arm_dir / "driver.json"
                        tail_receipt = arm_dir / "tail.json"
                        ready = arm_dir / "ready.json"
                        release = arm_dir / "release.json"
                        command = [
                            driver,
                            "--campaign",
                            "scanner-paired",
                            "--mode",
                            args.mode,
                            "--lane",
                            "shipping",
                            "--workload",
                            workload,
                            "--module",
                            module,
                            "--resources",
                            resources,
                            "--output",
                            native_receipt,
                            "--tail-run",
                            args.run_id,
                            "--tail-arm",
                            name,
                            "--tail-ready",
                            ready,
                            "--tail-release",
                            release,
                        ]
                        execution.write_json(
                            settings_path,
                            {
                                "mode": "tail",
                                "run": args.run_id,
                                "arm": name,
                                "last_ordinal": 2 if args.mode == "check" else 40,
                                "command": list(map(str, command)),
                                "ready": str(ready),
                                "release": str(release),
                                "receipt": str(tail_receipt),
                                "trace": str(arm_dir / "native.jsonl"),
                                "module": str(module),
                                "symbols": str(symbols) if os.name == "nt" else None,
                                "uuid": declaration["uuid"],
                                "points": points,
                                "scanner_schedule": scanner_entry,
                                "layout": declaration["layout"],
                            },
                        )
                        command = [
                            debugger,
                            "--batch",
                            "-o",
                            f'command script import "{observer.as_posix()}"',
                            "-o",
                            "script observe_seed.tail(lldb.debugger)",
                        ]
                        environment = {**os.environ, "FJ_SCANNER_OBSERVER": str(settings_path)}
                        command, environment = scanner_debugger_environment(command, environment, inputs)
                        run.command(f"{workload}-{name}", command, env=environment)
                        host = json.loads(native_receipt.read_text(encoding="utf-8"))
                        tail = json.loads(tail_receipt.read_text(encoding="utf-8"))
                        item = read_scanner_timing_arm(host, tail, workload, args.mode, arm, scanner_entry)
                        same_inputs = {key: item[key] for key in ("source_digest", "device", "controls")}
                        if binding is not None and same_inputs != binding:
                            raise ValueError(f"{workload} natural timing pair changed common workload inputs")
                        binding = same_inputs
                        samples = sorted(sample["completed_render_ns"] for sample in item["frame_samples"])
                        runs.append(
                            {
                                "arm": arm,
                                "receipt": str(arm_dir),
                                "result": item,
                                "median_ns": statistics.median(samples),
                                "p90_ns": statistics.quantiles(samples, n=10, method="inclusive")[-1]
                                if len(samples) > 1
                                else samples[0],
                            }
                        )
                    results.append(runs)
                paired = []
                aa_ratios = []
                for pair in results:
                    if len(pair) != 2:
                        continue
                    left, right = pair
                    reference, candidate = (left, right) if left["arm"] == "cubic" else (right, left)
                    if reference["arm"] == candidate["arm"]:
                        aa_ratios.append({key: right[key] / left[key] for key in ("median_ns", "p90_ns")})
                        continue
                    paired.append({key: candidate[key] / reference[key] for key in ("median_ns", "p90_ns")})
                gated = args.mode == "measure" and variant == "ordinary"
                passed = not gated or all(all(value <= 1.05 for value in pair.values()) for pair in paired)
                report["workloads"][workload] = {
                    "gated": gated,
                    "passed": passed,
                    "pairs": results,
                    "aa_ratios": aa_ratios,
                    "ratios": paired,
                }
                all_gates = all_gates and passed
            if not all_gates:
                raise ValueError("ordinary small-grid timing exceeds the 5% median or p90 limit")
            report["status"] = "passed"
        except Exception as error:
            report["error"] = str(error)
            raise
        finally:
            execution.write_json(directory / "result.json", report)


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
    command = sub.add_parser("g3-route")
    for option in ("preset", "run-id", "candidate-production-result", "mode"):
        command.add_argument("--" + option, required=True)
    command.add_argument("--inputs")
    command.add_argument("--module")
    command.add_argument("--resources")
    selection = command.add_mutually_exclusive_group(required=True)
    selection.add_argument("--workload")
    selection.add_argument("--g3-manifest")
    command = sub.add_parser("scanner-capacity")
    command.add_argument("--preset", required=True, choices=("windows-clang-release", "linux-release"))
    command.add_argument("--run-id", required=True)
    command.add_argument("--inputs", required=True, type=Path)
    command = sub.add_parser("spectral-seed-pilot")
    command.add_argument("--preset", required=True, choices=("windows-clang-release", "linux-release"))
    command.add_argument("--run-id", required=True)
    command.add_argument("--inputs", required=True, type=Path)
    command.add_argument("--route", required=True, choices=("negative-direct", "negative-print"))
    command.add_argument("--grain", required=True, choices=("current", "film-v1"))
    command = sub.add_parser("scanner-paired")
    command.add_argument("--preset", required=True, choices=("windows-clang-release", "linux-release"))
    command.add_argument("--run-id", required=True)
    command.add_argument("--inputs", required=True, type=Path)
    command.add_argument("--mode", required=True, choices=("check", "measure"))
    command.add_argument("--workloads", required=True, nargs="+", choices=SCANNER_PAIRED_ROWS)
    command.add_argument("--check-result", type=Path)
    command.add_argument("--check-arm", choices=("spectral", "cubic"), help="qualify one held tail in check mode")
    command = sub.add_parser("prepare-reference")
    for option in ("preset", "revision", "source-dir", "build-dir", "overlay", "run-id"):
        command.add_argument("--" + option, required=True)
    sub.choices["prepare-reference"].add_argument("--resume-from")
    command = sub.add_parser("report")
    command.add_argument("--runs", nargs=2, required=True)
    command.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.operation == "g3-route":
        g3_route(args)
    elif args.operation == "scanner-capacity":
        scanner_capacity(args)
    elif args.operation == "spectral-seed-pilot":
        spectral_seed_pilot(args)
    elif args.operation == "scanner-paired":
        scanner_paired(args)
    elif args.operation == "report":
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
