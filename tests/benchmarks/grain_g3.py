"""Freeze qualified grain inputs and run the bounded G3 comparison."""

from functools import cache
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess

import execution

from grain_g3_analysis import analyze_row, linear_quantile


PRESETS = ("linux-release", "windows-clang-release")
CHROMA = 0.2406350076198578
BUDGET = 1.05
ORDERS = ("AB", "BA", "AB")
QUALIFIED = {
    "portra-hd-36-default": (2.5, (0, 1)),
    "portra-hd-36-expensive": (4.0, (0, 1)),
    "portra-uhd-36-default": (1.25, (0, 1)),
    "portra-uhd-36-expensive": (2.0, (0, 1)),
    "gold-uhd-36-default": (1.0, (0,)),
    "provia-uhd-36-default": (1.0, (2,)),
    "portra-uhd-36-simple": (0.5, (0,)),
}
UNMATCHED = "portra-uhd-12-default"
ROUTES = {0: "direct", 1: "print", 2: "positive-direct"}
TIME_MODES = ("varied", "unchanged")
OFFSETS = {"integer": 0.0, "half": 0.5}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, data):
    if path.exists():
        raise FileExistsError(path)
    execution.write_json(path, data)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def native_paths(root, preset):
    build = root / "out/build" / preset
    exe = "JuicerGrainPrototype.exe" if os.name == "nt" else "JuicerGrainPrototype"
    return build, build / "tests/bin" / exe, build / "tests/Resources"


def relative(root, path):
    return path.resolve(strict=True).relative_to(root).as_posix()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def observe_device(run, name, smi):
    run.command(
        name,
        [
            smi,
            "--query-gpu=name,uuid,pci.bus_id,driver_version,pstate,power.draw,"
            "power.limit,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,"
            "memory.used",
            "--format=csv",
        ],
    )
    run.command(f"{name}-activity", [smi])


def timing_environment():
    return {key: value for key, value in os.environ.items() if key.upper() != "JUICER_DIAGNOSTICS"}


def receipt_captures(receipt):
    check(
        receipt["status"] == "passed"
        and receipt["schema"] == "grain-g2-research-v1"
        and all(command["returncode"] == 0 and command["drained"] for command in receipt["commands"]),
        "G2 capture receipt is incomplete",
    )
    captures = {capture["path"]: capture for capture in receipt["captures"]}
    check(len(captures) == len(receipt["captures"]), "duplicate G2 capture receipt")
    return captures


def g2_workloads(root, experiment_path, current_path, film_path):
    experiment = read_json(experiment_path)
    film_receipt = read_json(film_path)
    current_receipt = read_json(current_path)
    assets = {}
    experiment_suffix = "/" + relative(root, experiment_path)
    sources = [key for key in film_receipt["input_hashes"] if key.endswith(experiment_suffix)]
    check(len(sources) == 1, "G2 receipt does not identify the provided experiment")
    old_root = PurePosixPath(sources[0][: -len(experiment_suffix)])
    check(experiment["chroma_fraction"] == CHROMA, "unexpected G2 chroma")

    def local(old_path):
        path = root / Path(PurePosixPath(old_path).relative_to(old_root))
        return root / relative(root, path)

    @cache
    def bind(path, expected=None):
        path = path.resolve(strict=True)
        digest = execution.digest(path)
        check(expected is None or digest == expected, f"G2 input identity changed: {path}")
        key = relative(root, path)
        check(assets.setdefault(key, digest) == digest, f"input changed while freezing: {path}")
        return key

    bind(experiment_path, film_receipt["input_hashes"][sources[0]])
    bind(film_path)
    bind(current_path)
    current_captures = receipt_captures(current_receipt)
    film_captures = receipt_captures(film_receipt)
    check(not current_captures.keys() & film_captures.keys(), "duplicated G2 capture across receipts")
    receipts = current_captures | film_captures
    recheck = local(next(iter(film_captures))).parent

    @cache
    def capture(name):
        owner = receipts[name]
        directory = local(name)
        check(owner["status"] == "passed", f"G2 capture failed: {name}")
        for path, digest in owner["inputs"].items():
            bind(local(path), digest)
        bind(directory / "manifest.json", owner["manifest"])
        bind(directory / "frames.f32", owner["frames"])
        manifest = read_json(directory / "manifest.json")
        request = read_json(directory / "request.json")
        check(
            all(manifest.get(key) == value for key, value in request.items())
            and len(manifest["resolved"]) == len(request["requests"]),
            f"G2 manifest disagrees with request: {name}",
        )
        return relative(root, directory), manifest

    rows = {row["id"]: row for row in experiment["rows"]}
    check(rows.keys() == QUALIFIED.keys() | {UNMATCHED}, "unexpected G2 workload inventory")
    workloads = {}
    carriers = {}
    for row_id, (size, routes) in QUALIFIED.items():
        row = rows[row_id]
        frozen_path = recheck / f"{row_id}-frozen.json"
        report_path = recheck / f"{row_id}.json"
        frozen = read_json(frozen_path)
        report = read_json(report_path)
        check(
            report["row"] == row
            and report["disposition"] == "qualified"
            and report["chroma_fraction"] == CHROMA
            and report["frozen"] == frozen
            and frozen["size"] == size,
            f"unexpected G2 match for {row_id}",
        )
        fit_path, fit = capture(frozen["capture"])
        calibration_path, _ = capture(report["current_calibration"])
        provenance = {
            "frozen": bind(frozen_path),
            "report": bind(report_path),
            "fit_film": fit_path,
            "calibration_current": calibration_path,
        }
        check(
            frozen["calibration"]["matched"]
            and all(abs(x) <= 0.02 for x in frozen["calibration"]["strength"])
            and abs(frozen["calibration"]["length"]) <= 0.05,
            f"G2 calibration did not meet the frozen limits: {row_id}",
        )
        fit_keys = ("amounts", "size", "chroma_fraction", "bank", "tables", "footprint")
        for kind in ("evaluation", "fractional"):
            result = report[kind]
            current, current_manifest = capture(result["current_capture"])
            film, film_manifest = capture(result["film_capture"])
            differences = result["residuals"]
            check(
                differences["matched"]
                and all(abs(value) <= 0.02 for value in differences["strength"])
                and abs(differences["length"]) <= 0.05
                and all(film_manifest[key] == fit[key] for key in fit_keys)
                and film_manifest["requests"] == current_manifest["requests"],
                f"G2 {kind} evidence did not qualify {row_id}",
            )
            provenance[f"{kind}_current"] = current
            provenance[f"{kind}_film"] = film

        if routes == (0, 1):
            witness = report["original_current_route_witness"]
            direct_path, direct = capture(witness["direct"])
            print_path, printed = capture(witness["print"])
            check(
                direct["route"] == 0
                and printed["route"] == 1
                and direct["requests"] == printed["requests"]
                and all(
                    {key: value for key, value in a.items() if key != "route"}
                    == {key: value for key, value in b.items() if key != "route"}
                    for a, b in zip(direct["resolved"], printed["resolved"], strict=True)
                )
                and assets[f"{direct_path}/frames.f32"] == assets[f"{print_path}/frames.f32"],
                f"G2 Current direct/print equivalence failed for {row_id}",
            )
            provenance.update(route_direct=direct_path, route_print=print_path)

        stock = experiment["stocks"][row["stock"]]
        width, height = (1920, 1080) if row["geometry"] == "hd-36" else (3840, 2160)
        pitch = f32(36000 / width)
        keys = {
            "stock": row["stock"],
            "look": row["look"],
            "width": width,
            "height": height,
            "long_edge_mm": 36.0,
            "pitch_um": pitch,
            "session_seed": 17,
            "clip_token": 23,
            "fps": 24,
        }
        check(
            all(fit[key] == value for key, value in keys.items())
            and fit["route"] == stock["route"]
            and fit["engine"] == "film"
            and fit["amounts"] == frozen["amounts"]
            and fit["footprint"] == f32(pitch / f32(size))
            and local(fit["bank"]) == (root / stock["bank"]).resolve()
            and local(fit["tables"]) == (root / stock["tables"]).resolve(),
            f"G2 recipe changed for {row_id}",
        )
        bank, tables = relative(root, local(fit["bank"])), relative(root, local(fit["tables"]))
        table_files = {
            key: digest
            for key, digest in assets.items()
            if key.startswith(f"{tables}/footprint-")
            and f32(float(Path(key).stem.removeprefix("footprint-"))) == fit["footprint"]
        }
        check(len(table_files) == 1, f"missing or ambiguous frozen footprint: {row_id}")
        for route in routes:
            carrier_name = f"{row['stock'].replace('_', '-')}-{width}-{ROUTES[route]}"
            carrier = {key: keys[key] for key in ("stock", "width", "height", "long_edge_mm")}
            carrier["route"] = route
            carriers[carrier_name] = carrier
            workload_name = f"{row_id}-{ROUTES[route]}"
            workloads[workload_name] = {
                "g2_row": row_id,
                "g2_evidence": provenance,
                "carrier": carrier_name,
                "current": {**keys, "route": route},
                "film": {
                    "bank": bank,
                    "tables": tables,
                    "size": f32(size),
                    "amounts": frozen["amounts"],
                    "chroma_fraction": CHROMA,
                },
                "amounts_f32": [f32(value) for value in frozen["amounts"]],
                "footprint": fit["footprint"],
                "footprint_table": next(iter(table_files)),
            }

    unmatched_path = recheck / f"{UNMATCHED}.json"
    unmatched = read_json(unmatched_path)
    check(unmatched["disposition"] == "unmatched", "unexpected 12 mm disposition")
    for attempt in unmatched["attempts"]:
        capture(attempt["capture"])
    gaps = {
        UNMATCHED: {"disposition": "unmatched", "reason": unmatched["reason"], "report": bind(unmatched_path)},
        "vision3": {"disposition": "deferred", "reason": "requires a qualified measurement bank"},
        "roi-partial": {
            "disposition": "outside-matched-scope",
            "reason": "Current filtered partial frames unsupported",
        },
    }
    return workloads, carriers, assets, gaps


def expand_rows(workloads, carriers):
    rows = []
    for preset in PRESETS:
        for workload in workloads:
            for time_mode in TIME_MODES:
                for phase, offset in OFFSETS.items():
                    rows.append(
                        {
                            "id": f"{preset}-{workload}-{time_mode}-{phase}",
                            "preset": preset,
                            "workload": workload,
                            "qualification": "qualified",
                            "time_mode": time_mode,
                            "offset": offset,
                            "role": "decision",
                        }
                    )
        for carrier, facts in carriers.items():
            if facts["stock"] == "kodak_portra_400":
                for phase, offset in OFFSETS.items():
                    rows.append(
                        {
                            "id": f"{preset}-{carrier}-off-unchanged-{phase}",
                            "preset": preset,
                            "carrier": carrier,
                            "qualification": "control",
                            "time_mode": "unchanged",
                            "offset": offset,
                            "role": "grain-off",
                        }
                    )
    ids = {row["id"] for row in rows}
    check(
        len(ids) == len(rows) == 104 and sum(row["role"] == "decision" for row in rows) == 88,
        "incorrect G3 decision or control membership",
    )
    return rows


def freeze_inputs(root, args, run, binary, resources, smi):
    target = args.input.resolve()
    check(not target.exists(), "frozen G3 manifest already exists")
    workloads, carriers, assets, gaps = g2_workloads(
        root, args.experiment.resolve(), args.current_receipt.resolve(), args.film_receipt.resolve()
    )
    rows = expand_rows(workloads, carriers)
    data = {
        "schema": "grain-g3-workloads-v1",
        "budget": BUDGET,
        "pair_orders": ORDERS,
        "frame_indices": {"warmup": list(range(-10, 0)), "measurement": list(range(30)), "unchanged": 0},
        "workloads": workloads,
        "rows": rows,
        "coverage": gaps,
        "carriers": carriers,
        "g2_inputs": assets,
        "capture_receipt": (run.directory / "result.json").relative_to(root).as_posix(),
        "input_sequence": "fixed-image; frame time and seeds only vary",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with execution.OutputLease(root):
        observe_device(run, "device-before", smi)
        try:
            for name, geometry in carriers.items():
                directory = run.directory / name
                directory.mkdir()
                output = target.parent / f"{name}.f32"
                check(not output.exists(), f"carrier output already exists: {output}")
                request = directory / "request.json"
                save(request, {**geometry, "output": str(output)})
                run.command(f"develop-{name}", [binary, "develop-g3", resources, request], env=timing_environment())
                metadata = directory / "develop.json"
                resolved = read_json(metadata)
                expected_bytes = geometry["width"] * geometry["height"] * 12
                check(
                    resolved["resolved"]["route"] == geometry["route"]
                    and resolved["resolved"]["bytes"] == output.stat().st_size == expected_bytes,
                    f"incorrect carrier development: {name}",
                )
                geometry.update(
                    file=relative(root, output),
                    sha256=execution.digest(output),
                    metadata=relative(root, metadata),
                    metadata_sha256=execution.digest(metadata),
                )
        finally:
            observe_device(run, "device-after", smi)
    save(target, data)
    return {
        "manifest": relative(root, target),
        "manifest_sha256": execution.digest(target),
        "captured": len(carriers),
        "binary_sha256": execution.digest(binary),
    }


def manifest_rows(root, path, resources):
    manifest = read_json(path)
    check(
        manifest["schema"] == "grain-g3-workloads-v1"
        and manifest["budget"] == BUDGET
        and tuple(manifest["pair_orders"]) == ORDERS
        and manifest["rows"] == expand_rows(manifest["workloads"], manifest["carriers"]),
        "G3 workload manifest membership changed",
    )
    frozen = {path: execution.digest(path)}
    for key, digest in manifest["g2_inputs"].items():
        item = root / key
        if "/tests/Resources/profiles/" in key:
            item = resources / "profiles" / item.name
        check(execution.digest(item) == digest, f"frozen G2 identity mismatch: {item}")
        frozen[item] = digest
    for carrier in manifest["carriers"].values():
        for key in ("file", "metadata"):
            item = root / carrier[key]
            expected = carrier["sha256" if key == "file" else "metadata_sha256"]
            check(execution.digest(item) == expected, f"frozen carrier input mismatch: {item}")
            frozen[item] = expected
    return manifest, frozen


def stage_request(root, manifest, row, engine, identity):
    if row["role"] == "grain-off":
        carrier = manifest["carriers"][row["carrier"]]
        request = {key: carrier[key] for key in ("stock", "width", "height", "long_edge_mm", "route")}
        request.update(look="off", pitch_um=f32(36000 / carrier["width"]), session_seed=17, clip_token=23, fps=24)
    else:
        workload = manifest["workloads"][row["workload"]]
        carrier = manifest["carriers"][workload["carrier"]]
        request = dict(workload["current"])
        if engine == "film":
            request.update(workload["film"])
            request.update(bank=str(root / request["bank"]), tables=str(root / request["tables"]))
    return {
        **request,
        "engine": engine,
        "input": str(root / carrier["file"]),
        "offset": row["offset"],
        "time_mode": row["time_mode"],
        "invocation": identity,
    }


def invocation(root, run, binary, resources, manifest, row, identity, engine, *, operation="stage", verification=None):
    name = f"{row['id']}-{identity}"
    path = run.directory / name
    path.mkdir()
    request = stage_request(root, manifest, row, engine, identity)
    if verification:
        request.update(verification)
    request_path = path / "request.json"
    save(request_path, request)
    run.command(name, [binary, operation, resources, request_path], env=timing_environment())
    report_path = path / f"{operation}.json"
    report = read_json(report_path)
    check(report["request"] == request, f"native request identity mismatch: {name}")
    if operation == "stage":
        validate_samples(row, report)
    return {
        **report,
        "invocation": identity,
        "engine": engine,
        "request": request,
        "report": report_path.relative_to(run.directory).as_posix(),
        "sha256": execution.digest(report_path),
    }


def verify_stage(root, args, run, binary, resources, smi, result):
    manifest, frozen = manifest_rows(root, args.input.resolve(), resources)
    result.update(
        manifest=relative(root, args.input),
        manifest_sha256=frozen[args.input.resolve()],
        binary=relative(root, binary),
        binary_sha256=execution.digest(binary),
        checks={},
        film_variant=args.film_variant,
        film_block=args.film_block,
    )
    with execution.OutputLease(root):
        observe_device(run, "device-before", smi)
        try:
            for row in manifest["rows"]:
                if row["preset"] != args.preset or row["time_mode"] != "unchanged" or row["offset"] != 0:
                    continue
                engines = ("current", "film") if row["role"] == "decision" else ("current",)
                for engine in engines:
                    if row["role"] == "decision":
                        arm = verify_g2_stage(
                            root, run, binary, resources, manifest, row, engine, args.film_variant, args.film_block
                        )
                        initial_times = [-1, -0.5, 16, 16.5]
                    else:
                        arm = invocation(
                            root, run, binary, resources, manifest, row, engine, engine, operation="stage-check"
                        )
                        initial_times = [-1, -0.5, 0, 0.5]
                    samples = arm["samples"]
                    check(
                        [sample["time"] for sample in samples] == initial_times + initial_times[::-1],
                        "stage-check membership changed",
                    )
                    check(
                        all(
                            a["resolved"] == b["resolved"]
                            for a, b in zip(samples[:4], reversed(samples[4:]), strict=True)
                        ),
                        "stage-check order changed resolved fields",
                    )
                    check(arm["replay_bit_mismatches"] == 0, "stage-check bitwise replay disagrees")
                    result["checks"][f"{row['id']}-{engine}"] = {
                        "report": arm["report"],
                        "sha256": arm["sha256"],
                        "comparisons": arm.get("comparisons", []),
                    }
        finally:
            observe_device(run, "device-after", smi)
    check(len(result["checks"]) == 26, "incorrect stage-check membership")
    check(execution.digest(binary) == result["binary_sha256"], "G3 executable changed during verification")
    check(
        all(execution.digest(path) == digest for path, digest in frozen.items()),
        "G3 inputs changed during verification",
    )


def verify_g2_stage(root, run, binary, resources, manifest, row, engine, variant="reference", block="32x4"):
    workload = manifest["workloads"][row["workload"]]
    evidence = workload["g2_evidence"]
    references = [root / evidence[f"{kind}_{engine}"] for kind in ("evaluation", "fractional")]
    first = read_json(references[0] / "manifest.json")
    width, height = first["width"], first["height"]
    crop_w, crop_h = first["crop_width"], first["crop_height"]
    output = run.directory / f"{row['id']}-{engine}-verification"
    output.mkdir()
    stimulus = first["requests"][0]["density"]
    verification = {
        "verification": str(output),
        "crop_width": crop_w,
        "crop_height": crop_h,
        "verification_times": [-1, -0.5, 16, 16.5],
    }
    if engine == "film":
        film = workload["film"]
        verification.update(film_variant=variant, film_block=block)
        check(f"{film['bank']}/footprint-1.f32" in manifest["g2_inputs"], "missing frozen alternate table")
        verification.update(
            edit_tables=str(root / film["bank"]),
            edit_pitch=1.0,
            edit_amounts=[1.0, 1.0, 1.0],
            edit_time=17.5,
        )
    # Consume the exact G2 uniform density and crop for agreement. The
    # timed carrier is the separately frozen route-derived image.
    path = output / "flat-cmy.f32"
    with path.open("xb") as stream:
        for value in stimulus:
            line = struct.pack("<f", value) * width
            for _ in range(height):
                stream.write(line)
    arm = invocation(
        root,
        run,
        binary,
        resources,
        manifest,
        row,
        engine,
        engine,
        operation="stage-check",
        verification={**verification, "input": str(path)},
    )
    comparisons = []
    for index, reference in enumerate(references, 2):
        metadata = read_json(reference / "manifest.json")
        sample = arm["samples"][index]
        (reference_index,) = [
            i
            for i, request in enumerate(metadata["requests"])
            if request["time"] == sample["time"] and request["density"] == stimulus
        ]
        expected = metadata["resolved"][reference_index]
        check(
            all(sample["resolved"][key] == value for key, value in expected.items() if key != "route"),
            f"G3 resolved values differ from G2 for {row['id']} {engine} frame {sample['time']}",
        )
        frame_bytes = crop_w * crop_h * 12
        with (reference / "frames.f32").open("rb") as stream:
            stream.seek(reference_index * frame_bytes)
            data = stream.read(frame_bytes)
        observed = output / f"frame-{index}.f32"
        check(
            len(data) == frame_bytes and observed.read_bytes() == data,
            f"G3 stage output differs from G2 for {row['id']} {engine} frame {sample['time']}",
        )
        comparisons.append(
            {
                "time": sample["time"],
                "g2_capture": relative(root, reference),
                "g3_output": relative(root, observed),
                "sha256": execution.digest(observed),
            }
        )
    arm["comparisons"] = comparisons
    return arm


def validate_samples(row, report):
    samples = report["samples"]
    check(len(samples) == 40, "incomplete native stage samples")
    for index, sample in enumerate(samples):
        warmup = index < 10
        base = index - 10 if row["time_mode"] == "varied" else 0
        check(
            (sample["phase"], sample["index"], sample["time"])
            == ("warmup" if warmup else "measurement", index if warmup else index - 10, base + row["offset"]),
            "native stage sample identity disagrees with the frozen sequence",
        )
        resolved = sample["resolved"]
        if row["role"] == "grain-off":
            check(resolved["active"] is False, "grain-off frame entered active dispatch")
        else:
            check(
                resolved["frame_index"] == base and resolved["fraction"] == row["offset"],
                "native seed resolution disagrees with requested time",
            )
    if row["role"] == "grain-off":
        check(report["changed"] == [0, 0, 0], "grain-off control changed the carrier")
    else:
        check(all(value > 0 for value in report["changed"]), "active stage omitted one or more records")
    check(report["replay_bit_mismatches"] == 0, "replayed frame disagrees")


def stage_block(root, run, binary, resources, manifest, row, block, variant="reference", launch="32x4"):
    pairs = []
    for pair_index, order in enumerate(ORDERS, 1):
        pair = {"order": order.lower()}
        for arm in order:
            engine = "film" if block == "candidate" and arm == "B" else "current"
            identity = f"{block}-{pair_index}-{arm.lower()}"
            settings = {"film_variant": variant, "film_block": launch} if engine == "film" else None
            pair[arm.lower()] = invocation(
                root, run, binary, resources, manifest, row, identity, engine, verification=settings
            )
        pairs.append(pair)
    return pairs


def completed(command):
    return command["returncode"] == 0 and command["drained"] is True and command["process_status"] == "completed"


def film_selection(report):
    selection = (report.get("film_variant", "reference"), report.get("film_block", "32x4"))
    variant, block = selection
    check(
        variant in {"reference", "scalar", "reuse", "paired", "quad"}
        and block in {"32x4", "16x8", "32x8"}
        and (variant != "quad" or block == "32x8"),
        "invalid G3 Film selection",
    )
    return selection


def validate_stage_check(verification, preset, manifest_hash, binary_hash, selection):
    check(
        verification["schema"] == "grain-g3-native-v1"
        and verification["mode"] == "g3-check"
        and verification["status"] == "passed"
        and verification["preset"] == preset
        and verification["manifest_sha256"] == manifest_hash
        and verification["binary_sha256"] == binary_hash
        and film_selection(verification) == selection
        and len(verification["checks"]) == 26
        and bool(verification["commands"])
        and all(completed(command) for command in verification["commands"]),
        "G3 stage requires a completed check of the exact executable and frozen manifest",
    )


def stage_agreement(root, path, manifest, preset, binary_hash, manifest_hash, selection):
    from grain_optimization import validate_agreement

    evidence = read_json(path)
    check(
        bool(evidence["commands"]) and all(completed(command) for command in evidence["commands"]),
        "optimized stage agreement did not complete",
    )
    return validate_agreement(root, path, manifest, preset, binary_hash, manifest_hash, *selection, evidence=evidence)


def stage_campaign(root, args, run, binary, resources, smi, result):
    manifest, frozen = manifest_rows(root, args.input.resolve(), resources)
    binary_hash = execution.digest(binary)
    manifest_hash = frozen[args.input.resolve()]
    check_path = args.check_result.resolve()
    frozen[check_path] = execution.digest(check_path)
    selection = (args.film_variant, args.film_block)
    result.update(
        {
            "manifest": relative(root, args.input),
            "manifest_sha256": frozen[args.input.resolve()],
            "binary": relative(root, binary),
            "binary_sha256": binary_hash,
            "check_result": relative(root, check_path),
            "check_sha256": frozen[check_path],
            "rows": {},
            "controls": {},
            "film_variant": args.film_variant,
            "film_block": args.film_block,
            "identities_verified": False,
        }
    )
    validate_stage_check(read_json(check_path), args.preset, manifest_hash, binary_hash, selection)
    if selection != ("reference", "32x4"):
        check(args.agreement_result is not None, "optimized G3 requires --agreement-result")
        agreement_path = args.agreement_result.resolve()
        frozen[agreement_path] = execution.digest(agreement_path)
        agreement = stage_agreement(root, agreement_path, manifest, args.preset, binary_hash, manifest_hash, selection)
        result.update(agreement_result=relative(root, agreement_path), agreement_sha256=frozen[agreement_path])
        for member in agreement["observations"].values():
            path = agreement_path.parent / member["report"]
            frozen[path] = member["sha256"]
    with execution.OutputLease(root):
        observe_device(run, "device-before", smi)
        try:
            for row in manifest["rows"]:
                if row["preset"] != args.preset:
                    continue
                print(row["id"], flush=True)
                destination = run.directory / f"{row['id']}.json"
                measured = dict(row)
                try:
                    if row["role"] == "grain-off":
                        arm = invocation(root, run, binary, resources, manifest, row, "control", "current")
                        result["controls"][row["id"]] = {
                            "report": arm["report"],
                            "sha256": arm["sha256"],
                            "disposition": "descriptive",
                        }
                        continue
                    measured["baseline_pairs"] = stage_block(root, run, binary, resources, manifest, row, "baseline")
                    assessment = analyze_row(measured)
                    save(destination.with_suffix(".baseline.json"), assessment)
                    if assessment["decision"].get("candidate_admitted"):
                        measured["candidate_pairs"] = stage_block(
                            root, run, binary, resources, manifest, row, "candidate", args.film_variant, args.film_block
                        )
                    assessment = analyze_row(measured)
                except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as error:
                    execution.OutputLease.check_held()
                    assessment = {**measured, "decision": {"disposition": "invalid", "reason": str(error)}}
                save(destination, assessment)
                group = "controls" if row["role"] == "grain-off" else "rows"
                result[group][row["id"]] = {
                    "report": destination.name,
                    "sha256": execution.digest(destination),
                    "disposition": assessment["decision"]["disposition"],
                }
        finally:
            observe_device(run, "device-after", smi)
    check(execution.digest(binary) == binary_hash, "G3 executable changed during stage measurement")
    check(
        all(execution.digest(path) == digest for path, digest in frozen.items()),
        "G3 inputs changed during stage measurement",
    )
    check(
        all(completed(command) for command in run.commands)
        and all(item["disposition"] != "invalid" for group in ("rows", "controls") for item in result[group].values()),
        "G3 stage has failed commands or invalid rows",
    )
    result["identities_verified"] = True


def run_native(root, args, parser):
    if args.input is None or args.preset is None or args.run_id is None:
        parser.error(f"{args.mode} requires --input, --preset and --run-id")
    if args.preset.startswith("windows") != (os.name == "nt"):
        parser.error("use the preset's native platform")
    if args.mode == "g3-inputs" and not (args.experiment and args.current_receipt and args.film_receipt):
        parser.error("g3-inputs requires --experiment, --current-receipt and --film-receipt")
    if args.mode == "g3" and not args.check_result:
        parser.error("g3 requires --check-result")
    build, binary, resources = native_paths(root, args.preset)
    directory = root / "out/validation" / args.preset / "results" / execution.name_component(args.run_id)
    smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    with execution.OutputLease(root, build):
        directory.mkdir(parents=True)
        run = execution.NativeRun(root, directory)
        report = {
            "schema": "grain-g3-native-v1",
            "mode": args.mode,
            "preset": args.preset,
            "commands": run.commands,
            "status": "failed",
        }
        try:
            if args.mode == "g3-inputs":
                report.update(freeze_inputs(root, args, run, binary, resources, smi))
            elif args.mode == "g3-check":
                verify_stage(root, args, run, binary, resources, smi, report)
            else:
                stage_campaign(root, args, run, binary, resources, smi, report)
            report["status"] = "passed"
        except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as error:
            report["error"] = str(error)
            raise
        finally:
            run.publish(report)
    print(directory / "result.json")


def load_member(result_path, member):
    path = result_path.parent / member["report"]
    check(execution.digest(path) == member["sha256"], f"G3 result member changed: {path}")
    return read_json(path)


def distributions(values):
    check(all(math.isfinite(value) and value >= 0 for value in values), "nonfinite or negative measurement")
    return {name: linear_quantile(values, q) for name, q in (("median", 0.5), ("p90", 0.9), ("p95", 0.95))}


def stage_observations(arm):
    samples = arm["samples"][10:]
    metrics = ("gpu_ms", "host_prepare_ms", "host_submit_ms", "host_wait_ms", "reset_ms")
    observations = {metric: distributions([sample[metric] for sample in samples]) for metric in metrics} | {
        key: arm[key]
        for key in (
            "cold_setup_ms",
            "cold_first_completed_ms",
            "input_read_ms",
            "bank_read_ms",
            "table_read_ms",
            "bank_upload_ms",
            "carrier_allocation_upload_ms",
            "input_bytes",
            "carrier_bytes",
            "host_input_bytes",
            "device_bank_bytes",
            "film_table_bytes",
            "current_workspace_bytes",
        )
    }
    if arm["request"]["engine"] == "film":
        observations["film_variant"], observations["film_block"] = film_selection(arm["request"])
    observations.update(
        {
            key: arm[key]
            for key in (
                "device_quad_bytes",
                "device_sat_bytes",
                "quad_host_bytes",
                "quad_pack_source_and_staging_bytes",
                "quad_pack_ms",
                "quad_upload_ms",
            )
            if key in arm
        }
    )
    return observations


def collect_routes(paths, manifest, manifest_hash):
    expected = {row["id"]: row for row in manifest["rows"] if row["role"] == "decision"}
    routes = {}
    receipts = []
    for path in paths:
        result = read_json(path)
        check(
            result["operation"] == "g3-route"
            and result["mode"] == "measure"
            and result["manifest_sha256"] == manifest_hash
            and all(command["drained"] for command in result["commands"]),
            f"unexpected Current route evidence: {path}",
        )
        receipts.append({"report": str(path), "sha256": execution.digest(path), "status": result["status"]})
        if result["status"] != "passed" or not all(completed(command) for command in result["commands"]):
            continue
        for member in result["samples"]:
            row_id = member["row"]
            arm = member["workload"].split("-")[5]
            check(row_id in expected and result["preset"] == expected[row_id]["preset"], "route row outside manifest")
            if member["status"] != "passed":
                continue
            arms = routes.setdefault(row_id, {})
            check(arm in {"on", "off"} and arm not in arms, "unexpected or duplicate route arm")
            raw = load_member(path, {"report": member["file"], "sha256": member["sha256"]})
            check(raw["workload"] == member["workload"], "route workload report disagrees with receipt")
            cold, *samples = raw["renders"]
            check(len(samples) == 40, "incomplete route sample membership")
            row = expected[row_id]
            times = [base + row["offset"] for base in (range(-10, 30) if row["time_mode"] == "varied" else [0] * 40)]
            check([sample["frame_time"] for sample in samples] == times, "route time membership differs from stage")
            arms[arm] = {
                "report": str(path.parent / member["file"]),
                "sha256": member["sha256"],
                "stream_interval_ms": distributions([sample["stream_interval_ms"] for sample in samples[10:]]),
                "callback_ms": distributions([sample["callback_ns"] / 1e6 for sample in samples[10:]]),
                "cold": cold,
                "driver_owned": raw["driver_owned"],
            }
        if result["status"] == "passed":
            found = {member["row"] for member in result["samples"] if member["status"] == "passed"}
            check(
                found == {key for key, row in expected.items() if row["preset"] == result["preset"]}
                and len(result["samples"]) == 88,
                "complete route campaign has incorrect membership",
            )
    return routes, receipts


def stage_identity(root, path, result, manifest_path, manifest):
    preset = result["preset"]
    _, _, resources = native_paths(root, preset)
    _, frozen = manifest_rows(root, manifest_path, resources)
    manifest_hash = frozen[manifest_path]
    selection = film_selection(result)
    check(result["manifest"] == relative(root, manifest_path), "stage manifest path differs")
    check(
        result.get("identities_verified") is True
        or ("identities_verified" not in result and selection == ("reference", "32x4")),
        "stage input and binary closeout missing",
    )
    check_path = root / result["check_result"]
    check(relative(root, check_path) == result["check_result"], "invalid stage check path")
    check(execution.digest(check_path) == result["check_sha256"], "stage check receipt changed")
    validate_stage_check(read_json(check_path), preset, manifest_hash, result["binary_sha256"], selection)
    if selection != ("reference", "32x4"):
        check("agreement_result" in result and "agreement_sha256" in result, "optimized stage agreement missing")
        agreement_path = root / result["agreement_result"]
        check(relative(root, agreement_path) == result["agreement_result"], "invalid stage agreement path")
        check(execution.digest(agreement_path) == result["agreement_sha256"], "stage agreement receipt changed")
        stage_agreement(root, agreement_path, manifest, preset, result["binary_sha256"], manifest_hash, selection)
    check(path.resolve().is_relative_to(root), "stage receipt is outside the repository")


def read_stage_arm(root, path, result, manifest, row, member, identity, engine, commands):
    from grain_optimization import repository_path, requests_match

    name = f"{row['id']}-{identity}"
    check(member["report"] == f"{name}/stage.json", "stage arm report has the wrong invocation path")
    raw = load_member(path, member)
    expected = stage_request(root, manifest, row, engine, identity)
    actual = raw["request"]
    if engine == "film":
        check(film_selection(actual) == film_selection(result), "stage Film arm selection differs")
        expected.update({key: actual[key] for key in ("film_variant", "film_block") if key in actual})
    check(
        requests_match(root, actual, expected) and read_json(path.parent / name / "request.json") == actual,
        "stage raw request differs from the frozen invocation",
    )
    command = next(commands, None)
    exe = "JuicerGrainPrototype.exe" if result["preset"].startswith("windows") else "JuicerGrainPrototype"
    build = f"out/build/{result['preset']}/tests"
    expected_command = (
        f"{build}/bin/{exe}",
        "stage",
        f"{build}/Resources",
        relative(root, path.parent / name / "request.json"),
    )
    check(
        command is not None
        and command["name"] == name
        and len(command["command"]) == 4
        and tuple(repository_path(root, value) if i != 1 else value for i, value in enumerate(command["command"]))
        == expected_command
        and result["binary"] == expected_command[0]
        and completed(command),
        "stage process command is missing, reordered or mismatched",
    )
    validate_samples(row, raw)
    return raw


def read_stage_row(root, path, result, manifest, row, member, commands):
    if row["role"] == "grain-off":
        check(member["disposition"] == "descriptive", "invalid grain-off disposition")
        read_stage_arm(root, path, result, manifest, row, member, "control", "current", commands)
        return {"disposition": "descriptive"}, {}

    report = load_member(path, member)
    check(
        all(report.get(key) == value for key, value in row.items())
        and member["report"] == f"{row['id']}.json"
        and member["disposition"] == report["decision"]["disposition"]
        and analyze_row(report)["decision"] == report["decision"]
        and report["decision"]["disposition"] != "invalid",
        "stage report row or decision differs from the frozen row",
    )
    observations = {}
    for block in ("baseline", "candidate"):
        key = f"{block}_pairs"
        if key not in report:
            continue
        observations[key] = []
        for index, (order, pair) in enumerate(zip(ORDERS, report[key], strict=True), 1):
            summary = {}
            for arm in order.lower():
                engine = "film" if block == "candidate" and arm == "b" else "current"
                identity = f"{block}-{index}-{arm}"
                invocation_report = pair[arm]
                raw = read_stage_arm(root, path, result, manifest, row, invocation_report, identity, engine, commands)
                check(
                    invocation_report
                    == raw
                    | {
                        "invocation": identity,
                        "engine": engine,
                        "report": invocation_report["report"],
                        "sha256": invocation_report["sha256"],
                    },
                    "stage row differs from raw arm",
                )
                summary[arm] = stage_observations(raw)
            observations[key].append(summary)
    return report["decision"], observations


def offline_report(manifest_path, stage_paths, route_paths, output, *, root):
    manifest_path = manifest_path.resolve(strict=True)
    manifest = read_json(manifest_path)
    manifest_hash = execution.digest(manifest_path)
    check(
        manifest["rows"] == expand_rows(manifest["workloads"], manifest["carriers"]), "G3 manifest membership differs"
    )
    reports = {}
    stage_receipts = []
    selection = None
    measurements = []
    route_rows, route_receipts = collect_routes(route_paths, manifest, manifest_hash)
    for path in stage_paths:
        path = path.resolve(strict=True)
        result = read_json(path)
        preset = result["preset"]
        check(
            preset in PRESETS
            and preset not in reports
            and result["manifest_sha256"] == manifest_hash
            and result["schema"] == "grain-g3-native-v1"
            and result["mode"] == "g3"
            and result["status"] in {"passed", "failed"},
            f"unexpected G3 stage evidence: {path}",
        )
        expected = {
            group: {row["id"] for row in manifest["rows"] if row["preset"] == preset and row["role"] == role}
            for group, role in (("rows", "decision"), ("controls", "grain-off"))
        }
        observed = {group: result.get(group, {}).keys() for group in expected}
        check(
            all(observed[group] <= expected[group] for group in expected),
            f"G3 run has unexpected or incomplete membership: {path}",
        )
        admissible = (
            result["status"] == "passed"
            and all(observed[group] == expected[group] for group in expected)
            and bool(result["commands"])
            and all(completed(command) for command in result["commands"])
        )
        stage_receipts.append(
            {
                "report": str(path),
                "sha256": execution.digest(path),
                "status": result["status"],
                "preset": preset,
                "admitted": admissible,
            }
        )
        if "error" in result:
            stage_receipts[-1]["error"] = result["error"]
        commands = iter(command for command in result["commands"] if command["command"][1:2] == ["stage"])
        if admissible:
            stage_identity(root, path, result, manifest_path, manifest)
            selected = film_selection(result)
            check(selection is None or selection == selected, "G3 presets have incompatible Film selections")
            selection = selected
        reports[preset] = (path, result, admissible, commands)
    for row in manifest["rows"]:
        path, result, admissible, commands = reports.get(row["preset"], (None, None, False, None))
        group = "rows" if row["role"] == "decision" else "controls"
        member = result.get(group, {}).get(row["id"]) if result else None
        item = {**row, "decision": {"disposition": "unrun"}}
        if member:
            item["report"] = str(path.parent / member["report"])
            if admissible:
                item["decision"], observations = read_stage_row(root, path, result, manifest, row, member, commands)
                if observations:
                    item["stage"] = observations
            else:
                load_member(path, member)
                item["decision"] = {"disposition": "invalid", "reason": "stage parent did not complete successfully"}
                item["diagnostic_disposition"] = member["disposition"]
        if row["id"] in route_rows:
            item["current_route"] = route_rows[row["id"]]
            off = item["current_route"].get("off")
            film = item.get("stage", {}).get("candidate_pairs", [])
            if off and film:
                item["route_off_plus_film_stream_ms_estimate"] = [
                    off["stream_interval_ms"]["median"] + pair["b"]["gpu_ms"]["median"] for pair in film
                ]
        measurements.append(item)
    check(
        all(not admissible or next(commands, None) is None for _, _, admissible, commands in reports.values()),
        "stage has unexpected process commands",
    )
    counts = {}
    for row in measurements:
        disposition = row["decision"]["disposition"]
        counts[disposition] = counts.get(disposition, 0) + 1
    save(
        output,
        {
            "schema": "grain-g3-report-v1",
            "manifest_sha256": manifest_hash,
            "rows": measurements,
            "counts": counts,
            "coverage": manifest["coverage"],
            "stage_results": stage_receipts,
            "route_results": route_receipts,
            "route_estimate_interval": (
                "Median route-off stream event interval plus each Film arm's median grain stream event interval. "
                "Both intervals can include host submission gaps. Separate carrier images, process caches and "
                "production-owned route identities apply. Integration staging and synchronization are unmeasured."
            ),
            "all_qualified_decisions_within_budget": counts.get("within_budget") == 88,
            "scope": "qualified 36 mm full-frame stage only",
        },
    )
