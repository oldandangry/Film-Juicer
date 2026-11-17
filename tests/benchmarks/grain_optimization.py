"""Execute bounded output-preserving grain experiments under native ownership."""

import os
import math
import csv
import io
from pathlib import Path
import shutil
import subprocess
import sys
import importlib
import re
import locale
import hashlib

import execution
import grain_g3 as g3


PROFILE_CASES = (
    ("gold-uhd-36-default-direct", 0.0),
    ("gold-uhd-36-default-direct", 0.5),
    ("portra-uhd-36-default-direct", 0.5),
    ("provia-uhd-36-default-positive-direct", 0.5),
    ("portra-uhd-36-expensive-direct", 0.5),
)
SCREEN_CONTROLS = (
    ("gold-uhd-36-default-direct", 0.0),
    ("portra-uhd-36-default-direct", 0.5),
    ("portra-hd-36-default-direct", 0.5),
    ("provia-uhd-36-default-positive-direct", 0.5),
    ("portra-uhd-36-expensive-direct", 0.5),
    ("portra-uhd-36-simple-direct", 0.0),
    ("portra-uhd-36-simple-direct", 0.5),
)
SECTIONS = (
    "LaunchStats",
    "Occupancy",
    "MemoryWorkloadAnalysis",
    "WarpStateStats",
    "SchedulerStats",
    "ComputeWorkloadAnalysis",
    "InstructionStats",
)
SAT_SECTIONS = SECTIONS + ("MemoryWorkloadAnalysis_Tables", "SourceCounters")
SAT_METRICS = (
    tuple(
        f"l1tex__{counter}_pipe_lsu_mem_{space}_op_{operation}.sum"
        for space, operation in (("global", "ld"), ("global", "st"), ("local", "ld"), ("local", "st"))
        for counter in ("t_requests", "t_sectors")
    )
    + tuple(
        f"l1tex__t_output_wavefronts_pipe_lsu_mem_{space}_op_{operation}.sum"
        for space, operation in (("global", "ld"), ("global", "st"), ("local", "ld"), ("local", "st"))
    )
    + (
        "l1tex__data_pipe_lsu_wavefronts_mem_lg_cmd_read.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_lg_cmd_write.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_lg.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_lg.sum.pct_of_peak_sustained_elapsed",
    )
)


def profile_sections(sections):
    g3.check(tuple(sections) in (SECTIONS, SAT_SECTIONS), "unsupported profiler section set")
    return tuple(sections)


def repository_path(root, value):
    """Compare retained Windows/WSL paths without rewriting the evidence."""
    prefix = str(root).replace("\\", "/").rstrip("/")
    origins = [prefix]
    if re.match(r"^[A-Za-z]:/", prefix):
        origins.append(f"/mnt/{prefix[0].lower()}{prefix[2:]}")
    elif re.match(r"^/mnt/[a-z]/", prefix):
        origins.append(f"{prefix[5].upper()}:{prefix[6:]}")
    path = str(value).replace("\\", "/")
    g3.check(".." not in path.split("/"), "evidence path escapes repository")
    for origin in origins:
        if path.casefold().startswith(origin.casefold() + "/"):
            return path[len(origin) + 1 :]
    raise ValueError("evidence path is outside repository")


def requests_match(root, actual, expected):
    actual, expected = dict(actual), dict(expected)
    for key in ("input", "bank", "tables"):
        if key in expected and key in actual:
            actual[key] = repository_path(root, actual[key])
            expected[key] = repository_path(root, expected[key])
    return actual == expected


def validate_telemetry(directory, member):
    if "device_observations" not in member:
        return  # Earlier source screens retain campaign-level observations only.
    name = Path(member["report"]).parent.name
    expected = {f"{name}-{phase}{suffix}.log" for phase in ("before", "after") for suffix in ("", "-activity")}
    g3.check(set(member["device_observations"]) == expected, "screen telemetry membership differs")
    for name, digest in member["device_observations"].items():
        g3.check(execution.digest(directory / name) == digest, "screen telemetry changed")


def selected_row(manifest, preset, workload, offset):
    (row,) = [
        row
        for row in manifest["rows"]
        if row["preset"] == preset
        and row.get("workload") == workload
        and row["time_mode"] == "unchanged"
        and row["offset"] == offset
    ]
    return row


def profiler_command(profiler, directory, command, sections=SECTIONS):
    sections = profile_sections(sections)
    return [
        profiler,
        "--kernel-name-base",
        "function",
        "--kernel-name",
        "regex:apply_",
        "--launch-skip",
        "10",
        "--launch-count",
        "3",
        "--clock-control",
        "none",
        "--cache-control",
        "none",
        "--replay-mode",
        "kernel",
        "--export",
        directory / "profile",
        *[item for section in sections for item in ("--section", section)],
        *(["--import-source", "on", "--metrics", ",".join(SAT_METRICS)] if sections == SAT_SECTIONS else []),
        *command,
    ]


def native_invocation(
    root,
    run,
    binary,
    resources,
    manifest,
    row,
    identity,
    variant,
    block,
    operation="stage",
    profiler=None,
    sections=SECTIONS,
    baseline=None,
):
    name = f"{row['id']}-{identity}"
    directory = run.directory / name
    directory.mkdir()
    request = g3.stage_request(root, manifest, row, "film", identity)
    request.update(film_variant=variant, film_block=block)
    if baseline is not None and baseline != ("reference", "32x4"):
        request.update(baseline_variant=baseline[0], baseline_block=baseline[1])
    path = directory / "request.json"
    g3.save(path, request)
    command = [binary, operation, resources, path]
    telemetry = []
    smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    if profiler:
        command = profiler_command(profiler, directory, command, sections)
    if operation == "stage" and profiler is None:
        start = len(run.commands)
        g3.observe_device(run, f"{name}-before", smi)
        telemetry.extend(record["name"] for record in run.commands[start:])
    run.command(name, command, env=g3.timing_environment())
    if operation == "stage" and profiler is None:
        start = len(run.commands)
        g3.observe_device(run, f"{name}-after", smi)
        telemetry.extend(record["name"] for record in run.commands[start:])
    report = directory / f"{operation}.json"
    result = g3.read_json(report)
    g3.check(result["request"] == request, "native optimization request differs")
    if operation == "stage":
        g3.validate_samples(row, result)
    return {
        "report": report.relative_to(run.directory).as_posix(),
        "sha256": execution.digest(report),
        "variant": variant,
        "block": block,
        "device_observations": {f"{key}.log": execution.digest(run.directory / f"{key}.log") for key in telemetry},
    }


def profile_origin(path, candidate):
    prior = g3.read_json(path)
    profile_sections(prior["sections"])
    profile_sections(candidate["sections"])
    keys = (
        "schema",
        "mode",
        "preset",
        "binary_sha256",
        "manifest_sha256",
        "variant",
        "block",
        "profiler",
        "profiler_sha256",
        "sections",
    )
    g3.check(
        prior["schema"] == "grain-optimization-v1"
        and prior["mode"] == "g3-profile"
        and prior["status"] == "failed"
        and all(prior[key] == candidate[key] for key in keys)
        and all(c["returncode"] == 0 and c["drained"] for c in prior["commands"]),
        "closed profile receipt differs or has incomplete commands",
    )
    return prior


def profile_captures(root, path, candidate):
    captures, retained = {}, {}
    while path is not None:
        path = path.resolve(strict=True)
        g3.check(path not in retained, "cyclic profile continuation")
        prior = profile_origin(path, candidate)
        retained[path] = execution.digest(path)
        for command in prior["commands"]:
            if "regex:apply_" in command["command"]:
                g3.check(command["name"] not in captures, "profile case was captured repeatedly")
                captures[command["name"]] = (path, prior)
        origin = prior.get("continued_from")
        path = root / repository_path(root, origin["result"]) if origin else None
        if path is not None:
            g3.check(execution.digest(path) == origin["sha256"], "continued profile receipt changed")
    return captures, retained


def retained_profile(root, source, prior, candidate, manifest, row, *, member=None, directory=None):
    capture = source.parent / f"{row['id']}-profile"
    records = [c for c in prior["commands"] if c["name"] == capture.name]
    build = root / "out/build" / candidate["preset"]
    binary = (
        build
        / "tests/bin"
        / ("JuicerGrainPrototype.exe" if candidate["preset"].startswith("windows") else "JuicerGrainPrototype")
    )
    expected = profiler_command(
        prior["profiler"],
        capture,
        [binary, "stage", build / "tests/Resources", capture / "request.json"],
        prior["sections"],
    )

    def identity(command):
        values = list(map(str, command))
        for index in (values.index("--export") + 1, len(values) - 4, len(values) - 2, len(values) - 1):
            values[index] = repository_path(root, values[index])
        return values

    g3.check(
        len(records) == 1 and identity(records[0]["command"]) == identity(expected), "retained profile command differs"
    )
    report, raw = capture / "stage.json", capture / "profile.ncu-rep"
    stage = g3.read_json(report)
    request = g3.stage_request(root, manifest, row, "film", "profile")
    request.update(film_variant=candidate["variant"], film_block=candidate["block"])
    g3.check(requests_match(root, stage["request"], request) and raw.is_file(), "retained profile request differs")
    g3.validate_samples(row, stage)
    if member is not None:
        g3.check(
            repository_path(root, member["capture_receipt"]) == repository_path(root, source)
            and member["hash_binding"] == "readback"
            and (directory / member["report"]).resolve() == report.resolve()
            and execution.digest(report) == member["sha256"]
            and execution.digest(raw) == member["profile_sha256"],
            "retained profile member differs from origin",
        )
    return report, raw


def profile(root, args, run, binary, resources, manifest, result):
    g3.check(args.profiler is not None, "g3-profile requires --profiler")
    profiler = args.profiler.resolve(strict=True)
    sections = SAT_SECTIONS if args.profile_set == "sat" else SECTIONS
    result.update(profiler=str(profiler), profiler_sha256=execution.digest(profiler), sections=list(sections))
    run.command("profiler-version", [profiler, "--version"])
    inventory = run.command("profiler-sections", [profiler, "--list-sections"]).read_text(encoding="utf-8")
    g3.check(all(section in inventory for section in sections), "profiler section inventory is incomplete")
    reader = None
    if sections == SAT_SECTIONS:
        g3.check(args.profile_reader is not None, "SAT profiling requires --profile-reader")
        sys.path.insert(0, str(args.profile_reader.resolve(strict=True)))
        reader = importlib.import_module("ncu_report")
        inventory = run.command(
            "profiler-metric-inventory", [profiler, "--query-metrics-mode", "all", "--devices", "0"]
        ).read_text(encoding=locale.getencoding())
        g3.check(all(name in inventory for name in SAT_METRICS), "SAT metrics unavailable before capture")
    captures, retained = {}, {}
    if args.profile_result:
        path = args.profile_result.resolve(strict=True)
        captures, retained = profile_captures(root, path, result)
        result["continued_from"] = {"result": str(path), "sha256": retained[path]}
    expected = {
        f"{selected_row(manifest, args.preset, workload, offset)['id']}-profile" for workload, offset in PROFILE_CASES
    }
    g3.check(set(captures) <= expected, "retained profile case membership differs")
    for workload, offset in PROFILE_CASES:
        row = selected_row(manifest, args.preset, workload, offset)
        captured = captures.get(f"{row['id']}-profile")
        if captured:
            origin_path, prior = captured
            report, raw = retained_profile(root, origin_path, prior, result, manifest, row)
            member = {
                "report": os.path.relpath(report, run.directory).replace(os.sep, "/"),
                "sha256": execution.digest(report),
                "variant": args.film_variant,
                "block": args.film_block,
                "capture_receipt": str(origin_path),
                "hash_binding": "readback",
            }
            retained.update(
                {
                    report: member["sha256"],
                    raw: execution.digest(raw),
                }
            )
        else:
            member = native_invocation(
                root,
                run,
                binary,
                resources,
                manifest,
                row,
                "profile",
                args.film_variant,
                args.film_block,
                profiler=profiler,
                sections=sections,
            )
        result["observations"][row["id"]] = member
        report = run.directory / Path(member["report"]).parent / "profile.ncu-rep"
        g3.check(report.is_file(), "profiler omitted its raw report")
        log = run.command(
            f"{row['id']}-metrics", [profiler, "--import", report, "--page", "raw", "--csv"], stdout_only=True
        )
        member.update(
            profile=report.relative_to(run.directory).as_posix(),
            profile_sha256=execution.digest(report),
            metrics_log=log.name,
            metrics_sha256=execution.digest(log),
            **csv_metrics(log, args.film_variant, args.film_block),
        )
        if reader is not None:
            member["sat_metrics"] = read_sat_profile(report, reader, result["source_text_sha256"])
    g3.check(
        all(execution.digest(path) == digest for path, digest in retained.items()), "retained profile inputs changed"
    )


def profile_metrics(text, variant, block):
    rows = [row for row in csv.DictReader(io.StringIO(text)) if row.get("ID", "").isdigit()]
    kernel = {
        "reference": "apply_reference",
        "scalar": "apply_scalar",
        "reuse": "apply_reuse",
        "paired": "apply_paired",
        "quad": "apply_quad",
    }[variant]
    x, y = block.split("x")
    g3.check(len(rows) == 3 and len({row["ID"] for row in rows}) == 3, "incomplete profiler kernel membership")
    keys = (
        "launch__registers_per_thread",
        "launch__block_dim_x",
        "launch__block_dim_y",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
        "smsp__warps_eligible.avg.per_cycle_active",
        "l1tex__t_sector_hit_rate.pct",
        "lts__t_sector_hit_rate.pct",
        "derived__local_spilling_requests",
    )
    for row in rows:
        g3.check(
            kernel in row["Kernel Name"] and row["launch__block_dim_x"] == x and row["launch__block_dim_y"] == y,
            "profiled kernel differs from requested variant",
        )
        g3.check(all(row.get(key) for key in keys), "profile omitted required runtime metrics")
        g3.check(all(math.isfinite(float(row[key])) for key in keys), "profile contains nonfinite runtime metrics")
    return [{"id": row["ID"], "kernel": row["Kernel Name"], **{key: row[key] for key in keys}} for row in rows]


def csv_metrics(path, variant, block, encoding=None):
    encoding = encoding or locale.getencoding()
    return {"metrics_encoding": encoding, "metrics": profile_metrics(path.read_text(encoding=encoding), variant, block)}


def pc_values(metric):
    g3.check(metric is not None and metric.has_correlation_ids(), "missing correlated SAT metric")
    ids = metric.correlation_ids()
    g3.check(metric.num_instances() == ids.num_instances(), "SAT PC correlation count differs")
    instances = []
    for index in range(metric.num_instances()):
        # This SDK's has_value(int) resolves the value-kind overload. Validate
        # each instance's numeric kind rather than treating its index as a kind.
        g3.check(
            metric.kind(index)
            in (metric.ValueKind_FLOAT, metric.ValueKind_DOUBLE, metric.ValueKind_UINT32, metric.ValueKind_UINT64),
            "nonnumeric SAT PC sample",
        )
        g3.check(ids.kind(index) in (ids.ValueKind_UINT32, ids.ValueKind_UINT64), "nonnumeric SAT PC address")
        value = metric.as_double(index)
        g3.check(math.isfinite(value) and value >= 0, "invalid SAT PC sample")
        if value > 0:
            instances.append((ids.as_uint64(index), value))
    return instances


def source_text_digest(text):
    return hashlib.sha256(text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")).hexdigest()


def read_sat_profile(path, reader, source_digest):
    report = reader.load_report(str(path))
    g3.check(report.num_ranges() == 1 and report.range_by_idx(0).num_actions() == 3, "SAT capture membership differs")
    kernels = []
    for index in range(3):
        action = report.range_by_idx(0).action_by_idx(index)
        values = {}
        for name in SAT_METRICS:
            metric = action.metric_by_name(name)
            g3.check(metric is not None and metric.has_value(), f"missing SAT metric: {name}")
            value = metric.as_double()
            g3.check(math.isfinite(value), f"nonfinite SAT metric: {name}")
            values[name] = value
        invalid = sorted(name for name, value in values.items() if value < 0)
        g3.check(
            not any("global_op_ld" in name or "cmd_read" in name for name in invalid),
            "invalid required SAT read metric",
        )
        sources = {
            name: content
            for name, content in action.source_files().items()
            if name.replace("\\", "/").endswith("/tests/grain/film_reference.cu") and content
        }
        g3.check(bool(sources), "SAT capture omitted imported sampler source")
        imported = {name: source_text_digest(content) for name, content in sources.items()}
        g3.check(
            all(value == source_digest for value in imported.values()),
            "imported sampler source differs from frozen source",
        )
        hot = {}
        for name in (
            "memory_l1_tag_requests_global",
            "smsp__pcsamp_warps_issue_stalled_long_scoreboard_not_issued",
            "smsp__pcsamp_warps_issue_stalled_lg_throttle_not_issued",
        ):
            metric = action.metric_by_name(name)
            instances = []
            for pc, value in pc_values(metric):
                info = action.source_info(pc)
                instances.append(
                    {
                        "pc": pc,
                        "value": value,
                        "sass": action.sass_by_pc(pc),
                        "source": info.file_name() if info else None,
                        "line": info.line() if info else None,
                    }
                )
            hot[name] = sorted(instances, key=lambda item: (-item["value"], item["pc"]))[:12]
        g3.check(
            any(item["source"] in sources for item in hot["memory_l1_tag_requests_global"]),
            "SAT requests lack sampler source correlation",
        )
        ratios = {}
        for space, operation in (("global", "ld"), ("global", "st"), ("local", "ld"), ("local", "st")):
            requests = values[f"l1tex__t_requests_pipe_lsu_mem_{space}_op_{operation}.sum"]
            ratios[f"{space}_{operation}"] = {
                "sectors_per_request": values[f"l1tex__t_sectors_pipe_lsu_mem_{space}_op_{operation}.sum"] / requests
                if requests > 0 and values[f"l1tex__t_sectors_pipe_lsu_mem_{space}_op_{operation}.sum"] >= 0
                else None,
                "tag_wavefronts_per_request": values[
                    f"l1tex__t_output_wavefronts_pipe_lsu_mem_{space}_op_{operation}.sum"
                ]
                / requests
                if requests > 0 and values[f"l1tex__t_output_wavefronts_pipe_lsu_mem_{space}_op_{operation}.sum"] >= 0
                else None,
            }
        for operation, command in (("ld", "read"), ("st", "write")):
            counts = [
                values[f"l1tex__t_requests_pipe_lsu_mem_{space}_op_{operation}.sum"] for space in ("global", "local")
            ]
            requests = sum(counts)
            wavefronts = values[f"l1tex__data_pipe_lsu_wavefronts_mem_lg_cmd_{command}.sum"]
            ratios[f"combined_local_global_{operation}"] = {
                "data_wavefronts_per_request": wavefronts / requests
                if min(counts) >= 0 and requests > 0 and wavefronts >= 0
                else None
            }
        kernels.append(
            {
                "values": values,
                "ratios": ratios,
                "hot_pc_metrics": hot,
                "imported_source_text_sha256": imported,
                "invalid_counter_values": invalid,
            }
        )
    return kernels


def validate_agreement(root, path, manifest, preset, binary_hash, manifest_hash, variant, block, *, evidence=None):
    if evidence is None:
        evidence = g3.read_json(path)
    expected = {selected_row(manifest, preset, workload, 0.0)["id"] for workload in manifest["workloads"]}
    g3.check(
        evidence["schema"] == "grain-optimization-v1"
        and evidence["status"] == "passed"
        and evidence["mode"] == "g3-verify-opt"
        and evidence["preset"] == preset
        and evidence["binary_sha256"] == binary_hash
        and evidence["manifest_sha256"] == manifest_hash
        and (evidence["variant"], evidence["block"]) == (variant, block)
        and set(evidence["observations"]) == expected
        and all(command["returncode"] == 0 and command["drained"] for command in evidence["commands"]),
        "agreement receipt does not match candidate or workload membership",
    )
    for row_id, member in evidence["observations"].items():
        row = next(row for row in manifest["rows"] if row["id"] == row_id)
        raw = g3.load_member(path, member)
        expected_request = g3.stage_request(root, manifest, row, "film", "verify")
        expected_request.update(film_variant=variant, film_block=block)
        baseline = (evidence.get("baseline_variant", "reference"), evidence.get("baseline_block", "32x4"))
        if baseline != ("reference", "32x4"):
            expected_request.update(baseline_variant=baseline[0], baseline_block=baseline[1])
            g3.check(raw.get("baseline_bit_mismatches") == 0, "agreement baseline output differs")
        g3.check(
            requests_match(root, raw["request"], expected_request)
            and (member["variant"], member["block"]) == (variant, block),
            "agreement report request differs from manifest",
        )
        g3.check(
            raw["bit_mismatches"] == raw["replay_hash_mismatches"] == raw["nonfinite"] == raw["inactive_frames"] == 0
            and raw["frame_identities"] == 80
            and len(raw["observations"]) == 160,
            "agreement report has failed or missing frames",
        )
        for index, sample in enumerate(raw["observations"]):
            if baseline != ("reference", "32x4"):
                g3.check(sample.get("baseline_bit_mismatches") == 0, "agreement baseline frame differs")
            identity = index if index < 80 else 159 - index
            time = identity // 2 - 10 + (identity % 2) * 0.5
            g3.check(
                (sample["identity"], sample["order"], sample["time"])
                == (identity, "forward" if index < 80 else "reverse", time)
                and sample["bit_mismatches"] == sample["nonfinite"] == 0
                and sample["first_mismatch"] is None
                and not sample["replay_hash_mismatch"]
                and all(value > 0 for value in sample["changed"]),
                "agreement frame identity or output guard differs",
            )
    return evidence


def verify_variant(root, args, run, binary, resources, manifest, result):
    g3.check(args.film_variant != "reference" or args.film_block != "32x4", "select an optimized variant to verify")
    result.update(baseline_variant=args.baseline_variant, baseline_block=args.baseline_block)
    for workload in manifest["workloads"]:
        row = selected_row(manifest, args.preset, workload, 0.0)
        member = native_invocation(
            root,
            run,
            binary,
            resources,
            manifest,
            row,
            "verify",
            args.film_variant,
            args.film_block,
            operation="stage-verify",
            baseline=(args.baseline_variant, args.baseline_block),
        )
        report = g3.load_member(run.directory / "result.json", member)
        g3.check(report["bit_mismatches"] == 0 and report["frame_identities"] == 80, "incomplete optimized agreement")
        result["observations"][row["id"]] = member
    validate_agreement(
        root,
        run.directory / "result.json",
        manifest,
        args.preset,
        result["binary_sha256"],
        result["manifest_sha256"],
        args.film_variant,
        args.film_block,
        evidence={**result, "status": "passed"},
    )


def summarize_screen(directory, result, root, manifest):
    """Validate separate process arms and retain per-pair optimized/reference costs."""
    g3.check(result["schema"] == "grain-optimization-v1" and result["mode"] == "g3-screen", "unexpected screen result")
    g3.check(result["status"] == "passed", "incomplete descriptive screen")
    g3.check(len(result["screen_pairs"]) == 3, "screen must contain three pairs")
    g3.check(
        all(command["returncode"] == 0 and command["drained"] for command in result["commands"]),
        "screen has unsuccessful commands",
    )
    identities = set()
    expected_commands = []
    baseline_variant = result.get("baseline_variant", "reference")
    baseline_block = result.get("baseline_block", "32x4")

    def read_arm(member, variant, block, row, invocation):
        identity = member["report"]
        g3.check(identity not in identities, "screen reused a process observation")
        identities.add(identity)
        g3.check((member["variant"], member["block"]) == (variant, block), "screen variant identity differs")
        raw = g3.load_member(directory / "result.json", member)
        request = raw["request"]
        g3.check((request["film_variant"], request["film_block"]) == (variant, block), "native screen variant differs")
        expected_request = g3.stage_request(root, manifest, row, "film", invocation)
        expected_request.update(film_variant=variant, film_block=block)
        g3.check(requests_match(root, request, expected_request), "screen arm request differs from frozen workload")
        if result.get("device_observation_scope") == "per-invocation":
            g3.check("device_observations" in member, "screen omitted per-invocation telemetry")
        validate_telemetry(directory, member)
        g3.validate_samples(row, raw)
        values = [sample["gpu_ms"] for sample in raw["samples"][10:]]
        g3.check(all(math.isfinite(value) and value > 0 for value in values), "invalid descriptive stage time")
        return {label: g3.linear_quantile(values, q) for label, q in (("median", 0.5), ("p90", 0.9), ("p95", 0.95))}

    ratios = []
    gold = selected_row(manifest, result["preset"], "gold-uhd-36-default-direct", 0.5)
    for index, (pair, order) in enumerate(zip(result["screen_pairs"], g3.ORDERS, strict=True), 1):
        g3.check(pair["order"] == order.lower(), "screen pair order differs")
        reference = read_arm(pair["a"], baseline_variant, baseline_block, gold, f"pair-{index}-a")
        optimized = read_arm(pair["b"], result["variant"], result["block"], gold, f"pair-{index}-b")
        expected_commands.extend((f"{gold['id']}-pair-{index}-{arm.lower()}", pair[arm.lower()]) for arm in order)
        ratios.append({key: optimized[key] / reference[key] for key in reference})
    controls = {}
    expected = {selected_row(manifest, result["preset"], name, offset)["id"] for name, offset in SCREEN_CONTROLS}
    g3.check(set(result["observations"]) == expected, "screen control membership differs")
    for name, offset in SCREEN_CONTROLS:
        row = selected_row(manifest, result["preset"], name, offset)
        row_id = row["id"]
        arms = result["observations"][row_id]
        reference = read_arm(arms["reference"], baseline_variant, baseline_block, row, "control-reference")
        optimized = read_arm(arms["optimized"], result["variant"], result["block"], row, "control-optimized")
        expected_commands.extend((f"{row_id}-control-{arm}", arms[arm]) for arm in ("reference", "optimized"))
        controls[row_id] = {
            "reference_ms": reference,
            "optimized_ms": optimized,
            "optimized_over_reference": {key: optimized[key] / reference[key] for key in reference},
        }
    commands = [record for record in result["commands"] if record["command"][1:2] == ["stage"]]
    g3.check(
        [record["name"] for record in commands] == [name for name, _ in expected_commands],
        "screen process order differs",
    )
    for record, (_, member) in zip(commands, expected_commands, strict=True):
        expected = directory / Path(member["report"]).parent / "request.json"
        g3.check(
            len(record["command"]) == 4
            and repository_path(root, record["command"][-1]) == repository_path(root, expected),
            "screen process request differs",
        )
    return {"descriptive_only": True, "gold_half_pair_ratios": ratios, "controls": controls}


def validate_screen_baseline(result, agreement):
    baseline = (result.get("baseline_variant", "reference"), result.get("baseline_block", "32x4"))
    g3.check(
        baseline == ("reference", "32x4")
        or baseline == (agreement.get("baseline_variant"), agreement.get("baseline_block")),
        "screen baseline lacks exact-binary output agreement",
    )


def screen(root, args, run, binary, resources, manifest, result):
    g3.check(args.check_result is not None, "g3-screen requires --check-result")
    result["device_observation_scope"] = "per-invocation"
    result.update(baseline_variant=args.baseline_variant, baseline_block=args.baseline_block)
    g3.check(
        (args.baseline_variant, args.baseline_block) != (args.film_variant, args.film_block),
        "screen requires distinct configurations",
    )
    evidence = validate_agreement(
        root,
        args.check_result.resolve(),
        manifest,
        args.preset,
        execution.digest(binary),
        result["manifest_sha256"],
        args.film_variant,
        args.film_block,
    )
    validate_screen_baseline(result, evidence)
    result["check_result"] = str(args.check_result.resolve())
    result["check_sha256"] = execution.digest(args.check_result)
    agreement_inputs = {args.check_result.resolve(): result["check_sha256"]}
    agreement_inputs.update(
        {args.check_result.parent / member["report"]: member["sha256"] for member in evidence["observations"].values()}
    )
    row = selected_row(manifest, args.preset, "gold-uhd-36-default-direct", 0.5)
    pairs = []
    for index, order in enumerate(g3.ORDERS, 1):
        pair = {"order": order.lower()}
        for arm in order:
            variant = args.baseline_variant if arm == "A" else args.film_variant
            block = args.baseline_block if arm == "A" else args.film_block
            pair[arm.lower()] = native_invocation(
                root,
                run,
                binary,
                resources,
                manifest,
                row,
                f"pair-{index}-{arm.lower()}",
                variant,
                block,
            )
        pairs.append(pair)
    result["screen_pairs"] = pairs
    for workload, offset in SCREEN_CONTROLS:
        row = selected_row(manifest, args.preset, workload, offset)
        result["observations"][row["id"]] = {
            arm: native_invocation(
                root,
                run,
                binary,
                resources,
                manifest,
                row,
                f"control-{arm}",
                variant,
                args.baseline_block if arm == "reference" else args.film_block,
            )
            for arm, variant in (("reference", args.baseline_variant), ("optimized", args.film_variant))
        }
    result["descriptive_only"] = True
    # Publish the same validated arithmetic that offline reporting will reproduce.
    result["summary"] = summarize_screen(run.directory, {**result, "status": "passed"}, root, manifest)
    g3.check(
        all(execution.digest(path) == digest for path, digest in agreement_inputs.items()),
        "optimized agreement changed during screen",
    )


def run_native(root, args, parser):
    if args.input is None or args.preset is None or args.run_id is None:
        parser.error(f"{args.mode} requires --input, --preset and --run-id")
    if args.preset.startswith("windows") != (os.name == "nt"):
        parser.error("use the preset's native platform")
    build, binary, resources = g3.native_paths(root, args.preset)
    directory = root / "out/validation" / args.preset / "results" / execution.name_component(args.run_id)
    with execution.OutputLease(root, build):
        manifest, frozen = g3.manifest_rows(root, args.input.resolve(), resources)
        directory.mkdir(parents=True)
        run = execution.NativeRun(root, directory)
        inputs = [Path(__file__), Path(__file__).with_name("grain.py"), Path(g3.__file__), binary]
        frozen.update({path: execution.digest(path) for path in inputs})
        if args.mode == "g3-profile" and args.profile_set == "sat":
            frozen.update(
                {
                    root / name: execution.digest(root / name)
                    for name in (
                        "tests/grain/film_reference.cu",
                        "tests/grain/film_reference.h",
                        "tests/CMakeLists.txt",
                    )
                }
            )
        result = {
            "schema": "grain-optimization-v1",
            "mode": args.mode,
            "preset": args.preset,
            "variant": args.film_variant,
            "block": args.film_block,
            "status": "failed",
            "manifest": g3.relative(root, args.input),
            "manifest_sha256": frozen[args.input.resolve()],
            "binary_sha256": frozen[binary],
            "commands": run.commands,
            "observations": {},
        }
        if args.mode == "g3-profile" and args.profile_set == "sat":
            result["source_sha256"] = frozen[root / "tests/grain/film_reference.cu"]
            result["source_text_sha256"] = source_text_digest(
                (root / "tests/grain/film_reference.cu").read_text(encoding="utf-8")
            )
        smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
        try:
            with execution.OutputLease(root):
                g3.observe_device(run, "device-before", smi)
                try:
                    operation = {"g3-profile": profile, "g3-screen": screen, "g3-verify-opt": verify_variant}[args.mode]
                    operation(root, args, run, binary, resources, manifest, result)
                finally:
                    g3.observe_device(run, "device-after", smi)
            g3.check(
                all(execution.digest(path) == digest for path, digest in frozen.items()), "optimization inputs changed"
            )
            result["status"] = "passed"
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            result["error"] = str(error)
            raise
        finally:
            run.publish(result)
    print(directory / "result.json")


def read_profile(path, variant, block, reader):
    report = reader.load_report(str(path))
    g3.check(report.num_ranges() == 1, "profile has unexpected ranges")
    scope = report.range_by_idx(0)
    g3.check(scope.num_actions() == 3, "profile has incomplete kernel membership")
    expected = {
        "reference": "apply_reference",
        "scalar": "apply_scalar",
        "reuse": "apply_reuse",
        "paired": "apply_paired",
        "quad": "apply_quad",
    }[variant]
    metrics = (
        "launch__registers_per_thread",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
        "smsp__warps_eligible.avg.per_cycle_active",
        "l1tex__t_sector_hit_rate.pct",
        "lts__t_sector_hit_rate.pct",
        "derived__local_spilling_requests",
        "sass__inst_executed_register_spilling_mem_local_op_read",
        "sass__inst_executed_register_spilling_mem_local_op_write",
        "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__bytes.sum.per_second",
        "smsp__thread_inst_executed_per_inst_executed.ratio",
        "sm__instruction_throughput.avg.pct_of_peak_sustained_elapsed",
    )
    observations = []
    for index in range(3):
        action = scope.action_by_idx(index)
        x, y = (int(value) for value in block.split("x"))
        g3.check(
            expected in action.name()
            and action.metric_by_name("launch__block_dim_x").as_double() == x
            and action.metric_by_name("launch__block_dim_y").as_double() == y,
            "profile action does not match selected kernel",
        )
        values = {name: action.metric_by_name(name).as_double() for name in metrics}
        g3.check(all(math.isfinite(value) for value in values.values()), "invalid profile metric")
        observations.append({"kernel": action.name(), **values})
    return {"profile": str(path), "sha256_at_readback": execution.digest(path), "kernels": observations}


def offline_report(root, args):
    manifest = g3.read_json(args.input)
    manifest_hash = execution.digest(args.input)
    reader = None
    if args.profile_reader:
        sys.path.insert(0, str(args.profile_reader.resolve(strict=True)))
        reader = importlib.import_module("ncu_report")
    retained = []
    for source in args.results:
        source = source.resolve(strict=True)
        result = g3.read_json(source)
        g3.check(
            result["schema"] == "grain-optimization-v1"
            and result["manifest_sha256"] == manifest_hash
            and all(command["drained"] for command in result["commands"]),
            "unexpected optimization evidence",
        )
        item = {
            "result": str(source),
            "sha256": execution.digest(source),
            "mode": result["mode"],
            "preset": result["preset"],
            "variant": result["variant"],
            "block": result["block"],
            "status": result["status"],
        }
        item.update(
            {
                key: result[key]
                for key in ("baseline_variant", "baseline_block", "source_sha256", "source_text_sha256")
                if key in result
            }
        )
        captures = {}
        if "continued_from" in result:
            origin = result["continued_from"]
            original = root / repository_path(root, origin["result"])
            g3.check(execution.digest(original) == origin["sha256"], "continued profile receipt changed")
            captures, _ = profile_captures(root, original, result)
            item["continued_from"] = origin
        if result["status"] == "passed":
            g3.check(
                all(command["returncode"] == 0 for command in result["commands"]), "passed evidence has failed commands"
            )
            if result["mode"] == "g3-verify-opt":
                validate_agreement(
                    root,
                    source,
                    manifest,
                    result["preset"],
                    result["binary_sha256"],
                    manifest_hash,
                    result["variant"],
                    result["block"],
                )
                item.update(frame_identities=880, comparisons_with_reverse=1760, bit_mismatches=0)
                if result.get("baseline_variant") == "scalar":
                    item["baseline_comparisons_with_reverse"] = 1760
            elif result["mode"] == "g3-screen":
                agreement = root / repository_path(root, result["check_result"])
                g3.check(execution.digest(agreement) == result["check_sha256"], "screen agreement receipt changed")
                evidence = validate_agreement(
                    root,
                    agreement,
                    manifest,
                    result["preset"],
                    result["binary_sha256"],
                    manifest_hash,
                    result["variant"],
                    result["block"],
                )
                validate_screen_baseline(result, evidence)
                item["summary"] = summarize_screen(source.parent, result, root, manifest)
                g3.check(item["summary"] == result["summary"], "screen summary differs from original observations")
                members = [pair[arm] for pair in result["screen_pairs"] for arm in ("a", "b")]
                members.extend(member for arms in result["observations"].values() for member in arms.values())
                item["device_observations"] = {
                    str(source.parent / name): digest
                    for member in members
                    for name, digest in member.get("device_observations", {}).items()
                }
            elif result["mode"] == "g3-profile":
                g3.check(reader is not None, "profile evidence requires --profile-reader")
                expected = {
                    selected_row(manifest, result["preset"], name, offset)["id"] for name, offset in PROFILE_CASES
                }
                g3.check(set(result["observations"]) == expected, "profile workload membership differs")
                item["profiles"] = {}
                for row_id, member in result["observations"].items():
                    stage = g3.load_member(source, member)
                    row = next(row for row in manifest["rows"] if row["id"] == row_id)
                    inherited = f"{row_id}-profile" in captures
                    g3.check(inherited == ("capture_receipt" in member), "profile recovery membership differs")
                    if inherited:
                        original, prior = captures[f"{row_id}-profile"]
                        retained_profile(
                            root, original, prior, result, manifest, row, member=member, directory=source.parent
                        )
                    request = g3.stage_request(root, manifest, row, "film", "profile")
                    request.update(film_variant=result["variant"], film_block=result["block"])
                    g3.check(
                        requests_match(root, stage["request"], request)
                        and (member["variant"], member["block"]) == (result["variant"], result["block"]),
                        "profile request differs from workload",
                    )
                    g3.validate_samples(row, stage)
                    raw = source.parent / Path(member["report"]).parent / "profile.ncu-rep"
                    if "profile_sha256" in member:
                        g3.check(execution.digest(raw) == member["profile_sha256"], "raw profiler report changed")
                    if "metrics_log" in member:
                        log = source.parent / member["metrics_log"]
                        g3.check(execution.digest(log) == member["metrics_sha256"], "profiler CSV changed")
                        observed = csv_metrics(log, result["variant"], result["block"], member["metrics_encoding"])[
                            "metrics"
                        ]
                        g3.check(observed == member["metrics"], "profiler CSV metrics differ")
                    item["profiles"][row_id] = read_profile(raw, result["variant"], result["block"], reader)
                    if tuple(profile_sections(result["sections"])) == SAT_SECTIONS:
                        sat = read_sat_profile(raw, reader, result["source_text_sha256"])
                        g3.check(sat == member["sat_metrics"], "SAT profile readback differs")
                        item["profiles"][row_id]["sat_metrics"] = sat
        else:
            item["error"] = result.get("error", "native operation failed")
        retained.append(item)
    g3.save(
        args.output,
        {
            "schema": "grain-optimization-report-v1",
            "budget": g3.BUDGET,
            "manifest_sha256": manifest_hash,
            "experiments": retained,
            "coverage": manifest["coverage"],
            "decision_rows": [
                {"id": row["id"], "disposition": "unrun"} for row in manifest["rows"] if row["role"] == "decision"
            ],
            "scope": "bounded nonshipping optimization; screen observations are descriptive",
        },
    )
