#!/usr/bin/env python3
"""Collect retained matched scatter precision characterization."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
execution = importlib.import_module("execution")
VARIANTS = ("existing", "fp32_direct", "fp32_factored")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inventory(path):
    return {
        item.relative_to(path).as_posix(): execution.digest(item)
        for item in sorted(path.rglob("*"))
        if item.is_file() and item.name != "result.json"
    }


def expected_accuracy(manifest):
    expected = set()
    for row in manifest["rows"]:
        if "pixel_size_um" not in row:
            continue
        name = row["id"]
        if name == "wide-yvv-1024-above":
            expected.add((name, "admission", None, None))
            continue
        stages = ["post_reflection"]
        settings = row["parameters"]
        if settings["scatter_amount"] != 0 and settings["halation_amount"] != 0:
            stages.append("post_scatter")
        expected.update(
            (name, stage, variant, padding) for stage in stages for variant in VARIANTS for padding in (0, 7)
        )
    return expected


def validate_accuracy(report, manifest):
    require(report["schema"] == "scatter-halation-precision-accuracy.v1", "accuracy schema")
    rows = report["records"]
    keys = [(r["id"], r["stage"], r.get("variant"), r.get("padding")) for r in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected_accuracy(manifest), "accuracy membership")
    require(report["existing_gate_pass"] is True, "existing precision failed fixture gate")
    digests = {}
    fixtures = {row["id"]: row for row in manifest["rows"] if "pixel_size_um" in row}
    for row in rows:
        if row["stage"] == "admission":
            require(row["rejected"] is True, "radius rejection")
            continue
        require(len(row["channels"]) == 3 and all(c["count"] > 0 for c in row["channels"]), "RGB metrics")
        require(
            all(
                len(row[key]) == 64 and all(c in "0123456789abcdef" for c in row[key])
                for key in ("input_sha256", "output_sha256")
            ),
            "plane digest",
        )
        fixture = fixtures[row["id"]]
        require(
            row["reference_input_sha256"] == fixture["input"]["sha256"]
            and row["reference_output_sha256"] == fixture[row["stage"]]["sha256"],
            "reference digest",
        )
        key = (row["id"], row["stage"], row["variant"])
        digests.setdefault(key, []).append((row["input_sha256"], row["output_sha256"]))
        if row["variant"] == "existing":
            require(
                row["fixture_gate_pass"] is True and row["existing_ordinary_equal"] is True,
                "existing fixture or ordinary-launch failure",
            )
    require(all(len(values) == 2 and values[0] == values[1] for values in digests.values()), "compact/padded mismatch")
    require(
        all(len({digests[(row, stage, variant)][0][0] for variant in VARIANTS}) == 1 for row, stage, _ in digests),
        "variant source mismatch",
    )
    return {
        variant: sum(not r["fixture_gate_pass"] for r in rows if r.get("variant") == variant) for variant in VARIANTS
    }


def validate_timing(report, width, height):
    require(
        report["schema"] == "scatter-halation-precision-timing.v1" and report["mode"] == "precision-timing",
        "timing schema",
    )
    require((report["width"], report["height"]) == (width, height), "timing extent")
    require(
        report["warmupCount"] == 10 and report["precision_sample_count"] == 30 and report["admittedBytes"] > 0,
        "timing preparation",
    )
    rows = report["precision_rows"]
    keys = [(r["pair"], r["variant"]) for r in rows]
    require(
        len(keys) == 9 and len(set(keys)) == 9 and set(keys) == {(p, v) for p in range(3) for v in VARIANTS},
        "timing membership",
    )
    require(
        [(r["pair"], r["order"], r["variant"]) for r in rows]
        == [
            (pair, position, variant)
            for pair in range(3)
            for position, variant in enumerate(VARIANTS if pair % 2 == 0 else tuple(reversed(VARIANTS)))
        ],
        "timing order",
    )
    require(
        len(report["sourceChecksum"]) == 64 and all(c in "0123456789abcdef" for c in report["sourceChecksum"]),
        "timing source checksum",
    )
    require(bool(report["host_factor_derivation"]) and bool(report["gpu_event_interval"]), "timing interpretation")
    for row in rows:
        require(len(row["stage_samples_ms"]) == len(row["host_samples_ms"]) == 30, "timing sample count")
        require(
            all(math.isfinite(v) and v >= 0 for v in row["stage_samples_ms"] + row["host_samples_ms"]), "invalid timing"
        )
        for clock in ("stage", "host"):
            require(math.isfinite(row[f"cold_{clock}_ms"]) and row[f"cold_{clock}_ms"] >= 0, "invalid first sample")
            samples = sorted(row[f"{clock}_samples_ms"])
            for percentile in (50, 95):
                expected = samples[math.ceil(percentile * len(samples) / 100) - 1]
                require(row[f"{clock}_p{percentile}_ms"] == expected, "timing percentile disagrees with samples")
        require(
            set(row["registers"]) == {"horizontal", "vertical_accumulation", "vertical_core"}
            and all(isinstance(v, int) and v > 0 for v in row["registers"].values()),
            "kernel register attributes",
        )
    return {"source_checksum": report["sourceChecksum"], "recipe_hash": report["descriptorRecipeHash"]}


def collect(args):
    directory = args.bundle.resolve(strict=True)
    result = json.loads((directory / "result.json").read_text())
    require(result["schema"] == "scatter-precision-run.v1" and result["status"] == "passed", "incomplete run")
    require(all(result[key] for key in ("source_unchanged", "fixtures_unchanged", "probe_unchanged")), "changed inputs")
    require(inventory(directory) == result["members"], "bundle member identity")
    expected = (
        ["build-probe", "device-before", "accuracy"]
        + ([] if result["accuracy_only"] else ["timing-1080p", "timing-uhd"])
        + ["device-after"]
    )
    require(
        [c["name"] for c in result["commands"]] == expected
        and all(c["returncode"] == 0 and c["drained"] for c in result["commands"]),
        "command completion",
    )
    inputs = json.loads((directory / "inputs.json").read_text())
    require(execution.digest(Path(inputs["probe"]["retained_path"])) == inputs["probe"]["sha256"], "probe identity")
    for path, digest in inputs["fixtures"].items():
        require(execution.digest(Path(path)) == digest, "fixture identity")
    manifest = json.loads((ROOT / "tests/scatter_halation/parity-fixtures/manifest.json").read_text())
    failures = validate_accuracy(json.loads((directory / "accuracy.json").read_text()), manifest)
    require(failures == result["fixture_failures"], "accuracy summary")
    if not result["accuracy_only"]:
        for name, width, height in (("timing-1080p", 1920, 1080), ("timing-uhd", 3840, 2160)):
            require(
                validate_timing(json.loads((directory / f"{name}.json").read_text()), width, height)
                == result["timings"][name],
                "timing summary",
            )
    print(json.dumps({"status": "passed", "fixture_failures": failures, "bundle": str(directory)}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    return collect(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"Scatter precision failed: {error}", file=sys.stderr)
        raise SystemExit(1)
