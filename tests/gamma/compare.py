#!/usr/bin/env python3
"""Validate a candidate split/fused observation against a frozen bound."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


LEGACY_DIRECT_FUSED_CASE_IDS = {
    "negative-direct",
    "positive-direct",
    "dir-off-direct",
    "spatial-dir-ramp-edge-direct",
}


def fused_observation(
    record: dict[str, Any], label: str, historical: bool
) -> tuple[str, bool, float | None]:
    if not isinstance(record, dict):
        raise RuntimeError(f"{label}: fused capture is not an object")
    case_id = record.get("case")
    if not isinstance(case_id, str) or not case_id:
        raise RuntimeError(f"{label}: missing case")
    if "fused_path_applicable" in record:
        applicable = record["fused_path_applicable"]
        if type(applicable) is not bool:
            raise RuntimeError(f"{label}: fused_path_applicable is not Boolean")
    elif historical and case_id in LEGACY_DIRECT_FUSED_CASE_IDS:
        applicable = True
    else:
        raise RuntimeError(f"{label}: missing fused_path_applicable")

    difference = record.get("fused_maximum_absolute_difference")
    if not applicable:
        if not historical and difference is not None:
            raise RuntimeError(
                f"{label}: non-applicable fused difference must be null"
            )
        if historical and difference is not None:
            try:
                historical_difference = float(difference)
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"{label}: malformed historical non-applicable difference"
                ) from error
            if not math.isfinite(historical_difference):
                raise RuntimeError(
                    f"{label}: historical non-applicable difference is nonfinite"
                )
        return case_id, False, None

    if difference is None:
        raise RuntimeError(f"{label}: missing fused maximum difference")
    try:
        measured = float(difference)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label}: malformed fused maximum difference") from error
    if not math.isfinite(measured) or measured < 0.0:
        raise RuntimeError(
            f"{label}: fused maximum difference must be finite and nonnegative"
        )
    return case_id, True, measured


def compare_fused_case(
    candidate: dict[str, Any],
    frozen: dict[str, Any],
    expected_case: str,
    frozen_sources: list[dict[str, Any]],
    label: str,
) -> tuple[dict[str, Any], list[str]]:
    result: dict[str, Any] = {
        "case": expected_case,
        "frozen_sources": frozen_sources,
    }
    failures: list[str] = []
    try:
        candidate_case, candidate_applicable, candidate_difference = (
            fused_observation(candidate, f"{label}.candidate", False)
        )
        frozen_case, frozen_applicable, frozen_difference = fused_observation(
            frozen, f"{label}.frozen", True
        )
        if candidate_case != expected_case:
            failures.append(
                f"{label}: candidate case {candidate_case!r}, expected {expected_case!r}"
            )
        if frozen_case != expected_case:
            failures.append(
                f"{label}: frozen case {frozen_case!r}, expected {expected_case!r}"
            )
        if candidate_applicable != frozen_applicable:
            failures.append(
                f"{label}: fused applicability changed from {frozen_applicable} "
                f"to {candidate_applicable}"
            )
        result["candidate_applicable"] = candidate_applicable
        result["frozen_applicable"] = frozen_applicable
        if candidate_applicable and frozen_applicable:
            result["candidate_difference"] = candidate_difference
            result["frozen_bound"] = frozen_difference
            if candidate_difference > frozen_difference:
                failures.append(
                    f"{label}: fused difference {candidate_difference:.17g}, "
                    f"frozen bound {frozen_difference:.17g}"
                )
            result["status"] = "passed" if not failures else "failed"
        elif not candidate_applicable and not frozen_applicable:
            result["candidate_difference"] = None
            result["frozen_bound"] = None
            result["status"] = "not-applicable" if not failures else "failed"
        else:
            result["status"] = "failed"
    except RuntimeError as error:
        failures.append(str(error))
        result["status"] = "failed"
    result["failures"] = failures
    return result, failures


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"missing comparison input: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid comparison input {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"comparison input is not an object: {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--case", required=True)
    arguments = parser.parse_args(argv)
    try:
        candidate = load_json(arguments.candidate)
        frozen = load_json(arguments.frozen)
        result, failures = compare_fused_case(
            candidate,
            frozen,
            arguments.case,
            [{"path": str(arguments.frozen.resolve())}],
            arguments.case,
        )
    except RuntimeError as error:
        print(f"Gamma comparison failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
