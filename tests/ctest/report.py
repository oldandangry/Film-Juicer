"""Report a required CTest selection from an existing full-suite JUnit report."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def summarize(junit: Path, inventory: Path) -> dict:
    selection = json.loads(inventory.read_text(encoding="utf-8-sig"))
    if not isinstance(selection, dict) or not isinstance(selection.get("tests"), list):
        raise ValueError("inventory must contain a CTest tests array")
    names = []
    for test in selection["tests"]:
        if not isinstance(test, dict) or not isinstance(test.get("name"), str) or not test["name"]:
            raise ValueError("inventory contains a test without a name")
        names.append(test["name"])
    if not names:
        raise ValueError("required CTest selection is empty")
    if len(names) != len(set(names)):
        raise ValueError("inventory contains duplicate test names")

    root = ET.parse(junit).getroot()
    if root.tag not in {"testsuite", "testsuites"}:
        raise ValueError("report must be a JUnit testsuite or testsuites document")
    cases = {}
    for case in root.iter("testcase"):
        name = case.get("name")
        if not name or name in cases:
            raise ValueError("JUnit report contains missing or duplicate test names")
        cases[name] = case

    failed, skipped, missing = [], [], []
    for name in names:
        case = cases.get(name)
        if case is None:
            missing.append(name)
        elif case.find("failure") is not None or case.find("error") is not None or case.get("status") == "fail":
            failed.append(name)
        elif case.find("skipped") is not None or case.get("status") in {"notrun", "disabled"}:
            skipped.append(name)
        elif case.get("status") != "run":
            raise ValueError(f"JUnit case has no recognized execution status: {name}")

    return {
        "source": "selection extracted from full-suite CTest JUnit; no additional execution",
        "junit": str(junit.resolve()),
        "inventory": str(inventory.resolve()),
        "tests": len(names),
        "passed": len(names) - len(failed) - len(skipped) - len(missing),
        "failed": failed,
        "skipped": skipped,
        "missing": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path,
                        help="CTest --show-only=json-v1 output for the required selection")
    arguments = parser.parse_args()
    try:
        summary = summarize(arguments.junit, arguments.inventory)
    except (OSError, ValueError, ET.ParseError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))
    return int(summary["passed"] != summary["tests"])


if __name__ == "__main__":
    raise SystemExit(main())
