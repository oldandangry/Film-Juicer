"""Prepare a lossless A5 fixture-format candidate from an accepted Git revision."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = "tests/ffi/fixtures/exposure/"
PRESETS = ("linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release")
INPUT_FIELDS = frozenset("id linear illuminant reference window uv ir method apply_window reference_valid active".split())
EXPECTED_FIELDS = frozenset("built expected scale hash hash_failure_index failure".split())
REFERENCE_FIELDS = frozenset("source source_index source_value white blur built expected diagnostic".split())
BLOCK_FIELDS = ("scanner_tables", "exposure_tables", "sensitivity", "final_sensitivity")


def encode(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def compact_expected_blocks(fixtures: dict[str, dict], blocks_name: str) -> tuple[dict[str, dict], dict]:
    """Share exact expected arrays within one frozen fixture family."""
    blocks = {"schema_version": 1, **{field: [] for field in BLOCK_FIELDS}}
    indices = {field: {} for field in BLOCK_FIELDS}
    candidates = {}
    for name, original in fixtures.items():
        candidate = json.loads(json.dumps(original))
        for group in ("recipes", "products"):
            for row in candidate.get(group, []):
                expected = row.get("expected", {})
                for field in BLOCK_FIELDS:
                    if field not in expected:
                        continue
                    value = expected.pop(field)
                    key = encode(value)
                    if key not in indices[field]:
                        indices[field][key] = len(blocks[field])
                        blocks[field].append(value)
                    expected[field + "_record"] = indices[field][key]
        candidate["expected_blocks"] = blocks_name
        candidates[name] = candidate
        if expand_expected_blocks(candidate, blocks) != original:
            raise ValueError(f"expected block expansion changed capture: {name}")
    return candidates, blocks


def expand_expected_blocks(fixture: dict, blocks: dict) -> dict:
    if blocks["schema_version"] != 1:
        raise ValueError("unsupported shared expected-block schema")
    expanded = json.loads(json.dumps(fixture))
    expanded.pop("expected_blocks")
    for group in ("recipes", "products"):
        for row in expanded.get(group, []):
            expected = row.get("expected", {})
            for field in BLOCK_FIELDS:
                reference = field + "_record"
                if reference in expected:
                    index = expected.pop(reference)
                    if type(index) is not int or not 0 <= index < len(blocks[field]):
                        raise ValueError("invalid expected block reference")
                    expected[field] = copy.deepcopy(blocks[field][index])
    return expanded


def numerical_fixture(fixture: dict) -> dict:
    return {
        "parent": fixture["parent"],
        "contract": fixture["contract"],
        "reference": [{k: v for k, v in row.items() if k in REFERENCE_FIELDS} for row in fixture["reference"]],
        "sensitivity": [{
            "input": {k: v for k, v in row["input"].items() if k in INPUT_FIELDS},
            "expected": {k: v for k, v in row["expected"].items() if k in EXPECTED_FIELDS},
        } for row in fixture["sensitivity"]],
    }


def compact_products(fixtures: dict[str, dict]) -> tuple[dict[str, dict], list]:
    records = []
    indices = {}
    candidates = {}
    for preset, original in fixtures.items():
        candidate = json.loads(json.dumps(original))
        for row in candidate["products"]:
            if "expected" not in row:
                continue
            expected = row.pop("expected")
            key = encode(expected)
            if key not in indices:
                indices[key] = len(records)
                records.append(expected)
            row["expected_record"] = indices[key]
        candidates[preset] = candidate
    for preset, candidate in candidates.items():
        expanded = json.loads(json.dumps(candidate))
        for row in expanded["products"]:
            if "expected_record" in row:
                row["expected"] = records[row.pop("expected_record")]
        if expanded != fixtures[preset]:
            raise ValueError(f"complete product changed during compaction: {preset}")
    return candidates, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.relative_to((ROOT / "out/validation").resolve())
    if output.exists():
        raise ValueError("candidate output must be a new directory")
    revision = subprocess.check_output(["git", "rev-parse", args.source_ref], cwd=ROOT, text=True).strip()

    def read(name: str) -> bytes:
        return subprocess.check_output(["git", "show", f"{revision}:{FIXTURES}{name}"], cwd=ROOT)

    original_manifest = read("manifest.json")
    manifest = json.loads(original_manifest)
    numerical = None
    originals = {}
    source_hashes = {}
    for preset in PRESETS:
        for name in (f"{preset}.json", f"products-{preset}.json"):
            data = read(name)
            digest = hashlib.sha256(data).hexdigest()
            if digest != manifest["files"][name]:
                raise ValueError(f"source fixture differs from its manifest: {name}")
            source_hashes[name] = digest
            value = json.loads(data)
            if name.startswith("products-"):
                originals[preset] = value
            else:
                projected = numerical_fixture(value)
                if numerical is not None and projected != numerical:
                    raise ValueError(f"public numerical inputs/expectations differ: {preset}")
                numerical = projected
    products, records = compact_products(originals)
    files = {"numerical.json": encode(numerical), "product-values.json": encode({"schema_version": 1, "expected_records": records})}
    files.update({f"products-{preset}.json": encode(value) for preset, value in products.items()})
    for preset in PRESETS:
        del manifest["files"][f"{preset}.json"]
    manifest["files"].update({name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
    manifest["schema_version"] = 2
    manifest["format_migration"] = {
        "source_commit": revision,
        "source_manifest_sha256": hashlib.sha256(original_manifest).hexdigest(),
        "source_files": source_hashes,
        "verification": "All four public numerical consumer projections identical; every expanded complete-product document equals its source. All cases, asserted expected bits, hashes, reference/product diagnostics and controls retained. Unconsumed intermediate trace/diagnostic fields remain in original captures and the source revision.",
    }
    output.mkdir(parents=True)
    for name, data in files.items():
        (output / name).write_bytes(data)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Verified four numerical cohorts and {sum(len(v['products']) for v in products.values())} complete-product cases; {len(records)} shared result records.")


if __name__ == "__main__":
    main()
