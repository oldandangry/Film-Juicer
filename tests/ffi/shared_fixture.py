"""Read logical captures from the shared fixture manifest representation."""

import json

from compact_exposure_fixtures import expand_expected_blocks


def load_shared_fixture(logical_path, manifest_name="manifest.json"):
    directory = logical_path.parent
    manifest = json.loads((directory / manifest_name).read_bytes())
    name = logical_path.name
    fixture = json.loads((directory / manifest["payloads"][name]).read_bytes())
    if "contexts" in manifest:
        context = json.loads((directory / manifest["contexts"][name]).read_bytes())
        if fixture.keys() & context.keys():
            raise ValueError("shared fixture context overlaps payload")
        fixture.update(context)
    if "expected_blocks" in fixture:
        blocks = json.loads((directory / fixture["expected_blocks"]).read_bytes())
        fixture = expand_expected_blocks(fixture, blocks)
    return fixture


def original_capture_bytes(fixture, line_ending):
    if line_ending not in ("LF", "CRLF"):
        raise ValueError("unsupported original capture line ending")
    raw = (json.dumps(fixture, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()
    return raw if line_ending == "LF" else raw.replace(b"\n", b"\r\n")
