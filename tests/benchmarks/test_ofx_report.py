"""Offline OFX measurement evidence validation."""

import json
import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile
import unittest

import ofx


def sample(lane, row):
    renders = []
    phases = [(p, 0) for p in ["cold"] + ["warmup"] * 10 + ["warm"] * 30 + ["changed", "changed-reuse"] * 3]
    phases += [(p, c) for c in (1, 2) for p in ["cycle-cold"] + ["cycle-warmup"] * 10 + ["cycle-end"]]
    counts = {}
    for phase, cycle in phases:
        index = counts.get((phase, cycle), 0)
        counts[phase, cycle] = index + 1
        cold = phase in ("cold", "changed", "cycle-cold")
        renders.append(
            {
                "phase": phase,
                "cycle": cycle,
                "index": index,
                "callback_status": 0,
                "callback_ns": 100,
                "context_preserved": True,
                "output_checked": True,
                "output_valid": True,
                "stream_interval_ms": 1,
                "releases_valid": True,
                "construction": {
                    "starts": int(cold),
                    "pairs": int(cold),
                    "incomplete": 0,
                    "outcome": "success" if cold else "reused",
                    "duration_ns": 10 if cold else None,
                },
            }
        )
    endpoints = [("before-load", -1, 0), ("after-sample", 0, 30), ("after-unload", -1, 0)]
    endpoints += [(n, c, 0) for n in ("after-create", "after-cold", "after-warmups", "after-destroy") for c in range(3)]
    endpoints += [("after-edit", 0, i) for i in range(3)]
    memory = [{"name": n, "cycle": c, "index": i, "rss_bytes": 1, "peak_rss_bytes": 1} for n, c, i in endpoints]
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
        endpoint["observation"] = {
            key: ([0, 0, 0, 0] if key.endswith(("state_bytes", "state_counts")) else 0) for key in keys
        }
        endpoint["observation"].update({key: 1 for key in keys if "_peak_" in key})
        endpoint["observation"].update({"device_id": 0, "context_opaque": 1, "context_epoch": 1})
    return {
        "schema": 1,
        "status": "passed",
        "mode": "measure",
        "lane": lane,
        "workload": row,
        "width": 6048,
        "height": 4032,
        "controls": {"x": [1]},
        "control_types": {"x": "double"},
        "device": {"ordinal": 0, "context_opaque": 1},
        "renders": renders,
        "memory": memory,
        "edits": [{"gamma": g, "status": 0, "edit_ns": 1} for g in (1.1, 1.0, 1.1)],
        "timings": {
            "module_open_ns": 1,
            "load_ns": 1,
            "describe_ns": 1,
            "unload_ns": 1,
            "create_ns": [1] * 3,
            "author_ns": [1] * 3,
            "destroy_ns": [1] * 3,
        },
        "driver_owned": {"device_buffer_bytes": 1, "host_buffer_bytes": 1},
        "lifecycle": {
            "module_load_status": 0,
            "module_unload_status": 0,
            "context_restored": True,
            "module_closed": True,
            "callback_error": False,
            "effects_created": 3,
            "effects_destroyed": 3,
        },
        "drained": {"device_charged_bytes": 0, "device_records": 0, "pinned_upload_pool_physical_bytes": 0},
    }


def production_evidence(preset, module_hash, consumed_path):
    digest = ofx.execution.digest(consumed_path)
    contract = {
        "preset": preset,
        "build_testing": "OFF",
        "test_support": "OFF",
        "exports": ["OfxGetNumberOfPlugins", "OfxGetPlugin"],
        "loaded_module": {"sha256": module_hash},
        "consumed_inputs": {str(consumed_path): {"sha256_before": digest, "sha256_after": digest}},
    }
    contract_bytes = json.dumps(contract).encode()
    snapshot, patch = b"{}", b""

    def member(data):
        return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    result = {
        "status": "passed",
        "preset": preset,
        "production_contract": True,
        "candidate_unchanged": True,
        "commands": [{"drained": True, "returncode": 0, "process_status": "completed"}],
        "members": {
            "production-contract.json": member(contract_bytes),
            "candidate.json": member(snapshot),
            "candidate.patch": member(patch),
        },
    }
    return json.dumps(result).encode(), contract_bytes, snapshot, patch


class MeasurementReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_run(self, preset, *, qualified=False, workloads=None):
        selected = list(ofx.ROWS) if workloads is None else workloads
        directory = self.root / preset
        directory.mkdir()
        members = []
        commands = []
        inputs = {"driver": {"path": "driver", "sha256": "driver-hash"}, "overlay": {"sha256": "overlay-hash"}}
        for side in ofx.SIDES:
            module_hash = "ship-hash" if side == "candidate" and qualified else ofx.SHIPPING[preset][side]
            inputs[side] = {
                "shipping": {"path": side + "-shipping", "sha256": module_hash},
                "observed": {"path": side + "-observed", "sha256": "observer-hash"},
                "resources": {"path": side + "-resources", "files": {"x": "hash"}},
                "observed_resources": {"path": side + "-observed-resources", "files": {"x": "hash"}},
            }
        identities = {"driver": "driver-hash", "overlay": "overlay-hash"}
        for side in ofx.SIDES:
            identities[side] = {
                "shipping": inputs[side]["shipping"]["sha256"],
                "observed": "observer-hash",
                "resources": {"x": "hash"},
                "observed_resources": {"x": "hash"},
            }
        for name, value in (("candidate.patch", ""), ("candidate.json", "{}"), ("inputs.json", json.dumps(inputs))):
            (directory / name).write_text(value)
        original = (
            PureWindowsPath("C:/campaign/run") if preset.startswith("windows") else PurePosixPath("/campaign/run")
        )
        for lane, row, pair, side in (
            (lane, row, pair, side)
            for lane in ofx.LANES
            for row in selected
            for pair in range(3)
            for side in (ofx.SIDES if pair != 1 else ofx.SIDES[::-1])
        ):
            name = f"{lane}-{row}-{pair}-{side}.json"
            record = sample(lane, row)
            resource_key = "resources" if lane == "shipping" else "observed_resources"
            record["resources"] = inputs[side][resource_key]["path"]
            (directory / name).write_text(json.dumps(record))
            process = name.removesuffix(".json") + ".process.json"
            (directory / process).write_text(json.dumps({"status": "completed", "drained": True, "returncode": 0}))
            commands.append(
                {
                    "name": name.removesuffix(".json"),
                    "drained": True,
                    "returncode": 0,
                    "command": [
                        "driver",
                        "--mode",
                        "measure",
                        "--module",
                        inputs[side][lane]["path"],
                        "--resources",
                        inputs[side][resource_key]["path"],
                        "--lane",
                        lane,
                        "--workload",
                        row,
                        "--output",
                        str(original / name),
                    ],
                }
            )
            members.append(
                {"lane": lane, "row": row, "pair": pair, "side": side, "file": name, "drained": True, "returncode": 0}
            )
        evidence = {p.name: ofx.execution.digest(p) for p in directory.iterdir()}
        (directory / "result.json").write_text(
            json.dumps(
                {
                    "run_directory": str(original),
                    "source": {},
                    "commands": commands,
                    "evidence": evidence,
                    "status": "passed",
                    "operation": "measure",
                    "preset": preset,
                    "samples": members,
                    "source_unchanged": True,
                    "identities_unchanged": True,
                    "identities": identities,
                    "workloads": selected,
                }
            )
        )
        return directory

    def test_identity_mismatch(self):
        p = self.root / "file"
        p.write_text("x")
        with self.assertRaises(ValueError):
            ofx.identities({"driver": {"path": str(p), "sha256": "wrong"}})
        production = production_evidence("linux-release", "candidate-hash", p)
        binding = ofx.candidate_qualification(*production, "linux-release", "candidate-hash", check_inputs=True)
        self.assertEqual(binding["module_sha256"], "candidate-hash")
        with self.assertRaises(ValueError):
            ofx.candidate_qualification(*production, "linux-release", "other-hash", check_inputs=True)
        p.write_text("changed")
        with self.assertRaises(ValueError):
            ofx.candidate_qualification(*production, "linux-release", "candidate-hash", check_inputs=True)

    def test_missing_pair(self):
        p = self.make_run("linux-release")
        result = json.loads((p / "result.json").read_text())
        result["samples"].pop()
        (p / "result.json").write_text(json.dumps(result))
        with self.assertRaises(ValueError):
            ofx.report_run(p)

    def test_missing_workload(self):
        p = self.make_run("linux-release")
        result = json.loads((p / "result.json").read_text())
        result["samples"] = [m for m in result["samples"] if m["row"] != ofx.ROWS[0]]
        (p / "result.json").write_text(json.dumps(result))
        with self.assertRaises(ValueError):
            ofx.report_run(p)

    def test_failed_or_undrained_child(self):
        p = self.make_run("linux-release")
        result = json.loads((p / "result.json").read_text())
        result["samples"][0]["drained"] = False
        (p / "result.json").write_text(json.dumps(result))
        with self.assertRaises(ValueError):
            ofx.report_run(p)

    def test_incomplete_memory(self):
        out_of_order = sample("observed", ofx.ROWS[0])
        out_of_order["renders"][0], out_of_order["renders"][1] = out_of_order["renders"][1], out_of_order["renders"][0]
        with self.assertRaises(ValueError):
            ofx.validate_sample(out_of_order, "measure", "observed", ofx.ROWS[0])
        s = sample("observed", ofx.ROWS[0])
        s["memory"][0]["rss_bytes"] = None
        with self.assertRaises(ValueError):
            ofx.validate_sample(s, "measure", "observed", ofx.ROWS[0])

    def test_complete_offline_report(self):
        linux = self.make_run("linux-release", qualified=True)
        with self.assertRaises(ValueError):
            ofx.report_run(linux)
        unqualified = json.loads((linux / "result.json").read_text())
        unqualified["candidate_qualification"] = None
        (linux / "result.json").write_text(json.dumps(unqualified))
        with self.assertRaises(ValueError):
            ofx.report_run(linux)
        consumed = self.root / "consumed"
        consumed.write_text("source")
        production = production_evidence("linux-release", "ship-hash", consumed)
        result = json.loads((linux / "result.json").read_text())
        names = (
            "candidate-production-result.json",
            "candidate-production-contract.json",
            "candidate-production-source.json",
            "candidate-production-source.patch",
        )
        for name, data in zip(names, production):
            (linux / name).write_bytes(data)
            result["evidence"][name] = hashlib.sha256(data).hexdigest()
        result["candidate_qualification"] = ofx.candidate_qualification(
            *production, "linux-release", "ship-hash", check_inputs=True
        )
        (linux / "result.json").write_text(json.dumps(result))
        windows = self.make_run("windows-clang-release")
        report = ofx.offline_report([linux, windows])
        contract_file = linux / "candidate-production-contract.json"
        contract_file.write_bytes(contract_file.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            ofx.offline_report([linux, windows])
        self.assertEqual(len(report["comparisons"]), 2)
        self.assertEqual(len(report["comparisons"]["linux-release"]["rows"]), 8)
        for path in (linux, windows):
            result = json.loads((path / "result.json").read_text())
            result["workloads"] = ["negative-print-grain"]
            result["commands"] = [
                c
                for c in result["commands"]
                if c["name"].startswith(("shipping-negative-print-grain-", "observed-negative-print-grain-"))
            ]
            result["samples"] = [m for m in result["samples"] if m["row"] == "negative-print-grain"]
            (path / "result.json").write_text(json.dumps(result))
        contract_file.write_bytes(production[1])
        focused = ofx.offline_report([linux, windows])
        self.assertEqual(len(focused["comparisons"]["linux-release"]["rows"]), 2)
        result["workloads"] = ["negative-print-grain", "negative-print-grain"]
        (windows / "result.json").write_text(json.dumps(result))
        with self.assertRaises(ValueError):
            ofx.offline_report([linux, windows])


if __name__ == "__main__":
    unittest.main()
