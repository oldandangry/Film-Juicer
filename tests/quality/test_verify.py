"""Actual-edit, pending-failure and explicit scope/qualification controls."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from verify import edit_selection, source_snapshot
from qualification import require_checkpoint

ROOT = Path(__file__).resolve().parents[2]


class SelectionTests(unittest.TestCase):
    def test_unrelated_dirty_payloads_stay_pending_without_content_reads(self):
        import os
        import verify
        from hashlib import sha256
        from build_dependencies import file_identities
        from execution import digest

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            names = ["tests/ffi/fixtures/unowned.json", "Resources/unowned.bin", "tests/ffi/fixtures/selected.json"]
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("first")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Control", "-c", "user.email=control@example.invalid", "commit", "-qm", "initial"], check=True)
            prior = {"observed": source_snapshot(root), "pending": []}
            selected = root / names[2]
            stamp = selected.stat()
            before = file_identities([selected])[str(selected)]["sha256"]
            selected.write_text("other")
            os.utime(selected, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            after = file_identities([selected])[str(selected)]["sha256"]
            self.assertNotEqual(before, after)
            self.assertEqual(after, sha256(b"other").hexdigest())
            selected.write_text("first")
            os.utime(selected, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))

            def read_relevant(path):
                self.assertNotIn(path, [root / name for name in names])
                return digest(path)

            # Git membership must not rely on ctime for a same-size edit: on
            # Windows it can remain unchanged when mtime is restored. Relevant
            # content is checked independently above, without that assumption.
            for text in ["other payload", "third payload text"]:
                for name in names:
                    (root / name).write_text(text)
                os.utime(selected, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                subprocess.run(["git", "-C", str(root), "add", names[0]], check=True)
                # Keep an index-only edit visible even after restoring worktree bytes.
                (root / names[0]).write_text("first")
                with patch.object(verify, "digest", side_effect=read_relevant):
                    current = source_snapshot(root)
                # Selected fixture content belongs to preparation, independently
                # of selection-independent global pending observation.
                self.assertEqual(file_identities([selected])[str(selected)]["sha256"], digest(selected))
                pending = edit_selection(current, prior)
                self.assertEqual(pending, sorted(names))
                self.assertTrue(current[names[0]]["worktree"]["unresolved"])
                prior = {"observed": current, "pending": pending}
            for name in names[:2]:
                (root / name).unlink()
            with patch.object(verify, "digest", side_effect=read_relevant):
                current = source_snapshot(root)
            self.assertEqual(edit_selection(current, prior), sorted(names))

    def test_development_timing_starts_before_waiting_for_lease(self):
        import argparse
        import verify

        clock = [10.0]
        with patch.object(verify.time, "monotonic", side_effect=lambda: clock[0]), patch.object(verify, "OutputLease") as lease, patch.object(verify, "_development_owned") as owned:
            lease.return_value.__enter__.side_effect = lambda: clock.__setitem__(0, 17.0)
            args = argparse.Namespace(preset="linux-debug")
            verify.development(args)
        owned.assert_called_once_with(args, started=10.0, lease_seconds=7.0)

    def test_switching_route_and_color_keeps_clean_fixture_observation_stable(self):
        import verify

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            for name in verify.OWNERS:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((ROOT / name).read_bytes())
            files = ["src/ScanRoute.cpp", "rust/film-juicer-core/src/color/output.rs", "tests/ffi/fixtures/output-color/native.json"]
            for name in files:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Control", "-c", "user.email=control@example.invalid", "commit", "-qm", "initial"], check=True)
            observed = source_snapshot(root)
            for source in files[:2]:
                verify.select_consumers(root, root / "unneeded-build", [source])
                current = source_snapshot(root)
                self.assertEqual(edit_selection(current, {"observed": observed, "pending": []}), [])
                self.assertNotIn(files[2], current)
            (root / files[2]).write_text("changed")
            current = source_snapshot(root)
            self.assertTrue(current[files[2]]["unresolved"])
            self.assertEqual(edit_selection(current, {"observed": observed, "pending": []}), [files[2]])

    def test_reuse_still_rejects_consumed_junit_logs_and_process_corruption(self):
        import shutil
        import verify
        from build_dependencies import file_identities
        from build_evidence import SCOPED_PROTOCOL
        from execution import digest, write_json
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            build = root / "linux-debug"
            selection = {"files": ["src/ScanRoute.cpp"], "compiler_consumers": [], "rust_packages": [], "tests": ["Route.Host.Bridge"]}
            junit = root / "tests.xml"
            junit.write_text("retained passed report")
            commands = {
                "quality": verify.quality_command(build.name, selection),
                "tests": ["ctest", "--preset", build.name, "-R", "^(Route\\.Host\\.Bridge)$", "--output-on-failure", "--output-junit", str(junit)],
            }
            previous = {"protocol": SCOPED_PROTOCOL, "scope_status": "passed", "selection": selection, "build": "unused", "junit": str(junit)}
            payloads = [junit]
            for stage, command in commands.items():
                output = root / (stage + ".log")
                output.write_text("passed")
                process = root / (stage + ".process.json")
                proof = {"status": "completed", "drained": True, "returncode": 0}
                write_json(process, proof)
                receipt = root / (stage + ".json")
                executable = shutil.which(command[0])
                self.assertIsNotNone(executable)
                write_json(receipt, {"status": "passed", "returncode": 0, "output": str(output), "output_sha256": digest(output),
                                     "process_tree": {"receipt": str(process), **proof}, "identity": {"command": command, "executable_sha256": digest(Path(executable))}})
                previous[stage + "_command"] = str(receipt)
                payloads += [output, process]
            previous["evidence"] = file_identities(payloads)
            with patch.object(verify, "check_scoped_preparation"):
                verify.verify_development_receipt(SimpleNamespace(root=root), build, previous, selection=selection)
                for path, content, message in [(junit, "corrupt", "evidence or consumed"), (root / "quality.log", "corrupt", "command evidence"),
                                               (root / "quality.process.json", '{"drained":false}', "process evidence")]:
                    original = path.read_bytes()
                    path.write_text(content)
                    with self.assertRaisesRegex(ValueError, message):
                        verify.verify_development_receipt(SimpleNamespace(root=root), build, previous, selection=selection)
                    path.write_bytes(original)

    def test_second_edit_to_same_file_and_unrelated_edit_retain_failed_work(self):
        prior = {"observed": {"a.cpp": "first", "b.rs": "original"}, "pending": []}
        self.assertEqual(
            edit_selection({"a.cpp": "second", "b.rs": "original"}, prior), ["a.cpp"]
        )
        prior = {
            "observed": {"a.cpp": "second", "b.rs": "original"},
            "pending": ["a.cpp"],
        }
        self.assertEqual(
            edit_selection({"a.cpp": "second", "b.rs": "changed"}, prior),
            ["a.cpp", "b.rs"],
        )
        self.assertEqual(
            edit_selection({"a.cpp": "third", "b.rs": "original"}, prior), ["a.cpp"]
        )

    def test_new_deleted_and_renamed_membership_is_not_discarded(self):
        prior = {"observed": {"old.h": "same", "deleted.cpp": "bytes"}, "pending": []}
        self.assertEqual(
            edit_selection({"new.h": "same", "new.cpp": "new"}, prior),
            ["deleted.cpp", "new.cpp", "new.h", "old.h"],
        )
        self.assertEqual(
            edit_selection(
                {"deleted.cpp": None},
                {"observed": {"deleted.cpp": "bytes"}, "pending": []},
            ),
            ["deleted.cpp"],
        )

    def test_explicit_files_cannot_hide_initial_unsupported_changes(self):
        prior = {"observed": {"a.cpp": "old", "b.rs": "old"}, "pending": []}
        self.assertEqual(
            edit_selection({"a.cpp": "new", "b.rs": "new"}, prior, ["a.cpp"]), ["a.cpp", "b.rs"]
        )
        self.assertEqual(
            edit_selection({"a.cpp": "old", "b.rs": "old"}, prior, ["a.cpp"]), []
        )

    def test_content_snapshot_detects_staged_unstaged_untracked_deleted_and_same_timestamp(
        self,
    ):
        import os

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            paths = [root / name for name in ["staged.cpp", "unstaged.rs", "deleted.h"]]
            for path in paths:
                path.write_text("first")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Control", "-c", "user.email=control@example.invalid", "commit", "-qm", "initial"], check=True)
            before = source_snapshot(root)
            stamp = paths[0].stat()
            paths[0].write_text("other")
            os.utime(paths[0], ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            subprocess.run(["git", "-C", str(root), "add", "staged.cpp"], check=True)
            paths[1].write_text("other")
            paths[2].unlink()
            (root / "new.h").write_text("new")
            selected = edit_selection(
                source_snapshot(root), {"observed": before, "pending": []}
            )
            self.assertEqual(
                selected, ["deleted.h", "new.h", "staged.cpp", "unstaged.rs"]
            )

    def test_staged_changes_remain_visible_with_restored_worktree_bytes(self):
        from verify import dirty_paths, load_development_state

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            path = root / "unsupported.h"
            path.write_text("original")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Control", "-c", "user.email=control@example.invalid", "commit", "-qm", "initial"], check=True)
            before = source_snapshot(root)
            for text in ("first staged edit", "second staged edit"):
                path.write_text(text)
                subprocess.run(["git", "-C", str(root), "add", "unsupported.h"], check=True)
                path.write_text("original")
                current = source_snapshot(root)
                self.assertIn("unsupported.h", dirty_paths(root))
                self.assertEqual(edit_selection(current, {"observed": before, "pending": []}), ["unsupported.h"])
                initial = load_development_state(root, root / "absent-build", None, ["src/ScanRoute.cpp"])
                self.assertEqual(initial["pending"], ["unsupported.h"])
                self.assertNotIn("scope_status", initial)
                before = current

    def test_unknown_generated_resource_configuration_and_empty_scopes_are_explicit(
        self,
    ):
        from verify import select_consumers

        for path in [
            "src/new.cpp",
            "Resources/profile.json",
            "CMakeLists.txt",
            "rust/film-juicer-plugin/src/cuda/sys.rs",
            "tests/film/fixtures/new.json",
        ]:
            with (
                self.subTest(path=path),
                self.assertRaisesRegex(ValueError, "unresolved development scope"),
            ):
                select_consumers(ROOT, ROOT / "unused-build", [path])
        with self.assertRaisesRegex(ValueError, "empty development selection"):
            select_consumers(ROOT, ROOT / "unused-build", [])

    def test_cross_caller_pending_state_cannot_be_replaced_by_an_older_receipt(self):
        from execution import write_json, digest
        from verify import load_development_state

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            build = root / "out/build/linux-debug"
            build.mkdir(parents=True)
            first = root / "first.json"
            latest = root / "latest.json"
            record = {
                "mode": "development",
                "protocol": "scoped-development-v1",
                "preset": "linux-debug",
                "status": "failed",
                "observed": {"src/ScanRoute.cpp": "changed"},
                "pending": ["src/ScanRoute.cpp"],
            }
            write_json(first, {**record, "pending": []})
            write_json(latest, record)
            write_json(
                build / "verification-development.json",
                {"receipt": str(latest), "sha256": digest(latest)},
            )
            loaded = load_development_state(root, build, None, ())
            self.assertEqual(loaded["pending"], ["src/ScanRoute.cpp"])
            with self.assertRaisesRegex(ValueError, "not the latest"):
                load_development_state(root, build, first, ())
            write_json(latest, {**record, "pending": []})
            with self.assertRaisesRegex(ValueError, "receipt changed"):
                load_development_state(root, build, None, ())

    def test_established_scope_cannot_hide_a_new_unsupported_edit_with_files(self):
        self.assertEqual(
            edit_selection(
                {"a.cpp": "second", "config": "changed"},
                {
                    "observed": {"a.cpp": "first", "config": "original"},
                    "pending": [],
                    "established": True,
                },
                ["a.cpp"],
            ),
            ["a.cpp", "config"],
        )

    def test_initial_preparation_baseline_detects_new_edits_outside_explicit_files(
        self,
    ):
        from verify import load_development_state
        from execution import write_json, digest

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            build = root / "out/build/linux-debug"
            build.mkdir(parents=True)
            producer = root / "build.json"
            producer.write_text("{}")
            receipt = root / "preparation.json"
            write_json(
                receipt,
                {
                    "mode": "preparation",
                    "status": "passed",
                    "preset": build.name,
                    "source": {
                        "a.cpp": "old",
                        "config": "original",
                        "missing-generated": None,
                    },
                    "build": str(producer),
                    "build_sha256": digest(producer),
                },
            )
            write_json(
                build / "verification-development.json",
                {"receipt": str(receipt), "sha256": digest(receipt)},
            )
            current = {"a.cpp": "new", "config": "changed", "missing-generated": None}
            with (
                patch("verify.dirty_paths", return_value={"a.cpp", "config"}),
                patch("verify.source_snapshot", return_value=current),
            ):
                baseline = load_development_state(root, build, None, ["a.cpp"])
            self.assertEqual(
                edit_selection(current, baseline, ["a.cpp"]), ["a.cpp", "config"]
            )

    def test_complete_preparation_preserves_existing_pending_failure(self):
        import verify
        from execution import write_json, digest
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            build = root / "out/build/linux-debug"
            build.mkdir(parents=True)
            prior = root / "failed.json"
            write_json(
                prior,
                {
                    "mode": "development",
                    "status": "failed",
                    "pending": ["src/ScanRoute.cpp"],
                },
            )
            state = build / "verification-development.json"
            write_json(state, {"receipt": str(prior), "sha256": digest(prior)})
            saved = state.read_bytes()
            built = root / "built.json"
            built.write_text("{}")
            args = SimpleNamespace(mode="prepare", preset=build.name, files=["src/ScanRoute.cpp"])
            with (
                patch.object(verify, "ROOT", root),
                patch.object(verify, "source_snapshot", return_value={}),
                patch.object(verify, "select_consumers", return_value={"targets": ["JuicerRouteBridgeTests"], "inputs": ["src/ScanRoute.cpp"]}),
                patch.object(verify, "prepare_build", return_value=(None, built)),
                patch("argparse.ArgumentParser.parse_args", return_value=args),
            ):
                verify.main()
            self.assertEqual(state.read_bytes(), saved)
            self.assertEqual(
                json.loads(prior.read_text())["pending"], ["src/ScanRoute.cpp"]
            )

    def test_missing_receipts_initialize_observation_with_all_dirty_paths_pending(self):
        from verify import load_development_state

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            build = root / "build"
            with patch("verify.source_snapshot", return_value={"src/ScanRoute.cpp": "current"}), patch("verify.dirty_paths", return_value={"src/ScanRoute.cpp", "unsupported.h"}):
                state = load_development_state(root, build, None, ["src/ScanRoute.cpp"])
            self.assertEqual(state["pending"], ["src/ScanRoute.cpp", "unsupported.h"])
            self.assertNotIn("scope_status", state)
            self.assertFalse(build.exists())

    def test_unreadable_pending_state_is_preserved_without_scheduling(self):
        from verify import load_development_state

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state = root / "verification-development.json"
            state.write_text("unreadable pending state")
            with self.assertRaises(ValueError):
                load_development_state(root, root, None, ["src/ScanRoute.cpp"])
            self.assertEqual(state.read_text(), "unreadable pending state")

    def test_qualification_requires_explicit_scope_and_legacy_adapters_schedule_no_work(self):
        import build_evidence
        import qualification
        from types import SimpleNamespace
        from unittest.mock import Mock
        campaign = Mock(root=ROOT)
        campaign.attempts.return_value = []
        with patch.object(qualification, "Campaign") as scheduler:
            with self.assertRaisesRegex(ValueError, "mandatory"):
                qualification.qualify(ROOT, SimpleNamespace(reason="", selection=[]))
            scheduler.assert_not_called()
        with patch.object(build_evidence, "preparation_inventory") as inventory:
            for operation in (
                lambda: build_evidence.prepare_build(campaign, "prepare", source=ROOT, build=ROOT, inputs=[], timeout=1),
                lambda: build_evidence.run_ctest(campaign, "tests", build=ROOT, prepared=ROOT, inputs=[], timeout=1)):
                with self.assertRaisesRegex(ValueError, "legacy sealed-build"):
                    operation()
            campaign.run.assert_not_called()
            inventory.assert_not_called()
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/verify.py"), "qualification", "--preset", "linux-debug"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--reason", result.stderr)

    def test_legacy_expanded_adapters_reject_before_header_or_inventory_discovery(self):
        import build_evidence
        from types import SimpleNamespace

        campaign = SimpleNamespace(root=ROOT)
        with patch.object(build_evidence, "development_selection") as selection, patch.object(build_evidence, "preparation_inventory") as inventory:
            for preset, targets in [("linux-debug", ["FilmJuicer", "JuicerRouteBridgeTests"]), ("linux-release", ["JuicerRouteBridgeTests"])]:
                with self.assertRaisesRegex(ValueError, "legacy sealed-build"):
                    build_evidence._prepare_build_owned(campaign, "prepare", source=ROOT, build=ROOT / "out/build" / preset,
                                                      inputs=["src/ScanRoute.h"], timeout=1, targets=targets)
            with self.assertRaisesRegex(ValueError, "legacy sealed-build"):
                build_evidence.run_ctest(campaign, "tests", build=ROOT / "out/build/linux-debug", prepared=Path("unused"),
                                         inputs=["src/ScanRoute.h"], timeout=1, tests=["Route.Host.Bridge", "Ffi.Host.InputColorPreparation"])
            with self.assertRaisesRegex(TypeError, "checkpoint"):
                build_evidence.run_ctest(campaign, "tests", build=ROOT / "out/build/linux-debug", prepared=Path("unused"),
                                         inputs=["src/ScanRoute.cpp"], timeout=1, tests=["Route.Host.Bridge"], checkpoint=True)
            selection.assert_not_called()
            inventory.assert_not_called()

    def test_route_selection_is_exact_without_build_or_preparation_receipts(self):
        from verify import select_consumers

        selection = select_consumers(ROOT, ROOT / "absent-tree", ["src/ScanRoute.cpp"])
        self.assertEqual(selection["targets"], ["JuicerRouteBridgeTests"])
        self.assertEqual(selection["tests"], ["Route.Host.Bridge"])
        self.assertEqual(selection["compiler_consumers"], ["src/ScanRoute.cpp"])

    def test_rust_route_and_color_owners_select_only_named_checks(self):
        from verify import select_consumers

        route = select_consumers(ROOT, ROOT / "absent-tree", ["rust/film-juicer-core/src/route.rs"])
        self.assertEqual(route["targets"], ["JuicerRouteBridgeTests"])
        self.assertEqual(route["tests"], ["Quality.RustBoundaries", "Route.Host.Bridge"])
        self.assertEqual(route["rust_packages"], ["film-juicer-core"])
        self.assertEqual(route["rust_test_filters"], ["route::"])
        self.assertIn("tests/quality/test_rust_boundaries.py", route["inputs"])
        self.assertIn("tests/quality/fixtures/rust_boundaries/private_construction.rs", route["inputs"])
        color = select_consumers(ROOT, ROOT / "absent-tree", ["rust/film-juicer-core/src/color/output.rs"])
        self.assertEqual(color["targets"], ["JuicerOutputColorTests"])
        self.assertEqual(color["tests"], ["Ffi.Host.OutputColor", "Quality.RustBoundaries"])
        self.assertIn("tests/ffi/fixtures/output-color/native.json", color["inputs"])
        self.assertNotIn("Route.Host.Bridge", color["tests"])
        self.assertEqual(color["rust_packages"], ["film-juicer-core"])
        self.assertEqual(color["rust_test_filters"], ["color::output::"])

    def test_rust_interface_uncertainty_adds_plugin_checks(self):
        from verify import rust_body_edit

        name = "rust/film-juicer-core/src/route.rs"
        original = (ROOT / name).read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / name
            path.parent.mkdir(parents=True)
            path.write_bytes(original)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Control", "-c", "user.email=control@example.invalid", "commit", "-qm", "initial"], check=True)
            revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            path.write_bytes(original.replace(b"=> ScanRoute::NegativeDirect,", b"=> ScanRoute::NegativePrint,"))
            self.assertTrue(rust_body_edit(root, name, revision))
            for before, after in [(b"    Positive,", b"    Other,"), (b"pub enum ScanRoute", b"#[repr(u8)]\npub enum ScanRoute"), (b"selection: RouteSelection", b"selection: &RouteSelection")]:
                path.write_bytes(original.replace(before, after))
                self.assertFalse(rust_body_edit(root, name, revision))
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Control", "-c", "user.email=control@example.invalid", "commit", "-qm", "API edit"], check=True)
            self.assertFalse(rust_body_edit(root, name, revision))

    def test_unrelated_scope_reuses_saved_pass_without_union_or_losing_failure(self):
        import argparse
        import verify
        from execution import digest, write_json

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            build = root / "out/build/linux-debug"
            directory = root / "attempt"
            directory.mkdir()
            build.mkdir(parents=True)
            selection = {"files": ["src/ScanRoute.cpp"], "targets": ["route"], "tests": ["route"], "inputs": ["src/ScanRoute.cpp"]}
            passed = {"scope_status": "passed", "selection": selection, **{name: "saved" for name in ("build", "quality_command", "tests_command", "quality", "junit")}, "evidence": {}}
            receipt = root / "saved.json"
            write_json(receipt, passed)
            previous = {"observed": {}, "pending": ["unrelated.rs"], "scope_status": "failed",
                        "successful_scopes": {verify.scope_key(selection): {"receipt": str(receipt), "sha256": digest(receipt)}}}
            with patch.object(verify, "ROOT", root), patch.object(verify, "load_development_state", return_value=previous), patch.object(verify, "source_snapshot", return_value={}), patch.object(verify, "select_consumers", return_value=selection), patch.object(verify, "Campaign") as campaign, patch.object(verify, "verify_development_receipt") as consume, patch.object(verify, "check_scoped_preparation"), patch.object(verify, "prepare_build") as prepare:
                campaign.return_value.directory = directory
                result = verify._development_owned(argparse.Namespace(preset="linux-debug", previous=None, files=["src/ScanRoute.cpp"]))
            record = json.loads(result.read_text())
            consume.assert_called_once()
            prepare.assert_not_called()
            self.assertEqual(record["pending"], ["unrelated.rs"])
            self.assertEqual(record["selection"], selection)
            self.assertEqual(record["reused_from"], str(receipt))
            consume.assert_called_once_with(campaign.return_value, build, passed)
            self.assertEqual(record["bookkeeping_before_publication_seconds"], record["seconds"])
            self.assertIn("reuse_validation_seconds", record["timings"])
            with patch.object(verify, "ROOT", root), patch.object(verify, "load_development_state", return_value=previous), patch.object(verify, "source_snapshot", side_effect=[{}, {"new.h": "edited"}]), patch.object(verify, "select_consumers", return_value=selection), patch.object(verify, "Campaign") as campaign, patch.object(verify, "verify_development_receipt"), patch.object(verify, "check_scoped_preparation"), patch.object(verify, "prepare_build") as prepare:
                campaign.return_value.directory = directory
                with self.assertRaisesRegex(ValueError, "source membership or contents changed"):
                    verify._development_owned(argparse.Namespace(preset="linux-debug", previous=None, files=["src/ScanRoute.cpp"]))
            failed = json.loads(result.read_text())
            self.assertEqual(failed["scope_status"], "failed")
            self.assertIn("unrelated.rs", failed["pending"])
            self.assertEqual(digest(receipt), previous["successful_scopes"][verify.scope_key(selection)]["sha256"])
            prepare.assert_not_called()
            # A saved pass for these files cannot hide their later pending failure.
            previous["pending"].append("src/ScanRoute.cpp")
            with patch.object(verify, "ROOT", root), patch.object(verify, "load_development_state", return_value=previous), patch.object(verify, "source_snapshot", return_value={}), patch.object(verify, "select_consumers", return_value=selection), patch.object(verify, "Campaign") as campaign, patch.object(verify, "verify_development_receipt") as consume, patch.object(verify, "prepare_build", side_effect=ValueError("required fresh preparation")) as prepare:
                campaign.return_value.directory = directory
                with self.assertRaisesRegex(ValueError, "required fresh preparation"):
                    verify._development_owned(argparse.Namespace(preset="linux-debug", previous=None, files=["src/ScanRoute.cpp"]))
            consume.assert_not_called()
            prepare.assert_called_once()
            self.assertIn("src/ScanRoute.cpp", json.loads(result.read_text())["pending"])

    def test_qualification_rejects_development_partial_and_incomplete_receipts(self):
        for record in [
            {"mode": "development", "status": "passed"},
            {"mode": "preparation", "status": "passed"},
            {"mode": "qualification", "status": "passed", "stages": {"quality": {}}},
            {"mode": "qualification", "status": "failed"},
        ]:
            with self.subTest(record=json.dumps(record)), self.assertRaises(ValueError):
                require_checkpoint(record)


if __name__ == "__main__":
    unittest.main()
