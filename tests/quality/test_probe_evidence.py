from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import uuid

import probe_evidence as evidence


class ProbeEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / 'source.rs'
        self.source.write_text('original')
        self.run = self.root / 'run'
        self.run.mkdir()
        self.identity = str(uuid.uuid4())
        self.manifest = self.root / 'manifest.json'
        self.environment = evidence.compiler_environment()
        self.environment.pop('JUICER_RUST_PROBE_MANIFEST', None)
        self.environment.pop('JUICER_QUALIFICATION_ID', None)
        # Exercise the evidence protocol with real child processes; the context
        # inventory itself has an independent test below and real Rust runs.
        self.enterContext(patch.object(evidence, 'context', side_effect=lambda *args:
            {'files': {p.name: evidence.digest(p) for p in self.root.glob('*.rs')}}))
        self.enterContext(patch.object(evidence, 'process_token', return_value=['producer']))
        self.command = [sys.executable, '-X', 'utf8', '-c', 'import sys;print("diagnostic λ");print("stderr",file=sys.stderr)',
                        '--target-dir', str(self.root / 'cache')]
        self.case = {'test': 'real_consumer', 'package': 'film-juicer-plugin', 'profile': 'dev',
                     'features': [], 'fixtures': {'film-juicer-plugin/src/lib.rs': 'accepted'}}

    def capture(self):
        for suite in ('naming', 'boundaries'):
            artifacts = self.run / ('rust-' + suite)
            artifacts.mkdir()
            owner = evidence.ProbeEvidence(self.root, suite, sys.executable, 'target',
                                            (False, True), self.environment, artifacts)
            for profile in ('dev', 'release'):
                owner.run({**self.case, 'profile': profile}, self.command, self.root, artifacts / (profile + '.log'))
            owner.finish()
        self.receipt = self.run / 'run.json'
        self.seal()
        evidence.prepare_manifest(self.receipt, self.manifest, self.identity)
        self.token = self.enterContext(patch.object(evidence, 'process_token', return_value=None))
        evidence.publish(self.manifest, self.identity)

    def seal(self, status='passed'):
        evidence.write_json(self.run / 'run.json', {'status': status, 'finished_at': 'complete',
            'files': {p.relative_to(self.run).as_posix(): evidence.digest(p)
                      for p in self.run.rglob('*') if p.is_file() and p.name != 'run.json'}})

    def consume(self, profiles=(False,)):
        environment = {**self.environment, 'JUICER_RUST_PROBE_MANIFEST': str(self.manifest),
                       'JUICER_QUALIFICATION_ID': self.identity}
        return evidence.ProbeEvidence(self.root, 'naming', sys.executable, 'target', profiles,
                                       environment, self.root / 'consumer')

    def replace_ledger(self, update):
        path = self.run / 'rust-naming/cases.json'
        document = json.loads(path.read_text())
        update(document)
        evidence.write_json(path, document)
        self.seal()
        manifest = json.loads(self.manifest.read_text())
        manifest['producer_sha256'] = evidence.digest(self.receipt)
        evidence.write_json(self.manifest, manifest)

    def test_exact_output_is_verified_without_a_second_compiler_launch(self):
        self.capture()
        owner = self.consume()
        with patch.object(evidence.subprocess, 'run', side_effect=AssertionError('unexpected compiler')):
            result = owner.run(self.case, self.command, self.root, self.root / 'must-not-exist.log')
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, 'diagnostic λ\n')
            self.assertEqual(result.stderr, 'stderr\n')
            owner.finish()
        self.assertFalse((self.root / 'must-not-exist.log').exists())

    def test_standalone_runs_fresh_every_time(self):
        self.capture()
        for name in ('first', 'second'):
            artifacts = self.root / name
            artifacts.mkdir()
            owner = evidence.ProbeEvidence(self.root, 'naming', sys.executable, 'target',
                                            (False,), self.environment, artifacts)
            with patch.object(evidence.subprocess, 'run', wraps=subprocess.run) as command:
                owner.run(self.case, self.command, self.root, artifacts / 'fresh.log')
                command.assert_called_once()
            owner.finish()

    def test_source_edit_addition_and_deletion_invalidate_evidence(self):
        self.capture()
        for action in ('edit', 'add', 'delete'):
            with self.subTest(action=action):
                self.source.write_text('changed' if action == 'edit' else 'original')
                added = self.root / 'new.rs'
                if action == 'add':
                    added.write_text('new')
                if action == 'delete':
                    self.source.unlink()
                with self.assertRaisesRegex(ValueError, 'context changed'):
                    self.consume()
                self.source.write_text('original')
                added.unlink(missing_ok=True)

    def test_mutation_during_consumption_is_rejected(self):
        self.capture()
        owner = self.consume()
        owner.run(self.case, self.command, self.root, self.root / 'unused')
        self.source.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'inputs changed'):
            owner.finish()

    def test_changed_tool_dependency_or_environment_context_is_rejected(self):
        self.capture()
        for name in ('tool', 'dependency', 'environment', 'lint', 'cargo-config', 'host'):
            with self.subTest(name=name), patch.object(evidence, 'context', return_value={name: 'changed'}):
                with self.assertRaisesRegex(ValueError, 'context changed'):
                    self.consume()

    def test_profile_features_owner_and_flags_are_distinct_cases(self):
        self.capture()
        for change in ({'profile': 'release'}, {'features': ['test-support']},
                       {'fixtures': {'other/module.rs': 'accepted'}}, {'package': 'film-juicer-core'}):
            owner = self.consume()
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'matching probe'):
                owner.run({**self.case, **change}, self.command, self.root, self.root / 'unused')
        owner = self.consume()
        with self.assertRaisesRegex(ValueError, 'matching probe'):
            owner.run(self.case, [*self.command, '--all-targets'], self.root, self.root / 'unused')

    def test_missing_and_unexpected_cases_fail(self):
        self.capture()
        owner = self.consume()
        with self.assertRaisesRegex(ValueError, 'unconsumed'):
            owner.finish()
        self.replace_ledger(lambda d: d['cases'].__setitem__(slice(None), [r for r in d['cases'] if r['case']['profile'] == 'release']))
        with self.assertRaisesRegex(ValueError, 'no requested cases'):
            self.consume()

    def test_unrecognized_profile_and_empty_explicit_manifest_fail(self):
        self.capture()
        self.replace_ledger(lambda d: d['cases'][0]['case'].update(profile='unknown'))
        with self.assertRaisesRegex(ValueError, 'unexpected probe profile'):
            self.consume()
        with self.assertRaises((IsADirectoryError, PermissionError)):
            evidence.ProbeEvidence(self.root, 'naming', sys.executable, 'target', (False,),
                {**self.environment, 'JUICER_RUST_PROBE_MANIFEST': '', 'JUICER_QUALIFICATION_ID': self.identity}, self.root)

    def test_duplicate_cases_fail_even_in_another_profile(self):
        self.capture()
        self.replace_ledger(lambda d: d['cases'].append(copy.deepcopy(d['cases'][1])))
        with self.assertRaisesRegex(ValueError, 'duplicate probe'):
            self.consume()

    def test_modified_or_missing_logs_and_ledger_fail(self):
        self.capture()
        for path in (self.run / 'rust-naming/dev.log', self.run / 'rust-naming/cases.json'):
            original = path.read_bytes()
            with self.subTest(path=path), self.assertRaises((ValueError, json.JSONDecodeError)):
                path.write_bytes(b'changed')
                self.consume()
            path.write_bytes(original)
        (self.run / 'rust-naming/dev.log').unlink()
        with self.assertRaises(FileNotFoundError):
            self.consume()

    def test_log_and_manifest_changes_during_consumption_fail(self):
        self.capture()
        for path in (self.run / 'rust-naming/dev.log', self.run / 'rust-naming/cases.json', self.manifest):
            original = path.read_bytes()
            owner = self.consume()
            owner.run(self.case, self.command, self.root, self.root / 'unused')
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                owner.finish()
            path.write_bytes(original)

    def test_failed_incomplete_and_live_producers_cannot_be_consumed(self):
        self.capture()
        with patch.object(evidence, 'process_token', return_value=['producer']):
            with self.assertRaisesRegex(ValueError, 'has not completed'):
                self.consume()
        original = self.receipt.read_bytes()
        for status in ('failed', 'running'):
            self.seal(status)
            with self.assertRaisesRegex(ValueError, 'receipt changed'):
                self.consume()
            with self.assertRaisesRegex(ValueError, 'incomplete or failed'):
                evidence.prepare_manifest(self.receipt, self.root / 'no-export.json', self.identity)
        self.receipt.write_bytes(original)

    def test_wrong_qualification_and_empty_coverage_cannot_pass(self):
        self.capture()
        self.identity = str(uuid.uuid4())
        with self.assertRaisesRegex(ValueError, 'different qualification'):
            self.consume()
        self.identity = json.loads(self.manifest.read_text())['qualification']
        self.replace_ledger(lambda d: d.update(cases=[]))
        with self.assertRaisesRegex(ValueError, 'no requested cases'):
            self.consume()

    def test_additional_output_cargo_configuration_is_rejected(self):
        self.capture()
        workspace = self.root / 'scratch'
        (workspace / '.cargo').mkdir(parents=True)
        (workspace / '.cargo/config.toml').write_text('[build]\n')
        with self.assertRaisesRegex(ValueError, 'different Cargo'):
            self.consume().run(self.case, self.command, workspace, self.root / 'unused')

    def test_external_output_cannot_omit_root_cargo_configuration(self):
        self.capture()
        (self.root / '.cargo').mkdir()
        (self.root / '.cargo/config.toml').write_text('[build]\n')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'different Cargo'):
                self.consume().run(self.case, self.command, Path(directory), self.root / 'unused')

    def test_publication_requires_completed_producer_and_unchanged_evidence(self):
        self.capture()
        self.manifest.unlink()
        with patch.object(evidence, 'process_token', return_value=['producer']):
            with self.assertRaisesRegex(ValueError, 'has not completed'):
                evidence.publish(self.manifest, self.identity)
        (self.run / 'rust-naming/dev.log').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'changed before publication'):
            evidence.publish(self.manifest, self.identity)
        self.assertFalse(self.manifest.exists())

    def test_original_negative_assertions_reject_wrong_code_span_and_marker(self):
        from test_rust_boundaries import RustBoundaryTests
        diagnostic = {'level': 'error', 'code': {'code': 'E0616'}, 'spans': [
            {'is_primary': True, 'file_name': 'quality_boundary_probe.rs',
             'text': [{'text': 'private_storage'}]}]}
        message = json.dumps({'reason': 'compiler-message', 'message': diagnostic})
        self.command[4] = 'import sys; print(' + repr(message) + '); sys.exit(101)'
        self.capture()
        owner = self.consume()
        result = owner.run(self.case, self.command, self.root, self.root / 'unused')
        checked = [json.loads(result.stdout)['message']]
        consumer = RustBoundaryTests()
        expected = {'E0616': ('private_storage',)}
        consumer.assert_rejected(result.returncode, checked, expected)
        for fault in ('code', 'primary', 'file', 'marker'):
            bad = copy.deepcopy(checked)
            if fault == 'code':
                bad[0]['code']['code'] = 'E0000'
            elif fault == 'primary':
                bad[0]['spans'][0]['is_primary'] = False
            elif fault == 'file':
                bad[0]['spans'][0]['file_name'] = 'unrelated.rs'
            else:
                bad[0]['spans'][0]['text'][0]['text'] = 'unrelated'
            with self.subTest(fault=fault), self.assertRaises(AssertionError):
                consumer.assert_rejected(result.returncode, bad, expected)
        owner.finish()

    def test_original_positive_assertions_reject_tool_failure(self):
        from test_rust_naming import RustNamingTests
        self.command[4] = 'import sys; sys.exit(2)'
        self.capture()
        owner = self.consume()
        result = owner.run(self.case, self.command, self.root, self.root / 'unused')
        consumer = RustNamingTests()
        consumer.profiles = (False,)
        with patch.object(consumer, 'add_probe'), patch.object(consumer, 'clippy', return_value=(result.returncode, [])):
            with self.assertRaises(AssertionError):
                consumer.test_current_code_and_valid_names_pass()
        owner.finish()

    def test_execution_environment_preserves_compiler_settings(self):
        environment = {'RUSTUP_TOOLCHAIN': 'pinned', 'CARGO_PROFILE_RELEASE_LTO': 'true',
            'CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER': 'lld', 'GITHUB_ENV': 'step-path',
            'JUICER_RUST_PROBE_MANIFEST': 'evidence', 'UNKNOWN_COMPILER_INPUT': 'keep',
            'VSCONSOLEOUTPUT': '0'}
        for platform in ('posix', 'nt'):
            with self.subTest(platform=platform), patch.object(evidence.os, 'name', platform):
                result = evidence.execution_environment(environment)
            self.assertEqual(result, {'RUSTUP_TOOLCHAIN': 'pinned', 'CARGO_PROFILE_RELEASE_LTO': 'true',
                'CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER': 'lld', 'UNKNOWN_COMPILER_INPUT': 'keep',
                'VSCONSOLEOUTPUT': '1' if platform == 'nt' else '0'})


class WorkflowEnvironmentTests(unittest.TestCase):
    def test_both_jobs_disable_bytecode_for_the_entire_handoff(self):
        root = Path(__file__).resolve().parents[2]
        for platform in ('linux', 'windows'):
            with self.subTest(platform=platform):
                workflow = (root / f'.github/workflows/tests-{platform}-host.yml').read_text(encoding='utf-8')
                job = workflow.split(f'\n  {platform}-debug-host:\n', 1)[1].split('\n    steps:\n', 1)[0]
                block = re.search(r'^    env:\n((?:      [^\n]*\n)+)', job, re.MULTILINE)
                self.assertIsNotNone(block)
                environment = dict(line.strip().split(': ', 1) for line in block[1].splitlines())
                self.assertEqual(environment.get('PYTHONDONTWRITEBYTECODE'), '"1"')

    def test_linux_quality_and_ctest_share_compiler_environment(self):
        root = Path(__file__).resolve().parents[2]
        workflow = (root / '.github/workflows/tests-linux-host.yml').read_text(encoding='utf-8')
        job = workflow.split('\n  linux-debug-host:\n', 1)[1]
        job, steps = job.split('\n    steps:\n', 1)
        steps = re.split(r'^      - name: ', steps, flags=re.MULTILINE)
        quality, = [step for step in steps if 'python -u scripts/check-quality.py' in step]
        ctest, = [step for step in steps if 'ctest --preset linux-debug -L host' in step]

        def environment(section, indent):
            # These workflow env blocks contain literal single-line mappings.
            # Keep this check local to that format; no expression evaluation.
            block = re.search(rf'^{indent}env:\n((?:{indent}  [^\n]*\n)+)', section, re.MULTILINE)
            return dict(line.strip().split(': ', 1) for line in block[1].splitlines()) if block else {}

        shared = environment(job, '    ')
        producer = evidence.execution_environment(evidence.compiler_environment(
            {**shared, **environment(quality, '        ')}))
        consumer = evidence.execution_environment(evidence.compiler_environment(
            {**shared, **environment(ctest, '        ')}))
        self.assertEqual(producer, consumer, 'quality and CTest must bind the same compiler environment')
        for name, value in (('JUICER_CLANG_FORMAT', 'clang-format-22'),
                            ('JUICER_CLANG_TIDY', 'clang-tidy-22')):
            self.assertEqual(producer[name], value)


class ContextTests(unittest.TestCase):
    def test_real_inventory_binds_membership_configs_dependencies_and_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name in ('Cargo.toml', 'Cargo.lock', 'clippy.toml', 'rust-toolchain.toml',
                         'rustfmt.toml', 'scripts/check-quality.py', 'rust/core/lib.rs',
                         'tests/quality/fixture.rs', 'tools/rustc',
                         'tools/cargo', 'tools/cargo-clippy', 'tools/clippy-driver', 'dependency/Cargo.toml',
                         'python-runtime/unittest.py', 'python-runtime/__pycache__/unittest.pyc'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('original')
            (root / '.cargo').mkdir()
            (root / '.cargo/config.toml').write_text('[build]\n')
            def output(command, **kwargs):
                if 'metadata' in command:
                    return json.dumps({'packages': [{'source': 'registry', 'manifest_path': str(root / 'dependency/Cargo.toml')}]})
                if 'which' in command:
                    return str(root / 'tools' / command[-1])
                return str(root / 'sysroot')
            with patch.object(evidence.subprocess, 'check_output', side_effect=output), \
                    patch.dict(sys.modules, {'probe_runtime_fixture': types.SimpleNamespace(
                        __file__=str(root / 'python-runtime/unittest.py'),
                        __cached__=str(root / 'python-runtime/__pycache__/unittest.pyc'))}), \
                    patch.object(evidence.shutil, 'which', return_value=str(root / 'tools/cargo')):
                initial = evidence.context(root, str(root / 'tools/cargo'), 'target', {})
                for name in ('rust/core/lib.rs', '.cargo/config.toml', 'dependency/Cargo.toml', 'tools/rustc',
                             'python-runtime/unittest.py', 'python-runtime/__pycache__/unittest.pyc'):
                    path = root / name
                    original = path.read_text()
                    path.write_text('# changed')
                    self.assertNotEqual(initial, evidence.context(root, str(root / 'tools/cargo'), 'target', {}))
                    path.write_text(original)
                path = root / 'rust/core/new.rs'
                path.write_text('added')
                self.assertNotEqual(initial, evidence.context(root, str(root / 'tools/cargo'), 'target', {}))


if __name__ == '__main__':
    unittest.main()
