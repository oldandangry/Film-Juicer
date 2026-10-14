"""Share checked compiler output only within an explicitly named qualification."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import tomllib
import uuid


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def compiler_environment(environment=None) -> dict[str, str]:
    result = dict(os.environ if environment is None else environment)
    for name in ('RUSTFLAGS', 'CARGO_BUILD_RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CLIPPY_CONF_DIR'):
        result.pop(name, None)
    return result


def execution_environment(environment: dict) -> dict:
    # Remove orchestration variables from the compiler's actual environment,
    # not just its fingerprint. Unknown compiler/configuration variables remain.
    # CTest injects CMAKE_CONFIG_TYPE even for Ninja. Rust profiles are already
    # bound by each case and its explicit Cargo --release argument.
    controls = {'QUALITY_BASE', 'SHLVL', '_', 'CTEST_INTERACTIVE_DEBUG_MODE', 'CMAKE_CONFIG_TYPE',
                'JUICER_TEST_ARTIFACT_DIR', 'JUICER_RUST_PROBE_DIR', 'JUICER_RUST_PROBE_PROFILE',
                'JUICER_RUST_PROBE_MANIFEST', 'JUICER_QUALIFICATION_ID', 'JUICER_CARGO',
                'JUICER_RUST_TARGET'}
    result = {k: v for k, v in environment.items()
              if k not in controls and not k.startswith(('GITHUB_', 'RUNNER_'))}
    if os.name == 'nt':
        # Match CMake/CTest's console routing for captured Windows tool output.
        result['VSCONSOLEOUTPUT'] = '1'
    return result


def qualification_id(value: str) -> str:
    if str(uuid.UUID(value)) != value:
        raise ValueError('qualification identity must be a canonical UUID')
    return value


def process_token(pid: int):
    if os.name != 'nt':
        try:
            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                return None
            return [Path('/proc/sys/kernel/random/boot_id').read_text().strip(), fields[19]]
        except FileNotFoundError:
            return None
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    api.OpenProcess.restype = ctypes.c_void_p
    api.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    api.GetProcessTimes.argtypes = [ctypes.c_void_p, *([ctypes.POINTER(ctypes.c_uint64)] * 4)]
    api.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = api.OpenProcess(0x1000, False, pid)
    if not handle:
        if ctypes.get_last_error() == 87:
            return None
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        status = ctypes.c_uint32()
        if not api.GetExitCodeProcess(handle, ctypes.byref(status)):
            raise ctypes.WinError(ctypes.get_last_error())
        if status.value != 259:  # STILL_ACTIVE
            return None
        times = [ctypes.c_uint64() for _ in range(4)]
        if not api.GetProcessTimes(handle, *(ctypes.byref(value) for value in times)):
            raise ctypes.WinError(ctypes.get_last_error())
        return [times[0].value]
    finally:
        api.CloseHandle(handle)


def host_identity() -> str:
    if os.name == 'nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\Microsoft\Cryptography') as key:
            identity = winreg.QueryValueEx(key, 'MachineGuid')[0]
    else:
        identity = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    return hashlib.sha256((platform.node() + ':' + identity).encode()).hexdigest()


def context(root: Path, cargo: str, target: str, environment: dict) -> dict:
    """Bind actual files and membership; no Git timestamp or old-result lookup."""
    environment = execution_environment(environment)
    host = host_identity()
    paths = [root / name for name in ('Cargo.toml', 'Cargo.lock', 'clippy.toml',
             'rust-toolchain.toml', 'rustfmt.toml', 'scripts/check-quality.py')]
    for directory in ('rust', 'tests/quality', '.cargo'):
        paths.extend(p for p in (root / directory).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix not in {'.pyc', '.pyo'})
    files = {p.relative_to(root).as_posix(): digest(p) for p in sorted(paths)}
    resolved_parents = {}

    def resolved_file(path):
        # Resolve each owning directory once per inventory, while preserving
        # canonical identities for individual symlinks. No file hash is reused.
        if path.parent not in resolved_parents:
            resolved_parents[path.parent] = path.parent.resolve()
        canonical = resolved_parents[path.parent] / path.name
        return canonical.resolve() if canonical.is_symlink() else canonical

    def output(command):
        return subprocess.check_output(command, cwd=root, env=environment, text=True,
                                       encoding='utf-8', timeout=60).strip()

    metadata = json.loads(output([cargo, 'metadata', '--locked', '--offline', '--format-version', '1']))
    external = {Path(sys.executable).resolve(), Path(shutil.which(cargo) or cargo).resolve()}
    configurations = set()
    directories = [root, *root.parents]
    for directory in directories:
        for name in ('config', 'config.toml'):
            path = directory / '.cargo' / name
            if path.is_file():
                configurations.add(path.resolve())
    for name in ('config', 'config.toml'):
        path = Path(environment.get('CARGO_HOME', Path.home() / '.cargo')) / name
        if path.is_file():
            configurations.add(path.resolve())
    external.update(configurations)

    def bind_tool(value, directory=root):
        if value:
            path = Path(value)
            resolved = directory / path if path.parent != Path('.') else Path(shutil.which(value, path=environment.get('PATH')) or value)
            external.add(resolved.resolve(strict=True))

    for name, value in environment.items():
        if name in {'RUSTC', 'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER', 'RUSTDOC',
                    'CARGO_BUILD_RUSTC', 'CARGO_BUILD_RUSTC_WRAPPER', 'CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER'} or (
                name.startswith('CARGO_TARGET_') and name.endswith('_LINKER')):
            bind_tool(value)
    for config in configurations:
        configuration = tomllib.loads(config.read_text(encoding='utf-8'))
        for name in ('rustc', 'rustc-wrapper', 'rustc-workspace-wrapper', 'rustdoc'):
            bind_tool(configuration.get('build', {}).get(name), config.parent.parent)
        for target_config in configuration.get('target', {}).values():
            bind_tool(target_config.get('linker'), config.parent.parent)
    rustup = shutil.which('rustup', path=environment.get('PATH'))
    for name in ('cargo', 'rustc', 'cargo-clippy', 'clippy-driver'):
        path = Path(output([rustup, 'which', name])) if rustup else Path(cargo).resolve().with_name(name + ('.exe' if os.name == 'nt' else ''))
        external.add(path.resolve())
        if name == 'rustc':
            sysroot = Path(output([str(path), '--print', 'sysroot']))
    for directory in (sysroot / 'lib', sysroot / 'bin'):
        external.update(resolved_file(p) for p in directory.glob('*') if p.is_file() and
                        (p.suffix in {'.dll', '.so', '.dylib'} or '.so.' in p.name))
    external.update(resolved_file(p) for p in (sysroot / 'lib/rustlib' / target / 'lib').glob('*') if p.is_file())
    for package in metadata['packages']:
        if package.get('source'):
            external.update(resolved_file(p) for p in Path(package['manifest_path']).parent.rglob('*') if p.is_file())
    runtime = Path(sys.base_prefix)
    for pattern in ('*.dll', '*.pyd', 'python*.zip', '*._pth', 'sitecustomize.py', 'lib/libpython*.so*'):
        external.update(resolved_file(p) for p in runtime.glob(pattern) if p.is_file())
    # Bind code actually loaded by these consumers. The end check also rejects
    # a changed module set; unrelated standard-library files are not inputs.
    for module in tuple(sys.modules.values()):
        for attribute in ('__file__', '__cached__'):
            path = getattr(module, attribute, None)
            if path and Path(path).is_file():
                external.add(resolved_file(Path(path)))
    customization = getattr(sys.modules.get('sitecustomize'), '__file__', None)
    if customization:
        external.add(resolved_file(Path(customization)))
    for path in (Path(sys.prefix) / 'pyvenv.cfg',):
        if path.is_file():
            external.add(resolved_file(path))
    return {'root': str(root.resolve()), 'platform': sys.platform, 'machine': platform.machine(),
            'target': target, 'host': host, 'files': files, 'external': {str(p): digest(p) for p in sorted(external)},
            'python_flags': list(sys.flags), 'python_options': dict(sys._xoptions),
            'environment_sha256': hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()}


def read_producer(manifest: Path, identity: str, *, phase="published") -> tuple[dict, Path, dict]:
    document = json.loads(manifest.read_text(encoding='utf-8'))
    if (document.get('schema') != 1 or document.get('phase') != phase
            or document.get('qualification') != qualification_id(identity)):
        raise ValueError('probe evidence has a different qualification or schema')
    receipt = Path(document['producer'])
    if digest(receipt) != document['producer_sha256']:
        raise ValueError('probe producer receipt changed')
    producer = json.loads(receipt.read_text(encoding='utf-8'))
    if producer['status'] != 'passed' or not producer.get('finished_at'):
        raise ValueError('probe producer did not complete successfully')
    if process_token(document['pid']) == document['process_token']:
        raise ValueError('probe producer process has not completed')
    return document, receipt.parent, producer


class ProbeEvidence:
    """Existing test assertions consume either fresh or verified original output."""

    def __init__(self, root: Path, suite: str, cargo: str, target: str,
                 profiles, environment: dict, artifacts: Path):
        self.root, self.suite, self.cargo, self.target = root, suite, cargo, target
        self.environment, self.artifacts = environment, artifacts
        self.profiles = ['release' if release else 'dev' for release in profiles]
        self.started = time.perf_counter()
        self.initial = context(root, cargo, target, environment)
        self.context_seconds = time.perf_counter() - self.started
        self.records, self.used = {}, set()
        self.manifest = environment.get('JUICER_RUST_PROBE_MANIFEST')
        self.reusing = self.manifest is not None
        if self.reusing:
            self.manifest = Path(self.manifest).resolve(strict=True)
            self.manifest_hash = digest(self.manifest)
            document, directory, producer = read_producer(self.manifest, environment.get('JUICER_QUALIFICATION_ID', ''))
            ledger = directory / document['suites'][suite]
            ledger.resolve().relative_to(directory.resolve())
            self.ledger = ledger
            if digest(ledger) != producer['files'][ledger.relative_to(directory).as_posix()]:
                raise ValueError('probe case ledger changed')
            captured = json.loads(ledger.read_text(encoding='utf-8'))
            if captured['schema'] != 1 or captured['context'] != self.initial or captured['suite'] != suite:
                raise ValueError('probe compiler context changed; run fresh qualification')
            if not set(self.profiles) <= set(captured['profiles']):
                raise ValueError('probe evidence does not cover the requested profile')
            all_keys = set()
            for record in captured['cases']:
                if record['case']['profile'] not in captured['profiles']:
                    raise ValueError('unexpected probe profile')
                key = json.dumps(record['case'], sort_keys=True)
                if key in all_keys:
                    raise ValueError('duplicate probe case')
                all_keys.add(key)
                log = directory / record['log']
                log.resolve().relative_to(directory.resolve())
                if digest(log) != record['log_sha256'] or producer['files'][record['log']] != record['log_sha256']:
                    raise ValueError('probe compiler log changed')
                if record['case']['profile'] in self.profiles:
                    self.records[key] = record
            if not self.records:
                raise ValueError('probe evidence contains no requested cases')
            self.producer_directory = directory

    def run(self, case: dict, command: list[str], workspace: Path, log: Path) -> subprocess.CompletedProcess:
        def configurations(directory):
            return [path.resolve() for parent in (directory, *directory.parents)
                    for name in ('config', 'config.toml')
                    if (path := parent / '.cargo' / name).is_file()]

        if configurations(workspace) != configurations(self.root):
            raise ValueError('probe output directory has different Cargo configuration')
        # The only command difference allowed across profiles' cache trees is
        # --target-dir's output location. All compiler selection/flags stay bound.
        normalized = list(command[1:])
        normalized[normalized.index('--target-dir') + 1] = '<probe-cache>'
        case = {**case, 'command': normalized, 'target': self.target}
        key = json.dumps(case, sort_keys=True)
        if key in self.used:
            raise ValueError('probe case requested twice')
        self.used.add(key)
        if self.reusing:
            if key not in self.records:
                raise ValueError(f'missing matching probe case: {case}')
            record = self.records[key]
            path = self.producer_directory / record['log']
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != record['log_sha256']:
                raise ValueError('probe compiler log changed during consumption')
            begin, end = record['stdout_range']
            return subprocess.CompletedProcess(command, record['returncode'],
                                               data[begin:end].decode('utf-8'), data[end:].decode('utf-8'))
        started = time.perf_counter()
        result = subprocess.run(command, cwd=workspace, env=execution_environment(self.environment), capture_output=True,
                                text=True, encoding='utf-8', timeout=60, check=False)
        prefix = (subprocess.list2cmdline(command) + '\n').encode('utf-8')
        stdout = result.stdout.encode('utf-8')
        log.write_bytes(prefix + stdout + result.stderr.encode('utf-8'))
        self.records[key] = {'case': case, 'returncode': result.returncode,
                             'log': log.resolve().relative_to(self.artifacts.parent.resolve()).as_posix(),
                             'log_sha256': digest(log), 'stdout_range': [len(prefix), len(prefix) + len(stdout)],
                             'seconds': time.perf_counter() - started}
        return result

    def finish(self) -> None:
        started = time.perf_counter()
        if context(self.root, self.cargo, self.target, self.environment) != self.initial:
            raise ValueError('probe inputs changed during execution or consumption')
        self.context_seconds += time.perf_counter() - started
        if self.used != self.records.keys():
            raise ValueError('unexpected or unconsumed probe cases')
        if self.reusing:
            if digest(self.manifest) != self.manifest_hash:
                raise ValueError('probe manifest changed during consumption')
            _, directory, producer = read_producer(self.manifest, self.environment['JUICER_QUALIFICATION_ID'])
            if digest(self.ledger) != producer['files'][self.ledger.relative_to(directory).as_posix()]:
                raise ValueError('probe case ledger changed during consumption')
            for record in self.records.values():
                if digest(self.producer_directory / record['log']) != record['log_sha256']:
                    raise ValueError('probe compiler log changed during consumption')
            self.artifacts.mkdir(parents=True, exist_ok=True)
            write_json(self.artifacts / 'verification.json', {
                'manifest': str(self.manifest), 'manifest_sha256': self.manifest_hash,
                'suite': self.suite, 'profiles': self.profiles, 'cases': len(self.used),
                'case_set_sha256': hashlib.sha256(json.dumps(sorted(self.used)).encode()).hexdigest(),
                'context_seconds': self.context_seconds, 'seconds': time.perf_counter() - self.started})
            print(f'verified existing probe evidence: {self.suite}, {len(self.used)} cases, {self.manifest}', flush=True)
        else:
            write_json(self.artifacts / 'cases.json', {'schema': 1, 'suite': self.suite,
                       'profiles': self.profiles, 'context': self.initial,
                       'cases': list(self.records.values()), 'context_seconds': self.context_seconds,
                       'seconds': time.perf_counter() - self.started})


def prepare_manifest(receipt: Path, destination: Path, identity: str) -> bool:
    """Stage a request; it is not consumable while the producer is running."""
    qualification_id(identity)
    producer = json.loads(receipt.read_text(encoding='utf-8'))
    if producer['status'] != 'passed' or not producer.get('finished_at'):
        raise ValueError('cannot publish an incomplete or failed quality gate')
    suites = {suite: f'rust-{suite}/cases.json' for suite in ('naming', 'boundaries')}
    if not any((receipt.parent / name).exists() for name in suites.values()):
        return False  # This selection did not request Rust checks; CTest runs fresh.
    for suite, name in suites.items():
        path = receipt.parent / name
        if digest(path) != producer['files'][name]:
            raise ValueError('probe ledger is missing or changed')
        ledger = json.loads(path.read_text(encoding='utf-8'))
        if ledger['suite'] != suite or set(ledger['profiles']) != {'dev', 'release'} or not ledger['cases']:
            raise ValueError('complete both-profile probe evidence is required')
    document = {'schema': 1, 'phase': 'pending', 'qualification': identity, 'producer': str(receipt.resolve()),
                'pid': os.getpid(), 'process_token': process_token(os.getpid()),
                'producer_sha256': digest(receipt), 'suites': suites}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with pending_path(destination).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(document, indent=2) + '\n')
    return True


def pending_path(destination: Path) -> Path:
    return destination.with_name(destination.name + '.pending')


def publish(destination: Path, identity: str) -> None:
    """The caller invokes this only after its complete quality command drains."""
    document, directory, producer = read_producer(pending_path(destination), identity, phase='pending')
    for relative in producer['files']:
        path = directory / relative
        path.resolve().relative_to(directory.resolve())
        if digest(path) != producer['files'][relative]:
            raise ValueError('quality evidence changed before publication')
    document['phase'] = 'published'
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(document, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Publish explicit probe evidence after quality completion.')
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--qualification-id', required=True)
    options = parser.parse_args()
    publish(options.manifest, options.qualification_id)
    print(f'Published probe evidence: {options.manifest}', flush=True)
