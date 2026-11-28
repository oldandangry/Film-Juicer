# Building Film-Juicer

Open this folder directly in Visual Studio 2026 and select a CMake preset.
The shared build uses Ninja, Rust 1.98.1, C++20 and CUDA C++20 with CUDA
Toolkit 13.2. CMake invokes Cargo for the two-crate Rust workspace and links the
resulting `film-juicer-plugin` static library into the OFX module.
See the [ownership and maintenance map](../CONTRIBUTING.md#ownership-and-maintenance-map)
for model, host, ABI and native execution entry points.
The OpenFX 1.4 headers and the native tests’ nlohmann JSON headers are vendored.

Use CMake 4.3 or newer and Ninja. Install the Visual Studio C++/CMake and Clang
tools, MSVC v145, the Windows SDK, and CUDA 13.2. Set `CUDA_PATH_V13_2` to that toolkit.
Visual Studio supplies the x64 developer environment when selecting a Windows
preset. Command-line Windows builds need an x64 developer command prompt with
CMake, Ninja and the VS LLVM tools available. ClangCl builds use MSVC explicitly
as NVCC's host compiler; both configurations use ThinLTO for C++.
Linux also requires LLVM LLD on `PATH` as `ld.lld` (for example, LLVM 22's
`bin` directory). The Linux presets select LLD without changing the GCC or
NVCC compilers, CUDA runtime linkage, or exported OFX symbols.

Install the exact Rust/Cargo 1.98.1 toolchain selected by
`rust-toolchain.toml` before configuring. rustfmt and Clippy are also required
by the tracked quality gate. With the current platform's own rustup:

```sh
rustup toolchain install 1.98.1 --profile minimal --component rustfmt --component clippy
rustc --version --verbose
cargo --version
```

Install and run this separately on each supported target:
`x86_64-pc-windows-msvc` in Windows and `x86_64-unknown-linux-gnu` in WSL or
native Linux. The Windows installation does not supply Linux Cargo or rustc.
CMake rejects a missing tool, a different Rust release, or the wrong native
target instead of falling back.

| Preset | Compiler | Configuration |
| --- | --- | --- |
| `windows-clang-debug` | ClangCl + NVCC/MSVC | Optimised Debug with ThinLTO |
| `windows-clang-release` | ClangCl + NVCC/MSVC | Release with ThinLTO |
| `linux-debug` | GCC + NVCC/GCC | Optimised Debug |
| `linux-release` | GCC + NVCC/GCC | Release |

Debug and Release share compiler optimisation levels: `/O2` for Windows host
C++, `-O3` for Linux host C++ and CUDA, and Rust `opt-level = 3` with sixteen
code-generation units. Debug retains symbols, assertions, Rust debug assertions,
overflow checks, and invariant diagnostics. Both configurations retain standard
C/C++ assertions and Rust overflow checks. Optimised debugging can show inlined
calls, reordered stepping, or variables that have been optimised away. Fast math
remains disabled; the existing numerical fixtures are unchanged.
MSVC `/RTC` checks are incompatible with `/O2`; optimised Debug retains `/sdl`
and assertions instead of requesting those incompatible compiler options.

Native file diagnostics use the compile gate in [Logging.h](../src/Logging.h)
and the sink in [Logging.cpp](../src/Logging.cpp). Debug uses
`JUICER_DIAGNOSTICS` for the runtime level (0–3, default 0) and optionally
`JUICER_DIAGNOSTICS_PATH` for the output file. The standard Release presets
compile out the sink and trace calls.

For example:

```sh
cmake --preset linux-release
cmake --build --preset linux-release
cmake --install out/build/linux-release
cmake --build --preset linux-release --target bundle-archive
```

Replace the preset name for Windows. Configure caches live in `out/build`,
staged bundles and archives in `out/stage/<preset>` (the configured install prefix).
Install stages locally; it does not write to a system OFX directory. The archive
target rebuilds and stages into a clean bundle tree before packaging resources
and notices. Windows packaging includes the required CUDA and MSVC runtime
files; the existing installer may consume the staged bundle.

Linux presets use `/usr/bin/g++-13` for C++ and NVCC host compilation. They do
not encode a CUDA installation path. If CUDA 13.2 is not the system default,
select it before configuring:

```sh
export CUDAToolkit_ROOT=/absolute/path/to/cuda-13.2
export CUDACXX="$CUDAToolkit_ROOT/bin/nvcc"
```

Do not bypass NVCC's supported-compiler checks. WSL and native Linux use these
same presets in separate checkouts or build caches. WSL needs only the Linux
CUDA toolkit and Windows NVIDIA driver integration; never install a Linux
display driver into WSL.

Keep a Windows checkout and its build caches on the Windows filesystem, and a
WSL checkout and its caches on the Linux filesystem, such as under `/home`.
Compiling Linux sources and writing large static-linked test binaries through
`/mnt/c` adds cross-filesystem overhead. Configure fresh caches in the appropriate
checkout; existing CMake/Cargo caches contain absolute paths and cannot be moved
while retaining their preparation evidence.

Ninja uses separate pools of at most eight compile jobs and two link jobs.
Limit total concurrency with `cmake --build --parallel <count>` or
`CMAKE_BUILD_PARALLEL_LEVEL`. Use a fresh `-B` directory and separate
`CMAKE_INSTALL_PREFIX` to check an independent closure beside an existing stage.

For analysis, configure the native preset and build `juicer_analysis_inputs`
to generate headers. Run clang-tidy against its compile database.

```sh
cmake --preset linux-debug -DBUILD_TESTING=ON
cmake --build --preset linux-debug --target juicer_analysis_inputs
python3.13 scripts/check.py tidy --build-dir out/build/linux-debug --clang-tidy clang-tidy-22
```

Use `scripts/format_touched_files.sh` for native formatting. See
[native ABI and binding maintenance](ffi.md) for pinned bindgen commands.

The build preserves explicit SM 7.5 machine code plus compute_75 PTX. It does
not select the build machine's CPU/GPU architecture. Windows uses the CUDA
13.2 hybrid runtime through an explicit CMake integration; Linux links cudart
and cuFFT statically. The Linux module also links its GNU C++ support runtime
statically and localizes those implementation symbols at the OFX C boundary so
host libraries cannot interpose an incompatible libstdc++ implementation. See
[Linux installation](linux.md) for host requirements.

## Film v1 grain banks

See [Film v1 bank authoring](film-v1-bank-authoring.md) to generate and compare a
stock bank with the pinned Swift tool in scratch.

The standard Film-Juicer bundle includes the spectral scanner, Film v1 banks
for Kodak Portra 400 and Kodak Gold 200, and their notices. It uses the established
OFX identifier, `com.juicer.Juicer`. The grain model defaults to **Current**.
Use the Release preset and `bundle-archive`. See the
[Film v1 settings and limits](../README.md#portra-400-and-gold-200-film-v1).

Archived Grain Preview projects use `com.juicer.Juicer.GrainPreview`. On a
project copy, open the project with its archived plug-in available and record
the preview node's settings, keyframes and connections. Add the regular Juicer
effect, reproduce those settings and animations, and compare at proxy and Full
resolution before removing the preview node from the copy. Keep the source
project export and its corresponding bundle for recovery. Standard Juicer nodes
keep their existing identifier and start with **Current** when they have no
saved grain-model value.

## Public test suite

The default testing-enabled build supplies CTest's executables, modules and
resources. Use the matching native preset. Run Windows and WSL GPU checks
one OS at a time. CTest uses a 300-second per-test timeout.

The following example uses WSL Debug. The default build includes the
registered binaries, generated headers, runtime resources and the staged
module used by the production render probe.

```sh
cmake --preset linux-debug -DBUILD_TESTING=ON
cmake --build --preset linux-debug
ctest --preset linux-debug -L '^host$' --no-tests=error --output-on-failure
cargo nextest run --workspace --locked --test-threads 1
cargo test --workspace --doc --locked
ctest --preset linux-debug -L '^gpu$' -LE '^multi-gpu$' --no-tests=error --output-on-failure
```

Repeat the CMake and CTest steps with the Release preset. Run these Cargo
commands for Release.

```sh
cargo nextest run --workspace --locked --release --test-threads 1
cargo test --workspace --doc --locked --release
```

## Testing-OFF production package

Configure the production closure separately, with testing and benchmarks off.
Run the package build in a toolchain environment without Python. Run both
strict C11 probes with CUDA hidden. Use Python 3.13 afterwards to inspect
the archive, dependencies, exports, resources and notices.

```sh
cmake --preset linux-release -DBUILD_TESTING=OFF -DJUICER_BUILD_BENCHMARKS=OFF
cmake --build --preset linux-release --target bundle-archive JuicerProductionOfxProbe JuicerProductionCudaAbiProbe
CUDA_VISIBLE_DEVICES=-1 out/build/linux-release/tests/bin/JuicerProductionOfxProbe out/stage/linux-release/juicer.ofx.bundle/Contents/Linux-x86-64/juicer.ofx
CUDA_VISIBLE_DEVICES=-1 out/build/linux-release/tests/bin/JuicerProductionCudaAbiProbe
python3.13 scripts/check.py bundle --stage out/stage/linux-release
```

On Windows use `windows-clang-release`, both `.exe` probes and the staged
`Contents/Win64/juicer.ofx`. Set `$env:CUDA_VISIBLE_DEVICES = '-1'` for the
probe processes. Keep the CUDA toolkit's `bin/x64` and `bin` directories on
PATH for linked executables. Run the bundle check from the VS x64 developer
shell with `dumpbin` available.

When qualifying a native Linux Resolve environment, run the focused OFX
host-compatibility probe both normally and with Resolve's `libProResRAW.so`
preloaded as documented in [the OFX probe guide](../tests/ofx/README.md). This
protects Load/Describe/Unload from host C++ runtime symbol collisions; it still
does not replace a Resolve render.

Visual Studio uses this CMake project directly; the retired MSBuild solution and
project files are not required. MSVC v145 and the Windows SDK remain required as
the Windows ABI, standard-library, and NVCC host toolchain.
