# Building Film-Juicer

The preset and toolchain details below apply to the product build. For cleanup
on `testing`, the [contributor checks](../CONTRIBUTING.md#required-checks)
supersede the verifier commands and zero-timeout rules in this guide. The
first cleanup PR publishes the direct commands with its build-graph changes.

Open this folder directly in Visual Studio 2026 and select a CMake preset.
The shared build uses Ninja, Rust 1.98.1, C++20 and CUDA C++20 with CUDA
Toolkit 13.2. CMake invokes Cargo for the two-crate Rust workspace and links the
resulting `film-juicer-plugin` static library into the OFX module.
See the [ownership and maintenance map](../CONTRIBUTING.md#ownership-and-maintenance-map)
for model, host, ABI and native execution entry points.
The consumed OpenFX 1.4/Resolve OFXS variant and nlohmann JSON 3.12.0 are vendored;
no sibling SDK checkout, Eigen, or package manager is required.

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
The verification runner defaults to at most eight available CPU workers and
honours a positive `CMAKE_BUILD_PARALLEL_LEVEL`; ordinary CMake builds also accept
`--parallel <count>`. Changing this scheduling limit preserves compiled artifacts.
The desktop `CODEX_TOOL_CALL_ID` is removed from compiler/test subprocess environments and
freshness identities. Preparation consumes `CMAKE_BUILD_PARALLEL_LEVEL` before
removing it from configure/compiler subprocess environments, so it cannot become
an untracked compiler input. Other environment inputs remain tracked.
CMake creates the required `CACHEDIR.TAG` in its dedicated Cargo directory so
the pinned Cargo can perform package-scoped cleanup safely when preparation
requires it. CUDA library discovery uses the toolkit already selected by NVCC
and system library locations, avoiding unrelated environment `PATH` scans.

For local compiler analysis, prepare `juicer_analysis_inputs` through the native
owner instead of the complete product/test build:

```sh
python -B scripts/verify.py prepare --preset linux-debug --testing ON --targets juicer_analysis_inputs
python -B scripts/check-quality.py --base <accepted-base> --preset linux-debug
```

The same target works with the Windows preset. It builds the ordinary native
and retained host objects required by isolation checks, the production Rust
archive and generated diffusion reference header consumed by compiler analysis.
The compilation database still contains every configured test-hook variant.
Quality analysis defaults to at most four CPUs available to the process;
`--jobs <count>` provides an explicit limit, including `--jobs 1` for serial
diagnosis. Compiler commands, checks and before/after content verification are
unchanged by the worker count.
Preparation configures and builds the selected closure; it does not run tests or
create a qualification result. Select checks using the
[validation cadence](../tests/README.md#validation-cadence).

The build preserves explicit SM 7.5 machine code plus compute_75 PTX. It does
not select the build machine's CPU/GPU architecture. Windows uses the CUDA
13.2 hybrid runtime through an explicit CMake integration; Linux links cudart
and cuFFT statically. The Linux module also links its GNU C++ support runtime
statically and localizes those implementation symbols at the OFX C boundary so
host libraries cannot interpose an incompatible libstdc++ implementation. See
[Linux installation](linux.md) for host requirements.

## Portra 400 Film v1

The standard Film-Juicer bundle includes the spectral scanner, the Portra 400
Film v1 grain model, and the grain bank and notices. It uses the established
OFX identifier, `com.juicer.Juicer`. The grain model defaults to **Current**.
Use the normal Release preset and archive target. This example stages Windows
Release to a named directory so an existing staged bundle stays available.

```sh
python -B scripts/verify.py prepare --preset windows-clang-release \
  --testing OFF --targets bundle-archive --stage-subdirectory film-v1
```

The named stage contains `juicer.ofx.bundle` and `juicer-Win64.tar.gz`
under `out/stage/windows-clang-release/film-v1/`. Use `linux-release` for the
Linux bundle. See the [Film v1 settings and limits](../README.md#portra-400-film-v1).

Archived Grain Preview projects use `com.juicer.Juicer.GrainPreview`. On a
project copy, open the project with its archived plug-in available and record
the preview node's settings, keyframes and connections. Add the regular Juicer
effect, reproduce those settings and animations, and compare at proxy and Full
resolution before removing the preview node from the copy. Keep the source
project export and its corresponding bundle for recovery. Standard Juicer nodes
keep their existing identifier and start with **Current** when they have no
saved grain-model value.

## Public test suite

CTest registers native tests by domain. The `host` label needs no GPU at
runtime, although a product-linked build requires CUDA 13.2. The `gpu` label
requires a supported driver and device. The `reference` label identifies
fixture provenance. Python tests use Python 3.13 and `tests/requirements.txt`.

See the [public test guide](../tests/README.md) for fixture policy and
reference maintenance. The [tagged build guide](https://github.com/oldandangry/Film-Juicer/blob/archive/spectral-cutover-2026-11/docs/building.md#public-test-suite)
retains the original verifier examples. Fast math is forbidden for
correctness tests. Resolve render and lifecycle acceptance remain separate.

When qualifying a native Linux Resolve environment, run the focused OFX
host-compatibility probe both normally and with Resolve's `libProResRAW.so`
preloaded as documented in [the OFX probe guide](../tests/ofx/README.md). This
protects Load/Describe/Unload from host C++ runtime symbol collisions; it still
does not replace a Resolve render.

Visual Studio uses this CMake project directly; the retired MSBuild solution and
project files are not required. MSVC v145 and the Windows SDK remain required as
the Windows ABI, standard-library, and NVCC host toolchain.
