# Building Film-Juicer

Open this folder directly in Visual Studio 2026 and select a CMake preset.
The shared build uses Ninja, Rust 1.98.1, C++20 and CUDA C++20 with CUDA
Toolkit 13.2. CMake invokes Cargo for the two-crate Rust workspace and links the
resulting `film-juicer-plugin` static library into the OFX module.
The consumed OpenFX 1.4/Resolve OFXS variant and nlohmann JSON 3.12.0 are vendored;
no sibling SDK checkout, Eigen, or package manager is required.

Use CMake 4.3 or newer and Ninja. Install the Visual Studio C++/CMake and Clang
tools, MSVC v145, the Windows SDK, and CUDA 13.2. Set `CUDA_PATH_V13_2` to that toolkit.
Visual Studio supplies the x64 developer environment when selecting a Windows
preset. Command-line Windows builds need an x64 developer command prompt with
CMake, Ninja and the VS LLVM tools available. ClangCl builds use MSVC explicitly
as NVCC's host compiler; Release retains ThinLTO for C++.

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
| `windows-clang-debug` | ClangCl + NVCC/MSVC | Debug |
| `windows-clang-release` | ClangCl + NVCC/MSVC | Release with ThinLTO |
| `linux-debug` | GCC + NVCC/GCC | Debug |
| `linux-release` | GCC + NVCC/GCC | Release |

For example:

```sh
cmake --preset linux-release
cmake --build --preset linux-release
cmake --install out/build/linux-release
cmake --build --preset linux-release --target bundle-archive
```

Replace the preset name for Windows. Configure caches live in `out/build`,
staged bundles in `out/stage`, and archives in the selected build directory.
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

The build preserves explicit SM 7.5 machine code plus compute_75 PTX. It does
not select the build machine's CPU/GPU architecture. Windows uses the CUDA
13.2 hybrid runtime through an explicit CMake integration; Linux links cudart
and cuFFT statically. The Linux module also links its GNU C++ support runtime
statically and localizes those implementation symbols at the OFX C boundary so
host libraries cannot interpose an incompatible libstdc++ implementation. See
[Linux installation](linux.md) for host requirements.

## Public test suite

Public tests are organized by domain under `tests/` and use CTest as the
scheduler. Enable them on the normal build preset, build, and run the matching
test preset:

```sh
cmake --preset linux-release -DBUILD_TESTING=ON
cmake --build --preset linux-release
ctest --preset linux-release
```

Replace the preset name for Windows or Debug. The unfiltered run is the
complete automated suite for a supported GPU machine. `ctest --preset
linux-release -L host` selects tests that require no NVIDIA driver/device at
runtime; configuration and compilation still require CUDA 13.2. The `gpu`
label requires a supported driver/device, while `reference` describes fixture
authority and is not an execution requirement. Python tests require Python
3.13 and `tests/requirements.txt`.

See [the public test guide](../tests/README.md) for environment setup, domain
selection, fixture policy, diagnostics, benchmarks, and reference maintenance.
Fast math is forbidden for correctness tests. Automated results do not replace
Resolve render and lifecycle acceptance when a change touches that boundary.

When qualifying a native Linux Resolve environment, run the focused OFX
host-compatibility probe both normally and with Resolve's `libProResRAW.so`
preloaded as documented in [the OFX probe guide](../tests/ofx/README.md). This
protects Load/Describe/Unload from host C++ runtime symbol collisions; it still
does not replace a Resolve render.

Visual Studio uses this CMake project directly; the retired MSBuild solution and
project files are not required. MSVC v145 and the Windows SDK remain required as
the Windows ABI, standard-library, and NVCC host toolchain.
