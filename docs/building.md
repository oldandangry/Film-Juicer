# Building Film-Juicer

Open this folder directly in Visual Studio 2026 and select a CMake preset.
The shared build uses Ninja, C++20 and CUDA C++20 with CUDA Toolkit 13.2.
The consumed OpenFX 1.4/Resolve OFXS variant and nlohmann JSON 3.12.0 are vendored;
no sibling SDK checkout, Eigen, or package manager is required.

Use CMake 4.3.1 and Ninja 1.13.2 (the tested Visual Studio versions are
4.3.1-msvc1 and 1.13.2). Install the Visual Studio C++/CMake and Clang tools,
MSVC v145, the Windows SDK, and CUDA 13.2. Set `CUDA_PATH_V13_2` to that toolkit.
Visual Studio supplies the x64 developer environment when selecting a Windows
preset. Command-line Windows builds need an x64 developer command prompt with
CMake, Ninja and the VS LLVM tools available. ClangCl builds use MSVC explicitly
as NVCC's host compiler and retain ThinLTO for C++.

| Preset | Compiler | Configuration |
| --- | --- | --- |
| `windows-msvc-debug` | MSVC + NVCC/MSVC | Debug |
| `windows-msvc-release` | MSVC + NVCC/MSVC | Release |
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

Linux uses `/usr/local/cuda-13.2/bin/nvcc` and `/usr/bin/g++-13` by default.
Select a compatible installed GCC 10–15 and toolkit path in ignored
`CMakeUserPresets.json` if needed. Both the C++ and NVCC host compiler must use
that GCC. Do not bypass NVCC's supported-compiler checks. WSL and native Linux
use these same presets in separate checkouts or build caches. WSL needs only
the Linux CUDA toolkit and Windows NVIDIA driver integration; never install a
Linux display driver into WSL.

The build preserves explicit SM 7.5 machine code plus compute_75 PTX. It does
not select the build machine's CPU/GPU architecture. Windows uses the CUDA
13.2 hybrid runtime through an explicit CMake integration; Linux links cudart
and cuFFT statically. The Linux module also links its GNU C++ support runtime
statically and localizes those implementation symbols at the OFX C boundary so
host libraries cannot interpose an incompatible libstdc++ implementation. See
[Linux installation](linux.md) for host requirements.

To build the opt-in tracked acceptance executable:

```sh
cmake --preset linux-release -DJUICER_BUILD_VALIDATION=ON
cmake --build --preset linux-release --target ScatterHalationValidation
out/build/linux-release/validation/ScatterHalationValidation \
  --case-group focused-cuda-gate-3 \
  --fixture-root validation/scatter_halation/fixtures \
  --resource-root Resources --scratch-root out/validation/linux-release
```

Other migration groups are `host-gate-1`, `prepared-gate-2`,
`unpublished-routes`, `zero-work`, and `lifecycle`. Keep the committed fixture
manifest and binary together; the manifest's reference revision remains the
authority. Fast math is forbidden for this acceptance build. Harness results
do not replace Resolve render and lifecycle acceptance.

On a native Resolve installation, run the focused OFX host-compatibility probe
both normally and with Resolve's `libProResRAW.so` preloaded as documented in
`validation/ofx_host_compat/README.md`. This protects Load/Describe/Unload from
host C++ runtime symbol collisions; it still does not replace a Resolve render.

The existing MSBuild solution remains available during migration. A native
Linux bundle is now working on the primary Arch/CachyOS machine, but retirement
still requires the current Windows CMake/Resolve revalidation and final migration
cleanup.
