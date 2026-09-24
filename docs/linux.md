# Linux bundle

Linux rendering requires x86-64, an NVIDIA Turing or newer GPU, and a working
Resolve CUDA installation. The public host and GPU CTest suite has passed in
WSL2 and native Arch/CachyOS Debug and Release configurations. A native
Arch/CachyOS bundle has also been installed and reported working in Resolve.
This is primary-machine evidence, not a general distro compatibility claim.
See [the migration record](../tests/MIGRATION.md) for the completed validation
and measured toolchain/driver context.

Extract the archive, close Resolve, and copy the complete `juicer.ofx.bundle`
directory into `/usr/OFX/Plugins` (or a host-configured OFX search directory).
Keep the `Contents/Resources` and `Contents/Legal` directories alongside the
binary. Remove the same bundle directory while Resolve is closed to uninstall.
Preserve any previously installed bundle outside the OFX search path before
replacing it.

The Linux binary links the CUDA runtime, cuFFT, and its GNU C++ support runtime
statically. GNU runtime implementation symbols remain local to the OFX module
to avoid collisions with C++ runtimes already loaded by Resolve. The binary
still requires the NVIDIA driver and compatible system glibc. Do not copy the
NVIDIA driver, glibc, or a system dynamic loader into the bundle, or set a
global `LD_LIBRARY_PATH` to a development toolkit.

After installation, check discovery, all four negative/positive direct/print
routes, alpha, grain and physical artifacts, camera/enlarger diffusion, halation,
scanner effects, cancellation, instance deletion and clean Resolve shutdown.
A standalone loader check or GPU harness result does not establish Resolve
acceptance. If loading fails, retain the Resolve log and inspect the trusted
binary with `ldd`, `readelf -d`, and `readelf --version-info` before rebuilding.

The current native Arch artifact was linked against glibc 2.44 and requires
symbols through GLIBC 2.43. It is intended for the tested primary machine; use
the planned older-baseline release builder before distributing one archive
across Linux distributions.

The [build guide](building.md) describes the shared WSL/native `linux-debug`
and `linux-release` presets, while [the public test guide](../tests/README.md)
covers setup and execution. Always use a fresh build cache when moving between
WSL and a native Linux installation.
