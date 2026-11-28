# OFX host fixtures

The linked host cases call the production discovery shim, Rust action dispatcher
and native CUDA owner. The [production render probe](../ffi/production-ofx-render.cpp)
loads the staged module and its resources. It compares guarded Direct and Print
output against the selected `fixtures/spectral-v1/ordinary-${preset}.json` and
exercises edited Print and Grain v2 thumbnail returns, post-enqueue cancellation,
and exactly-once release of both acquired images. The Grain v2 staged cases use
the natural OFX instance seed.

Use the [test guide](../README.md) for the host and serialized GPU selections.
The [tagged OFX guide](https://github.com/oldandangry/Film-Juicer/blob/archive/spectral-cutover-2026-11/tests/ofx/README.md)
records the migration captures and their replay commands.

## Linux OFX host compatibility probe

`probe.py` exercises Juicer's OFX Load, Describe, and Unload actions with a
narrow synthetic host. Pass one or more `--preload` paths to reproduce symbol
resolution in a host that has already loaded those libraries globally.

The Linux host CTest group runs the synthetic lifecycle automatically. The
installed-Resolve-library form below is a host-qualification compatibility
check rather than an ordinary automated test. It passed for this migration on
2026-09-19; repeat it when qualifying a different Linux Resolve environment.

For DaVinci Resolve on Linux, run both forms against the trusted build output:

```sh
PYTHONDONTWRITEBYTECODE=1 python3.13 tests/ofx/probe.py \
  out/build/linux-release/juicer.ofx

PYTHONDONTWRITEBYTECODE=1 python3.13 tests/ofx/probe.py \
  out/build/linux-release/juicer.ofx \
  --preload /opt/resolve/libs/libProResRAW.so
```

The preloaded check protects the plug-in's C++ runtime boundary from host
libraries that expose incompatible libstdc++ implementation symbols. It is a
focused lifecycle and symbol-compatibility check, not Resolve render acceptance.

The probe intentionally stops before `DescribeInContext`, where Juicer first
needs its process root. A Debug build can therefore verify that unload remains
non-constructing by running the probe under GDB with a pending constructor
breakpoint:

```sh
gdb -q -batch \
  -ex 'set debuginfod enabled off' \
  -ex 'set breakpoint pending on' \
  -ex 'break JuicerProcess::Root::Root()' \
  -ex run --args python3.13 tests/ofx/probe.py \
  out/build/linux-debug/juicer.ofx
```

The expected result is normal inferior exit without the breakpoint being hit.
