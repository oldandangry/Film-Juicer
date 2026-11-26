# GNU runtime linkage

The Linux Film-Juicer OFX module statically links the GNU C++ and GCC support
runtime libraries so their implementation symbols remain local to the plug-in.
This prevents host libraries that contain another libstdc++ implementation from
interposing incompatible C++ runtime symbols across the OFX C API boundary.

The Linux presets select GCC 13. The archives come from the selected
compiler's GNU runtime development packages. On Ubuntu 24.04 these are
provided by `libstdc++-13-dev` and `libgcc-13-dev`. Package inspection checks
that the module has no dynamic GNU C++ dependency. The Rust toolchain links
the system `libgcc_s.so.1` for unwinding.

The linked runtime code is governed by GPL version 3 together with the GCC
Runtime Library Exception version 3.1. Copies of both texts are retained in this
directory and installed into the bundle's legal notices.

Upstream source: <https://gcc.gnu.org/>
