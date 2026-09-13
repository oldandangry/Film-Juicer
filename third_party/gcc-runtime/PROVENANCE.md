# GNU runtime linkage

The Linux Film-Juicer OFX module statically links the GNU C++ and GCC support
runtime libraries so their implementation symbols remain local to the plug-in.
This prevents host libraries that contain another libstdc++ implementation from
interposing incompatible C++ runtime symbols across the OFX C API boundary.

The validated native build uses the Arch Linux `gcc13-libs` package version
`13.4.1+r80+gd6ebfe4-1`. The distributable module must be rebuilt with the
supported GCC selected by the CMake preset and rechecked for dynamic GNU runtime
dependencies and exported symbols.

The linked runtime code is governed by GPL version 3 together with the GCC
Runtime Library Exception version 3.1. Copies of both texts are retained in this
directory and installed into the bundle's legal notices.

Upstream source: <https://gcc.gnu.org/>
