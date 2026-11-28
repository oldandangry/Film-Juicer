# Diffusion checks

The frozen [native frame payload](frame-fixtures/native-payload.json) is shared
by the Rust diffusion frame cases and `Scanner.Gpu.Routes`. The Rust cases select
it through the [core diffusion frame fixtures](../../rust/film-juicer-core/tests/fixtures/optics/diffusion-frame/README.md),
which also hold the native contexts, edge payload and checked failures.
The strict C11 CUDA probe verifies the layout of all 36 resolved diffusion lanes.
The [tagged diffusion guide](https://github.com/oldandangry/Film-Juicer/blob/archive/spectral-cutover-2026-11/tests/diffusion/README.md)
records the original reference environment and capture commands.
