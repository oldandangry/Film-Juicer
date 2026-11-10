# Locked Rust dependency notices

These notices cover every external registry release in the repository's
`Cargo.lock`. Film-Juicer statically links its production Rust
archive; this inventory includes runtime dependencies and a small superset of
compile-time proc-macro/build dependencies. It does not describe twelve shipped
dynamic libraries or change any dependency's license terms.

The files in each `<name>-<version>/` directory are complete, unmodified
top-level members of that release's crates.io `.crate` archive. Each archive
was checked with SHA-256 against its Cargo.lock registry checksum before copying
the listed members. The release manifest supplies the declared license expression.
This establishes consistency with the locked release, without claiming an
independent publisher-signature audit.

Copy source: the corresponding crates.io archive in
`C:/Users/Alexi/.cargo/registry/cache/index.crates.io-1949cf8c6b5b557f/`.
The local cache is only a copy source. Build, installation and runtime consume
the repository-owned files here and require neither that cache nor network access.
CMake installs this document and the notices under `Contents/Legal/rust/`.

The complete libm combined notice, memchr COPYING and alternative licenses, and
Unicode notice are retained alongside the MIT/Apache texts. Existing project and
native/vendor notices remain separately distributed under `Contents/Legal/`.

## itoa 1.0.18

Declared license: `MIT OR Apache-2.0`. Runtime JSON integer formatting.

Release archive: [itoa-1.0.18.crate](https://static.crates.io/crates/itoa/itoa-1.0.18.crate).
Cargo.lock SHA-256: `8f42a60cbdf9a97f5d2305f08a87dc4e09308d1276d28c869c684d7777685682`.

| Archive member / repository file | SHA-256 |
| --- | --- |
| [itoa-1.0.18/LICENSE-MIT](itoa-1.0.18/LICENSE-MIT) | `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3` |
| [itoa-1.0.18/LICENSE-APACHE](itoa-1.0.18/LICENSE-APACHE) | `62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a` |

## libm 0.2.15

Declared license: `MIT`. Runtime mathematics; also runs a build script.

Release archive: [libm-0.2.15.crate](https://static.crates.io/crates/libm/libm-0.2.15.crate).
Cargo.lock SHA-256: `f9fbbcab51052fe104eb5e5d351cf728d30a5be1fe14d9be8a3b097481fb97de`.

| Archive member / repository file | SHA-256 |
| --- | --- |
| [libm-0.2.15/LICENSE.txt](libm-0.2.15/LICENSE.txt) | `3823dda7cf046602f4b4e77ec8e227863dc4736037cc85bb33d9f19febe16bb7` |

## memchr 2.8.3

Declared license: `Unlicense OR MIT`. Runtime JSON parsing.

Release archive: [memchr-2.8.3.crate](https://static.crates.io/crates/memchr/memchr-2.8.3.crate).
Cargo.lock SHA-256: `cf8baf1c55e62ffcace7a9f06f4bd9cd3f0c4beb022d3b367256b91b87513d98`.

| Archive member / repository file | SHA-256 |
| --- | --- |
| [memchr-2.8.3/COPYING](memchr-2.8.3/COPYING) | `01c266bced4a434da0051174d6bee16a4c82cf634e2679b6155d40d75012390f` |
| [memchr-2.8.3/LICENSE-MIT](memchr-2.8.3/LICENSE-MIT) | `0f96a83840e146e43c0ec96a22ec1f392e0680e6c1226e6f3ba87e0740af850f` |
| [memchr-2.8.3/UNLICENSE](memchr-2.8.3/UNLICENSE) | `7e12e5df4bae12cb21581ba157ced20e1986a0508dd10d0e8a4ab9a4cf94e85c` |

## proc-macro2 1.0.107

Declared license: `MIT OR Apache-2.0`. Compile-time proc-macro support.

Release archive: [proc-macro2-1.0.107.crate](https://static.crates.io/crates/proc-macro2/proc-macro2-1.0.107.crate).
Cargo.lock SHA-256: `985e7ec9bb745e6ce6535b544d84d6cd6f7ad8bd711c398938ae983b91a766d9`.

| Archive member / repository file | SHA-256 |
| --- | --- |
| [proc-macro2-1.0.107/LICENSE-MIT](proc-macro2-1.0.107/LICENSE-MIT) | `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3` |
| [proc-macro2-1.0.107/LICENSE-APACHE](proc-macro2-1.0.107/LICENSE-APACHE) | `62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a` |

## quote 1.0.47

Declared license: `MIT OR Apache-2.0`. Compile-time code generation support.

Release archive: [quote-1.0.47.crate](https://static.crates.io/crates/quote/quote-1.0.47.crate).
Cargo.lock SHA-256: `1fbf4db142a473a8d80c26bbf18454ed458bf8d26c8219c331daecfdbd079001`.

| Archive member / repository file | SHA-256 |
| --- | --- |
| [quote-1.0.47/LICENSE-MIT](quote-1.0.47/LICENSE-MIT) | `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3` |
| [quote-1.0.47/LICENSE-APACHE](quote-1.0.47/LICENSE-APACHE) | `62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a` |

## serde 1.0.229

Declared license: `MIT OR Apache-2.0`. Runtime serialization/deserialization, with derive support.

Release archive: [serde-1.0.229.crate](https://static.crates.io/crates/serde/serde-1.0.229.crate).
Cargo.lock SHA-256: `4148590afebada386688f18773da617792bf2ef03ffc1e4cbd2b1d45b023e0ba`.

| Archive member / repository file | SHA-256 |
| --- | --- |
| [serde-1.0.229/LICENSE-MIT](serde-1.0.229/LICENSE-MIT) | `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3` |
| [serde-1.0.229/LICENSE-APACHE](serde-1.0.229/LICENSE-APACHE) | `62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a` |

## serde_core 1.0.229

Declared license: `MIT OR Apache-2.0`. Runtime serialization/deserialization core.

Release archive: [serde_core-1.0.229.crate](https://static.crates.io/crates/serde_core/serde_core-1.0.229.crate).
Cargo.lock SHA-256: `67dca2c9c51e58a4791a4b1ed58308b39c64224d349a935ab5039aa360942a48`.

| Archive member / repository file | SHA-256 |
| --- | --- |
| [serde_core-1.0.229/LICENSE-MIT](serde_core-1.0.229/LICENSE-MIT) | `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3` |
| [serde_core-1.0.229/LICENSE-APACHE](serde_core-1.0.229/LICENSE-APACHE) | `62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a` |

## serde_derive 1.0.229

Declared license: `MIT OR Apache-2.0`. Compile-time derive proc macros.

Release archive: [serde_derive-1.0.229.crate](https://static.crates.io/crates/serde_derive/serde_derive-1.0.229.crate).
Cargo.lock SHA-256: `e7a5d71263a5a7d47b41f6b3f06ba276f10cc18b0931f1799f710578e2309348`.

| Archive member / repository file | SHA-256 |
| --- | --- |
| [serde_derive-1.0.229/LICENSE-MIT](serde_derive-1.0.229/LICENSE-MIT) | `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3` |
| [serde_derive-1.0.229/LICENSE-APACHE](serde_derive-1.0.229/LICENSE-APACHE) | `62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a` |

## serde_json 1.0.151

Declared license: `MIT OR Apache-2.0`. Runtime JSON, including the selected float_roundtrip feature.

Release archive: [serde_json-1.0.151.crate](https://static.crates.io/crates/serde_json/serde_json-1.0.151.crate).
Cargo.lock SHA-256: `c841b55ecdae098c80dcae9cf767f6f8a0c2cdb3416bbef72181df4d0fe73f14`.

| Archive member / repository file | SHA-256 |
| --- | --- |
| [serde_json-1.0.151/LICENSE-MIT](serde_json-1.0.151/LICENSE-MIT) | `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3` |
| [serde_json-1.0.151/LICENSE-APACHE](serde_json-1.0.151/LICENSE-APACHE) | `62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a` |

## syn 3.0.6

Declared license: `MIT OR Apache-2.0`. Compile-time derive syntax support.

Release archive: [syn-3.0.6.crate](https://static.crates.io/crates/syn/syn-3.0.6.crate).
Cargo.lock SHA-256: `8593e8e72159ed2257d083c7a454a85cbf854f37a0966d8d483aff8c8a3ebcee`.

| Archive member / repository file | SHA-256 |
| --- | --- |
| [syn-3.0.6/LICENSE-MIT](syn-3.0.6/LICENSE-MIT) | `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3` |
| [syn-3.0.6/LICENSE-APACHE](syn-3.0.6/LICENSE-APACHE) | `62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a` |

## unicode-ident 1.0.26

Declared license: `(MIT OR Apache-2.0) AND Unicode-3.0`. Compile-time syntax/identifier support.

Release archive: [unicode-ident-1.0.26.crate](https://static.crates.io/crates/unicode-ident/unicode-ident-1.0.26.crate).
Cargo.lock SHA-256: `d245f478577f809a851594d02313b640fb437e0bb33866753cff937863096954`.

| Archive member / repository file | SHA-256 |
| --- | --- |
| [unicode-ident-1.0.26/LICENSE-MIT](unicode-ident-1.0.26/LICENSE-MIT) | `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3` |
| [unicode-ident-1.0.26/LICENSE-APACHE](unicode-ident-1.0.26/LICENSE-APACHE) | `62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a` |
| [unicode-ident-1.0.26/LICENSE-UNICODE](unicode-ident-1.0.26/LICENSE-UNICODE) | `f7db81051789b729fea528a63ec4c938fdcb93d9d61d97dc8cc2e9df6d47f2a1` |

## zmij 1.0.23

Declared license: `MIT`. Runtime JSON number formatting.

Release archive: [zmij-1.0.23.crate](https://static.crates.io/crates/zmij/zmij-1.0.23.crate).
Cargo.lock SHA-256: `29666d0abbfad1e3dc4dcf6144730dd3a3ab225bbbdac83319345b1b44ccfc1b`.

| Archive member / repository file | SHA-256 |
| --- | --- |
| [zmij-1.0.23/LICENSE-MIT](zmij-1.0.23/LICENSE-MIT) | `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3` |
