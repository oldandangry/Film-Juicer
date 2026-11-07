#!/usr/bin/env python3
"""Regenerate the committed CUDA or OFX declarations; never invoked by product builds."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import sys


BINDGEN_VERSION = "0.72.1"
CLANG_VERSIONS = {"linux": "22.1.8", "win32": "22.1.3"}
TARGETS = {"linux": "x86_64-unknown-linux-gnu", "win32": "x86_64-pc-windows-msvc"}


# Exact declarations used by the current native OFX descriptor, dispatch,
# image, parameter and message paths. Complete suites bring their handle types.
OFX_TYPES = (
    "OfxHost", "OfxPlugin", "OfxPluginEntryPoint", "OfxStatus", "OfxTime",
    "OfxPropertySuiteV1", "OfxParameterSuiteV1", "OfxImageEffectSuiteV1",
    "OfxMessageSuiteV1", "OfxRectI", "OfxRectD", "OfxPointI", "OfxPointD", "OfxRangeD",
)
OFX_CONSTANTS = (
    "kOfxImageComponentRGB",
    "kOfxImageOpaque",
    "kOfxImagePreMultiplied",
    "kOfxImageUnPreMultiplied",
    "kOfxImageFieldNone",
    "kOfxImageEffectPluginApiVersion",

    "kOfxStatOK",
    "kOfxStatFailed",
    "kOfxStatErrFatal",
    "kOfxStatErrUnknown",
    "kOfxStatErrMissingHostFeature",
    "kOfxStatErrUnsupported",
    "kOfxStatErrExists",
    "kOfxStatErrFormat",
    "kOfxStatErrMemory",
    "kOfxStatErrBadHandle",
    "kOfxStatErrBadIndex",
    "kOfxStatErrValue",
    "kOfxStatReplyYes",
    "kOfxStatReplyNo",
    "kOfxStatReplyDefault",
    "kOfxStatErrImageFormat",
    "kOfxImageEffectPluginApi",
    "kOfxPropertySuite",
    "kOfxParameterSuite",
    "kOfxImageEffectSuite",
    "kOfxMessageSuite",
    "kOfxActionLoad",
    "kOfxActionUnload",
    "kOfxActionDescribe",
    "kOfxActionCreateInstance",
    "kOfxActionDestroyInstance",
    "kOfxActionPurgeCaches",
    "kOfxActionSyncPrivateData",
    "kOfxActionBeginInstanceChanged",
    "kOfxActionInstanceChanged",
    "kOfxActionEndInstanceChanged",
    "kOfxActionBeginInstanceEdit",
    "kOfxActionEndInstanceEdit",
    "kOfxImageEffectActionDescribeInContext",
    "kOfxImageEffectActionRender",
    "kOfxImageEffectActionBeginSequenceRender",
    "kOfxImageEffectActionEndSequenceRender",
    "kOfxImageEffectActionIsIdentity",
    "kOfxImageEffectActionGetRegionOfDefinition",
    "kOfxImageEffectActionGetRegionsOfInterest",
    "kOfxImageEffectActionGetFramesNeeded",
    "kOfxImageEffectActionGetClipPreferences",
    "kOfxImageEffectActionGetTimeDomain",
    "kOfxPropType",
    "kOfxPropName",
    "kOfxPropLabel",
    "kOfxPropShortLabel",
    "kOfxPropLongLabel",
    "kOfxPropVersion",
    "kOfxPropVersionLabel",
    "kOfxPropPluginDescription",
    "kOfxPropInstanceData",
    "kOfxPropTime",
    "kOfxPropChangeReason",
    "kOfxPropIsInteractive",
    "kOfxPropAPIVersion",
    "kOfxTypeParameter",
    "kOfxTypeClip",
    "kOfxTypeImageEffect",
    "kOfxChangePluginEdited",
    "kOfxChangeUserEdited",
    "kOfxChangeTime",
    "kOfxImageEffectContextFilter",
    "kOfxImageEffectSimpleSourceClipName",
    "kOfxImageEffectOutputClipName",
    "kOfxBitDepthFloat",
    "kOfxImageComponentRGBA",
    "kOfxImageEffectRenderUnsafe",
    "kOfxImageEffectPluginPropGrouping",
    "kOfxImageEffectPluginPropHostFrameThreading",
    "kOfxImageEffectPluginPropSingleInstance",
    "kOfxImageEffectPluginRenderThreadSafety",
    "kOfxImageEffectPropSupportedContexts",
    "kOfxImageEffectPropSupportedPixelDepths",
    "kOfxImageEffectPropSupportsMultiResolution",
    "kOfxImageEffectPropSupportsTiles",
    "kOfxImageEffectPropSupportedComponents",
    "kOfxImageEffectPropTemporalClipAccess",
    "kOfxImageClipPropOptional",
    "kOfxImageEffectPropCudaRenderSupported",
    "kOfxImageEffectPropCudaStreamSupported",
    "kOfxImageEffectPropCudaEnabled",
    "kOfxImageEffectPropCudaStream",
    "kOfxImageEffectPropContext",
    "kOfxImageEffectPropRenderScale",
    "kOfxImageEffectPropRenderWindow",
    "kOfxImageEffectPropFrameRange",
    "kOfxImageEffectPropFrameRate",
    "kOfxImageEffectPropFrameStep",
    "kOfxImageEffectPropRegionOfDefinition",
    "kOfxImageEffectPropRegionOfInterest",
    "kOfxImageEffectPropPixelDepth",
    "kOfxImageEffectPropComponents",
    "kOfxImageEffectPropPreMultiplication",
    "kOfxImageEffectPropFieldToRender",
    "kOfxImageEffectPropSequentialRenderStatus",
    "kOfxImageEffectPropInteractiveRenderStatus",
    "kOfxImageEffectPropRenderQualityDraft",
    "kOfxImageEffectPropProjectSize",
    "kOfxImageEffectPropProjectOffset",
    "kOfxImageEffectPropProjectExtent",
    "kOfxImageEffectPropProjectPixelAspectRatio",
    "kOfxImageClipPropConnected",
    "kOfxImagePropData",
    "kOfxImagePropBounds",
    "kOfxImagePropRowBytes",
    "kOfxImagePropRegionOfDefinition",
    "kOfxImagePropPixelAspectRatio",
    "kOfxImagePropField",
    "kOfxImagePropUniqueIdentifier",
    "kOfxParamTypeBoolean",
    "kOfxParamTypeChoice",
    "kOfxParamTypeStrChoice",
    "kOfxParamTypeDouble",
    "kOfxParamTypeDouble2D",
    "kOfxParamTypeDouble3D",
    "kOfxParamTypeInteger",
    "kOfxParamTypeGroup",
    "kOfxParamTypePage",
    "kOfxParamTypePushButton",
    "kOfxParamPropType",
    "kOfxParamPropDefault",
    "kOfxParamPropMin",
    "kOfxParamPropMax",
    "kOfxParamPropDisplayMin",
    "kOfxParamPropDisplayMax",
    "kOfxParamPropIncrement",
    "kOfxParamPropDigits",
    "kOfxParamPropDoubleType",
    "kOfxParamPropDimensionLabel",
    "kOfxParamPropChoiceOption",
    "kOfxParamPropChoiceEnum",
    "kOfxParamHostPropSupportsStrChoiceAnimation",
    "kOfxParamPropParent",
    "kOfxParamPropPageChild",
    "kOfxParamPropGroupOpen",
    "kOfxParamPropHint",
    "kOfxParamPropScriptName",
    "kOfxParamPropAnimates",
    "kOfxParamPropEnabled",
    "kOfxParamPropSecret",
    "kOfxParamPropEvaluateOnChange",
    "kOfxParamPropPersistant",
    "kOfxParamPropCanUndo",
    "kOfxParamPropCacheInvalidation",
    "kOfxParamDoubleTypePlain",
    "kOfxParamDoubleTypeScale",
    "kOfxParamInvalidateValueChange",
    "kOfxMessageFatal",
    "kOfxMessageError",
    "kOfxMessageWarning",
    "kOfxMessageMessage",
    "kOfxMessageLog",
    "kOfxMessageQuestion",
)


class ClangString(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("private_flags", ctypes.c_uint)]


def libclang_version(path: Path) -> str:
    library = ctypes.CDLL(str(path))
    library.clang_getClangVersion.restype = ClangString
    library.clang_getCString.argtypes = [ClangString]
    library.clang_getCString.restype = ctypes.c_char_p
    library.clang_disposeString.argtypes = [ClangString]
    value = library.clang_getClangVersion()
    try:
        return library.clang_getCString(value).decode("utf-8")
    finally:
        library.clang_disposeString(value)


def run(command: list[str], **kwargs: object) -> str:
    print("COMMAND:", subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, **kwargs)
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindgen", required=True, type=Path)
    parser.add_argument("--clang", required=True, type=Path)
    parser.add_argument("--libclang", required=True, type=Path)
    parser.add_argument("--rustfmt", required=True, type=Path)
    parser.add_argument("--api", choices=("cuda", "ofx"), default="cuda")
    parser.add_argument("--check", action="store_true", help="compare without changing the committed output")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if sys.platform not in TARGETS:
        parser.error("binding generation is qualified only on native Linux and Windows x64")
    bindgen = str(arguments.bindgen.resolve(strict=True))
    clang = str(arguments.clang.resolve(strict=True))
    arguments.rustfmt.resolve(strict=True)
    libclang = arguments.libclang.resolve(strict=True)
    version = run([bindgen, "--version"]).strip()
    if version != f"bindgen {BINDGEN_VERSION}":
        raise RuntimeError(f"expected bindgen {BINDGEN_VERSION}, got {version}")
    clang_version = run([clang, "--version"])
    library_version = libclang_version(libclang)
    required = CLANG_VERSIONS[sys.platform]
    if f"clang version {required} " not in clang_version or f"clang version {required} " not in library_version:
        raise RuntimeError(f"expected Clang and libclang {required}: {clang_version}; {library_version}")
    print(version, clang_version.strip(), library_version, sep="\n")
    # rustfmt is a rustup proxy in qualified installations. Preserve its filename
    # (resolving the symlink to rustup changes argv[0] and selects the wrong tool).
    rustfmt = os.path.abspath(arguments.rustfmt)
    rustfmt_version = run([rustfmt, "--version"], cwd=root).strip()
    print(rustfmt_version)
    if rustfmt_version != "rustfmt 1.9.0-stable (48a229ceae 2026-09-01)":
        raise RuntimeError(f"expected the Rust 1.98.1 rustfmt component: {rustfmt_version}")
    resource_dir = run([clang, "-print-resource-dir"]).strip()
    environment = os.environ.copy()
    environment["CLANG_PATH"] = clang
    environment["LIBCLANG_PATH"] = str(libclang)
    # Do not inherit hidden target/include changes from a developer's shell.
    for name in list(environment):
        if name.startswith("BINDGEN_EXTRA_CLANG_ARGS") or name in {"CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"}:
            environment.pop(name)
    if arguments.api == "cuda":
        selection = [
            "native/juicer_cuda_api.h", "--allowlist-type", "Fj.*",
            "--allowlist-function", "fj_cuda_.*", "--allowlist-var", "FJ_.*",
        ]
        destination = root / "rust/film-juicer-plugin/src/cuda/sys.rs"
        includes = []
    else:
        selection = ["native/ofx_bindings.h", "--clang-macro-fallback", "--default-macro-constant-type", "signed"]
        for name in OFX_TYPES:
            selection.extend(["--allowlist-type", f"^{name}$"])
        for name in OFX_CONSTANTS:
            selection.extend(["--allowlist-var", f"^{name}$"])
        destination = root / "rust/film-juicer-plugin/src/ofx/sys.rs"
        includes = ["-I", "third_party/ofxs/OpenFX-1.4/include"]
    command = [
        bindgen, *selection, "--no-doc-comments", "--no-derive-debug",
        "--use-core", "--rust-target", "1.85", "--rust-edition", "2024",
        "--no-include-path-detection", "--formatter", "none", "--",
        "-x", "c", "-std=c11", f"--target={TARGETS[sys.platform]}",
        "-ffreestanding", "-nostdinc", "-isystem", str(Path(resource_dir) / "include"),
        *includes,
    ]
    if arguments.api == "ofx":
        inputs = [root / "native/ofx_bindings.h"]
        inputs.extend(sorted((root / "third_party/ofxs/OpenFX-1.4/include").glob("*.h")))
        for path in inputs:
            print(f"INPUT SHA256: {hashlib.sha256(path.read_bytes()).hexdigest()} {path.relative_to(root)}")
    generated = run(command, cwd=root, env=environment)
    formatted = run(
        [rustfmt, "--edition", "2024", "--config-path", str(root / "rustfmt.toml")],
        cwd=root, input=generated,
    ).replace("\r\n", "\n").encode("utf-8")
    if arguments.check:
        if destination.read_bytes() != formatted:
            raise RuntimeError(f"{arguments.api.upper()} bindings differ; regenerate and review the change")
        print("Bindings reproduce the committed file byte-for-byte.")
    else:
        destination.write_bytes(formatted)
        print(f"Wrote {destination}")
    print("SHA256:", hashlib.sha256(formatted).hexdigest())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
