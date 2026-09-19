#!/usr/bin/env python3
"""Generate Task 5 convolution fixtures through unmodified Spektrafilm calls."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tmp/diffusion/matplotlib"))

from spektrafilm.model.diffusion import (  # noqa: E402
    _bloom_max_lambda_um,
    apply_diffusion_filter_um,
    diffusion_filter_psf,
)
from spektrafilm.runtime.params_schema import DiffusionFilterParams  # noqa: E402


SCHEMA = "film-juicer.diffusion-cuda-convolution.v1"
HASH_SCHEMA = "film-juicer.diffusion-cuda-convolution-hashes.v1"
REFERENCE_SHA256 = "640006bef2fc89d4861f69add2a5c388658b0dde1fc694ac75d0e7a9c2b13d88"
FAMILIES = ("glimmerglass", "black_pro_mist", "pro_mist", "cinebloom")
ADVANCED_CASES = (
    (
        "default",
        {
            "core_intensity": 1.0,
            "core_size": 1.0,
            "halo_intensity": 1.0,
            "halo_size": 1.0,
            "bloom_intensity": 1.0,
            "bloom_size": 1.0,
        },
    ),
    (
        "proportional",
        {
            "core_intensity": 2.0,
            "core_size": 1.0,
            "halo_intensity": 4.0,
            "halo_size": 1.0,
            "bloom_intensity": 6.0,
            "bloom_size": 1.0,
        },
    ),
    (
        "all_zero_nondefault_sizes",
        {
            "core_intensity": 0.0,
            "core_size": 0.25,
            "halo_intensity": 0.0,
            "halo_size": 2.0,
            "bloom_intensity": 0.0,
            "bloom_size": 4.0,
        },
    ),
    (
        "runtime_outside_widget",
        {
            "core_intensity": -1.0,
            "core_size": 1.0e-8,
            "halo_intensity": 5.0,
            "halo_size": 6.0,
            "bloom_intensity": 8.0,
            "bloom_size": 0.05,
        },
    ),
)
SMALL_EXTENTS = ((16, 18), (17, 19), (19, 12), (12, 19))
CANDIDATES = (
    ("1024x1024", 1024, 1024, 128, 773, 771),
    ("2048x2048", 2048, 2048, 512, 1027, 1031),
    ("3072x3072", 3072, 3072, 1024, 2051, 2053),
    ("4096x4096", 4096, 4096, 1024, 2051, 2053),
    ("4608x4608", 4608, 4608, 2159, 4320, 4322),
    ("5120x5120", 5120, 5120, 2159, 4320, 4322),
    ("6144x6144", 6144, 6144, 2159, 4320, 4322),
    ("8192x6144", 8192, 6144, 2159, 4320, 4322),
    ("6144x8192", 6144, 8192, 2159, 4320, 4322),
    ("8192x8192", 8192, 8192, 2159, 4320, 4322),
)
PRODUCTION_SOURCES = (
    "src/Cuda/Diffusion/JuicerCudaDiffusion.h",
    "src/Cuda/Diffusion/JuicerCudaDiffusion.cu",
    "src/DiffusionHostBehavior.h",
    "src/DiffusionHostBehavior.cpp",
    "src/RenderRecipe.h",
    "src/RenderRecipe.cpp",
    "src/DiffusionExecution.h",
    "src/DiffusionExecution.cpp",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _production_digest() -> str:
    digest = hashlib.sha256()
    for relative in PRODUCTION_SOURCES:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((ROOT / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class BundleWriter:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.records: list[dict[str, object]] = []
        self.ids: set[str] = set()

    def array(self, identifier: str, value: np.ndarray, role: str) -> str:
        if identifier in self.ids:
            raise ValueError(f"duplicate array ID: {identifier}")
        array = np.ascontiguousarray(value, dtype="<f4")
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite array: {identifier}")
        relative = f"arrays/{identifier}.f32"
        path = self.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        array.tofile(path)
        self.records.append(
            {
                "id": identifier,
                "path": relative,
                "role": role,
                "dtype": "<f4",
                "shape": list(array.shape),
                "sha256": _sha256(path),
            }
        )
        self.ids.add(identifier)
        return identifier


def _capture_apply(
    image: np.ndarray,
    params: DiffusionFilterParams,
    pixel_size_um: float,
) -> tuple[np.ndarray, int]:
    captured: dict[str, int] = {}
    target_code = apply_diffusion_filter_um.__code__
    previous = sys.getprofile()

    def profiler(frame, event: str, _argument) -> None:
        if event == "return" and frame.f_code is target_code and "radius" in frame.f_locals:
            captured["radius"] = int(frame.f_locals["radius"])

    sys.setprofile(profiler)
    try:
        output = apply_diffusion_filter_um(image, params, pixel_size_um)
    finally:
        sys.setprofile(previous)
    if "radius" not in captured:
        raise RuntimeError("Spektrafilm did not expose the applied radius")
    return np.ascontiguousarray(output, dtype="<f4"), captured["radius"]


def _normalized_input(height: int, width: int, seed: int) -> np.ndarray:
    image = np.empty((height, width, 3), dtype="<f4")
    image[..., 0] = np.float32(0.25 + (seed % 5) * 0.025)
    image[..., 1] = 0.0
    image[(height * 2) // 5, (width * 3) // 7, 1] = np.float32(0.75)
    x = np.arange(width, dtype=np.float32)[None, :]
    y = np.arange(height, dtype=np.float32)[:, None]
    image[..., 2] = np.remainder(x * 13.0 + y * 7.0 + seed * 11.0, 251.0) / 250.0
    return image


def _hdr_input(height: int, width: int, seed: int) -> np.ndarray:
    image = np.empty((height, width, 3), dtype="<f4")
    x = np.arange(width, dtype=np.float32)[None, :]
    y = np.arange(height, dtype=np.float32)[:, None]
    image[..., 0] = np.remainder(x * 0.75 + y * 0.25 + seed, 16.0)
    image[..., 1] = np.remainder(x * 0.125 + y * 1.25 + seed * 0.5, 16.0)
    image[..., 2] = np.remainder(x * 1.5 + y * 0.0625 + seed * 0.25, 16.0)
    image[height // 3, width // 4, :] = np.array((16.0, 8.0, 4.0), dtype=np.float32)
    return image


def _params(family: str, advanced: dict[str, float], *, strength: float = 0.5) -> DiffusionFilterParams:
    return DiffusionFilterParams(
        active=True,
        filter_family=family,
        strength=strength,
        spatial_scale=1.0,
        halo_warmth=0.35,
        **advanced,
    )


def _stage_record(
    writer: BundleWriter,
    case_id: str,
    stage_index: int,
    label: str,
    params: DiffusionFilterParams,
    pixel_size_um: float,
    radius: int,
) -> dict[str, object]:
    overrides = {
        "core_intensity": params.core_intensity,
        "core_size": params.core_size,
        "halo_intensity": params.halo_intensity,
        "halo_size": params.halo_size,
        "bloom_intensity": params.bloom_intensity,
        "bloom_size": params.bloom_size,
    }
    psf = diffusion_filter_psf(
        (2 * radius + 1, 2 * radius + 1),
        family=params.filter_family,
        spatial_scale=params.spatial_scale,
        pixel_size_um=pixel_size_um,
        halo_warmth=params.halo_warmth,
        overrides=overrides,
    )
    psf_id = writer.array(f"{case_id}-stage-{stage_index}-psf", psf, "psf")
    del psf
    return {
        "label": label,
        "params": asdict(params),
        "pixel_size_um": pixel_size_um,
        "radius": radius,
        "psf": psf_id,
    }


def _generate_case(
    writer: BundleWriter,
    *,
    case_id: str,
    height: int,
    width: int,
    seed: int,
    advanced_class: str,
    stages: list[tuple[str, DiffusionFilterParams, float, int | None]],
    candidate_consumers: list[str],
) -> dict[str, object]:
    normalized = _normalized_input(height, width, seed)
    normalized_input = writer.array(f"{case_id}-normalized-input", normalized, "normalized_input")
    current = normalized
    observed_radii: list[int] = []
    for _label, params, pixel_size_um, expected_radius in stages:
        current, observed_radius = _capture_apply(current, params, pixel_size_um)
        if expected_radius is not None and observed_radius != expected_radius:
            raise RuntimeError(
                f"{case_id} radius {observed_radius} did not equal {expected_radius}"
            )
        observed_radii.append(observed_radius)
    normalized_output = writer.array(f"{case_id}-normalized-output", current, "normalized_output")
    del normalized, current
    gc.collect()

    hdr = _hdr_input(height, width, seed)
    hdr_input = writer.array(f"{case_id}-hdr-input", hdr, "hdr_input")
    current = hdr
    for stage_index, (_label, params, pixel_size_um, expected_radius) in enumerate(stages):
        current, observed_radius = _capture_apply(current, params, pixel_size_um)
        if (
            (expected_radius is not None and observed_radius != expected_radius)
            or observed_radius != observed_radii[stage_index]
        ):
            raise RuntimeError(f"{case_id} HDR radius disagrees with normalized radius")
    hdr_output = writer.array(f"{case_id}-hdr-output", current, "hdr_output")
    del hdr, current
    gc.collect()

    stage_rows = [
        _stage_record(
            writer,
            case_id,
            index,
            label,
            params,
            pixel_size_um,
            observed_radii[index],
        )
        for index, (label, params, pixel_size_um, _expected_radius) in enumerate(stages)
    ]
    gc.collect()
    return {
        "id": case_id,
        "height": height,
        "width": width,
        "advanced_class": advanced_class,
        "stages": stage_rows,
        "candidate_consumers": candidate_consumers,
        "normalized_input": normalized_input,
        "normalized_output": normalized_output,
        "hdr_input": hdr_input,
        "hdr_output": hdr_output,
    }


def _candidate_rows() -> list[dict[str, object]]:
    rows = []
    for candidate_id, width, height, radius, frame_height, frame_width in CANDIDATES:
        valid_width = width - 2 * radius
        valid_height = height - 2 * radius
        rows.append(
            {
                "id": candidate_id,
                "width": width,
                "height": height,
                "radius": radius,
                "frame_height": frame_height,
                "frame_width": frame_width,
                "valid_tile_width": valid_width,
                "valid_tile_height": valid_height,
                "reference_case": f"stress-r{radius}-{frame_height}x{frame_width}",
                "seams": {
                    "horizontal": True,
                    "vertical": True,
                    "partial_bottom_right": True,
                },
            }
        )
    return rows


def _prepare_output(output: Path, replace: bool) -> None:
    resolved = output.resolve()
    expected_parent = (ROOT / ".tmp/diffusion").resolve()
    if expected_parent not in resolved.parents:
        raise ValueError("fixture output must remain below .tmp/diffusion")
    if output.exists():
        if not replace:
            raise FileExistsError(f"{output} already exists; pass --replace to regenerate")
        shutil.rmtree(output)
    output.mkdir(parents=True)


def generate(output: Path, replace: bool) -> None:
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("the accepted diffusion reference requires Python 3.13")
    origin = Path(inspect.getsourcefile(apply_diffusion_filter_um) or "").resolve()
    expected_source = (ROOT / "external/spektrafilm/src/spektrafilm/model/diffusion.py").resolve()
    if origin != expected_source:
        raise RuntimeError(f"ineligible Spektrafilm import origin: {origin}")
    if _sha256(origin) != REFERENCE_SHA256:
        raise RuntimeError("Spektrafilm diffusion authority hash changed")
    _prepare_output(output, replace)
    writer = BundleWriter(output)
    cases: list[dict[str, object]] = []

    case_index = 0
    for family in FAMILIES:
        for advanced_class, advanced in ADVANCED_CASES:
            height, width = SMALL_EXTENTS[case_index % len(SMALL_EXTENTS)]
            labels = ["camera", "enlarger"] if case_index == 0 else ["camera" if case_index % 2 == 0 else "enlarger"]
            params = _params(family, advanced, strength=0.5 + 0.125 * (case_index % 3))
            pixel_size_um = 1000.0
            stages = [(label, params, pixel_size_um, None) for label in labels]
            cases.append(
                _generate_case(
                    writer,
                    case_id=f"behavior-{family}-{advanced_class}",
                    height=height,
                    width=width,
                    seed=case_index + 1,
                    advanced_class=advanced_class,
                    stages=stages,
                    candidate_consumers=["1024x1024"],
                )
            )
            case_index += 1

    groups: dict[tuple[int, int, int], list[str]] = {}
    for candidate_id, _width, _height, radius, frame_height, frame_width in CANDIDATES:
        groups.setdefault((radius, frame_height, frame_width), []).append(candidate_id)
    default_advanced = dict(ADVANCED_CASES[0][1])
    for group_index, ((radius, height, width), consumers) in enumerate(groups.items()):
        params = _params("cinebloom", default_advanced)
        outer_lambda = _bloom_max_lambda_um("cinebloom", default_advanced)
        pixel_size_um = 8.0 * outer_lambda / (radius - 0.5)
        if not math.isfinite(pixel_size_um) or pixel_size_um <= 0.0:
            raise RuntimeError("invalid derived stress pixel size")
        cases.append(
            _generate_case(
                writer,
                case_id=f"stress-r{radius}-{height}x{width}",
                height=height,
                width=width,
                seed=100 + group_index,
                advanced_class="default",
                stages=[("camera", params, pixel_size_um, radius)],
                candidate_consumers=consumers,
            )
        )

    manifest = {
        "schema": SCHEMA,
        "semantic_channel_order": ["R", "G", "B"],
        "family_order": list(FAMILIES),
        "advanced_classes": [row[0] for row in ADVANCED_CASES],
        "reference_source": {
            "path": origin.relative_to(ROOT).as_posix(),
            "import_origin": str(origin),
            "sha256": REFERENCE_SHA256,
        },
        "production_source_sha256": _production_digest(),
        "production_sources": list(PRODUCTION_SOURCES),
        "candidates": _candidate_rows(),
        "cases": cases,
        "arrays": writer.records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hash_rows = [
        {"path": "manifest.json", "sha256": _sha256(manifest_path)},
        {
            "path": "tests/diffusion/reference/generate_cuda_convolution_cases.py",
            "sha256": _sha256(Path(__file__).resolve()),
        },
    ]
    hash_rows.extend({"path": row["path"], "sha256": row["sha256"]} for row in writer.records)
    (output / "bundle.sha256.json").write_text(
        json.dumps({"schema": HASH_SCHEMA, "files": hash_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "CUDA_CONVOLUTION_FIXTURES=GENERATED "
        f"cases={len(cases)} candidates={len(CANDIDATES)} arrays={len(writer.records)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / ".tmp/diffusion/cuda-convolution-fixtures",
    )
    parser.add_argument("--replace", action="store_true")
    arguments = parser.parse_args()
    generate(arguments.output_dir, arguments.replace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
