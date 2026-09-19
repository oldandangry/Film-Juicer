#!/usr/bin/env python3
"""Generate deterministic diffusion fixtures from unmodified spektrafilm imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np


_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
# spektrafilm imports Matplotlib transitively; keep its cache with the managed environment.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(_DEFAULT_REPO_ROOT / ".tmp/diffusion/matplotlib"),
)

from spektrafilm.model import diffusion
from spektrafilm.runtime import params_schema
from spektrafilm.runtime.params_schema import DiffusionFilterParams


FAMILIES = ("glimmerglass", "black_pro_mist", "pro_mist", "cinebloom")
STRENGTHS = (-1.0, 0.0, 0.0625, 0.125, 0.25, 0.375, 0.5, 1.0, 2.0, 4.0)
WARMTHS = (-3.0, -1.5, 0.0, 1.5, 3.0)
ADVANCED_CASES = (
    {
        "name": "default",
        "core_intensity": 1.0,
        "halo_intensity": 1.0,
        "bloom_intensity": 1.0,
        "core_size": 1.0,
        "halo_size": 1.0,
        "bloom_size": 1.0,
    },
    {
        "name": "proportional",
        "core_intensity": 2.0,
        "halo_intensity": 4.0,
        "bloom_intensity": 6.0,
        "core_size": 1.0,
        "halo_size": 1.0,
        "bloom_size": 1.0,
    },
    {
        "name": "all_zero_nondefault_sizes",
        "core_intensity": 0.0,
        "halo_intensity": 0.0,
        "bloom_intensity": 0.0,
        "core_size": 0.25,
        "halo_size": 2.0,
        "bloom_size": 4.0,
    },
    {
        "name": "runtime_outside_widget",
        "core_intensity": -1.0,
        "halo_intensity": 5.0,
        "bloom_intensity": 8.0,
        "core_size": 1.0e-8,
        "halo_size": 6.0,
        "bloom_size": 0.05,
    },
)
PATTERNS = ("center", "asymmetric", "constant", "edge", "corner", "random")
RADIUS_EXTENTS = (
    ("even", 16, 18),
    ("odd", 17, 19),
    ("portrait", 23, 11),
    ("landscape", 11, 23),
)
SEMANTIC_CHANNEL_ORDER = ("R", "G", "B")
NORMAL_PIXEL_SIZE_UM = 1000.0


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(cwd: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _store_array(arrays: dict[str, np.ndarray], name: str, value: np.ndarray) -> str:
    if name in arrays:
        raise ValueError(f"duplicate array name: {name}")
    array = np.ascontiguousarray(value)
    if not np.isfinite(array).all():
        raise ValueError(f"non-finite fixture array: {name}")
    arrays[name] = array
    return name


def _array_records(arrays: dict[str, np.ndarray]) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "shape": list(arrays[name].shape),
            "dtype": arrays[name].dtype.name,
            "sha256": hashlib.sha256(arrays[name].tobytes(order="C")).hexdigest(),
        }
        for name in sorted(arrays)
    ]


def _overrides(case: dict[str, float | str]) -> dict[str, float]:
    return {key: float(value) for key, value in case.items() if key != "name"}


def _expanded_group(group: dict[str, object], kind: str) -> dict[str, object]:
    lambdas, weights = diffusion._expand_group(group, kind=kind)
    result: dict[str, object] = {
        "lambda_um": float(group["lambda_um"]),
        "spread": float(group.get("spread", 1.0)),
        "n_components": int(group.get("n_components", 1)),
        "expanded_lambdas_um": [float(value) for value in lambdas],
        "expanded_weights": [float(value) for value in weights],
    }
    if "alpha" in group:
        result["alpha"] = float(group["alpha"])
    return result


def _apply_with_radius(
    image: np.ndarray,
    params: DiffusionFilterParams,
    pixel_size_um: float,
) -> tuple[np.ndarray, int]:
    captured: dict[str, int] = {}
    target_code = diffusion.apply_diffusion_filter_um.__code__
    previous_profiler = sys.getprofile()

    def capture_radius(frame, event: str, _argument):
        if event == "return" and frame.f_code is target_code and "radius" in frame.f_locals:
            captured["radius"] = int(frame.f_locals["radius"])

    sys.setprofile(capture_radius)
    try:
        output = diffusion.apply_diffusion_filter_um(image, params, pixel_size_um)
    finally:
        sys.setprofile(previous_profiler)
    if "radius" not in captured:
        raise RuntimeError("reference apply_diffusion_filter_um did not expose its PSF radius")
    return np.ascontiguousarray(output), captured["radius"]


def _image_row(
    *,
    arrays: dict[str, np.ndarray],
    input_array: str,
    output_array: str,
    output: np.ndarray,
    params: DiffusionFilterParams,
    stage: str,
    family: str,
    pattern: str,
    height: int,
    width: int,
    pixel_size_um: float,
    psf_radius: int,
) -> dict[str, object]:
    _store_array(arrays, output_array, output)
    return {
        "stage": stage,
        "family": family,
        "pattern": pattern,
        "extent": [height, width],
        "pixel_size_um": float(pixel_size_um),
        "params": asdict(params),
        "input_array": input_array,
        "output_array": output_array,
        "input_dtype": arrays[input_array].dtype.name,
        "output_dtype": arrays[output_array].dtype.name,
        "psf_radius": int(psf_radius),
        "semantic_channel_order": list(SEMANTIC_CHANNEL_ORDER),
    }


def source_provenance(repo_root: Path) -> list[dict[str, str]]:
    repo_root = repo_root.resolve()
    source_root = (repo_root / "external/spektrafilm").resolve()
    rows: list[dict[str, str]] = []
    for module_name, module in sorted(sys.modules.items()):
        if module_name != "spektrafilm" and not module_name.startswith("spektrafilm."):
            continue
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        path = Path(module_file).resolve()
        if source_root not in path.parents or path.suffix != ".py":
            raise RuntimeError(f"ineligible imported spektrafilm source path: {path}")
        rows.append(
            {
                "module": module_name,
                "path": path.relative_to(repo_root).as_posix(),
                "sha256": _sha256_file(path),
            }
        )
    if not rows:
        raise RuntimeError("normal imports produced no spektrafilm source provenance")
    return sorted(rows, key=lambda row: (row["path"], row["module"]))


def make_pattern(name: str, height: int, width: int) -> np.ndarray:
    if height <= 0 or width <= 0:
        raise ValueError("pattern extent must be positive")
    image = np.zeros((height, width, 3), dtype=np.float32)
    asymmetric = np.array([1.0, 0.5, 0.25], dtype=np.float32)
    if name == "center":
        image[height // 2, width // 2] = 1.0
    elif name == "asymmetric":
        image[height // 2, width // 2] = asymmetric
    elif name == "constant":
        image[:] = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    elif name == "edge":
        image[0, width // 2] = asymmetric
    elif name == "corner":
        image[0, 0] = asymmetric
    elif name == "random":
        image = np.random.default_rng(0xD1FF0510).random(
            (height, width, 3), dtype=np.float32
        )
    elif name == "hdr_epsilon":
        image = np.linspace(0.0, 16.0, image.size, dtype=np.float32).reshape(image.shape)
        image.reshape(-1)[:7] = np.array(
            [
                0.0,
                np.nextafter(np.float32(0.0), np.float32(1.0)),
                0.5e-10,
                1.0e-10,
                2.0e-10,
                1.0,
                16.0,
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"unknown fixture pattern: {name}")
    upper_bound = 16.0 if name == "hdr_epsilon" else 1.0
    if image.dtype != np.float32 or not np.isfinite(image).all():
        raise RuntimeError(f"invalid pattern dtype or samples: {name}")
    if float(image.min()) < 0.0 or float(image.max()) > upper_bound:
        raise RuntimeError(f"pattern outside authored range: {name}")
    return np.ascontiguousarray(image)


def build_reference_bundle(
    repo_root: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    repo_root = repo_root.resolve()
    reference_root = (repo_root / "external/spektrafilm").resolve()
    expected_interpreter = (repo_root / ".tmp/diffusion/reference-venv/bin/python").resolve()
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError(f"reference generation requires Python 3.13, got {sys.version}")
    if Path(sys.executable).resolve() != expected_interpreter:
        raise RuntimeError(
            f"reference generation requires {expected_interpreter}, got {sys.executable}"
        )
    if tuple(diffusion.DIFFUSION_FILTER_FAMILIES) != FAMILIES:
        raise RuntimeError(
            "spektrafilm family order changed: "
            f"expected {FAMILIES}, got {diffusion.DIFFUSION_FILTER_FAMILIES}"
        )

    reference_status = _git_output(reference_root, "status", "--porcelain")
    if reference_status:
        raise RuntimeError(f"dirty external/spektrafilm checkout:\n{reference_status}")

    freeze_path = repo_root / ".tmp/diffusion/evidence/reference/python-freeze.txt"
    uv_path = repo_root / ".tmp/diffusion/evidence/reference/uv-version.txt"
    lock_path = repo_root / "tests/diffusion/reference/requirements.lock"
    for required in (freeze_path, uv_path, lock_path):
        if not required.is_file():
            raise RuntimeError(f"missing Task 1 environment evidence: {required}")
    expected_editable = f"-e file://{reference_root}"
    freeze_lines = freeze_path.read_text(encoding="utf-8").splitlines()
    if expected_editable not in freeze_lines:
        raise RuntimeError(
            "freeze does not contain the editable reference checkout: "
            f"{expected_editable}"
        )

    arrays: dict[str, np.ndarray] = {}
    strength_cases: list[dict[str, object]] = []
    family_cases: list[dict[str, object]] = []
    warmth_cases: list[dict[str, object]] = []
    advanced_cases: list[dict[str, object]] = []
    psf_cases: list[dict[str, object]] = []
    image_cases: list[dict[str, object]] = []
    radius_cases: list[dict[str, object]] = []
    hdr_cases: list[dict[str, object]] = []

    for family in FAMILIES:
        for strength in STRENGTHS:
            strength_cases.append(
                {
                    "family": family,
                    "strength": float(strength),
                    "scatter_fraction": float(diffusion._strength_to_scatter(strength, family)),
                }
            )

        default_cfg = diffusion._resolve_family_cfg(family)
        family_cases.append(
            {
                "family": family,
                "w_c": float(default_cfg["w_c"]),
                "w_h": float(default_cfg["w_h"]),
                "w_b": float(default_cfg["w_b"]),
                "halo_warmth_base": float(default_cfg["halo_warmth_base"]),
                "core": _expanded_group(default_cfg["core"], "core"),
                "halo": _expanded_group(default_cfg["halo"], "halo"),
                "bloom": _expanded_group(default_cfg["bloom"], "bloom"),
            }
        )

        _, halo_weights = diffusion._expand_group(default_cfg["halo"], kind="halo")
        for warmth in WARMTHS:
            effective = float(default_cfg["halo_warmth_base"]) + float(warmth)
            channel_weights = diffusion._halo_channel_weights(halo_weights, effective)
            warmth_cases.append(
                {
                    "family": family,
                    "halo_warmth": float(warmth),
                    "family_base": float(default_cfg["halo_warmth_base"]),
                    "effective_unclamped": effective,
                    "effective_clamped": float(np.clip(effective, -1.5, 1.5)),
                    "channel_weights_rgb": channel_weights.tolist(),
                }
            )

        for advanced_case in ADVANCED_CASES:
            overrides = _overrides(advanced_case)
            resolved = diffusion._resolve_family_cfg(family, overrides)
            advanced_cases.append(
                {
                    "family": family,
                    "name": advanced_case["name"],
                    "overrides": overrides,
                    "resolved_w_c": float(resolved["w_c"]),
                    "resolved_w_h": float(resolved["w_h"]),
                    "resolved_w_b": float(resolved["w_b"]),
                    "core": _expanded_group(resolved["core"], "core"),
                    "halo": _expanded_group(resolved["halo"], "halo"),
                    "bloom": _expanded_group(resolved["bloom"], "bloom"),
                }
            )
            for warmth_index, warmth in enumerate(WARMTHS):
                psf = diffusion.diffusion_filter_psf(
                    (17, 19),
                    family=family,
                    spatial_scale=1.0,
                    pixel_size_um=NORMAL_PIXEL_SIZE_UM,
                    halo_warmth=warmth,
                    overrides=overrides,
                )
                array_name = (
                    f"psf__{family}__advanced_{advanced_case['name']}__warmth_{warmth_index}"
                )
                _store_array(arrays, array_name, psf)
                psf_cases.append(
                    {
                        "family": family,
                        "halo_warmth": float(warmth),
                        "advanced_case": advanced_case["name"],
                        "overrides": overrides,
                        "extent": [17, 19],
                        "pixel_size_um": NORMAL_PIXEL_SIZE_UM,
                        "spatial_scale": 1.0,
                        "array": array_name,
                        "semantic_channel_order": list(SEMANTIC_CHANNEL_ORDER),
                    }
                )

    normalized_inputs = {
        pattern: _store_array(arrays, f"input__17x19__{pattern}", make_pattern(pattern, 17, 19))
        for pattern in PATTERNS
    }
    for family in FAMILIES:
        for stage in ("camera", "enlarger"):
            for pattern in PATTERNS:
                params = DiffusionFilterParams(
                    active=True,
                    filter_family=family,
                    strength=0.5,
                    spatial_scale=1.0,
                )
                input_name = normalized_inputs[pattern]
                output, radius = _apply_with_radius(
                    arrays[input_name].copy(), params, NORMAL_PIXEL_SIZE_UM
                )
                output_name = f"output__{stage}__{family}__{pattern}"
                image_cases.append(
                    _image_row(
                        arrays=arrays,
                        input_array=input_name,
                        output_array=output_name,
                        output=output,
                        params=params,
                        stage=stage,
                        family=family,
                        pattern=pattern,
                        height=17,
                        width=19,
                        pixel_size_um=NORMAL_PIXEL_SIZE_UM,
                        psf_radius=radius,
                    )
                )

    radius_inputs: dict[tuple[int, int], str] = {}
    for extent_name, height, width in RADIUS_EXTENTS:
        radius_inputs[(height, width)] = _store_array(
            arrays,
            f"input__radius_{extent_name}__{height}x{width}",
            make_pattern("asymmetric", height, width),
        )
    for family in FAMILIES:
        for extent_name, height, width in RADIUS_EXTENTS:
            params = DiffusionFilterParams(
                active=True,
                filter_family=family,
                strength=0.5,
                spatial_scale=1.0,
            )
            input_name = radius_inputs[(height, width)]
            output, radius = _apply_with_radius(
                arrays[input_name].copy(), params, NORMAL_PIXEL_SIZE_UM
            )
            output_name = f"output__radius_{extent_name}__{family}__{height}x{width}"
            row = _image_row(
                arrays=arrays,
                input_array=input_name,
                output_array=output_name,
                output=output,
                params=params,
                stage="camera",
                family=family,
                pattern="asymmetric",
                height=height,
                width=width,
                pixel_size_um=NORMAL_PIXEL_SIZE_UM,
                psf_radius=radius,
            )
            row["extent_class"] = extent_name
            row["image_radius_cap"] = max(min(height, width) // 2 - 1, 1)
            radius_cases.append(row)

    hdr_input = _store_array(
        arrays,
        "input__17x19__hdr_epsilon",
        make_pattern("hdr_epsilon", 17, 19),
    )
    for family in FAMILIES:
        for stage in ("camera", "enlarger"):
            params = DiffusionFilterParams(
                active=True,
                filter_family=family,
                strength=0.5,
                spatial_scale=1.0,
            )
            output, radius = _apply_with_radius(
                arrays[hdr_input].copy(), params, NORMAL_PIXEL_SIZE_UM
            )
            output_name = f"output__{stage}__{family}__hdr_epsilon"
            hdr_cases.append(
                _image_row(
                    arrays=arrays,
                    input_array=hdr_input,
                    output_array=output_name,
                    output=output,
                    params=params,
                    stage=stage,
                    family=family,
                    pattern="hdr_epsilon",
                    height=17,
                    width=19,
                    pixel_size_um=NORMAL_PIXEL_SIZE_UM,
                    psf_radius=radius,
                )
            )

    manifest: dict[str, object] = {
        "schema": "film-juicer.diffusion-reference.v1",
        "array_schema": "film-juicer.diffusion-reference-arrays.v1",
        "family_order": list(FAMILIES),
        "semantic_channel_order": list(SEMANTIC_CHANNEL_ORDER),
        "python": {
            "major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": ".tmp/diffusion/reference-venv/bin/python",
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "uv": {
            "version": uv_path.read_text(encoding="utf-8").strip(),
            "evidence_path": uv_path.relative_to(repo_root).as_posix(),
            "sha256": _sha256_file(uv_path),
        },
        "dependency_lock": {
            "path": lock_path.relative_to(repo_root).as_posix(),
            "sha256": _sha256_file(lock_path),
        },
        "python_freeze": {
            "path": freeze_path.relative_to(repo_root).as_posix(),
            "sha256": _sha256_file(freeze_path),
            "line_count": len(freeze_lines),
        },
        "repository_git": {"commit": _git_output(repo_root, "rev-parse", "HEAD")},
        "reference_git": {
            "commit": _git_output(reference_root, "rev-parse", "HEAD"),
            "status_porcelain": reference_status,
        },
        "source_files": source_provenance(repo_root),
        "strength_cases": strength_cases,
        "family_cases": family_cases,
        "warmth_cases": warmth_cases,
        "advanced_cases": advanced_cases,
        "psf_cases": psf_cases,
        "image_cases": image_cases,
        "radius_cases": radius_cases,
        "hdr_cases": hdr_cases,
        "arrays": _array_records(arrays),
    }
    return manifest, arrays


def write_bundle(
    output_dir: Path,
    manifest: dict[str, object],
    arrays: dict[str, np.ndarray],
) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {row["name"]: row for row in manifest["arrays"]}
    if set(records) != set(arrays):
        raise ValueError("manifest array inventory does not match payload")

    with tempfile.TemporaryDirectory(
        prefix=".diffusion-reference-", dir=output_dir.parent
    ) as temporary:
        temporary_dir = Path(temporary)
        temporary_manifest = temporary_dir / "diffusion_reference_v1.json"
        temporary_arrays = temporary_dir / "diffusion_reference_v1.npz"
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(
            temporary_arrays,
            **{name: arrays[name] for name in sorted(arrays)},
        )

        reloaded_manifest = json.loads(temporary_manifest.read_text(encoding="utf-8"))
        if reloaded_manifest != manifest:
            raise RuntimeError("reloaded manifest differs from generated manifest")
        with np.load(temporary_arrays, allow_pickle=False) as reloaded_arrays:
            if set(reloaded_arrays.files) != set(records):
                raise RuntimeError("reloaded payload inventory differs from manifest")
            for name, record in records.items():
                array = reloaded_arrays[name]
                if list(array.shape) != record["shape"] or array.dtype.name != record["dtype"]:
                    raise RuntimeError(f"reloaded payload metadata mismatch: {name}")
                actual_hash = hashlib.sha256(array.tobytes(order="C")).hexdigest()
                if actual_hash != record["sha256"]:
                    raise RuntimeError(f"reloaded payload hash mismatch: {name}")

        os.replace(temporary_arrays, output_dir / "diffusion_reference_v1.npz")
        os.replace(temporary_manifest, output_dir / "diffusion_reference_v1.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_DEFAULT_REPO_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/diffusion/fixtures"),
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    manifest, arrays = build_reference_bundle(repo_root)
    write_bundle(output_dir, manifest, arrays)
    print(
        "reference_bundle=PASS "
        f"arrays={len(arrays)} output={output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
