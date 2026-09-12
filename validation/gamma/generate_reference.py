#!/usr/bin/env python3
"""Generate pinned spektrafilm reference captures for gamma validation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from spektrafilm.runtime.params_builder import digest_params, init_params
from spektrafilm.runtime.pipeline import SimulationPipeline
from spektrafilm.runtime.topology import Tap
from spektrafilm.model.couplers import compute_exposure_correction_dir_couplers
from spektrafilm.profiles.io import DensityCurvesModel, load_profile
from spektrafilm.utils.fast_interp import fast_interp
from spektrafilm.utils.gamut_compression import (
    InputGamutCompressSpec,
    OutputGamutCompressSpec,
)
from spektrafilm.utils.morph_curves import (
    PrintCurvesMorphParams,
    apply_print_curves_morph,
)


CASE_NAME = "default-paper"
WIDTH = 16
HEIGHT = 4
FILM_PROFILE = "kodak_portra_400"
PRINT_PROFILE = "kodak_portra_endura"
PRODUCT_NEUTRAL_CMY = (0.0, 51.43162770877449, 55.26070894686862)
PRINT_STOCKS = (
    "fujifilm_crystal_archive_typeii",
    "kodak_2383",
    "kodak_2393",
    "kodak_ektacolor_edge",
    "kodak_endura_premier",
    "kodak_portra_endura",
    "kodak_supra_endura",
    "kodak_ultra_endura",
)
PRINT_FACTORS = (0.5, 0.75, 1.0, 1.1, 1.25, 1.5, 2.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments],
        text=True,
    ).strip()


def build_chart() -> np.ndarray:
    chart = np.zeros((HEIGHT, WIDTH, 4), dtype=np.float32)
    neutral = np.array(
        [
            0.0,
            0.0001,
            0.0003,
            0.001,
            0.003,
            0.01,
            0.018,
            0.03,
            0.06,
            0.1,
            0.18,
            0.3,
            0.5,
            0.7,
            0.9,
            1.0,
        ],
        dtype=np.float32,
    )
    chart[0, :, :3] = neutral[:, None]
    patches = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.18, 0.02],
            [0.02, 0.18, 1.0],
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7],
            [0.8, 0.55, 0.25],
            [0.25, 0.55, 0.8],
            [0.9, 0.75, 0.7],
            [0.04, 0.02, 0.01],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    chart[1, :, :3] = patches
    chart[2, :, :3] = patches * np.float32(0.18)
    chart[3, :, :3] = np.sqrt(patches, dtype=np.float32) * np.float32(0.65)
    chart[:, :, 3] = np.float32(1.0)
    return chart


def _configure_diffusion_filter(filter_params, *, active: bool) -> None:
    filter_params.active = active
    filter_params.filter_family = "black_pro_mist"
    filter_params.strength = 0.5
    filter_params.spatial_scale = 1.0
    filter_params.halo_warmth = 0.0
    filter_params.core_intensity = 1.0
    filter_params.core_size = 1.0
    filter_params.halo_intensity = 1.0
    filter_params.halo_size = 1.0
    filter_params.bloom_intensity = 1.0
    filter_params.bloom_size = 1.0


def g09_effective_settings(params, *, width: int = WIDTH, height: int = HEIGHT) -> dict:
    long_edge_pixels = max(width, height)
    pixel_size_um = params.camera.film_format_mm * 1000.0 / long_edge_pixels
    enlarger = params.enlarger.diffusion_filter
    camera = params.camera.diffusion_filter
    return {
        "scan_film": params.io.scan_film,
        "output_cctf_encoding": params.io.output_cctf_encoding,
        "frame_width": width,
        "frame_height": height,
        "camera_film_format_mm": params.camera.film_format_mm,
        "pixel_size_um": pixel_size_um,
        "dir_active": params.film_render.dir_couplers.active,
        "dir_diffusion_size_um": (
            params.film_render.dir_couplers.diffusion_size_um
        ),
        "camera_lens_blur_um": params.camera.lens_blur_um,
        "enlarger_lens_blur": params.enlarger.lens_blur,
        "scanner_lens_blur": params.scanner.lens_blur,
        "camera_diffusion_active": camera.active,
        "camera_diffusion_family": camera.filter_family,
        "camera_diffusion_strength": camera.strength,
        "camera_diffusion_spatial_scale": camera.spatial_scale,
        "enlarger_diffusion_active": enlarger.active,
        "enlarger_diffusion_family": enlarger.filter_family,
        "enlarger_diffusion_strength": enlarger.strength,
        "enlarger_diffusion_spatial_scale": enlarger.spatial_scale,
        "enlarger_diffusion_halo_warmth": enlarger.halo_warmth,
        "enlarger_diffusion_core_intensity": enlarger.core_intensity,
        "enlarger_diffusion_core_size": enlarger.core_size,
        "enlarger_diffusion_halo_intensity": enlarger.halo_intensity,
        "enlarger_diffusion_halo_size": enlarger.halo_size,
        "enlarger_diffusion_bloom_intensity": enlarger.bloom_intensity,
        "enlarger_diffusion_bloom_size": enlarger.bloom_size,
        "grain_active": params.film_render.grain.active,
        "halation_active": params.film_render.halation.active,
        "film_glare_active": params.film_render.glare.active,
        "print_glare_active": params.print_render.glare.active,
        "scanner_unsharp_mask": list(params.scanner.unsharp_mask),
        "debug_deactivate_spatial_effects": (
            params.debug.deactivate_spatial_effects
        ),
    }


def assert_g09_effective_settings(
    params,
    *,
    scan_film: bool,
    enlarger_diffusion: bool,
    dir_active: bool,
    dir_diffusion_size_um: float,
    output_cctf_encoding: bool,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> dict:
    settings = g09_effective_settings(params, width=width, height=height)
    expected = {
        "scan_film": scan_film,
        "output_cctf_encoding": output_cctf_encoding,
        "frame_width": width,
        "frame_height": height,
        "camera_film_format_mm": 35.0,
        "pixel_size_um": 35000.0 / max(width, height),
        "dir_active": dir_active,
        "dir_diffusion_size_um": dir_diffusion_size_um,
        "camera_lens_blur_um": 0.0,
        "enlarger_lens_blur": 0.0,
        "scanner_lens_blur": 0.0,
        "camera_diffusion_active": False,
        "camera_diffusion_family": "black_pro_mist",
        "camera_diffusion_strength": 0.5,
        "camera_diffusion_spatial_scale": 1.0,
        "enlarger_diffusion_active": enlarger_diffusion,
        "enlarger_diffusion_family": "black_pro_mist",
        "enlarger_diffusion_strength": 0.5,
        "enlarger_diffusion_spatial_scale": 1.0,
        "enlarger_diffusion_halo_warmth": 0.0,
        "enlarger_diffusion_core_intensity": 1.0,
        "enlarger_diffusion_core_size": 1.0,
        "enlarger_diffusion_halo_intensity": 1.0,
        "enlarger_diffusion_halo_size": 1.0,
        "enlarger_diffusion_bloom_intensity": 1.0,
        "enlarger_diffusion_bloom_size": 1.0,
        "grain_active": False,
        "halation_active": False,
        "film_glare_active": False,
        "print_glare_active": False,
        "scanner_unsharp_mask": [0.0, 0.0],
        "debug_deactivate_spatial_effects": False,
    }
    mismatches = {
        name: {"measured": settings.get(name), "expected": value}
        for name, value in expected.items()
        if settings.get(name) != value
    }
    if mismatches:
        raise RuntimeError(
            "G09 effective settings do not match the requested isolated case: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return settings


def configured_params(
    film_profile=FILM_PROFILE,
    print_profile=PRINT_PROFILE,
    *,
    scan_film=False,
    neutral_cmy=PRODUCT_NEUTRAL_CMY,
    film_gamma=1.0,
    print_gamma=1.0,
    camera_exposure_ev=0.0,
    print_exposure_compensation=True,
    normalize_print_exposure=True,
    print_exposure=1.0,
    scanner_black_correction=False,
    scanner_white_correction=False,
    preflash_exposure=0.0,
    enlarger_diffusion=False,
    dir_active=True,
    dir_diffusion_size_um=0.0,
):
    params = init_params(film_profile, print_profile)
    params.io.input_color_space = "sRGB"
    params.io.input_cctf_decoding = False
    params.io.output_color_space = "sRGB"
    params.io.output_cctf_encoding = True
    params.io.scan_film = scan_film
    params.io.input_gamut_compress = InputGamutCompressSpec(active=False)
    params.io.output_gamut_compress = OutputGamutCompressSpec(algorithm="off")
    params.camera.auto_exposure = False
    params.camera.exposure_compensation_ev = camera_exposure_ev
    params.camera.film_format_mm = 35.0
    params.camera.lens_blur_um = 0.0
    _configure_diffusion_filter(params.camera.diffusion_filter, active=False)
    params.film_render.density_curve_gamma = film_gamma
    params.film_render.dir_couplers.active = dir_active
    params.film_render.dir_couplers.diffusion_size_um = dir_diffusion_size_um
    params.film_render.grain.active = False
    params.film_render.halation.active = False
    params.film_render.glare.active = False
    params.print_render.density_curves_morph = PrintCurvesMorphParams(
        active=print_gamma != 1.0,
        gamma_factor=print_gamma,
    )
    params.settings.neutral_print_filters_from_database = False
    params.enlarger.c_filter_neutral = neutral_cmy[0]
    params.enlarger.m_filter_neutral = neutral_cmy[1]
    params.enlarger.y_filter_neutral = neutral_cmy[2]
    params.settings.use_enlarger_lut = False
    params.settings.use_scanner_lut = True
    params.settings.lut_resolution = 17
    params.enlarger.print_exposure_compensation = print_exposure_compensation
    params.enlarger.normalize_print_exposure = normalize_print_exposure
    params.enlarger.print_exposure = print_exposure
    params.enlarger.preflash_exposure = preflash_exposure
    params.enlarger.lens_blur = 0.0
    _configure_diffusion_filter(
        params.enlarger.diffusion_filter,
        active=enlarger_diffusion,
    )
    params.print_render.glare.active = False
    params.scanner.lens_blur = 0.0
    params.scanner.unsharp_mask = (0.0, 0.0)
    params.debug.deactivate_spatial_effects = False
    params.debug.deactivate_stochastic_effects = True
    params.scanner.black_correction = scanner_black_correction
    params.scanner.white_correction = scanner_white_correction
    params = digest_params(params)
    assert_g09_effective_settings(
        params,
        scan_film=scan_film,
        enlarger_diffusion=enlarger_diffusion,
        dir_active=dir_active,
        dir_diffusion_size_um=dir_diffusion_size_um,
        output_cctf_encoding=True,
    )
    return params


def save_array(path: Path, values: np.ndarray, dtype: np.dtype) -> None:
    converted = np.asarray(values, dtype=dtype)
    converted.tofile(path)


def generate_route_reference(
    root: Path,
    identifier: str,
    chart: np.ndarray,
    film_profile: str,
    *,
    print_profile: str = PRINT_PROFILE,
    scan_film: bool,
    neutral_cmy,
    **settings,
) -> dict:
    route_root = root / identifier
    route_root.mkdir(parents=True)
    params = configured_params(
        film_profile,
        print_profile,
        scan_film=scan_film,
        neutral_cmy=neutral_cmy,
        **settings,
    )
    pipeline = SimulationPipeline(params)
    rgb_input = np.asarray(chart[:, :, :3], dtype=np.float64)
    film_density = pipeline.process(rgb_input, collect=Tap.CMY_FILM)
    print_density = None
    print_log_exposure = None
    if not scan_film:
        print_log_exposure = pipeline.process(rgb_input, collect=Tap.LOG_E_PRINT)
        print_density = pipeline.process(rgb_input, collect=Tap.CMY_PRINT)
    output = pipeline.process(rgb_input, collect=Tap.RGB_OUT)
    linear_params = configured_params(
        film_profile,
        print_profile,
        scan_film=scan_film,
        neutral_cmy=neutral_cmy,
        **settings,
    )
    linear_params.io.output_cctf_encoding = False
    linear_settings = assert_g09_effective_settings(
        linear_params,
        scan_film=scan_film,
        enlarger_diffusion=settings.get("enlarger_diffusion", False),
        dir_active=settings.get("dir_active", True),
        dir_diffusion_size_um=settings.get("dir_diffusion_size_um", 0.0),
        output_cctf_encoding=False,
    )
    linear_output = SimulationPipeline(linear_params).process(
        rgb_input,
        collect=Tap.RGB_OUT,
    )
    save_array(route_root / "film_density_cmy_interleaved.f64", film_density, np.float64)
    if print_density is not None:
        save_array(
            route_root / "print_log_exposure_rgb_interleaved.f64",
            print_log_exposure,
            np.float64,
        )
        save_array(
            route_root / "print_density_cmy_interleaved.f64",
            print_density,
            np.float64,
        )
    save_array(
        route_root / "scanner_linear_rgb_interleaved.f64",
        linear_output,
        np.float64,
    )
    save_array(
        route_root / "output_rgb_clipped_interleaved.f64",
        np.clip(output, 0.0, 1.0),
        np.float64,
    )
    return {
        "id": identifier,
        "film_profile": film_profile,
        "print_profile": None if scan_film else print_profile,
        "scan_film": scan_film,
        "neutral_cmy_kodak_cc": list(neutral_cmy),
        "settings": settings,
        "effective_settings": g09_effective_settings(params),
        "linear_output_effective_settings": linear_settings,
    }


def reference_curve_samples(
    axis: np.ndarray,
    density: np.ndarray,
    queries: np.ndarray,
    factors: np.ndarray,
) -> list[float]:
    output = []
    density_rgb = np.repeat(np.asarray(density)[:, None], 3, axis=1)
    for query, factor in zip(queries, factors, strict=True):
        image = np.full((1, 1, 3), query, dtype=np.float64)
        scaled_axis = np.repeat(
            (np.asarray(axis, dtype=np.float64) / float(factor))[:, None],
            3,
            axis=1,
        )
        output.append(float(fast_interp(image, scaled_axis, density_rgb)[0, 0, 0]))
    return output


def curve_probe_case(
    identifier: str,
    classification: str,
    axis,
    density,
    queries,
    factors,
) -> dict:
    axis_float = np.asarray(axis, dtype=np.float32)
    density_float = np.asarray(density, dtype=np.float32)
    query_float = np.asarray(queries, dtype=np.float32)
    factor_float = np.asarray(factors, dtype=np.float32)
    return {
        "id": identifier,
        "classification": classification,
        "axis_float32": axis_float.tolist(),
        "density_float32": density_float.tolist(),
        "queries_float32": query_float.tolist(),
        "gamma_float32": factor_float.tolist(),
        "domain_begin": 0,
        "domain_end": int(axis_float.size - 1),
        "expected_float64": reference_curve_samples(
            axis_float.astype(np.float64),
            density_float.astype(np.float64),
            query_float.astype(np.float64),
            factor_float.astype(np.float64),
        ),
    }


def dir_probe_case(
    identifier: str,
    classification: str,
    positive: bool,
    axis,
    curves_bgr,
    log_exposure_bgr,
    density_bgr,
    matrix,
    maxima,
    factors,
) -> dict:
    axis_float = np.asarray(axis, dtype=np.float32)
    curves_float = np.asarray(curves_bgr, dtype=np.float32)
    log_float = np.asarray(log_exposure_bgr, dtype=np.float32)
    density_float = np.asarray(density_bgr, dtype=np.float32)
    matrix_float = np.asarray(matrix, dtype=np.float32)
    maxima_float = np.asarray(maxima, dtype=np.float32)
    factors_float = np.asarray(factors, dtype=np.float32)
    corrected = compute_exposure_correction_dir_couplers(
        log_float.astype(np.float64)[None, :, :],
        density_float.astype(np.float64)[None, :, :],
        maxima_float.astype(np.float64),
        matrix_float.astype(np.float64),
        0.0,
        positive=positive,
    )[0]
    curve_matrix = curves_float.T.astype(np.float64)
    scaled_axis = axis_float.astype(np.float64)[:, None] / factors_float.astype(
        np.float64
    )[None, :]
    final_density = fast_interp(
        np.ascontiguousarray(corrected[None, :, :]),
        np.ascontiguousarray(scaled_axis),
        np.ascontiguousarray(curve_matrix),
    )[0]
    return {
        "id": identifier,
        "classification": classification,
        "positive": positive,
        "axis_float32": axis_float.tolist(),
        "curve_density_bgr_float32": curves_float.tolist(),
        "log_exposure_bgr_float32": log_float.tolist(),
        "initial_density_bgr_float32": density_float.tolist(),
        "matrix_float32": matrix_float.reshape(-1).tolist(),
        "dmax_float32": maxima_float.tolist(),
        "gamma_bgr_float32": factors_float.tolist(),
        "expected_corrected_log_exposure_bgr_float64": corrected.tolist(),
        "expected_final_density_bgr_float64": final_density.tolist(),
    }


def generate_sampling_dir_probes() -> dict:
    curve_cases = [
        curve_probe_case(
            "internal-repeat-rightmost",
            "supported-semantic-defect",
            [-1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 2.0, 3.0],
            [-2.0, -1.0, -0.5, 0.0, 0.0001, 0.5, 1.0, 2.0],
            [1.0] * 8,
        ),
        curve_probe_case(
            "endpoint-repeats",
            "supported-endpoint-semantics",
            [-1.0, -1.0, 0.0, 1.0, 1.0],
            [10.0, 11.0, 20.0, 30.0, 31.0],
            [-2.0, -1.0, -0.9999, 0.0, 0.9999, 1.0, 2.0],
            [1.0] * 7,
        ),
        curve_probe_case(
            "single-sample",
            "supported-single-sample",
            [0.25],
            [3.5],
            [-10.0, 0.25, 10.0],
            [0.05, 1.0, 4.0],
        ),
        curve_probe_case(
            "nonuniform-factor-sweep",
            "supported-float32-sampling-arithmetic",
            [-3.0, -1.25, -0.5, 0.2, 2.4],
            [-0.25, 0.4, 1.1, 1.8, 4.2],
            [-12.0, -2.5, -0.7, 0.0, 0.18, 0.91, 3.0, 12.0],
            [0.05, 0.5, 1.0, 1.1, 1.25, 2.0, 4.0, 4.0],
        ),
    ]
    curves = [
        [0.0, 4.0],
        [0.0, 8.0],
        [0.0, 12.0],
    ]
    dir_cases = [
        dir_probe_case(
            "negative-boundary-crossing-gamma-half",
            "supported-semantic-defect",
            False,
            [-2.0, 2.0],
            curves,
            [[-2.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            np.eye(3),
            [4.0, 8.0, 12.0],
            [0.5, 1.0, 1.0],
        ),
        dir_probe_case(
            "positive-boundary-crossing-gamma-half",
            "supported-polarity-mapping",
            True,
            [-2.0, 2.0],
            curves,
            [[-2.0, 0.0, 0.0]],
            [[3.0, 8.0, 12.0]],
            np.eye(3),
            [4.0, 8.0, 12.0],
            [0.5, 1.0, 1.0],
        ),
        dir_probe_case(
            "finite-correction-above-old-cap",
            "operator-only-semantic-sentinel",
            False,
            [-30.0, 30.0],
            [[0.0, 60.0], [10.0, 70.0], [20.0, 80.0]],
            [[1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0]],
            np.diag([20.0, 21.0, 22.0]),
            [60.0, 70.0, 80.0],
            [1.0, 1.0, 1.0],
        ),
        dir_probe_case(
            "positive-finite-silver-below-zero",
            "operator-only-semantic-sentinel",
            True,
            [-4.0, 4.0],
            [[0.0, 8.0], [1.0, 9.0], [2.0, 10.0]],
            [[0.0, 0.0, 0.0]],
            [[2.0, 3.0, 4.0]],
            [[1.0, 0.25, 0.5], [0.75, 1.0, 0.2], [0.1, 0.3, 1.0]],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ),
        dir_probe_case(
            "negative-donor-receiver-mapping",
            "supported-channel-order",
            False,
            [-5.0, 5.0],
            [[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]],
            [[2.0, 2.0, 2.0], [-1.0, 0.5, 1.5]],
            [[0.2, 0.4, 0.6], [1.0, 0.5, 0.25]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [10.0, 11.0, 12.0],
            [1.0, 2.0, 4.0],
        ),
    ]
    return {"schema_version": 1, "curve_cases": curve_cases, "dir_cases": dir_cases}


def fitted_table(profile, factor: float) -> np.ndarray:
    active = factor != 1.0
    return apply_print_curves_morph(
        profile.data.log_exposure,
        profile.data.density_curves_model,
        PrintCurvesMorphParams(active=active, gamma_factor=factor),
        profile_type=profile.info.type,
    )


def make_cdf_case(identifier, model, positive, factor, exposure) -> dict:
    expected = apply_print_curves_morph(
        np.asarray([exposure], dtype=np.float64),
        model,
        PrintCurvesMorphParams(active=factor != 1.0, gamma_factor=factor),
        profile_type="positive" if positive else "negative",
    )[0]
    return {
        "id": identifier,
        "positive": positive,
        "gamma": float(factor),
        "log_exposure": float(exposure),
        "centers": model.centers.tolist(),
        "amplitudes": model.amplitudes.tolist(),
        "sigmas": model.sigmas.tolist(),
        "expected_float64": expected.tolist(),
        "expected_float32": expected.astype(np.float32).tolist(),
    }


def generate_print_backend_probes() -> dict:
    sampler_cases = []
    cdf_cases = []
    for stock in PRINT_STOCKS:
        profile = load_profile(stock)
        axis_float = profile.data.log_exposure.astype(np.float32)
        midpoint = (axis_float[:-1] + axis_float[1:]) * np.float32(0.5)
        queries = np.unique(
            np.concatenate(
                ([axis_float[0] - 1.0], axis_float, midpoint, [axis_float[-1] + 1.0])
            )
        ).astype(np.float32)
        query_rgb = np.repeat(queries[:, None], 3, axis=1)
        for factor in PRINT_FACTORS:
            table_double = fitted_table(profile, factor)
            table_float = table_double.astype(np.float32)
            expected = fast_interp(
                np.ascontiguousarray(query_rgb),
                np.ascontiguousarray(axis_float),
                np.ascontiguousarray(table_float),
            )
            sampler_cases.append(
                {
                    "id": f"{stock}-gamma-{factor:.17g}",
                    "stock": stock,
                    "gamma": factor,
                    "axis_double": profile.data.log_exposure.tolist(),
                    "axis_float32": axis_float.tolist(),
                    "density_double": table_double.tolist(),
                    "density_cmy_float32": table_float.tolist(),
                    "queries_cmy_float32": query_rgb.tolist(),
                    "expected_float64": expected.astype(np.float64).tolist(),
                }
            )
        for factor in (0.5, 1.0, 1.00000001, 1.1, 2.0):
            for exposure in (
                float(profile.data.log_exposure[0]),
                0.0,
                float(profile.data.log_exposure[-1]),
            ):
                cdf_cases.append(
                    make_cdf_case(
                        f"{stock}-gamma-{factor:.17g}-x-{exposure:.17g}",
                        profile.data.density_curves_model,
                        profile.info.type == "positive",
                        factor,
                        exposure,
                    )
                )

    synthetic = DensityCurvesModel(
        centers=np.asarray(
            [[-1.2, 0.1, 1.4], [-0.9, 0.0, 1.8], [-1.7, 0.4, 1.1]],
            dtype=np.float64,
        ),
        amplitudes=np.asarray(
            [[0.0, 1.5, -0.25], [0.3, 0.0, 2.0], [-0.2, 0.75, 0.0]],
            dtype=np.float64,
        ),
        sigmas=np.asarray(
            [[0.01, 0.08, 0.3], [0.049, 0.12, 0.5], [0.02, 0.07, 0.25]],
            dtype=np.float64,
        ),
    )
    adjacent_one = (
        float(np.nextafter(np.float64(1.0), np.float64(0.0))),
        1.0,
        float(np.nextafter(np.float64(1.0), np.float64(2.0))),
        1.00000001,
    )
    for positive in (False, True):
        for factor in (0.5, *adjacent_one, 1.1, 2.0):
            for exposure in (-8.0, -1.0, 0.0, 0.75, 8.0):
                cdf_cases.append(
                    make_cdf_case(
                        f"synthetic-{'positive' if positive else 'negative'}-"
                        f"gamma-{factor:.17g}-x-{exposure:.17g}",
                        synthetic,
                        positive,
                        factor,
                        exposure,
                    )
                )
    for channel in range(3):
        for layer in range(3):
            amplitudes = np.zeros((3, 3), dtype=np.float64)
            amplitudes[channel, layer] = 1.0
            one_hot = DensityCurvesModel(
                centers=synthetic.centers,
                amplitudes=amplitudes,
                sigmas=synthetic.sigmas,
            )
            cdf_cases.append(
                make_cdf_case(
                    f"one-hot-channel-{channel}-layer-{layer}",
                    one_hot,
                    False,
                    1.25,
                    0.0,
                )
            )
    return {
        "schema_version": 1,
        "cdf_cases": cdf_cases,
        "sampler_cases": sampler_cases,
    }


def generate_model_ingestion_inputs() -> dict:
    return {
        "schema_version": 1,
        "selected_print_profile": PRINT_PROFILE,
        "cases": [
            {"id": "missing-model", "mutation": "delete data.density_curves_model"},
            {
                "id": "wrong-model-tag",
                "mutation": "set data.density_curves_model.model_type to gaussians",
            },
            {
                "id": "wrong-centers-shape",
                "mutation": "replace data.density_curves_model.centers with a 2x3 array",
            },
            {
                "id": "nonfinite-center",
                "mutation": "replace first center JSON number with 1e400",
            },
            {
                "id": "nonpositive-sigma",
                "mutation": "replace first sigma with 0",
            },
            {
                "id": "float32-overflow-derived-table",
                "mutation": "replace first amplitude with 3.5e38",
            },
            {
                "id": "source-double-axis-digest-change",
                "mutation": (
                    "replace an axis double with an adjacent double that retains the same "
                    "Float32 value"
                ),
            },
            {
                "id": "duplicate-axis",
                "mutation": "replace one axis value with its preceding value",
            },
            {
                "id": "malformed-unused-print-model-on-direct",
                "mutation": "delete data.density_curves_model and select a direct route",
            },
        ],
        "allowed": {
            "finite_signed_amplitudes": True,
            "finite_zero_amplitudes": True,
            "duplicate_nondecreasing_axis": True,
        },
    }


def generate_state_inputs() -> dict:
    return {
        "schema_version": 1,
        "film_authored_values": [
            0.05,
            float(np.nextafter(np.float64(0.05), -np.inf)),
            1.0,
            1.00000001,
            4.0,
            float(np.nextafter(np.float64(4.0), np.inf)),
        ],
        "print_authored_values": [
            0.5,
            float(np.nextafter(np.float64(0.5), -np.inf)),
            float(np.nextafter(np.float64(1.0), 0.0)),
            1.0,
            float(np.nextafter(np.float64(1.0), 2.0)),
            1.00000001,
            2.0,
            float(np.nextafter(np.float64(2.0), np.inf)),
        ],
        "nonfinite_values": ["negative-infinity", "positive-infinity", "nan"],
        "required_transitions": [
            "initial-valid",
            "initial-invalid",
            "ordinary-edit",
            "invalid-with-old-publication",
            "recovery",
            "retained-admitted-snapshot",
            "direct-unused-invalid-print",
            "route-switch-validation",
        ],
    }


def generate_lifetime_inputs() -> dict:
    return {
        "schema_version": 1,
        "current_baseline": [
            "finished print submission",
            "direct route switch",
            "prepared-frame abort",
            "idle exact-context retirement",
            "print recovery",
        ],
        "candidate_only": [
            "print gamma 1 to 1.25 to 1 equal-size update",
            "different-N table replacement",
            "queued old/new streams",
            "context reset and shutdown",
        ],
        "constraint": "never hold two simultaneous active retained-scratch leases",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    arguments = parser.parse_args()

    reference_root = arguments.reference_root.resolve()
    fixture_root = arguments.fixture_root.resolve()
    if fixture_root.exists():
        raise RuntimeError(f"reference destination already exists: {fixture_root}")
    revision = git_output(reference_root, "rev-parse", "HEAD")
    if git_output(reference_root, "status", "--short"):
        raise RuntimeError("reference checkout is not clean")

    import spektrafilm

    installed_root = Path(spektrafilm.__file__).resolve().parents[2]
    if not installed_root.samefile(reference_root):
        raise RuntimeError(
            f"installed spektrafilm root {installed_root} does not match {reference_root}"
        )

    case_root = fixture_root / "reference" / CASE_NAME
    case_root.mkdir(parents=True)
    chart = build_chart()
    params = configured_params()
    pipeline = SimulationPipeline(params)
    rgb_input = np.asarray(chart[:, :, :3], dtype=np.float64)
    log_e_film = pipeline.process(rgb_input, collect=Tap.LOG_E_FILM)
    film_density = pipeline.process(rgb_input, collect=Tap.CMY_FILM)
    log_e_print = pipeline.process(rgb_input, collect=Tap.LOG_E_PRINT)
    print_density = pipeline.process(rgb_input, collect=Tap.CMY_PRINT)
    rgb_output = pipeline.process(rgb_input, collect=Tap.RGB_OUT)

    linear_params = configured_params()
    linear_params.io.output_cctf_encoding = False
    default_linear_settings = assert_g09_effective_settings(
        linear_params,
        scan_film=False,
        enlarger_diffusion=False,
        dir_active=True,
        dir_diffusion_size_um=0.0,
        output_cctf_encoding=False,
    )
    linear_output = SimulationPipeline(linear_params).process(
        rgb_input,
        collect=Tap.RGB_OUT,
    )

    save_array(case_root / "input_rgba_interleaved.f32", chart, np.float32)
    save_array(case_root / "log_e_film_rgb_interleaved.f64", log_e_film, np.float64)
    save_array(case_root / "film_density_cmy_interleaved.f64", film_density, np.float64)
    save_array(case_root / "log_e_print_rgb_interleaved.f64", log_e_print, np.float64)
    save_array(case_root / "print_density_cmy_interleaved.f64", print_density, np.float64)
    save_array(case_root / "scanner_linear_rgb_interleaved.f64", linear_output, np.float64)
    save_array(case_root / "output_rgb_interleaved.f64", rgb_output, np.float64)
    save_array(
        case_root / "output_rgb_clipped_interleaved.f64",
        np.clip(rgb_output, 0.0, 1.0),
        np.float64,
    )

    routes_root = fixture_root / "reference" / "routes"
    route_cases = [
        generate_route_reference(
            routes_root,
            "negative-direct",
            chart,
            "kodak_portra_400",
            scan_film=True,
            neutral_cmy=PRODUCT_NEUTRAL_CMY,
        ),
        generate_route_reference(
            routes_root,
            "negative-print",
            chart,
            "kodak_portra_400",
            scan_film=False,
            neutral_cmy=PRODUCT_NEUTRAL_CMY,
        ),
        generate_route_reference(
            routes_root,
            "positive-direct",
            chart,
            "kodak_ektachrome_100",
            scan_film=True,
            neutral_cmy=(0.0, 50.0, 50.0),
        ),
        generate_route_reference(
            routes_root,
            "positive-print",
            chart,
            "kodak_ektachrome_100",
            scan_film=False,
            neutral_cmy=(0.0, 50.0, 50.0),
        ),
    ]

    routes_g09_root = fixture_root / "reference" / "routes-g09"
    routes_g09_cases = [
        generate_route_reference(
            routes_g09_root,
            "compensation-on-manual-plus-one",
            chart,
            FILM_PROFILE,
            scan_film=False,
            neutral_cmy=PRODUCT_NEUTRAL_CMY,
            camera_exposure_ev=1.0,
            print_exposure_compensation=True,
        ),
        generate_route_reference(
            routes_g09_root,
            "compensation-off-manual-plus-one",
            chart,
            FILM_PROFILE,
            scan_film=False,
            neutral_cmy=PRODUCT_NEUTRAL_CMY,
            camera_exposure_ev=1.0,
            print_exposure_compensation=False,
        ),
        generate_route_reference(
            routes_g09_root,
            "scanner-black-white-correction-on",
            chart,
            FILM_PROFILE,
            scan_film=False,
            neutral_cmy=PRODUCT_NEUTRAL_CMY,
            scanner_black_correction=True,
            scanner_white_correction=True,
        ),
        generate_route_reference(
            routes_g09_root,
            "preflash-0.1",
            chart,
            FILM_PROFILE,
            scan_film=False,
            neutral_cmy=PRODUCT_NEUTRAL_CMY,
            preflash_exposure=0.1,
        ),
        generate_route_reference(
            routes_g09_root,
            "enlarger-diffusion-black-pro-mist-half",
            chart,
            FILM_PROFILE,
            scan_film=False,
            neutral_cmy=PRODUCT_NEUTRAL_CMY,
            enlarger_diffusion=True,
        ),
        generate_route_reference(
            routes_g09_root,
            "cine-default",
            chart,
            "kodak_vision3_250d",
            scan_film=False,
            print_profile="kodak_2383",
            neutral_cmy=(0.0, 52.121908804298464, 54.92168330594596),
        ),
        generate_route_reference(
            routes_g09_root,
            "dir-off-direct",
            chart,
            FILM_PROFILE,
            scan_film=True,
            neutral_cmy=PRODUCT_NEUTRAL_CMY,
            dir_active=False,
        ),
        generate_route_reference(
            routes_g09_root,
            "spatial-dir-ramp-edge-direct",
            chart,
            FILM_PROFILE,
            scan_film=True,
            neutral_cmy=PRODUCT_NEUTRAL_CMY,
            dir_diffusion_size_um=2187.5,
        ),
    ]

    route_probe_root = fixture_root / "reference" / "route-probe"
    route_probe_specs = (
        ("paper-film-1.25", FILM_PROFILE, PRINT_PROFILE, PRODUCT_NEUTRAL_CMY, 1.25, 1.0),
        ("paper-print-1.25", FILM_PROFILE, PRINT_PROFILE, PRODUCT_NEUTRAL_CMY, 1.0, 1.25),
        ("paper-combined-0.75-1.25", FILM_PROFILE, PRINT_PROFILE, PRODUCT_NEUTRAL_CMY, 0.75, 1.25),
        (
            "cine-film-1.25",
            "kodak_vision3_250d",
            "kodak_2383",
            (0.0, 52.121908804298464, 54.92168330594596),
            1.25,
            1.0,
        ),
        (
            "cine-print-1.25",
            "kodak_vision3_250d",
            "kodak_2383",
            (0.0, 52.121908804298464, 54.92168330594596),
            1.0,
            1.25,
        ),
        (
            "cine-combined-0.75-1.25",
            "kodak_vision3_250d",
            "kodak_2383",
            (0.0, 52.121908804298464, 54.92168330594596),
            0.75,
            1.25,
        ),
    )
    route_probe_cases = [
        generate_route_reference(
            route_probe_root,
            identifier,
            chart,
            film_profile,
            scan_film=False,
            print_profile=print_profile,
            neutral_cmy=neutral_cmy,
            film_gamma=film_gamma,
            print_gamma=print_gamma,
        )
        for (
            identifier,
            film_profile,
            print_profile,
            neutral_cmy,
            film_gamma,
            print_gamma,
        ) in route_probe_specs
    ]

    probe_root = fixture_root / "reference" / "probes"
    probe_root.mkdir(parents=True)
    sampling_dir_path = probe_root / "sampling_dir.json"
    sampling_dir_path.write_text(
        json.dumps(generate_sampling_dir_probes(), indent=2) + "\n",
        encoding="utf-8",
    )
    print_backend_path = probe_root / "print_backend.json"
    print_backend_path.write_text(
        json.dumps(generate_print_backend_probes(), indent=2) + "\n",
        encoding="utf-8",
    )
    (probe_root / "model_ingestion.json").write_text(
        json.dumps(generate_model_ingestion_inputs(), indent=2) + "\n",
        encoding="utf-8",
    )
    (probe_root / "state.json").write_text(
        json.dumps(generate_state_inputs(), indent=2) + "\n",
        encoding="utf-8",
    )
    (probe_root / "lifetime.json").write_text(
        json.dumps(generate_lifetime_inputs(), indent=2) + "\n",
        encoding="utf-8",
    )

    performance_width = 512
    performance_height = 256
    performance_root = fixture_root / "reference" / "performance"
    performance_root.mkdir(parents=True)
    performance_chart = np.tile(
        chart,
        (
            performance_height // HEIGHT,
            performance_width // WIDTH,
            1,
        ),
    )
    save_array(
        performance_root / "input_rgba_interleaved.f32",
        performance_chart,
        np.float32,
    )

    profile_root = reference_root / "src" / "spektrafilm" / "data" / "profiles"
    package_versions = {}
    for name in (
        "spektrafilm",
        "numpy",
        "scipy",
        "numba",
        "colour-science",
        "lensfunpy",
        "exiv2",
    ):
        package_versions[name] = importlib.metadata.version(name)
    files = {}
    for path in sorted((fixture_root / "reference").rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(fixture_root / "reference").as_posix()
        files[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    manifest = {
        "schema_version": 1,
        "status": "reference-generated-baseline-pending",
        "reference": {
            "revision": revision,
            "root": str(reference_root),
            "python": sys.executable,
            "python_version": sys.version,
            "uv_lock_sha256": sha256_file(reference_root / "uv.lock"),
            "packages": package_versions,
            "profile_sha256": {
                FILM_PROFILE: sha256_file(profile_root / f"{FILM_PROFILE}.json"),
                PRINT_PROFILE: sha256_file(profile_root / f"{PRINT_PROFILE}.json"),
            },
        },
        "case": {
            "id": CASE_NAME,
            "width": WIDTH,
            "height": HEIGHT,
            "input_dtype": "float32",
            "reference_dtype": "float64",
            "input_layout": "R,G,B,A interleaved little-endian",
            "density_layout": "C,M,Y interleaved little-endian",
            "film_profile": FILM_PROFILE,
            "print_profile": PRINT_PROFILE,
            "input_pixels_rgba": chart.reshape(-1, 4).tolist(),
            "settings": {
                "route": "negative_print_scan",
                "input_color_space": "sRGB",
                "input_cctf_decoding": False,
                "input_gamut_compression": "off",
                "output_color_space": "sRGB",
                "output_cctf_encoding": True,
                "output_gamut_compression": "off",
                "final_comparison_clip": [0.0, 1.0],
                "auto_exposure": False,
                "film_gamma": 1.0,
                "print_gamma": 1.0,
                "dir_active": True,
                "dir_diffusion_size_um": 0.0,
                "visual_grain": False,
                "camera_diffusion_active": False,
                "enlarger_diffusion_active": False,
                "halation_active": False,
                "stochastic_effects": False,
                "scanner_black_correction": False,
                "scanner_white_correction": False,
                "scanner_lut": True,
                "scanner_lut_resolution": 17,
                "enlarger_lut": False,
                "neutral_cmy_kodak_cc": PRODUCT_NEUTRAL_CMY,
            },
            "effective_settings": g09_effective_settings(params),
            "linear_output_effective_settings": default_linear_settings,
        },
        "default_routes": route_cases,
        "routes_g09": routes_g09_cases,
        "route_probe": {
            "status": "inputs-and-upstream-frozen; limits-pending-phase-3a",
            "injection": (
                "validation-owned pinned Float32 print curves replace only the local "
                "PrintDevelopPayload curve views after production packing; sampling gamma is one"
            ),
            "cases": route_probe_cases,
        },
        "model_ingestion_inputs": generate_model_ingestion_inputs(),
        "state_inputs": generate_state_inputs(),
        "lifetime_inputs": generate_lifetime_inputs(),
        "performance": {
            "width": performance_width,
            "height": performance_height,
            "warmup_count": 3,
            "measured_count": 11,
            "input_construction": "tile the 16x4 default-paper chart to 512x256",
            "timing": (
                "host prepare_cuda_frame interval plus CUDA events around fused production "
                "launch; stream synchronization; numerical boundary capture costs reported separately"
            ),
        },
        "files": files,
    }
    manifest_path = fixture_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "reference_revision": revision}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
