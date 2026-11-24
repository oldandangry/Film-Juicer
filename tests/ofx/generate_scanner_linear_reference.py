"""Generate offline spectral scanner references from linked OFX input captures."""

import argparse
import ast
import hashlib
import json
import math
import struct
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import opt_einsum


SPEKTRAFILM_REVISION = "3bb2c2d2801ff68b92019cf1dbcbb133d60832bc"
REGISTRATION = "Ofx.Gpu.ScannerReferenceInputs"
RENDER_REGISTRATION = "Ofx.Gpu.RenderReferenceInputs"
ACCEPTED_REGISTRATION = "Ofx.Gpu.AcceptedCleanReferenceInputs"
PROCESSOR_REGISTRATION = "Ofx.Gpu.ProcessorReferenceInputs"
EXTENDED_PROCESSOR_REGISTRATION = "Ofx.Gpu.ProcessorScannerReferenceInputs"
WORKSPACE_REGISTRATION = "Ofx.Gpu.WorkspaceReferenceInputs"
ACCEPTED_SOURCE = Path(__file__).with_name("accepted_cuda_clean_samples.json")
ROUTES = ("negative-direct", "negative-print", "positive-direct", "positive-print")
NATURAL_GLARE_TRACE = "fixed/negative-print"
FIXED_TRACES = {
    NATURAL_GLARE_TRACE: (1, 4, 1, (0, 0)),
    "fixed/positive-direct": (2, 3, 2, (11, -3)),
    "fixed/positive-print": (3, 4, 0, (0, 0)),
    "fixed/combined-print": (1, 3, 0, (0, 0)),
    "fixed/glare-plus-zero": (1, 3, 0, (0, 0)),
    "fixed/glare-minus-zero": (1, 3, 0, (0, 0)),
}
RENDER_CASES = {
    "route/negative-direct": (0, 3, 0, (0, 0)),
    "route/negative-print": (1, 4, 1, (0, 0)),
    "route/positive-direct": (2, 3, 2, (11, -3)),
    "route/positive-print": (3, 4, 0, (0, 0)),
    "format/rgba": (0, 4, 0, (0, 0)),
    "geometry/bounds": (0, 3, 0, (11, -3)),
}
PROCESSOR_CASES = {name: RENDER_CASES[f"route/{name}"] for name in ROUTES}
PROCESSOR_CASES.update(
    {name: FIXED_TRACES[f"fixed/{name}"] for name in ("combined-print", "glare-plus-zero", "glare-minus-zero")}
)
EXTENDED_PROCESSOR_CASES = {
    "negative-direct-ektar": (0, 3, 0, (0, 0)),
    "negative-print-crystal-archive-ii": (1, 4, 1, (0, 0)),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def expected_print_glare_hash(row):
    if not row["glare"]:
        return 0
    key = row["paper"].encode("utf-8")
    data = b"print-glare-v1" + struct.pack("<BQ", row["route"], len(key)) + key
    data += struct.pack("<BIII", 1, *row["glare_controls_bits"])
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def reference_primitives(repository):
    git = ["git", "-C", str(repository)]
    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if revision != SPEKTRAFILM_REVISION:
        raise ValueError(f"unexpected Spektrafilm revision {revision}")
    sources = {
        "model/develop.py": "compute_density_spectral",
        "utils/conversions.py": "density_to_light",
    }
    namespace = {"np": np, "contract": opt_einsum.contract}
    digests = {}
    for relative, name in sources.items():
        path = f"src/spektrafilm/{relative}"
        # These numerical functions depend only on NumPy and opt-einsum. Load
        # their exact committed definitions without the runtime's other stages.
        source = subprocess.check_output([*git, "show", f"{revision}:{path}"])
        matching = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(matching) != 1:
            raise ValueError(f"expected one definition of {name} at {revision}:{path}")
        exec(compile(ast.Module(body=matching, type_ignores=[]), path, "exec"), namespace)
        digests[path] = sha256(source)
    return namespace["compute_density_spectral"], namespace["density_to_light"], digests


def read_capture(result_dir, registration):
    result = json.loads((result_dir / "result.json").read_bytes())
    if (
        result.get("status") != "passed"
        or not result.get("candidate_unchanged")
        or registration not in result.get("selection", ())
    ):
        raise ValueError("expected a completed, unchanged scanner input capture")
    test_log = (result_dir / "ctest.xml").read_bytes()
    (test,) = [entry for entry in ET.fromstring(test_log).iter("testcase") if entry.get("name") == registration]
    stdout = test.findtext("system-out", "")
    records = (json.loads(line) for line in stdout.splitlines() if line.startswith("{"))
    rows = [
        record
        for record in records
        if record.get("schema")
        in {
            "scanner-reference-input-v1",
            "scanner-reference-input-v2",
            "scanner-reference-input-v3",
            "scanner-reference-input-v4",
            "scanner-reference-input-v5",
            "scanner-reference-input-v6",
            "scanner-reference-render-gpu-v1",
            "scanner-reference-accepted-clean-v1",
            "scanner-reference-processor-v1",
            "scanner-reference-processor-extended-v1",
        }
    ]
    schema = rows[0]["schema"] if rows else None
    fixed = ()
    if schema == "scanner-reference-input-v6":
        fixed = tuple(FIXED_TRACES)
    if schema == "scanner-reference-input-v5":
        fixed = tuple(name for name in FIXED_TRACES if name != "fixed/combined-print")
    if schema == "scanner-reference-input-v4":
        fixed = (NATURAL_GLARE_TRACE,)
    ids = (*ROUTES, *fixed)
    routes = (0, 1, 2, 3, *(FIXED_TRACES[name][0] for name in fixed))
    if registration == RENDER_REGISTRATION:
        if schema != "scanner-reference-render-gpu-v1":
            raise ValueError("unexpected OFX higher-level render input schema")
        ids = tuple(RENDER_CASES)
        routes = tuple(case[0] for case in RENDER_CASES.values())
    if registration == ACCEPTED_REGISTRATION:
        if schema != "scanner-reference-accepted-clean-v1":
            raise ValueError("unexpected accepted clean capture input schema")
        ids = tuple(f"{name}-clean" for name in ROUTES)
        routes = (0, 1, 2, 3)
    if registration in {PROCESSOR_REGISTRATION, EXTENDED_PROCESSOR_REGISTRATION, WORKSPACE_REGISTRATION}:
        schedule = "processor" if registration == PROCESSOR_REGISTRATION else "workspace"
        if registration == EXTENDED_PROCESSOR_REGISTRATION:
            if schema != "scanner-reference-processor-extended-v1" or any(
                row.get("schedule") != "processor" for row in rows
            ):
                raise ValueError("unexpected extended processor scan input schema or schedule")
            ids = tuple(EXTENDED_PROCESSOR_CASES)
            routes = tuple(EXTENDED_PROCESSOR_CASES[name][0] for name in ids)
        elif schema != "scanner-reference-processor-v1" or any(row.get("schedule") != schedule for row in rows):
            raise ValueError("unexpected processor scan input schema or schedule")
        elif schedule == "processor":
            ids = tuple(PROCESSOR_CASES)
            routes = tuple(PROCESSOR_CASES[name][0] for name in ids)
        else:
            ids = ROUTES
            routes = (0, 1, 2, 3)
    if (
        tuple(row.get("id") for row in rows) != ids
        or tuple(row.get("route") for row in rows) != routes
        or len({row["schema"] for row in rows}) != 1
    ):
        raise ValueError("unexpected reference input membership")
    return (
        result["preset"],
        rows,
        {
            "candidate_sha256": sha256((result_dir / "candidate.json").read_bytes()),
            "capture_sha256": sha256(test_log),
        },
    )


def promote(words, shape):
    return np.asarray(words, dtype="<u4").view("<f4").astype(np.float64).reshape(shape)


def corrected_xyz(xyz, slope, offset):
    if slope is None:
        return xyz.copy()
    luminance = np.clip(slope * xyz[:, 1] + offset, 0.0, 1.0)
    return xyz * (luminance / (xyz[:, 1] + 1e-10))[:, None]


def device_stage(words, expected, width, limit=None):
    observed = np.asarray(words, dtype="<u8").view("<f8").reshape(expected.shape)
    if not np.isfinite(observed).all():
        raise ValueError("missing or nonfinite CUDA stage sample")
    delta = observed - expected
    absolute = np.abs(delta)
    pixel, channel = np.unravel_index(np.argmax(absolute), expected.shape)
    return {
        "worst_xy": [int(pixel % width), int(pixel // width)],
        "channel": int(channel),
        "observed": float(observed[pixel, channel]),
        "expected": float(expected[pixel, channel]),
        "max_abs": float(absolute[pixel, channel]),
        "delta": delta.tolist(),
        **(
            {"limit": float(limit[pixel, channel]), "passed": bool(np.all(absolute <= limit))}
            if limit is not None
            else {}
        ),
    }


def reinhard_knee(values, parameters):
    threshold, limit, power = parameters
    normalized = np.maximum(values - threshold, 0.0) / (limit - threshold)
    compressed = threshold + (limit - threshold) * normalized / (1.0 + normalized**power) ** (1.0 / power)
    return np.where(values > threshold, compressed, values)


def output_gamut(row, signed_rgb):
    native_to_d65 = promote(row["native_to_d65_bits"], (3, 3))
    d65_to_native = promote(row["d65_to_native_bits"], (3, 3))
    xyz_to_lms, lms_to_xyz, roots_to_lab, lab_to_roots = promote(row["oklab_matrices_bits"], (4, 3, 3))
    cmax = promote(row["gamut_cmax_bits"], (64, 720))
    lab = np.cbrt(signed_rgb @ native_to_d65.T @ xyz_to_lms.T) @ roots_to_lab.T
    lab[:, 0] = reinhard_knee(lab[:, 0], promote(row["gamut_lightness_bits"], (3,)))
    hue = np.arctan2(lab[:, 2], lab[:, 1])
    chroma = np.hypot(lab[:, 1], lab[:, 2])

    minimum = float(np.float32(0.02))
    lightness_position = 63 * np.clip((lab[:, 0] - minimum) / (1.0 - minimum), 0.0, 1.0)
    hue_position = 720 * ((hue + math.pi) % math.tau) / math.tau
    lightness = np.floor(lightness_position).astype(int)
    hue_bin = np.floor(hue_position).astype(int)
    lightness_fraction = lightness_position - lightness
    hue_fraction = hue_position - hue_bin
    next_lightness = np.minimum(lightness + 1, 63)
    h0, h1 = hue_bin % 720, (hue_bin + 1) % 720
    lo = (1 - hue_fraction) * cmax[lightness, h0] + hue_fraction * cmax[lightness, h1]
    hi = (1 - hue_fraction) * cmax[next_lightness, h0] + hue_fraction * cmax[next_lightness, h1]
    boundary = np.maximum((1 - lightness_fraction) * lo + lightness_fraction * hi, 1e-9)
    compressed = boundary * reinhard_knee(chroma / boundary, promote(row["gamut_chroma_bits"], (3,)))
    lab[:, 1:] *= np.divide(compressed, chroma, out=np.ones_like(chroma), where=chroma > 0)[:, None]
    return (lab @ lab_to_roots.T) ** 3 @ lms_to_xyz.T @ d65_to_native.T


def scanner_unsharp(row, linear):
    radius = row.get("unsharp_radius", 0)
    if radius == 0:
        return linear
    height, width, _ = row["shape"]
    sigma, amount = promote(row["unsharp_controls_bits"], (2,))
    if radius < 0 or sigma <= 0 or amount <= 0 or not np.isfinite((sigma, amount)).all():
        raise ValueError("invalid held unsharp parameters")
    weights = [math.exp(-(index * index) / (2 * sigma * sigma)) for index in range(-radius, radius + 1)]
    # Gaussian preparation rounds samples before normalizing by the unrounded sum.
    weights = (np.asarray(weights, dtype="<f4").astype(np.float64) / math.fsum(weights)).astype("<f4")
    original = linear.astype("<f4").reshape(height, width, 3).astype(np.float64)
    horizontal = np.pad(original, ((0, 0), (radius, radius), (0, 0)), mode="symmetric")
    filtered_x = sum(float(weight) * horizontal[:, tap : tap + width] for tap, weight in enumerate(weights))
    vertical = np.pad(filtered_x.astype("<f4"), ((radius, radius), (0, 0), (0, 0)), mode="symmetric").astype(np.float64)
    filtered_y = sum(float(weight) * vertical[tap : tap + height] for tap, weight in enumerate(weights)).astype("<f4")
    return (original + amount * (original - filtered_y)).astype("<f4").reshape(-1, 3).astype(np.float64)


def delivered_reference(row, source, signed_rgb):
    output_flags = row["output_color_flags"]
    if row["schema"] == "scanner-reference-accepted-clean-v1":
        if row["output_color_space"] != 4 or output_flags not in (2, 6):
            raise ValueError("expected already-linear Rec.2020 input to output encoding")
    elif (row["output_color_space"], output_flags, row["cctf_kind"]) != (0, 7, 2):
        raise ValueError("expected prepared sRGB with gamut compression and encoding enabled")
    mapped = output_gamut(row, signed_rgb) if output_flags & 4 else signed_rgb
    linear = scanner_unsharp(row, mapped)
    encoded = linear
    if output_flags & 1:
        encoded = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.maximum(linear, 0) ** (1 / 2.4) - 0.055)
    expected = np.empty_like(source, dtype="<f4")
    expected[:, :3] = np.clip(encoded, 0.0, 1.0)
    observed = promote(row["device_output_bits"], source.shape)
    if source.shape[1] == 4:
        expected[:, 3] = source[:, 3]
    alpha_matches = np.array_equal(observed[:, 3:].view("<u8"), source[:, 3:].view("<u8"))
    absolute = np.abs(observed[:, :3] - expected[:, :3])
    mean = np.mean(absolute, axis=0)
    p999 = np.sort(absolute, axis=0)[math.ceil(0.999 * len(absolute)) - 1]
    maximum = np.max(absolute, axis=0)
    native_pixel_bound = 2e-4 + 3e-4 * np.abs(expected[:, :3])
    return {
        "output_bits": expected.view("<u4").reshape(-1).tolist(),
        "mae": mean.tolist(),
        "p99_9": p999.tolist(),
        "max_abs": maximum.tolist(),
        "alpha_matches": bool(alpha_matches),
        "passed": bool(
            alpha_matches
            and np.isfinite(observed).all()
            and np.all(absolute <= native_pixel_bound)
            and np.all(mean <= 0.25 / 1023)
            and np.all(p999 <= 1 / 1023)
            and np.all(maximum <= 2 / 1023)
        ),
    }


def wavelength_precision(row, reference, cmy, dye, base, weights, inverse_y):
    """Check the actual loop records and attribute the evaluator residual on the worst pixels."""
    height, width, components = row["shape"]
    expected = promote(reference["delivered"]["output_bits"], (height * width, components))
    observed = promote(row["device_output_bits"], expected.shape)
    pixel, _ = np.unravel_index(np.argmax(np.abs(observed[:, :3] - expected[:, :3])), (height * width, 3))
    coordinates = {(int(pixel % width), int(pixel // width))}
    for tap in ("xyz", "signed_rgb"):
        coordinates.add(tuple(reference["device_stages"][tap]["worst_xy"]))
    captured = row["wavelength_arithmetic"]
    xyz_stage = np.asarray(row["device_xyz_bits"], dtype="<u8").view("<f8").reshape(-1, 3)
    corrected_stage = np.asarray(row["device_corrected_xyz_bits"], dtype="<u8").view("<f8").reshape(-1, 3)
    adapted_stage = np.asarray(row["device_adapted_xyz_bits"], dtype="<u8").view("<f8").reshape(-1, 3)
    rgb_stage = np.asarray(row["device_signed_rgb_bits"], dtype="<u8").view("<f8").reshape(-1, 3)
    cat02 = promote(row["cat02_bits"], (3, 3))
    xyz_to_rgb = promote(row["xyz_to_rgb_bits"], (3, 3))
    if len(captured) != height * width:
        raise ValueError("scanner arithmetic trace has the wrong raster shape")

    output = []
    for x, y in sorted(coordinates):
        index = y * width + x
        trace = captured[index]
        prior = [0.0, 0.0, 0.0]
        terms = {name: [[], [], []] for name in ("device_dot", "narrowed_exponent", "device_intrinsic")}
        dot_error = narrowing_error = intrinsic_error = 0.0
        distinct_steps = 0
        fused = split = True
        for step, samples, floor, lanes in zip(trace["steps"], dye, base, weights, strict=True):
            mask = sum((1 << channel) for channel, val in enumerate(lanes) if math.isfinite(val))
            if not mask or any(map(math.isnan, (*samples, floor))):
                if step is not None:
                    raise ValueError("scanner device arithmetic did not omit an inactive wavelength")
                continue
            if step is None or step[0] != mask:
                raise ValueError("scanner device arithmetic omitted an active wavelength or lane")
            _, density_bits, exponent_bits, transmission_bits, accumulated_bits = step
            observed_density = struct.unpack("<d", struct.pack("<Q", density_bits))[0]
            exponent, transmission = promote([exponent_bits, transmission_bits], (2,))
            accumulated = promote(accumulated_bits, (3,)).tolist()
            dot_terms = [float(a) * float(b) for a, b in zip(cmy[index], samples, strict=True)] + [float(floor)]
            expected_density = math.fsum(dot_terms)
            residual = abs(expected_density - observed_density)
            if residual > 4.0 * sys.float_info.epsilon * math.fsum(map(abs, dot_terms)):
                raise ValueError("scanner device dye dot differs from the Float64 arithmetic")
            dot_error = max(dot_error, residual)
            if np.float32(-observed_density).view("<u4") != exponent_bits:
                raise ValueError("scanner exponent differs from the rounded device dye dot")
            narrowing_error = max(narrowing_error, abs(float(exponent) + observed_density))
            rounded_transmission = 10.0 ** float(exponent)
            intrinsic_error = max(intrinsic_error, abs(float(transmission) - rounded_transmission))
            for channel, weight in enumerate(map(float, lanes)):
                fused_sum = split_sum = prior[channel]
                if mask & (1 << channel):
                    for name, value in zip(
                        terms, (10.0**-observed_density, rounded_transmission, float(transmission)), strict=True
                    ):
                        terms[name][channel].append(value * weight)
                    fused_sum = float(np.float32(math.fma(float(transmission), weight, prior[channel])))
                    split_sum = float(np.float32(float(np.float32(float(transmission) * weight)) + prior[channel]))
                    distinct_steps += fused_sum != split_sum
                fused &= accumulated[channel] == fused_sum
                split &= accumulated[channel] == split_sum
            prior = accumulated
        if not (fused or split):
            raise ValueError("scanner wavelength accumulation differs from both ordered Float32 forms")
        scaled = np.asarray(np.asarray(prior, dtype="<f4") * np.float32(inverse_y), dtype="<f4")
        if scaled.view("<u4").tolist() != trace["scaled_xyz_bits"]:
            raise ValueError("scanner inverse-Y multiplication differs from the ordered device accumulation")
        device_xyz = np.maximum(scaled.astype(np.float64), 0.0) + 1e-10
        if not np.array_equal(device_xyz, xyz_stage[index]):
            raise ValueError("scanner output differs from the captured nonnegative Float64 XYZ floor")

        cat02_terms = cat02 * corrected_stage[index]
        device_rgb_terms = xyz_to_rgb * adapted_stage[index]
        for label, transform_terms, actual in (
            ("CAT02", cat02_terms, adapted_stage[index]),
            ("XYZ to RGB", device_rgb_terms, rgb_stage[index]),
        ):
            for lane, value in zip(transform_terms, actual, strict=True):
                if abs(math.fsum(lane) - value) > 4.0 * sys.float_info.epsilon * math.fsum(map(abs, lane)):
                    raise ValueError(f"{label} device matrix residual exceeds ordered Float64 rounding")

        def normalized(lanes):
            return [max(0.0, math.fsum(lane) * float(inverse_y)) + 1e-10 for lane in lanes]

        output.append(
            {
                "xy": [x, y],
                "accumulation": "fused"
                if fused and not split
                else "split"
                if split and not fused
                else "indistinguishable",
                "steps_distinguishing_fused_from_split": distinct_steps,
                "max_abs_dye_dot": dot_error,
                "max_abs_exponent_narrowing": narrowing_error,
                "max_abs_intrinsic_vs_float64_pow": intrinsic_error,
                "xyz_by_precision_stage": {
                    "float64_reference": reference["xyz"][index],
                    **{name: normalized(lanes) for name, lanes in terms.items()},
                    "device_accumulation": normalized([[val] for val in prior]),
                    "device_normalization_and_floor": device_xyz.tolist(),
                },
                "rgb_terms": reference["rgb_terms"][index],
                "device_cat02_terms": cat02_terms.tolist(),
                "device_rgb_terms": device_rgb_terms.tolist(),
            }
        )
    return output


def reference_row(row, density_spectral, density_to_light):
    height, width, components = row["shape"]
    trace = FIXED_TRACES.get(row["id"])
    explicit = trace or RENDER_CASES.get(row["id"])
    processor = row["schema"] in {"scanner-reference-processor-v1", "scanner-reference-processor-extended-v1"}
    if processor:
        explicit = (
            EXTENDED_PROCESSOR_CASES if row["schema"] == "scanner-reference-processor-extended-v1" else PROCESSOR_CASES
        )[row["id"]]
    if row["schema"] == "scanner-reference-accepted-clean-v1":
        explicit = (row["route"], 3, 0, (0, 0))
    natural_glare = (trace is not None or processor) and row["route"] in (1, 3)
    combined = row["id"] in {"fixed/combined-print", "combined-print"}
    expected_components = explicit[1] if explicit else 3 + row["route"] % 2
    if (
        (height, width, components) != (5, 7, expected_components)
        or bool(row["glare"]) != natural_glare
        or bool(row["grain"]) != combined
        or (row.get("unsharp_radius", 0) > 0) != combined
    ):
        raise ValueError(f"invalid deterministic scanner capture for {row['id']}")
    if explicit and (
        row["reconstruction"] != explicit[2] or tuple(row["source_origin"]) != explicit[3] or row["frame_token"] != 37
    ):
        raise ValueError(f"fixed trace controls changed for {row['id']}")
    if "print_glare_hash" in row and row["print_glare_hash"] != expected_print_glare_hash(row):
        raise ValueError(f"prepared Print glare identity differs from the independent value for {row['id']}")
    pixels = height * width
    source = promote(row["source_bits"], (pixels, components))
    cmy = promote(row["developed_cmy_bits"], (pixels, 3))
    dye = promote(row["dye_bits"], (3, 81)).T
    weights = promote(row["weight_bits"], (3, 81)).T
    base = promote(row["base_bits"], (81,)) if row["base_bits"] else np.zeros(81)
    (inverse_y,) = promote([row["inverse_y_bits"]], (1,))
    cat02 = promote(row["cat02_bits"], (3, 3))
    xyz_to_rgb = promote(row["xyz_to_rgb_bits"], (3, 3))
    slope, offset = promote([row["slope_bits"], row["offset_bits"]], (2,))
    if not row["correction_active"]:
        slope = None
    if not np.isfinite(source).all() or not np.isfinite(cmy).all() or not np.isfinite(inverse_y):
        raise ValueError(f"nonfinite capture input in {row['id']}")

    scalar_xyz = []
    for densities in cmy:
        terms = [[], [], []]
        for sample, floor, weight in zip(dye, base, weights, strict=True):
            if not any(map(math.isfinite, weight)) or any(map(math.isnan, (*sample, floor))):
                continue
            absorbance = math.fsum(float(d) * float(a) for d, a in zip(densities, sample, strict=True)) + float(floor)
            transmission = 10.0**-absorbance
            for lane, value in zip(terms, weight, strict=True):
                if math.isfinite(value):
                    lane.append(transmission * float(value))
        scalar_xyz.append([max(0.0, math.fsum(lane) * float(inverse_y)) + 1e-10 for lane in terms])
    scalar_xyz = np.asarray(scalar_xyz)
    active_weights = np.where(np.isfinite(weights), weights, 0.0)
    primitive_light = density_to_light(density_spectral(dye, cmy[None, :, :], base), 1.0)[0]
    primitive_xyz = np.maximum(primitive_light @ active_weights * inverse_y, 0.0) + 1e-10
    scalar_corrected = corrected_xyz(scalar_xyz, slope, offset)
    primitive_corrected = corrected_xyz(primitive_xyz, slope, offset)
    if natural_glare:
        percent = promote(row["glare_field_bits"], (pixels,))
        white = promote(row["white_bits"], (3,))
        if not np.isfinite(percent).all() or not np.all(percent > 0) or not np.isfinite(white).all():
            raise ValueError("invalid natural glare field or scanner white")
        # Freeze the realized post-blur field as an input to the optical oracle.
        glare_xyz = 0.01 * percent[:, None] * white[None, :]
        scalar_corrected += glare_xyz
        primitive_corrected += glare_xyz
    elif "glare_field_bits" in row and list(row["glare_field_bits"]) != [0] * pixels:
        raise ValueError(f"inactive scanner glare field changed for {row['id']}")
    adapted = scalar_corrected @ cat02.T
    rgb_terms = adapted[:, None, :] * xyz_to_rgb[None, :, :]
    signed_rgb = np.asarray([[math.fsum(lane) for lane in pixel] for pixel in rgb_terms])
    primitive_rgb = primitive_corrected @ cat02.T @ xyz_to_rgb.T
    if not np.allclose(scalar_xyz, primitive_xyz, rtol=1e-13, atol=1e-15) or not np.allclose(
        signed_rgb, primitive_rgb, rtol=1e-13, atol=1e-15
    ):
        raise ValueError(f"shared-input Spektrafilm and scalar references diverged for {row['id']}")

    reference = {
        "input": row,
        "xyz": scalar_xyz.tolist(),
        "corrected_xyz": scalar_corrected.tolist(),
        "adapted_xyz": adapted.tolist(),
        "signed_rgb": signed_rgb.tolist(),
        "rgb_terms": rgb_terms.tolist(),
        "shared_reference_max_abs_xyz_delta": float(np.max(np.abs(scalar_xyz - primitive_xyz))),
        "shared_reference_max_abs_rgb_delta": float(np.max(np.abs(signed_rgb - primitive_rgb))),
    }
    if row["schema"] != "scanner-reference-input-v1":
        reference["device_stages"] = {
            "xyz": device_stage(row["device_xyz_bits"], scalar_xyz, width, 1e-10 + 0.01 * np.abs(scalar_xyz)),
            "corrected_xyz": device_stage(row["device_corrected_xyz_bits"], scalar_corrected, width),
            "adapted_xyz": device_stage(row["device_adapted_xyz_bits"], adapted, width),
            "signed_rgb": device_stage(
                row["device_signed_rgb_bits"], signed_rgb, width, 1e-6 + 1e-4 * np.sum(np.abs(rgb_terms), axis=-1)
            ),
        }
    if row["schema"] in {
        "scanner-reference-input-v3",
        "scanner-reference-input-v4",
        "scanner-reference-input-v5",
        "scanner-reference-input-v6",
        "scanner-reference-render-gpu-v1",
        "scanner-reference-accepted-clean-v1",
        "scanner-reference-processor-v1",
        "scanner-reference-processor-extended-v1",
    }:
        reference["delivered"] = delivered_reference(row, source, signed_rgb)
    if "wavelength_arithmetic" in row:
        reference["wavelength_precision"] = wavelength_precision(row, reference, cmy, dye, base, weights, inverse_y)
    return reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="completed native scanner reference input run directory")
    parser.add_argument("spektrafilm", type=Path, help="Git checkout containing the approved Spektrafilm revision")
    parser.add_argument("output", type=Path, help="new offline reference document")
    parser.add_argument(
        "--trace-fixture", type=Path, action="append", default=[], help="freeze matched fixed trace pixels"
    )
    parser.add_argument("--ordinary-fixtures", type=Path, help="freeze the matching native ordinary smoke fixtures")
    parser.add_argument("--render-fixtures", type=Path, help="freeze the matched OFX render fixture cohort")
    parser.add_argument("--accepted-fixtures", type=Path, help="freeze the accepted clean capture fixture cohort")
    parser.add_argument("--processor-fixtures", type=Path, help="freeze the processor callback fixture cohort")
    parser.add_argument("--processor-inputs", action="store_true", help="read the processor callback capture cohort")
    parser.add_argument(
        "--processor-scanner-inputs", action="store_true", help="read the extended scanner capture cohort"
    )
    parser.add_argument(
        "--processor-scanner-fixtures", type=Path, help="freeze the extended grain-off scanner fixture cohort"
    )
    parser.add_argument("--workspace-fixtures", type=Path, help="freeze the workspace actual-schedule fixture cohort")
    args = parser.parse_args()
    if args.accepted_fixtures:
        registration = ACCEPTED_REGISTRATION
    elif args.render_fixtures:
        registration = RENDER_REGISTRATION
    elif args.processor_scanner_inputs or args.processor_scanner_fixtures:
        registration = EXTENDED_PROCESSOR_REGISTRATION
    elif args.processor_inputs or args.processor_fixtures:
        registration = PROCESSOR_REGISTRATION
    elif args.workspace_fixtures:
        registration = WORKSPACE_REGISTRATION
    else:
        registration = REGISTRATION
    preset, captures, provenance = read_capture(args.capture, registration)
    density_spectral, density_to_light, source_hashes = reference_primitives(args.spektrafilm)
    report = {
        "schema": "scanner-numerical-reference-v3",
        "preset": preset,
        "spektrafilm_revision": SPEKTRAFILM_REVISION,
        "spektrafilm_sources": source_hashes,
        "generator_sha256": sha256(Path(__file__).read_bytes()),
        "dependencies": {"python": sys.version, "numpy": np.__version__, "opt-einsum": opt_einsum.__version__},
        **provenance,
        "rows": [reference_row(row, density_spectral, density_to_light) for row in captures],
    }
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    stages_passed = all(
        stage.get("passed", True) for row in report["rows"] for stage in row.get("device_stages", {}).values()
    )
    delivered_passed = all(row.get("delivered", {}).get("passed", True) for row in report["rows"])
    if args.ordinary_fixtures and stages_passed and delivered_passed:
        write_ordinary_fixtures(report, sha256(args.output.read_bytes()), args.ordinary_fixtures)
    if stages_passed and delivered_passed:
        if args.render_fixtures:
            write_render_fixtures(report, sha256(args.output.read_bytes()), args.render_fixtures)
        if args.accepted_fixtures:
            write_accepted_fixtures(report, sha256(args.output.read_bytes()), args.accepted_fixtures)
        if args.processor_fixtures:
            write_processor_fixtures(report, sha256(args.output.read_bytes()), args.processor_fixtures, "processor")
        if args.processor_scanner_fixtures:
            write_extended_processor_fixtures(report, sha256(args.output.read_bytes()), args.processor_scanner_fixtures)
        if args.workspace_fixtures:
            write_processor_fixtures(report, sha256(args.output.read_bytes()), args.workspace_fixtures, "workspace")
        for path in args.trace_fixture:
            write_trace_fixture(report, sha256(args.output.read_bytes()), path)
    return int(not (stages_passed and delivered_passed))


def write_trace_fixture(report, digest, path):
    refs = {row["input"]["id"]: row for row in report["rows"] if row["input"]["id"] in FIXED_TRACES}
    if tuple(refs) != tuple(FIXED_TRACES):
        raise ValueError("fixed scanner trace fixture requires the complete cohort")
    reference_preset = report["preset"]
    preset = path.stem.removeprefix("fixed-trace-")
    if preset not in {reference_preset, reference_preset.removesuffix("-debug") + "-release"}:
        raise ValueError("fixed scanner trace output path must select the matching native preset")
    fields = (
        "id",
        "shape",
        "route",
        "film",
        "paper",
        "reconstruction",
        "exposure_bits",
        "frame_token",
        "frame_rate",
        "source_origin",
        "session_token",
        "instance_token",
        "clip_token",
        "grain",
        "glare",
        "glare_controls_bits",
        "glare_blur_radius",
        "unsharp_radius",
        "unsharp_controls_bits",
        "recipe_hash",
        "scanner_hash",
        "print_glare_hash",
        "source_bits",
        "glare_field_bits",
    )
    rows = {
        name: {
            "inputs": {key: reference["input"][key] for key in fields},
            "output_bits": reference["delivered"]["output_bits"],
        }
        for name, reference in refs.items()
    }
    fixture = {
        "schema": "spectral-scanner-trace-v4",
        "preset": preset,
        "reference_preset": reference_preset,
        "reference_sha256": digest,
        "rows": rows,
    }
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(fixture, output, separators=(",", ":"), allow_nan=False)
        output.write("\n")


def write_ordinary_fixtures(report, digest, directory):
    native_preset = report["preset"]
    if native_preset not in {"linux-debug", "windows-clang-debug"}:
        raise ValueError("ordinary fixture preparation requires a native Debug reference capture")
    rows = {}
    reconstructions = (0, 1, 0, 0)
    for reference, name, reconstruction in zip(report["rows"][: len(ROUTES)], ROUTES, reconstructions, strict=True):
        capture = reference["input"]
        if capture["id"] != name or capture["reconstruction"] != reconstruction or capture["grain"] or capture["glare"]:
            raise ValueError(f"ordinary smoke controls changed in {name}")
        rows[name] = {
            key: capture[key]
            for key in (
                "shape",
                "route",
                "reconstruction",
                "film",
                "paper",
                "grain",
                "glare",
                "frame_token",
                "frame_rate",
                "source_bits",
            )
        }
        rows[name]["output_bits"] = reference["delivered"]["output_bits"]
    for preset in (native_preset, native_preset.removesuffix("-debug") + "-release"):
        fixture = {
            "schema": "spectral-ordinary-v1",
            "preset": preset,
            "reference_preset": native_preset,
            "reference_sha256": digest,
            "rows": rows,
        }
        path = directory / f"ordinary-{preset}.json"
        with path.open("x", encoding="utf-8", newline="\n") as output:
            json.dump(fixture, output, separators=(",", ":"), allow_nan=False)
            output.write("\n")


def write_render_fixtures(report, digest, directory):
    reference_preset = report["preset"]
    rows = []
    for reference in report["rows"]:
        capture = reference["input"]
        row = {
            key: capture[key]
            for key in (
                "id",
                "shape",
                "route",
                "reconstruction",
                "film",
                "paper",
                "source_origin",
                "frame_token",
                "frame_rate",
                "grain",
                "glare",
                "unsharp_radius",
                "output_color_flags",
                "source_bits",
            )
        }
        row["output_bits"] = reference["delivered"]["output_bits"]
        rows.append(row)
    if tuple(row["id"] for row in rows) != tuple(RENDER_CASES):
        raise ValueError("unexpected OFX render fixture membership")
    for preset in (reference_preset, reference_preset.removesuffix("-debug") + "-release"):
        fixture = {
            "schema": "spectral-render-gpu-v1",
            "preset": preset,
            "reference_preset": reference_preset,
            "reference_sha256": digest,
            "rows": rows,
        }
        with (directory / f"render-gpu-{preset}.json").open("x", encoding="utf-8", newline="\n") as output:
            json.dump(fixture, output, separators=(",", ":"), allow_nan=False)
            output.write("\n")


def write_accepted_fixtures(report, digest, directory):
    archive_bytes = ACCEPTED_SOURCE.read_bytes()
    archive = json.loads(archive_bytes)
    original_fields = (
        "capture_baseline",
        "source_shape",
        "layout",
        "source_color_space",
        "frame",
        "fps",
        "camera_format_long_edge_mm",
        "coordinate_map",
        "input_bits",
        "input_sha256",
        "pixel_bound",
        "compaction",
    )
    reference_preset = report["preset"]
    rows = []
    for index, (reference, original) in enumerate(zip(report["rows"], archive["cases"], strict=True)):
        capture = reference["input"]
        if (
            capture["id"] != original["name"]
            or capture["settings"] != original["settings"]
            or capture["route"] != index
            or capture["frame"] != archive["frame"]
            or capture["frame_rate"] != archive["fps"]
            or capture["resolved_format_mm"] != archive["camera_format_long_edge_mm"]
            or capture["source_shape"] != archive["source_shape"]
            or capture["source_bits"] != archive["input_bits"]
            or capture["coordinate_map"] != archive["coordinate_map"]
        ):
            raise ValueError(f"accepted input or control changed for {original['name']}")
        rows.append(
            {
                "name": original["name"],
                "settings": original["settings"],
                "output_bits": reference["delivered"]["output_bits"],
            }
        )
    for preset in (reference_preset, reference_preset.removesuffix("-debug") + "-release"):
        fixture = {
            "schema": "spectral-accepted-clean-v1",
            "preset": preset,
            "reference_preset": reference_preset,
            "reference_sha256": digest,
            "archive_sha256": sha256(archive_bytes),
            **{key: archive[key] for key in original_fields},
            "cases": rows,
        }
        with (directory / f"accepted-clean-{preset}.json").open("x", encoding="utf-8", newline="\n") as output:
            json.dump(fixture, output, separators=(",", ":"), allow_nan=False)
            output.write("\n")


def write_processor_fixtures(report, digest, directory, schedule):
    reference_preset = report["preset"]
    rows = []
    for reference in report["rows"]:
        capture = reference["input"]
        row = {
            key: capture[key]
            for key in (
                "id",
                "shape",
                "route",
                "reconstruction",
                "film",
                "paper",
                "source_origin",
                "frame",
                "frame_rate",
                "grain",
                "glare",
                "exposure_bits",
                "source_bits",
            )
        }
        row["output_bits"] = reference["delivered"]["output_bits"]
        rows.append(row)
    if tuple(row["id"] for row in rows) != (tuple(PROCESSOR_CASES) if schedule == "processor" else ROUTES):
        raise ValueError("unexpected processor fixture membership")
    for preset in (reference_preset, reference_preset.removesuffix("-debug") + "-release"):
        fixture = {
            "schema": "spectral-processor-v1",
            "preset": preset,
            "reference_preset": reference_preset,
            "reference_sha256": digest,
            "schedule": schedule,
            "rows": rows,
        }
        with (directory / f"{schedule}-{preset}.json").open("x", encoding="utf-8", newline="\n") as output:
            json.dump(fixture, output, separators=(",", ":"), allow_nan=False)
            output.write("\n")


def write_extended_processor_fixtures(report, digest, directory):
    reference_preset = report["preset"]
    rows = []
    for reference in report["rows"]:
        capture = reference["input"]
        rows.append(
            {
                key: capture[key]
                for key in (
                    "id",
                    "shape",
                    "route",
                    "reconstruction",
                    "film",
                    "paper",
                    "source_origin",
                    "frame",
                    "frame_rate",
                    "grain",
                    "glare",
                    "exposure_bits",
                    "source_bits",
                )
            }
            | {"output_bits": reference["delivered"]["output_bits"]}
        )
    if tuple(row["id"] for row in rows) != tuple(EXTENDED_PROCESSOR_CASES):
        raise ValueError("unexpected extended processor fixture membership")
    presets = (reference_preset,)
    if reference_preset.endswith("-debug"):
        presets += (reference_preset.removesuffix("-debug") + "-release",)
    for preset in presets:
        fixture = {
            "schema": "spectral-processor-extended-v1",
            "preset": preset,
            "reference_preset": reference_preset,
            "reference_sha256": digest,
            "schedule": "processor",
            "rows": rows,
        }
        with (directory / f"processor-scanner-{preset}.json").open("x", encoding="utf-8", newline="\n") as output:
            json.dump(fixture, output, separators=(",", ":"), allow_nan=False)
            output.write("\n")


if __name__ == "__main__":
    sys.exit(main())
