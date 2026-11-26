#!/usr/bin/env python3
"""Analyze frozen planar grain captures and optionally create paired previews.

Each capture directory contains request.json, manifest.json, and frames.f32.
The output records source paths, frame requests and resolved controls. Previews
are exploratory scalar-record transmission mapped directly to RGB with sRGB
encoding, without a photographic color pipeline.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import grain_analysis as analysis
import numpy as np
from PIL import Image, ImageDraw


def read_capture(path):
    path = Path(path)
    request = json.loads((path / "request.json").read_text(encoding="utf-8"))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    requests = request["requests"]
    width, height = int(manifest["width"]), int(manifest["height"])
    crop_width, crop_height = int(manifest["crop_width"]), int(manifest["crop_height"])
    analysis_width = int(manifest["analysis_width"])
    analysis_height = int(manifest["analysis_height"])
    pitch = float(manifest["pitch_um"])
    if (
        not requests
        or min(width, height, crop_width, crop_height, analysis_width, analysis_height) <= 0
        or crop_width > width
        or crop_height > height
        or analysis_width > crop_width
        or analysis_height > crop_height
        or not math.isfinite(pitch)
        or pitch <= 0
    ):
        raise ValueError(f"invalid capture geometry: {path}")
    size = len(requests) * 3 * crop_height * crop_width * 4
    if (path / "frames.f32").stat().st_size != size:
        raise ValueError(f"capture size does not match request: {path}")
    frames = np.memmap(path / "frames.f32", dtype="<f4", mode="r", shape=(len(requests), 3, crop_height, crop_width))
    window = ((crop_height - analysis_height) // 2, (crop_width - analysis_width) // 2, analysis_height, analysis_width)
    return {
        "path": str(path.resolve()),
        "request": request,
        "manifest": manifest,
        "frames": frames,
        "window": window,
        "pitch": pitch,
    }


def uncertainty(values):
    """Mean, seed standard deviation and standard error, over independent frames."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("invalid observations")
    deviation = values.std(axis=0, ddof=1) if len(values) > 1 else None
    return {
        "count": len(values),
        "mean": values.mean(axis=0).tolist(),
        "seed_sd": None if deviation is None else deviation.tolist(),
        "standard_error": None if deviation is None else (deviation / np.sqrt(len(values))).tolist(),
    }


def analyze_capture(capture, diameters, max_lag, spectrum_path):
    y, x, h, w = capture["window"]
    pitch = capture["pitch"]
    results = []
    spectra = None
    sequence_length = int(capture["request"].get("sequence_length", len(capture["frames"])))
    if not 1 <= sequence_length <= len(capture["frames"]):
        raise ValueError("invalid sequence_length")
    hashes = []
    differences = []
    adjacent_correlations = []
    seams = []
    previous = None
    for index, (frame, request) in enumerate(zip(capture["frames"], capture["request"]["requests"], strict=True)):
        baseline = request["density"]
        hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
        if previous is not None and index < sequence_length:
            adjacent_correlations.append(record_pearson(previous, frame))
            differences.append(
                np.sqrt(np.mean((np.asarray(frame, dtype=np.float64) - previous) ** 2, axis=(1, 2))).tolist()
            )
        previous = np.asarray(frame, dtype=np.float64) if index < sequence_length else None
        seams.append(seam_diagnostics(frame, capture))
        raw = np.asarray(frame[:, y : y + h, x : x + w], dtype=np.float64)
        entry = {"index": index, "request": request, "raw": analysis.descriptors(raw, baseline), "apertures": {}}
        centered = raw - raw.mean(axis=(1, 2), keepdims=True)
        entry["cross_covariance"] = (centered.reshape(3, -1) @ centered.reshape(3, -1).T / (h * w)).tolist()
        try:
            entry["correlation"] = analysis.autocorrelation(raw, max_lag).tolist()
        except ValueError as error:
            entry["correlation_error"] = str(error)
        for diameter in diameters:
            measured = analysis.aperture_density(frame, pitch, capture["window"], diameter)
            entry["apertures"][str(diameter)] = analysis.descriptors(measured, baseline)
        spectrum = analysis.spatial_spectrum(raw, pitch)
        if index < sequence_length:
            spectra = spectrum["power"] if spectra is None else spectra + spectrum["power"]
        results.append(entry)
    np.savez_compressed(spectrum_path, mean_power=spectra / sequence_length, fx=spectrum["fx"], fy=spectrum["fy"])
    groups = {}
    for entry in results[:sequence_length]:
        key = json.dumps(entry["request"]["density"], separators=(",", ":"))
        groups.setdefault(key, []).append(entry)
    summary = {}
    for key, entries in groups.items():
        summary[key] = {
            "raw": {
                name: uncertainty([entry["raw"][name] for entry in entries])
                for name in (
                    "mean_density",
                    "density_shift",
                    "mean_transmission",
                    "transmission_shift",
                    "effective_density",
                    "rms",
                    "skew",
                    "quantiles",
                )
            },
            "apertures": {
                str(d): {
                    name: uncertainty([entry["apertures"][str(d)][name] for entry in entries])
                    for name in ("mean_density", "mean_transmission", "effective_density", "rms", "skew", "quantiles")
                }
                for d in diameters
            },
        }
    return {
        "source": capture["path"],
        "manifest": capture["manifest"],
        "window": capture["window"],
        "pitch_um": pitch,
        "spectrum_file": str(spectrum_path.name),
        "frames": results,
        "by_input_density": summary,
        "temporal": {
            "frame_sha256": hashes,
            "adjacent_density_rms": differences,
            "adjacent_record_pearson": adjacent_correlations,
            "seam_diagnostics": seams,
            "mean_density_correlation": temporal_means(results[:sequence_length]),
            "sequence_length": sequence_length,
            "repeat_requests": repeat_witness(capture["request"]["requests"], hashes),
            "half_variance": half_variance(capture, sequence_length),
        },
    }


def record_pearson(first, second):
    first = np.array(first, dtype=np.float64, copy=True).reshape(3, -1)
    second = np.array(second, dtype=np.float64, copy=True).reshape(3, -1)
    first -= first.mean(axis=1, keepdims=True)
    second -= second.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(first**2, axis=1) * np.sum(second**2, axis=1))
    return [float(np.dot(a, b) / d) if d > 0 else None for a, b, d in zip(first, second, denominator, strict=True)]


def seam_diagnostics(frame, capture):
    """Energy of differences nearest actual Film placement/fade coordinates."""
    request, manifest = capture["request"], capture["manifest"]
    if request.get("engine", manifest.get("engine")) != "film":
        return None
    pitch = capture["pitch"] / float(request["size"])
    if pitch <= 0 or not math.isfinite(pitch):
        raise ValueError("invalid Film size or pitch")
    if "crop_x" not in manifest or "crop_y" not in manifest:
        raise ValueError("Film seam diagnostic requires manifest crop_x and crop_y")
    result = []
    for laying, shift in enumerate((0, 64)):
        axes = {}
        for axis, origin in ((2, int(manifest["crop_x"])), (1, int(manifest["crop_y"]))):
            length = frame.shape[axis]
            # A difference spans neighboring sample centers; evaluate its midpoint.
            coordinates = (np.arange(1, length) + origin) * pitch + shift
            phase = (coordinates + 64) % 128 - 64
            gradient = np.diff(np.asarray(frame, dtype=np.float64), axis=axis) ** 2
            bands = {}
            for name, distance in (("block", np.abs(phase)), ("fade_start", np.abs(np.abs(phase) - 16))):
                mask = distance <= pitch
                bands[name] = {
                    "count": int(mask.sum()),
                    "mean_squared_gradient": np.take(gradient, np.flatnonzero(mask), axis=axis)
                    .mean(axis=(1, 2))
                    .tolist()
                    if np.any(mask)
                    else None,
                }
            mask = (np.abs(phase) > pitch) & (np.abs(np.abs(phase) - 16) > pitch)
            bands["outside"] = {
                "count": int(mask.sum()),
                "mean_squared_gradient": np.take(gradient, np.flatnonzero(mask), axis=axis).mean(axis=(1, 2)).tolist()
                if np.any(mask)
                else None,
            }
            axes[str(axis)] = bands
        result.append({"laying": laying, "shift_texels": shift, "axes": axes})
    return {"block_texels": 128, "fade_texels": 16, "band_half_width_samples": 1, "layings": result}


def repeat_witness(requests, hashes):
    """Compare every repeated request with its first occurrence bit for bit."""
    seen = {}
    comparisons = []
    for index, (request, digest) in enumerate(zip(requests, hashes, strict=True)):
        key = json.dumps(request, sort_keys=True, separators=(",", ":"))
        if key in seen:
            prior = seen[key]
            comparisons.append({"first": prior, "repeat": index, "bitwise_equal": digest == hashes[prior]})
        else:
            seen[key] = index
    return {"comparisons": comparisons, "passed": all(item["bitwise_equal"] for item in comparisons)}


def half_variance(capture, sequence_length):
    """Inspect integer/half/integer triples without assuming independent seeds."""
    requests = capture["request"]["requests"][:sequence_length]
    by_time = {float(request["time"]): index for index, request in enumerate(requests)}
    triples = []
    y, x, h, w = capture["window"]
    for time, first in sorted(by_time.items()):
        if not time.is_integer() or time + 0.5 not in by_time or time + 1 not in by_time:
            continue
        indices = [first, by_time[time + 0.5], by_time[time + 1]]
        frames = [capture["frames"][i] for i in indices]
        raw = [np.asarray(f[:, y : y + h, x : x + w], dtype=np.float64) for f in frames]
        centered = [a - a.mean(axis=(1, 2), keepdims=True) for a in raw]
        variance = [np.mean(a * a, axis=(1, 2)) for a in centered]
        covariance = np.mean(centered[0] * centered[2], axis=(1, 2))
        denominator = (variance[0] + variance[2]) / 2
        ratio = [float(v / d) if d > 0 else None for v, d in zip(variance[1], denominator, strict=True)]
        triples.append(
            {
                "times": [time, time + 0.5, time + 1],
                "indices": indices,
                "cross_seed_covariance": covariance.tolist(),
                "cross_seed_pearson": record_pearson(raw[0], raw[2]),
                "half_to_mean_integer_variance": ratio,
                "raw": [analysis.descriptors(a) for a in raw],
                "aperture_48um": [
                    analysis.descriptors(analysis.aperture_density(f, capture["pitch"], capture["window"]))
                    for f in frames
                ],
            }
        )
    return triples


def analyze_patches(capture, diameters, tile_size=96):
    """Measure central 96-square of each frozen row-major patch for every seed."""
    requests = capture["request"]["requests"]
    first = requests[0]
    columns, rows = int(first["columns"]), int(first["rows"])
    patches, names = first["patches"], first["patch_names"]
    if columns <= 0 or rows <= 0 or len(patches) != rows * columns or len(names) != len(patches):
        raise ValueError("invalid patch layout")
    h, w = capture["frames"].shape[2:]
    manifest = capture["manifest"]
    if (h, w) != (int(manifest["height"]), int(manifest["width"])):
        raise ValueError("patches require retained whole-frame output")
    metrics = []
    for frame, request in zip(capture["frames"], requests, strict=True):
        if any(request[key] != first[key] for key in ("patches", "patch_names", "columns", "rows")):
            raise ValueError("patch layout differs between frames")
        per_patch = []
        for i, (baseline, name) in enumerate(zip(patches, names, strict=True)):
            row, col = divmod(i, columns)
            y0, y1 = row * h // rows, (row + 1) * h // rows
            x0, x1 = col * w // columns, (col + 1) * w // columns
            if min(y1 - y0, x1 - x0) < tile_size:
                raise ValueError("tile smaller than analysis window")
            y, x = (y0 + y1 - tile_size) // 2, (x0 + x1 - tile_size) // 2
            max_radius = analysis.aperture_kernel(capture["pitch"], max(diameters)).shape[0] // 2
            if (
                y - max_radius < y0
                or x - max_radius < x0
                or y + tile_size + max_radius > y1
                or x + tile_size + max_radius > x1
            ):
                raise ValueError("aperture support crosses patch boundary")
            window = (y, x, tile_size, tile_size)
            raw = frame[:, y : y + tile_size, x : x + tile_size]
            halo = frame[:, y - max_radius : y + tile_size + max_radius, x - max_radius : x + tile_size + max_radius]
            local_window = (max_radius, max_radius, tile_size, tile_size)
            per_patch.append(
                {
                    "name": name,
                    "baseline": baseline,
                    "window": window,
                    "raw": analysis.descriptors(raw, baseline),
                    "apertures": {
                        str(d): analysis.descriptors(
                            analysis.aperture_density(halo, capture["pitch"], local_window, d), baseline
                        )
                        for d in diameters
                    },
                }
            )
        metrics.append({"request": request, "patches": per_patch})
    names_to_summarize = (
        "mean_density",
        "density_shift",
        "mean_transmission",
        "transmission_shift",
        "effective_density",
        "rms",
        "skew",
        "quantiles",
    )
    summaries = []
    for i, name in enumerate(names):
        summaries.append(
            {
                "name": name,
                "baseline": patches[i],
                "raw": {k: uncertainty([m["patches"][i]["raw"][k] for m in metrics]) for k in names_to_summarize},
                "apertures": {
                    str(d): {
                        k: uncertainty([m["patches"][i]["apertures"][str(d)][k] for m in metrics])
                        for k in names_to_summarize
                    }
                    for d in diameters
                },
            }
        )
    return {
        "source": capture["path"],
        "manifest": manifest,
        "pitch_um": capture["pitch"],
        "frames": metrics,
        "patches": summaries,
    }


def analyze_scale(captures, extent_um=512):
    """Compare aligned frames against transmission reductions of 0.25 and 1 µm captures."""
    by_pitch = {c["pitch"]: c for c in captures}
    if len(by_pitch) != len(captures) or 0.25 not in by_pitch or 1.0 not in by_pitch:
        raise ValueError("scale captures need unique pitches including 0.25 and 1 µm")
    expected = {0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0}
    if set(by_pitch) != expected:
        raise ValueError("scale captures must cover 0.25, 0.5, 1, 2, 4, 8 and 16 µm")
    requests = captures[0]["request"]["requests"]
    if any(c["request"]["requests"] != requests for c in captures):
        raise ValueError("scale captures require identical ordered requests")

    def patch(capture, index):
        n = extent_um / capture["pitch"]
        if not n.is_integer():
            raise ValueError("physical extent is not an integer number of pixels")
        n = int(n)
        frame = capture["frames"][index]
        h, w = frame.shape[1:]
        if min(h, w) < n:
            raise ValueError("capture is smaller than physical extent")
        y, x = (h - n) // 2, (w - n) // 2
        return np.asarray(frame[:, y : y + n, x : x + n], dtype=np.float64)

    rows = []
    for index, request in enumerate(requests):
        refs = {p: patch(by_pitch[p], index) for p in (0.25, 1.0)}
        for pitch, capture in sorted(by_pitch.items()):
            observed = patch(capture, index)
            observed_stats = analysis.descriptors(observed)
            comparisons = {}
            for base, reference in refs.items():
                if pitch < base:
                    continue
                ratio = pitch / base
                if not ratio.is_integer():
                    raise ValueError("nonintegral scale ratio")
                reduced = analysis.downsample_transmission(reference, int(ratio))
                ref_stats = analysis.descriptors(reduced)
                comparisons[str(base)] = {
                    "pearson": record_pearson(observed, reduced),
                    "density_mean_drift": (
                        np.array(observed_stats["mean_density"]) - ref_stats["mean_density"]
                    ).tolist(),
                    "density_rms_drift": (np.array(observed_stats["rms"]) - ref_stats["rms"]).tolist(),
                    "transmission_mean_drift": (
                        np.array(observed_stats["mean_transmission"]) - ref_stats["mean_transmission"]
                    ).tolist(),
                    "reference": ref_stats,
                }
            rows.append(
                {
                    "index": index,
                    "request": request,
                    "pitch_um": pitch,
                    "source": capture["path"],
                    "observed": observed_stats,
                    "comparisons": comparisons,
                }
            )
    return {
        "extent_um": extent_um,
        "alignment": "centered retained crops; assumes identical physical center and seed",
        "rows": rows,
    }


def temporal_means(results):
    means = np.asarray([entry["raw"]["mean_density"] for entry in results])
    curves = []
    for record in range(3):
        try:
            curves.append(analysis.temporal_correlation(means[:, record], min(32, len(means) - 1)))
        except ValueError:
            curves.append(None)
    return curves


def preview_rgb(frame, window):
    y, x, h, w = window
    light = np.clip(10.0 ** (-np.asarray(frame[:, y : y + h, x : x + w], dtype=np.float64)), 0, 1)
    srgb = np.where(light <= 0.0031308, 12.92 * light, 1.055 * light ** (1 / 2.4) - 0.055)
    return np.rint(np.moveaxis(srgb, 0, -1) * 255).astype(np.uint8)


def previews(captures, output, ffmpeg, fps, seed):
    if len(captures) != 2:
        raise ValueError("paired previews require exactly two captures")
    a, b = captures
    if a["window"][2:] != b["window"][2:] or len(a["frames"]) != len(b["frames"]):
        raise ValueError("paired captures must have equal frame counts and crop sizes")
    if a["request"]["requests"] != b["request"]["requests"]:
        raise ValueError("paired captures must have identical ordered requests")
    rng = np.random.default_rng(seed)
    swapped = bool(rng.integers(2))
    ordered = [b, a] if swapped else [a, b]
    w, h = a["window"][3], a["window"][2]
    first = np.concatenate([preview_rgb(c["frames"][0], c["window"]) for c in ordered], axis=1)
    sheet = Image.fromarray(first)
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), "A", fill="white", stroke_width=1, stroke_fill="black")
    draw.text((w + 8, 8), "B", fill="white", stroke_width=1, stroke_fill="black")
    sheet.save(output / "contact.png")
    video = None
    if ffmpeg:
        movie = output / "paired.mp4"
        frame_count = int(a["request"].get("sequence_length", len(a["frames"])))
        rgb_hash = hashlib.sha256()
        command = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{2 * w}x{h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            "-preset",
            "fast",
            "-pix_fmt",
            "rgb24",
            "-movflags",
            "+faststart",
            str(movie),
        ]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
            try:
                for i in range(frame_count):
                    pixels = np.concatenate(
                        [preview_rgb(c["frames"][i], c["window"]) for c in ordered], axis=1
                    ).tobytes()
                    rgb_hash.update(pixels)
                    process.stdin.write(pixels)
            finally:
                process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("ffmpeg preview failed")
        decoded = subprocess.check_output(
            [str(ffmpeg), "-v", "error", "-i", str(movie), "-c:v", "rawvideo", "-pix_fmt", "rgb24", "-f", "hash", "-"],
            text=True,
        ).strip()
        if decoded != f"SHA256={rgb_hash.hexdigest()}":
            raise ValueError("decoded preview changed the display RGB samples")
        video = {
            "path": movie.name,
            "frames": frame_count,
            "encoding": "lossless H.264 RGB",
            "rgb_sha256": rgb_hash.hexdigest(),
        }
    return {
        "seed": seed,
        "A": ordered[0]["path"],
        "B": ordered[1]["path"],
        "fps": fps,
        "conversion": "CMY record transmission mapped directly to RGB, clipped to [0,1], sRGB transfer",
        "contact_frame": 0,
        "video": video,
    }


def main():
    mode = "generic"
    if len(sys.argv) > 1 and sys.argv[1] in {"patches", "scale"}:
        mode = sys.argv.pop(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("captures", type=Path, nargs="+")
    parser.add_argument("--diameters", type=float, nargs="+", default=[12, 24, 48, 96])
    parser.add_argument("--max-lag", type=int, default=128)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fps", type=float, default=24)
    args = parser.parse_args()
    if args.preview_only and mode != "generic":
        parser.error("--preview-only requires a pair of generic captures")
    args.output.mkdir(parents=True, exist_ok=False)
    captures = [read_capture(path) for path in args.captures]
    if args.preview_only:
        key = previews(captures, args.output, args.ffmpeg, args.fps, args.seed)
        (args.output / "preview-key.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
        return
    if mode == "scale":
        result = analyze_scale(captures)
        (args.output / "scale.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return
    if mode == "patches":
        result = {
            "captures": [analyze_patches(c, args.diameters) for c in captures],
            "diameters_um": args.diameters,
            "uncertainty": "standard error across frames; only independent seeds justify seed uncertainty",
        }
        (args.output / "patches.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return
    reports = [
        analyze_capture(capture, args.diameters, args.max_lag, args.output / f"spectrum-{i}.npz")
        for i, capture in enumerate(captures)
    ]
    result = {
        "captures": reports,
        "diameters_um": args.diameters,
        "max_lag": args.max_lag,
        "uncertainty": "standard error across frames; interpretable as seed uncertainty only for independent seeds",
    }
    if args.preview:
        result["preview"] = previews(captures, args.output, args.ffmpeg, args.fps, args.seed)
    (args.output / "quality.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
