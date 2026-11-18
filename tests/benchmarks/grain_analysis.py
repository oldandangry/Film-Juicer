"""Offline density and spatial measurements for the grain research capture.

Frames are planar CMY density arrays. Window coordinates are (y, x, height,
width) in the supplied full frame or explicitly recorded halo crop.
"""

import math

import numpy as np
from scipy import fft, signal


def aperture_kernel(pitch_um, diameter_um=48.0):
    """Circle/pixel intersection areas, normalized to one, at pixel centers."""
    if not (math.isfinite(pitch_um) and math.isfinite(diameter_um) and pitch_um > 0 and diameter_um > 0):
        raise ValueError("aperture diameter and pitch must be finite and positive")
    radius = diameter_um / (2 * pitch_um)
    extent = math.ceil(radius - 0.5)
    edges = np.arange(-extent, extent + 2, dtype=np.float64) - 0.5

    def integral(x, y):
        # Oriented rectangle from the origin; integrate min(y, circle height).
        ax = np.minimum(np.abs(x), radius)
        ay = np.minimum(np.abs(y), radius)
        split = np.sqrt(np.maximum(0.0, radius * radius - ay * ay))
        flat = np.minimum(ax, split)

        def primitive(t):
            return 0.5 * (
                t * np.sqrt(np.maximum(0.0, radius * radius - t * t))
                + radius * radius * np.arcsin(np.clip(t / radius, -1, 1))
            )

        return np.sign(x) * np.sign(y) * (ay * flat + primitive(ax) - primitive(flat))

    x = edges[None, :]
    y = edges[:, None]
    corners = integral(x, y)
    areas = np.diff(np.diff(corners, axis=0), axis=1)
    # Roundoff near tangencies can produce tiny negative intersections.
    areas = np.maximum(areas, 0.0)
    areas /= areas.sum()
    return areas


def _window(frame, window):
    if frame.ndim != 3 or frame.shape[0] != 3 or not np.all(np.isfinite(frame)):
        raise ValueError("expected finite planar CMY density")
    if window is None:
        return (0, 0, frame.shape[1], frame.shape[2])
    y, x, h, w = window
    if min(y, x) < 0 or min(h, w) <= 0 or y + h > frame.shape[1] or x + w > frame.shape[2]:
        raise ValueError("window outside frame")
    return y, x, h, w


def aperture_density(frame, pitch_um, window, diameter_um=48.0):
    """Measure density through an aperture with full support around the window."""
    frame = np.asarray(frame)
    y, x, h, w = _window(frame, window)
    kernel = aperture_kernel(pitch_um, diameter_um)
    r = kernel.shape[0] // 2
    if y < r or x < r or y + h + r > frame.shape[1] or x + w + r > frame.shape[2]:
        raise ValueError("aperture support outside supplied frame")
    patch = frame[:, y - r : y + h + r, x - r : x + w + r].astype(np.float64)
    transmission = np.power(10.0, -patch)
    filtered = np.stack([signal.fftconvolve(channel, kernel, mode="valid") for channel in transmission])
    if not np.all(np.isfinite(filtered)) or np.any(filtered <= 0):
        raise ValueError("invalid aperture transmission")
    return -np.log10(filtered)


def descriptors(field, baseline=None):
    """Per-record density and transmission moments; baseline is ungrained CMY."""
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 3 or field.shape[0] != 3 or not np.all(np.isfinite(field)):
        raise ValueError("expected finite planar CMY density")
    axes = (1, 2)
    mean = field.mean(axis=axes)
    delta = field - mean[:, None, None]
    rms = np.sqrt(np.mean(delta * delta, axis=axes))
    skew = np.divide(np.mean(delta**3, axis=axes), rms**3, out=np.zeros(3), where=rms > 0)
    light = np.power(10.0, -field).mean(axis=axes)
    result = {
        "mean_density": mean.tolist(),
        "rms": rms.tolist(),
        "skew": skew.tolist(),
        "quantiles": np.quantile(field, [0.001, 0.01, 0.5, 0.99, 0.999], axis=axes).T.tolist(),
        "mean_transmission": light.tolist(),
        "effective_density": (-np.log10(light)).tolist(),
    }
    if baseline is not None:
        baseline = np.asarray(baseline, dtype=np.float64)
        if baseline.shape != (3,) or not np.all(np.isfinite(baseline)):
            raise ValueError("baseline must be three finite densities")
        result["density_shift"] = (mean - baseline).tolist()
        result["transmission_shift"] = (light - 10.0 ** (-baseline)).tolist()
    return result


def autocorrelation(field, max_lag=128):
    """Pair-count normalized directional autocorrelation, shape (3, 2, lag)."""
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 3 or field.shape[0] != 3 or not np.all(np.isfinite(field)):
        raise ValueError("expected finite planar CMY density")
    if max_lag < 0 or max_lag >= min(field.shape[1:]):
        raise ValueError("lag exceeds field dimensions")
    centered = field - field.mean(axis=(1, 2), keepdims=True)
    variance = np.mean(centered**2, axis=(1, 2))
    if np.any(variance <= 0) or not np.all(np.isfinite(variance)):
        raise ValueError("zero or nonfinite density variance")
    curves = []
    for axis in (2, 1):
        length = field.shape[axis]
        # Padding by the largest requested lag prevents circular pairs from
        # contributing. Sum across lines before the linear inverse transform.
        lines = np.ascontiguousarray(np.moveaxis(centered, axis, -1))
        padded = fft.next_fast_len(length + max_lag, real=True)
        transform = fft.rfft(lines, n=padded, axis=-1)
        power = (transform.real**2 + transform.imag**2).sum(axis=1)
        cov = fft.irfft(power, n=padded, axis=-1)[:, : max_lag + 1]
        counts = (length - np.arange(max_lag + 1)) * field.shape[1 if axis == 2 else 2]
        curves.append(cov / counts / variance[:, None])
    return np.stack(curves, axis=1)


def crossing(curve, pitch_um):
    curve = np.asarray(curve, dtype=np.float64)
    if curve.ndim != 1 or not np.all(np.isfinite(curve)) or not math.isfinite(pitch_um) or pitch_um <= 0:
        raise ValueError("invalid correlation or pitch")
    target = math.exp(-1)
    hits = np.flatnonzero(curve[1:] <= target)
    if not len(hits):
        raise ValueError("correlation has no exp(-1) crossing")
    i = int(hits[0])
    return float((i + (curve[i] - target) / (curve[i] - curve[i + 1])) * pitch_um)


def analyze_frame(density, pitch_um, window, diameter_um=48.0, max_lag=128, baseline=None):
    """Analyze one frame; callers retain only the compact returned metrics."""
    density = np.asarray(density)
    y, x, h, w = _window(density, window)
    raw = density[:, y : y + h, x : x + w].astype(np.float64)
    aperture = aperture_density(density, pitch_um, window, diameter_um)
    centered = raw - raw.mean(axis=(1, 2), keepdims=True)
    covariance = centered.reshape(3, -1) @ centered.reshape(3, -1).T / (h * w)
    return {
        "raw": descriptors(raw, baseline),
        "aperture": descriptors(aperture, baseline),
        "correlation": autocorrelation(raw, max_lag).tolist(),
        "cross_covariance": covariance.tolist(),
    }


def summarize_frames(frames, pitch_um):
    """Pool equally over frames; retain per-frame strengths and directional curves."""
    if not frames:
        raise ValueError("no frames")
    rms = np.asarray([frame["aperture"]["rms"] for frame in frames], dtype=np.float64)
    curves = np.asarray([frame["correlation"] for frame in frames], dtype=np.float64)
    strength = np.sqrt(np.mean(rms**2, axis=0))
    if not np.all(np.isfinite(strength)) or np.any(strength <= 0):
        raise ValueError("zero or nonfinite strength")
    direction = curves.mean(axis=0)
    pooled = direction.mean(axis=(0, 1))
    return {
        "strength": strength.tolist(),
        "length_um": crossing(pooled, pitch_um),
        "per_seed_strength": rms.tolist(),
        "directional_correlation": direction.tolist(),
        "pooled_correlation": pooled.tolist(),
        "frames": frames,
    }


def match_residuals(candidate, current):
    """Return signed fractional residuals and the fixed G2 match disposition."""
    strength = np.asarray(candidate["strength"], dtype=np.float64)
    reference = np.asarray(current["strength"], dtype=np.float64)
    length = float(candidate["length_um"])
    reference_length = float(current["length_um"])
    if (
        strength.shape != (3,)
        or reference.shape != (3,)
        or not np.all(np.isfinite(strength))
        or not np.all(np.isfinite(reference))
        or np.any(strength <= 0)
        or np.any(reference <= 0)
        or not math.isfinite(length)
        or not math.isfinite(reference_length)
        or length <= 0
        or reference_length <= 0
    ):
        raise ValueError("invalid match statistics")
    residual = strength / reference - 1
    scale = length / reference_length - 1
    return {
        "strength": residual.tolist(),
        "length": scale,
        "matched": bool(np.all(np.abs(residual) <= 0.02) and abs(scale) <= 0.05),
    }


def spatial_spectrum(field, pitch_um):
    """Two-dimensional per-record power density and frequency axes in cycles/µm."""
    field = np.asarray(field, dtype=np.float64)
    if (
        field.ndim != 3
        or field.shape[0] != 3
        or not np.all(np.isfinite(field))
        or not math.isfinite(pitch_um)
        or pitch_um <= 0
    ):
        raise ValueError("invalid spectrum input")
    centered = field - field.mean(axis=(1, 2), keepdims=True)
    power = np.abs(np.fft.rfft2(centered, axes=(1, 2))) ** 2 / (field.shape[1] * field.shape[2])
    return {
        "power": power,
        "fy": np.fft.fftfreq(field.shape[1], pitch_um),
        "fx": np.fft.rfftfreq(field.shape[2], pitch_um),
    }


def temporal_correlation(values, max_lag):
    """Pair-normalized temporal correlation of a sequence of scalar observations."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or max_lag < 0 or max_lag >= len(values):
        raise ValueError("invalid temporal sequence or lag")
    centered = values - values.mean()
    variance = np.mean(centered**2)
    if variance <= 0:
        raise ValueError("zero temporal variance")
    return [float(np.mean(centered[: len(values) - lag] * centered[lag:]) / variance) for lag in range(max_lag + 1)]


def downsample_transmission(density, factor):
    """Average aligned integer blocks in transmission and return density."""
    density = np.asarray(density, dtype=np.float64)
    if (
        density.ndim != 3
        or density.shape[0] != 3
        or not np.all(np.isfinite(density))
        or not isinstance(factor, int)
        or factor <= 0
    ):
        raise ValueError("invalid density or factor")
    _, h, w = density.shape
    if h % factor or w % factor:
        raise ValueError("dimensions must be divisible by factor")
    light = (10.0**-density).reshape(3, h // factor, factor, w // factor, factor).mean(axis=(2, 4))
    if np.any(light <= 0) or not np.all(np.isfinite(light)):
        raise ValueError("invalid transmission")
    return -np.log10(light)


def analyze_match_frame(density, pitch_um, window, max_lag=128):
    """Calculate only the quantities needed for the G2 match."""
    density = np.asarray(density)
    y, x, h, w = _window(density, window)
    raw = density[:, y : y + h, x : x + w]
    aperture = aperture_density(density, pitch_um, window)
    rms = aperture.std(axis=(1, 2))
    return {"aperture": {"rms": rms.tolist()}, "correlation": autocorrelation(raw, max_lag).tolist()}


def strength_for_amounts(raw, input_density, pitch_um, window, multipliers):
    """RMS of affine record perturbations, reusing a unit-amount density frame.

    Multipliers are ratios to the amounts used to capture raw, including any
    size compensation already applied in that capture.
    """
    raw = np.asarray(raw)
    baseline = np.asarray(input_density, dtype=np.float64)
    amounts = np.asarray(multipliers, dtype=np.float64)
    if (
        baseline.shape != (3,)
        or amounts.shape != (3,)
        or not np.all(np.isfinite(baseline))
        or not np.all(np.isfinite(amounts))
        or np.any(amounts <= 0)
    ):
        raise ValueError("expected finite baseline and positive record multipliers")
    density = baseline[:, None, None] + (raw - baseline[:, None, None]) * amounts[:, None, None]
    return aperture_density(density, pitch_um, window).std(axis=(1, 2)).tolist()
