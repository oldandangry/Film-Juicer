"""Independent known cases for the G2 offline grain measurements."""

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import grain_analysis as analysis
import grain_campaign as campaign
import numpy as np
from grain_chroma import coupled_strength, mix_density
from scipy.integrate import quad


class GrainAnalysisTests(unittest.TestCase):
    def test_chroma_projection_preserves_luminance(self):
        density = np.array([0.2, 0.5, 1.0])
        amounts = np.array([1.0, 0.75, 1.5])
        frame = density[:, None, None] + np.array([0.04, -0.02, 0.1])[:, None, None]
        weights = np.array([0.2126, 0.7152, 0.0722])
        base = 10.0**-density
        full = 10.0 ** -(density + amounts * (frame[:, 0, 0] - density))
        neutral = base * ((weights @ full) / (weights @ base))
        for chroma in (0.0, 0.240635, 1.0):
            output = 10.0 ** -mix_density(density, amounts, frame, chroma)[:, 0, 0]
            np.testing.assert_allclose(weights @ output, weights @ full, atol=2e-15, rtol=0)
            np.testing.assert_allclose(output - neutral, chroma * (full - neutral), atol=2e-15, rtol=0)

    def test_coupled_strength_boundaries(self):
        density = np.array([0.2, 0.5, 1.0])
        sigma = np.array([0.02, 0.03, 0.05])
        amounts = np.array([0.5, 1.0, 2.0])
        covariance = np.diag(sigma**2)
        np.testing.assert_allclose(coupled_strength(covariance, density, 1.0, amounts), sigma * amounts)
        weights = np.array([0.2126, 0.7152, 0.0722]) * 10.0**-density
        expected = np.linalg.norm(weights * sigma * amounts) / weights.sum()
        np.testing.assert_allclose(coupled_strength(covariance, density, 0.0, amounts), expected)

    def test_circle_area_and_pixel_intersections(self):
        radius = 2.4
        kernel = analysis.aperture_kernel(1, 2 * radius)
        self.assertAlmostEqual(kernel.sum(), 1.0, places=14)
        # Independent one-dimensional quadrature of the circle/square overlap.
        for iy, ix in [(0, 1), (1, 1), (2, 2), (2, 4)]:
            y, x = iy - 2, ix - 2

            def height(t, *, y=y):
                half = math.sqrt(max(0.0, radius**2 - t**2))
                return max(0.0, min(y + 0.5, half) - max(y - 0.5, -half))

            area = quad(
                height, max(-radius, x - 0.5), min(radius, x + 0.5), epsabs=1e-11, points=[0] if x == 0 else None
            )[0]
            self.assertAlmostEqual(kernel[iy, ix], area / (math.pi * radius**2), places=8)

    def test_constant_and_impulse_transmission(self):
        field = np.ones((3, 11, 11))
        result = analysis.aperture_density(field, 1, (4, 4, 3, 3), 4)
        np.testing.assert_allclose(result, 1, atol=1e-14)
        field[:, 5, 5] = 0
        result = analysis.aperture_density(field, 1, (5, 5, 1, 1), 4)
        center_weight = analysis.aperture_kernel(1, 4)[2, 2]
        np.testing.assert_allclose(result[:, 0, 0], -math.log10(0.1 + 0.9 * center_weight), atol=1e-14)

    def test_pair_normalized_correlation(self):
        field = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5) ** 2
        actual = analysis.autocorrelation(field, 3)
        for c in range(3):
            centered = field[c] - field[c].mean()
            for axis in range(2):
                for lag in range(4):
                    if axis == 0:
                        a, b = centered[:, : 5 - lag], centered[:, lag:]
                    else:
                        a, b = centered[: 4 - lag, :], centered[lag:, :]
                    self.assertAlmostEqual(actual[c, axis, lag], (a * b).mean() / (centered**2).mean(), places=12)

    def test_crossing_and_match(self):
        self.assertAlmostEqual(analysis.crossing([1, 0.5, 0.25], 2), 2 * (1 + (0.5 - math.exp(-1)) / 0.25))
        current = {"strength": [1, 2, 3], "length_um": 5}
        self.assertTrue(analysis.match_residuals(current, current)["matched"])
        self.assertFalse(analysis.match_residuals({"strength": [1.03, 2, 3], "length_um": 5}, current)["matched"])
        with self.assertRaises(ValueError):
            analysis.crossing([1, 0.9], 1)

    def test_cached_amounts_match_direct_density(self):
        rng = np.random.default_rng(7)
        baseline = np.array([0.5, 1.0, 1.5])
        raw = baseline[:, None, None] + rng.normal(0, 0.1, (3, 30, 30))
        amounts = np.array([0.5, 1.25, 2.0])
        adjusted = baseline[:, None, None] + (raw - baseline[:, None, None]) * amounts[:, None, None]
        cached = analysis.strength_for_amounts(raw, baseline, 4, (7, 7, 16, 16), amounts)
        direct = analysis.analyze_match_frame(adjusted, 4, (7, 7, 16, 16), 4)
        np.testing.assert_allclose(cached, direct["aperture"]["rms"])

    def test_halo_and_zero_variance_rejected(self):
        field = np.ones((3, 10, 10))
        with self.assertRaises(ValueError):
            analysis.aperture_density(field, 1, (0, 0, 5, 5), 4)
        with self.assertRaises(ValueError):
            analysis.autocorrelation(field, 3)

    def test_calibration_overlap_rejected_before_any_render(self):
        experiment = {"rows": [{"id": "known"}], "calibration": [0, 1], "evaluation": [1, 2]}
        with self.assertRaisesRegex(ValueError, "overlapping"):
            campaign.calibrate(experiment, None, None, None)

    def test_held_out_failure_preserves_frozen_controls(self):
        experiment = {
            "rows": [{"id": "known", "stock": campaign.GOLD, "geometry": "uhd-36", "look": "default"}],
            "stocks": {campaign.GOLD: {"bank": "bank", "tables": "tables", "density": [1, 1, 1], "route": 0}},
            "calibration": list(range(16)),
            "evaluation": list(range(16, 32)),
            "fractional": [i + 0.5 for i in range(16, 32)],
            "sizes": [1.0],
            "session_seed": 17,
            "clip_token": 23,
        }
        amounts = [0.5, 1.0, 1.5]
        current = {"strength": [1, 1, 1], "length_um": 5}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            def render(name, request):
                times = [sample["time"] for sample in request["requests"]]
                if min(times) >= 16:
                    frozen = campaign.read_json(output / "known-frozen.json")
                    self.assertEqual(frozen["amounts"], amounts)
                    if request["engine"] == "film" and "reference" not in name:
                        self.assertEqual(request["amounts"], amounts)
                path = output / name
                path.mkdir()
                return path

            def statistics(path):
                if path.name == "known-evaluation-film":
                    return {"strength": [1, 1, 1.03], "length_um": 5}
                return current

            with (
                patch.object(campaign, "statistics", side_effect=statistics),
                patch.object(campaign, "tune_amounts", return_value=(amounts, [])) as tune,
            ):
                campaign.calibrate(experiment, render, output, None)
            tune.assert_called_once()
            report = campaign.read_json(output / "known.json")
            self.assertEqual(report["disposition"], "unmatched")
            self.assertFalse(report["evaluation"]["matched"])
            self.assertTrue(report["fractional"]["residuals"]["matched"])


if __name__ == "__main__":
    unittest.main()
