#!/usr/bin/env python3
"""Regression checks for the pinned G09 reference configuration."""

from __future__ import annotations

import unittest

import generate_reference as reference


class G09ReferenceConfigurationTests(unittest.TestCase):
    def assert_encoded_and_linear_settings(
        self,
        params,
        *,
        scan_film: bool,
        enlarger_diffusion: bool,
        dir_diffusion_size_um: float,
    ) -> None:
        reference.assert_g09_effective_settings(
            params,
            scan_film=scan_film,
            enlarger_diffusion=enlarger_diffusion,
            dir_active=True,
            dir_diffusion_size_um=dir_diffusion_size_um,
            output_cctf_encoding=True,
        )
        params.io.output_cctf_encoding = False
        reference.assert_g09_effective_settings(
            params,
            scan_film=scan_film,
            enlarger_diffusion=enlarger_diffusion,
            dir_active=True,
            dir_diffusion_size_um=dir_diffusion_size_um,
            output_cctf_encoding=False,
        )

    def test_ordinary_case_disables_every_isolated_spatial_component(self) -> None:
        params = reference.configured_params()
        self.assert_encoded_and_linear_settings(
            params,
            scan_film=False,
            enlarger_diffusion=False,
            dir_diffusion_size_um=0.0,
        )

    def test_enlarger_diffusion_survives_digest_with_exact_controls(self) -> None:
        params = reference.configured_params(enlarger_diffusion=True)
        self.assert_encoded_and_linear_settings(
            params,
            scan_film=False,
            enlarger_diffusion=True,
            dir_diffusion_size_um=0.0,
        )
        settings = reference.g09_effective_settings(params)
        self.assertEqual("black_pro_mist", settings["enlarger_diffusion_family"])
        self.assertEqual(0.5, settings["enlarger_diffusion_strength"])
        self.assertEqual(1.0, settings["enlarger_diffusion_spatial_scale"])

    def test_spatial_dir_survives_digest_without_halation_or_unsharp(self) -> None:
        params = reference.configured_params(
            scan_film=True,
            dir_diffusion_size_um=2187.5,
        )
        self.assert_encoded_and_linear_settings(
            params,
            scan_film=True,
            enlarger_diffusion=False,
            dir_diffusion_size_um=2187.5,
        )
        settings = reference.g09_effective_settings(params)
        self.assertEqual(2187.5, settings["pixel_size_um"])
        self.assertFalse(settings["halation_active"])
        self.assertEqual([0.0, 0.0], settings["scanner_unsharp_mask"])


if __name__ == "__main__":
    unittest.main()
