//! Immediate, borrowed projection of a completed core state into the CUDA ABI.
//!
//! The spans in this value refer to `state` or `noise`; callers must consume the
//! record before either borrow ends. This module performs field transport only.
#![forbid(unsafe_code)]

use film_juicer_core::{color, recipe, route::ScanRoute};

use super::sys;

fn float_span(values: &[f32]) -> sys::FjFloatSpan {
    if values.is_empty() {
        sys::FjFloatSpan {
            data: std::ptr::null(),
            count: 0,
        }
    } else {
        sys::FjFloatSpan {
            data: values.as_ptr(),
            count: values.len(),
        }
    }
}

fn byte_span(values: &[u8]) -> sys::FjByteSpan {
    if values.is_empty() {
        sys::FjByteSpan {
            data: std::ptr::null(),
            count: 0,
        }
    } else {
        sys::FjByteSpan {
            data: values.as_ptr(),
            count: values.len(),
        }
    }
}

fn string_view(value: &str) -> sys::FjStringView {
    if value.is_empty() {
        sys::FjStringView {
            data: std::ptr::null(),
            count: 0,
        }
    } else {
        sys::FjStringView {
            data: value.as_ptr().cast(),
            count: value.len(),
        }
    }
}

fn route_tag(route: ScanRoute) -> u32 {
    match route {
        ScanRoute::NegativeDirect => sys::FJ_ROUTE_NEGATIVE_DIRECT,
        ScanRoute::NegativePrint => sys::FJ_ROUTE_NEGATIVE_PRINT,
        ScanRoute::PositiveDirect => sys::FJ_ROUTE_POSITIVE_DIRECT,
        ScanRoute::PositivePrint => sys::FJ_ROUTE_POSITIVE_PRINT,
    }
}

fn capture_polarity(route: ScanRoute) -> u32 {
    match route {
        ScanRoute::NegativeDirect | ScanRoute::NegativePrint => sys::FJ_POLARITY_NEGATIVE,
        ScanRoute::PositiveDirect | ScanRoute::PositivePrint => sys::FJ_POLARITY_POSITIVE,
    }
}

fn input_space(space: color::InputSpace) -> u32 {
    match space {
        color::InputSpace::DavinciWideGamut => sys::FJ_INPUT_DWG,
        color::InputSpace::Bt2020 => sys::FJ_INPUT_BT2020,
        color::InputSpace::Aces2065_1 => sys::FJ_INPUT_ACES2065_1,
        color::InputSpace::SrgbRec709 => sys::FJ_INPUT_SRGB_REC709,
    }
}

fn method_tag(method: film_juicer_core::film::input::Method) -> u32 {
    use film_juicer_core::film::input::Method;
    match method {
        Method::Hanatos2025 => sys::FJ_RAW_HANATOS_2025,
        Method::Mallett2019 => sys::FJ_RAW_MALLETT_2019,
        Method::Arctic2026Beta04 => sys::FJ_RAW_ARCTIC_2026_BETA04,
    }
}

fn empty_film_exposure() -> sys::FjFilmExposure {
    sys::FjFilmExposure {
        input_color_space: 0,
        method: 0,
        flags: 0,
        manual_exposure_ev: 0.0,
        route_correction_scale: 0.0,
        mallett_green_midgray_scale: 0.0,
        input_rgb_to_xyz: [0.0; 9],
        input_xyz_adapt: [0.0; 9],
        xyz_to_linear_srgb: [0.0; 9],
        sensitivity_rgb: float_span(&[]),
        tc_lut_rgba: float_span(&[]),
        mallett_illuminant: float_span(&[]),
        mallett_basis: float_span(&[]),
        sensitivity_hash: 0,
        tc_lut_hash: 0,
        hash: 0,
    }
}

fn empty_development() -> sys::FjFilmDevelopment {
    sys::FjFilmDevelopment {
        log_exposure: float_span(&[]),
        density_rgb: float_span(&[]),
        density_layers: [[float_span(&[]); 3]; 3],
        gamma_rgb: [0.0; 3],
        density_curves_hash: 0,
        density_layers_hash: 0,
        hash: 0,
    }
}

fn empty_dir() -> sys::FjDirCouplers {
    sys::FjDirCouplers {
        mode: sys::FJ_DIR_INACTIVE,
        matrix_rgb: [0.0; 9],
        density_max_rgb: [0.0; 3],
        density_ref_rgb: [0.0; 3],
        donor_k_rgb: [0.0; 3],
        receiver_c_ref_rgb: [0.0; 3],
        receiver_kr_rgb: [0.0; 3],
        compensated_axes_rgb: [float_span(&[]); 3],
        compensated_axes_hash: 0,
        hash: 0,
    }
}

fn empty_bounds() -> sys::FjDensityBounds {
    sys::FjDensityBounds {
        min_cmy: [0.0; 3],
        inv_span_cmy: [0.0; 3],
        hash: 0,
    }
}

fn empty_print() -> sys::FjPrint {
    sys::FjPrint {
        film_density_cmy: float_span(&[]),
        film_base_density: float_span(&[]),
        sensitivity_cmy: float_span(&[]),
        log_exposure: float_span(&[]),
        density_cmy: float_span(&[]),
        main_illuminant: float_span(&[]),
        preflash_illuminant: float_span(&[]),
        flags: 0,
        main_cc_cmy: [0.0; 3],
        preflash_cc_cmy: [0.0; 3],
        exposure: 0.0,
        preflash_exposure: 0.0,
        preflash_raw_cmy: [0.0; 3],
        normalizer: 0.0,
        film_density_hash: 0,
        profile_tables_hash: 0,
        main_illuminant_hash: 0,
        preflash_illuminant_hash: 0,
        preflash_raw_hash: 0,
        balance_hash: 0,
        hash: 0,
    }
}

fn empty_noise() -> sys::FjStaticNoise {
    sys::FjStaticNoise {
        stbn: byte_span(&[]),
        stbn_width: 0,
        stbn_height: 0,
        stbn_frames: 0,
        wang_tiles: byte_span(&[]),
        wang_lut: byte_span(&[]),
        wang_width: 0,
        wang_height: 0,
        wang_tile_count: 0,
        wang_colors: 0,
    }
}

fn density_bounds(bounds: &film_juicer_core::film::density::Bounds) -> sys::FjDensityBounds {
    sys::FjDensityBounds {
        min_cmy: *bounds.minimum_cmy(),
        inv_span_cmy: *bounds.inv_span_cmy(),
        hash: bounds.hash(),
    }
}

fn cctf_kind(kind: color::CctfKind) -> u32 {
    match kind {
        color::CctfKind::Linear => sys::FJ_CCTF_LINEAR,
        color::CctfKind::Gamma => sys::FJ_CCTF_GAMMA,
        color::CctfKind::Srgb => sys::FJ_CCTF_SRGB,
        color::CctfKind::Bt2020 => sys::FJ_CCTF_BT2020,
        color::CctfKind::ProPhoto => sys::FJ_CCTF_PROPHOTO,
        color::CctfKind::DavinciIntermediate => sys::FJ_CCTF_DWG_INTERMEDIATE,
    }
}

fn scanner_correction(value: &film_juicer_core::scanner::Correction) -> sys::FjScannerCorrection {
    sys::FjScannerCorrection {
        flags: if value.active() {
            sys::FJ_CORRECTION_ACTIVE
        } else {
            0
        },
        xyz_slope: value.xyz_slope(),
        xyz_offset: value.xyz_offset(),
        exposure_scale: value.exposure_scale(),
    }
}

pub(super) fn host_data(
    state: &recipe::RenderState,
    optics: &sys::FjOptics,
    descriptors: &super::edge::FjNativeFrameDescriptors,
    preview: &sys::FjAutoExposure,
    noise: Option<&crate::asset_noise::NoiseView<'_>>,
) -> Result<sys::FjPreparedHostData, &'static str> {
    let route = state.development().route();
    let print_state = match state.route() {
        recipe::PreparedRoute::Direct(_) => None,
        recipe::PreparedRoute::Print(print) => Some(print),
    };
    let input = state.input();
    let color = input.color();
    let development = state.development();
    let film_tables = state.film_profile().tables();
    let scanner = state.scanner();
    let scanner_tables = scanner.tables();

    let mut film_exposure = empty_film_exposure();
    film_exposure.input_color_space = input_space(color.space());
    film_exposure.method = method_tag(input.method());
    film_exposure.flags = (if color.decode_cctf() {
        sys::FJ_FILM_DECODE_CCTF
    } else {
        0
    }) | sys::FJ_FILM_ADAPT_INPUT
        | (if input.auto_exposure_enabled() {
            sys::FJ_FILM_AUTO_EXPOSURE
        } else {
            0
        });
    film_exposure.manual_exposure_ev = input.manual_exposure_ev();
    film_exposure.route_correction_scale = match state.route() {
        recipe::PreparedRoute::Direct(direct) => direct.correction().exposure_scale(),
        recipe::PreparedRoute::Print(_) => 1.0,
    };
    film_exposure.mallett_green_midgray_scale = input.sensitivity().mallett_green_scale();
    film_exposure.input_rgb_to_xyz = *color.rgb_to_xyz();
    film_exposure.input_xyz_adapt = *color.xyz_adapt();
    film_exposure.xyz_to_linear_srgb = *color.xyz_to_linear_srgb();
    film_exposure.sensitivity_rgb = float_span(input.sensitivity().values_rgb().as_flattened());
    film_exposure.sensitivity_hash = input.sensitivity().hash();
    film_exposure.tc_lut_hash = input.tc_hash();
    film_exposure.hash = input.hash();
    match input.method() {
        film_juicer_core::film::input::Method::Mallett2019 => {
            film_exposure.mallett_illuminant = float_span(state.exposure_tables().illuminant());
            let basis = state
                .reconstruction()
                .mallett_basis()
                .ok_or("missing Mallett basis")?;
            film_exposure.mallett_basis = float_span(basis.as_flattened());
        }
        film_juicer_core::film::input::Method::Hanatos2025
        | film_juicer_core::film::input::Method::Arctic2026Beta04 => {
            let lut = state.reconstruction().tc_lut().ok_or("missing TC LUT")?;
            film_exposure.tc_lut_rgba = float_span(lut.samples());
        }
    }

    let mut film_development = empty_development();
    film_development.log_exposure = float_span(development.log_exposure());
    film_development.density_rgb = float_span(development.normalized_curves_cmy().as_flattened());
    film_development.gamma_rgb = *development.gamma_rgb();
    film_development.density_curves_hash = development.normalized_curves_hash();
    film_development.hash = development.hash();
    if let Some(layers) = development.density_curves_layers() {
        for (layer_index, layer_values) in layers.iter().enumerate() {
            for (channel_index, channel_values) in layer_values.iter().enumerate() {
                film_development.density_layers[layer_index][channel_index] =
                    float_span(channel_values);
            }
        }
        film_development.density_layers_hash = development.layers_hash();
    }

    let mut dir = empty_dir();
    let dir_recipe = state.dir();
    if dir_recipe.active() {
        dir.mode = match dir_recipe.mode() {
            film_juicer_core::film::dir::Mode::Inactive => sys::FJ_DIR_INACTIVE,
            film_juicer_core::film::dir::Mode::NegativeDonorLangmuir => {
                sys::FJ_DIR_NEGATIVE_DONOR_LANGMUIR
            }
            film_juicer_core::film::dir::Mode::PositiveReceiverLangmuir => {
                sys::FJ_DIR_POSITIVE_RECEIVER_LANGMUIR
            }
        };
        let matrix = dir_recipe.matrix_rgb();
        dir.matrix_rgb = [
            matrix[0][0],
            matrix[0][1],
            matrix[0][2],
            matrix[1][0],
            matrix[1][1],
            matrix[1][2],
            matrix[2][0],
            matrix[2][1],
            matrix[2][2],
        ];
        dir.density_max_rgb = *dir_recipe.density_max_rgb();
        dir.density_ref_rgb = *dir_recipe.density_ref_rgb();
        dir.donor_k_rgb = *dir_recipe.donor_k_rgb();
        dir.receiver_c_ref_rgb = *dir_recipe.receiver_c_ref_rgb();
        dir.receiver_kr_rgb = *dir_recipe.receiver_kr_rgb();
        for channel in 0..3 {
            dir.compensated_axes_rgb[channel] =
                float_span(&dir_recipe.compensated_axes_rgb()[channel]);
        }
        dir.compensated_axes_hash = dir_recipe.compensated_axes_hash();
        dir.hash = dir_recipe.hash();
    }

    let active_correction = match state.route() {
        recipe::PreparedRoute::Direct(direct) => direct.correction(),
        recipe::PreparedRoute::Print(print) => print.correction(),
    };
    let correction = scanner_correction(active_correction);
    let post_fields = scanner.post_effects().fields();
    let post_values = post_fields.values();
    let post_radii = post_fields.radii();
    let scanner_post = sys::FjScannerPostEffects {
        flags: if post_fields.glare_active() {
            sys::FJ_POST_GLARE
        } else {
            0
        },
        glare_percent: post_values[0],
        glare_roughness: post_values[1],
        glare_blur_sigma_px: post_values[2],
        glare_blur_radius: post_radii[0],
        lens_blur_sigma_px: post_values[3],
        lens_blur_radius: post_radii[1],
        unsharp_sigma_px: post_values[4],
        unsharp_radius: post_radii[2],
        unsharp_amount: post_values[5],
        hash: scanner.post_effects().hash(),
    };

    let color_recipe = scanner.color();
    let cctf = color_recipe.cctf();
    let transform = scanner.gamut().transform();
    let mut output_flags = sys::FJ_COLOR_INPUT_IS_OUTPUT_SPACE;
    if color_recipe.apply_cctf() {
        output_flags |= sys::FJ_COLOR_ENCODE_CCTF;
    }
    if scanner.gamut().enabled() {
        output_flags |= sys::FJ_COLOR_GAMUT_COMPRESSION;
    }
    let (gamut_cmax, gamut_table_hash) = scanner
        .gamut()
        .boundary()
        .map_or((float_span(&[]), 0), |boundary| {
            (float_span(boundary.cmax()), boundary.hash())
        });
    let output_color = sys::FjOutputColor {
        cat02: *color_recipe.cat02(),
        xyz_to_rgb: *color_recipe.xyz_to_rgb(),
        illuminant_xyz: *color_recipe.illuminant_xyz(),
        color_space: u32::from(color_recipe.output_space()),
        flags: output_flags,
        native_rgb_to_d65_xyz: *transform.native_rgb_to_d65_xyz(),
        d65_xyz_to_native_rgb: *transform.d65_xyz_to_native_rgb(),
        lightness_knee: *scanner.gamut().lightness_knee(),
        chroma_knee: *scanner.gamut().chroma_knee(),
        gamut_cmax,
        gamut_table_hash,
        gamut_recipe_hash: scanner.gamut().hash(),
        encoding: sys::FjOutputEncoding {
            schema: sys::FJ_OUTPUT_ENCODING_SCHEMA,
            cctf_kind: cctf_kind(cctf.kind),
            gamma: cctf.gamma,
            a: cctf.a,
            b: cctf.b,
            c: cctf.c,
            d: cctf.d,
            linear_cutoff: cctf.linear_cutoff,
            dwg_to_output: *color_recipe.dwg_to_output(),
        },
    };

    let mut print = empty_print();
    let mut scanner_bounds = density_bounds(state.film_bounds());
    let mut enlarger_film_bounds = empty_bounds();
    let mut print_profile_key = sys::FjStringView {
        data: std::ptr::null(),
        count: 0,
    };
    let mut print_profile_asset_version = 0;
    if let Some(print_state) = print_state {
        let profile = print_state.profile();
        let profile_tables = profile.tables();
        let descriptors = print_state.resource_descriptors();
        let exposure_params = print_state.exposure().params();
        let balance = print_state.balance();
        print_profile_key = string_view(print_state.key());
        print_profile_asset_version = profile.asset_token();
        scanner_bounds = sys::FjDensityBounds {
            min_cmy: *print_state.bounds().minimum_cmy(),
            inv_span_cmy: *print_state.bounds().inv_span_cmy(),
            hash: print_state.bounds().hash(),
        };
        enlarger_film_bounds = density_bounds(state.film_bounds());
        print.film_density_cmy = float_span(film_tables.channel_density().as_flattened());
        print.film_base_density = float_span(film_tables.base_density());
        print.sensitivity_cmy = float_span(profile_tables.linear_sensitivity().as_flattened());
        print.log_exposure = float_span(profile_tables.interpolation_log_exposure());
        print.density_cmy = float_span(print_state.curves().totals().as_flattened());
        print.main_illuminant = float_span(print_state.main_illuminant().samples());
        print.flags = if print_state.preflash().is_some() {
            sys::FJ_PRINT_PREFLASH
        } else {
            0
        };
        print.main_cc_cmy = *print_state.filters().main_cmy_cc();
        print.preflash_cc_cmy = *print_state.filters().preflash_cmy_cc();
        print.exposure = exposure_params.print_exposure;
        print.preflash_exposure = exposure_params.preflash_exposure;
        print.normalizer = balance.normalizer();
        if let Some(preflash) = print_state.preflash() {
            print.preflash_illuminant = float_span(preflash.illuminant().samples());
            print.preflash_raw_cmy = *preflash.raw().cmy();
        }
        print.film_density_hash = descriptors.film_density_hash();
        print.profile_tables_hash = descriptors.profile_tables_hash();
        print.main_illuminant_hash = descriptors.main_illuminant_hash();
        print.preflash_illuminant_hash = descriptors.preflash_illuminant_hash();
        print.preflash_raw_hash = descriptors.preflash_raw_hash();
        print.balance_hash = descriptors.balance_hash();
        print.hash = descriptors.hash();
    }

    let dyes = scanner_tables.dyes_cmy();
    let weighted = scanner_tables.weighted_xyz();
    let scanner_spectra = sys::FjScannerSpectra {
        dye_y: float_span(&dyes[2]),
        dye_m: float_span(&dyes[1]),
        dye_c: float_span(&dyes[0]),
        weighted_x: float_span(&weighted[0]),
        weighted_y: float_span(&weighted[1]),
        weighted_z: float_span(&weighted[2]),
        base_density: if scanner_tables.has_baseline() {
            float_span(scanner_tables.baseline_min())
        } else {
            float_span(&[])
        },
        inverse_y_normalization: scanner_tables.inv_yn(),
    };
    let lut = scanner.lut();
    let scanner_lut = sys::FjScannerLut {
        density_bounds_hash: lut.density_bounds_hash(),
        resolution: lut.resolution(),
        hash: lut.hash(),
    };

    let static_noise = if let Some(noise) = noise {
        let [stbn_width, stbn_height, stbn_frames] = noise.stbn_dimensions;
        let [wang_width, wang_height, wang_tile_count] = noise.wang_dimensions;
        sys::FjStaticNoise {
            stbn: byte_span(noise.stbn),
            stbn_width: i32::try_from(stbn_width).map_err(|_| "STBN width exceeds native range")?,
            stbn_height: i32::try_from(stbn_height)
                .map_err(|_| "STBN height exceeds native range")?,
            stbn_frames: i32::try_from(stbn_frames)
                .map_err(|_| "STBN frame count exceeds native range")?,
            wang_tiles: byte_span(noise.wang_tiles),
            wang_lut: byte_span(noise.wang_lut),
            wang_width: i32::try_from(wang_width).map_err(|_| "Wang width exceeds native range")?,
            wang_height: i32::try_from(wang_height)
                .map_err(|_| "Wang height exceeds native range")?,
            wang_tile_count,
            wang_colors: i32::try_from(noise.wang_colors)
                .map_err(|_| "Wang color count exceeds native range")?,
        }
    } else {
        empty_noise()
    };

    Ok(sys::FjPreparedHostData {
        route: route_tag(route),
        capture_polarity: capture_polarity(route),
        film_profile_key: string_view(state.film_key()),
        print_profile_key,
        film_profile_asset_version: state.film_profile().asset_token(),
        print_profile_asset_version,
        recipe_hash: state.hash(),
        film_exposure,
        auto_exposure: *preview,
        film_development,
        dir_couplers: dir,
        scanner_bounds,
        enlarger_film_bounds,
        scanner_spectra,
        scanner_lut,
        scanner_correction: correction,
        scanner_post,
        output_color,
        print,
        optics: *optics,
        spatial_dir: descriptors.spatial_dir,
        grain: descriptors.grain,
        effects: descriptors.effects,
        noise: static_noise,
    })
}
