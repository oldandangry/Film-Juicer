//! Completed film-input policy; resources are borrowed only during construction.

use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::{color, exposure, hash, profile};

#[derive(Clone, Copy, Debug)]
pub enum Method {
    Hanatos2025,
    Mallett2019,
    Arctic2026Beta04,
}
#[derive(Clone, Copy, Debug)]
pub enum Meter {
    CenterWeighted,
    Average,
    Median,
    Partial,
    Matrix,
    MultiZone,
    HighlightWeighted,
}
#[derive(Clone, Copy, Debug)]
pub enum ProjectionIlluminant {
    ProfileReference,
    D65,
}
pub struct Params {
    pub input_space: color::InputSpace,
    pub decode_cctf: bool,
    pub method: Method,
    pub auto_exposure_enabled: bool,
    pub meter: Meter,
    pub manual_exposure_ev: f32,
    pub film_format_long_edge_mm: f32,
    pub filter_override: bool,
    pub uv: [f64; 3],
    pub ir: [f64; 3],
    pub apply_hanatos_window: bool,
    pub apply_hanatos_surface: bool,
    pub input_compression: bool,
}
pub struct Resources<'a> {
    pub reference_illuminant: &'a [f32; 81],
    pub reference_white_xyz: Option<&'a [f32; 3]>,
    pub integrated_d65_xyz: Option<&'a [f32; 3]>,
    pub reconstructed_reference_white: Option<&'a [f32; 81]>,
    pub tc_source_hash: u64,
    pub input_hull_hash: u64,
}
#[derive(Clone, Copy, Debug)]
pub enum Operation {
    InputMatrices,
    Cat16,
    Cat02,
    Sensitivity,
}
#[derive(Debug)]
pub enum Error {
    MissingHanatosWindow,
    MissingHanatosSurface,
    InvalidHanatosBlur,
    MissingProjectionWhite,
    MissingTcSource(Method),
    MissingInputHull,
    Sensitivity(exposure::Error),
    ZeroIdentity,
    Panic(Operation),
    #[cfg(feature = "test-support")]
    Boundary(Operation, u32),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::MissingHanatosWindow => "MalformedRequiredProfileData phase=3D-2 field=data.hanatos2025_adaptation_window_params",
            Self::MissingHanatosSurface => "MalformedRequiredProfileData phase=3D-2 field=data.hanatos2025_adaptation_surface_params",
            Self::InvalidHanatosBlur => "MalformedRequiredProfileData phase=3D-2 field=settings.spectral_gaussian_blur",
            Self::MissingProjectionWhite => "MissingRequiredResource component=film_tc_lut requirement=integrated_projection_white",
            Self::MissingTcSource(Method::Hanatos2025) => "MissingRequiredResource resource=hanatos_2025_spectra_lut.npy requirement=finite_192x192x81",
            Self::MissingTcSource(Method::Arctic2026Beta04) => "MissingRequiredResource resource=arctic2026beta04_reflectance_xy_tc.npy requirement=finite_float16_c_order_192x192x81",
            Self::MissingTcSource(Method::Mallett2019) => "film input boundary integrity failure",
            Self::MissingInputHull => "MissingRequiredResource component=film_tc_lut requirement=input_compression_hull",
            Self::Sensitivity(error) => return error.fmt(f),
            Self::ZeroIdentity => "MalformedRequiredProfileData phase=3B field=data.log_sensitivity",
            Self::Panic(Operation::Sensitivity) => "asset boundary panic",
            Self::Panic(_) => "film input boundary panic",
            #[cfg(feature = "test-support")]
            Self::Boundary(_, _) => "exposure production boundary fault",
        })
    }
}
impl std::error::Error for Error {}

pub struct Recipe {
    color: color::InputColor,
    method: Method,
    auto_exposure_enabled: bool,
    meter: Meter,
    manual_exposure_ev: f32,
    film_format_long_edge_mm: f32,
    band_pass_active: bool,
    uv: [f32; 3],
    ir: [f32; 3],
    apply_hanatos_window: bool,
    apply_hanatos_surface: bool,
    hanatos_spectral_blur: f32,
    hanatos_window: [f32; 4],
    hanatos_surface_rgb: [[f32; 15]; 3],
    projection_illuminant: ProjectionIlluminant,
    input_compression_active: bool,
    input_hull_hash: u64,
    tc_source_hash: u64,
    sensitivity: exposure::Sensitivity,
    tc_hash: u64,
    hash: u64,
}

fn leaf<T>(operation: Operation, call: impl FnOnce() -> T) -> Result<T, Error> {
    catch_unwind(AssertUnwindSafe(|| {
        #[cfg(feature = "test-support")]
        testing::check(operation)?;
        Ok(call())
    }))
    .map_err(|_| Error::Panic(operation))?
}

fn narrow_filter(values: [f64; 3], active: bool) -> [f32; 3] {
    [
        if active { values[0] as f32 } else { 0.0 },
        values[1] as f32,
        values[2] as f32,
    ]
}
fn method_byte(method: Method) -> u8 {
    match method {
        Method::Hanatos2025 => 0,
        Method::Mallett2019 => 1,
        Method::Arctic2026Beta04 => 2,
    }
}
fn meter_byte(meter: Meter) -> u8 {
    match meter {
        Meter::CenterWeighted => 0,
        Meter::Average => 1,
        Meter::Median => 2,
        Meter::Partial => 3,
        Meter::Matrix => 4,
        Meter::MultiZone => 5,
        Meter::HighlightWeighted => 6,
    }
}
fn space_tag(space: color::InputSpace) -> i32 {
    match space {
        color::InputSpace::DavinciWideGamut => 0,
        color::InputSpace::Bt2020 => 1,
        color::InputSpace::Aces2065_1 => 2,
        color::InputSpace::SrgbRec709 => 3,
    }
}
fn hash_float_bits(hash: &mut u64, values: &[f32]) {
    for value in values {
        hash::update_bytes(hash, &value.to_bits().to_le_bytes());
    }
}
impl Recipe {
    pub fn new(
        profile: &profile::FilmProfile,
        params: Params,
        resources: Resources<'_>,
    ) -> Result<Self, Error> {
        let uv = narrow_filter(params.uv, params.filter_override);
        let ir = narrow_filter(params.ir, params.filter_override);
        let matrices = leaf(Operation::InputMatrices, || {
            color::input_matrices(params.input_space)
        })?;
        let hanatos = matches!(params.method, Method::Hanatos2025);
        let tc = !matches!(params.method, Method::Mallett2019);
        let (mut apply_window, mut apply_surface, mut blur, mut window, mut surface) =
            (true, false, 0.0, [0.0; 4], [[0.0; 15]; 3]);
        if hanatos {
            apply_window = params.apply_hanatos_window;
            apply_surface = params.apply_hanatos_surface;
            let tables = profile.tables();
            if apply_window && tables.hanatos2025_adaptation_window_params().is_none() {
                return Err(Error::MissingHanatosWindow);
            }
            if apply_surface && tables.hanatos2025_adaptation_surface_params().is_none() {
                return Err(Error::MissingHanatosSurface);
            }
            blur = profile
                .processing_defaults()
                .hanatos_spectral_gaussian_blur_default;
            if !blur.is_finite() || blur < 0.0 {
                return Err(Error::InvalidHanatosBlur);
            }
            window = tables
                .hanatos2025_adaptation_window_params()
                .copied()
                .unwrap_or([0.0; 4]);
            surface = tables
                .hanatos2025_adaptation_surface_params()
                .copied()
                .unwrap_or([[0.0; 15]; 3]);
        }
        let projection_white = match params.method {
            Method::Hanatos2025 => *resources
                .reference_white_xyz
                .ok_or(Error::MissingProjectionWhite)?,
            Method::Arctic2026Beta04 => *resources
                .integrated_d65_xyz
                .ok_or(Error::MissingProjectionWhite)?,
            Method::Mallett2019 => matrices.d65_white_xyz,
        };
        if tc && resources.tc_source_hash == 0 {
            return Err(Error::MissingTcSource(params.method));
        }
        let whites = color::Whites {
            source_xyz: matrices.nominal_white_xyz,
            destination_xyz: projection_white,
        };
        let adaptation = if tc {
            leaf(Operation::Cat16, || color::cat16_matrix(whites))?
        } else {
            leaf(Operation::Cat02, || color::cat02_matrix(whites))?
        };
        let compression = tc && params.input_compression;
        if compression && resources.input_hull_hash == 0 {
            return Err(Error::MissingInputHull);
        }
        let color = color::InputColor::from_matrices(
            params.input_space,
            params.decode_cctf,
            matrices,
            adaptation,
            projection_white,
        );
        let sensitivity = leaf(Operation::Sensitivity, || {
            exposure::prepare_sensitivity(exposure::Input {
                linear_sensitivity_rgb: profile.tables().linear_sensitivity(),
                reference_illuminant: resources.reference_illuminant,
                band_pass: params
                    .filter_override
                    .then_some(exposure::BandPass { uv, ir }),
                method: match params.method {
                    Method::Hanatos2025 => exposure::Method::Hanatos {
                        window: apply_window.then_some(exposure::Window {
                            params: &window,
                            reference_white: resources.reconstructed_reference_white,
                        }),
                    },
                    Method::Mallett2019 => exposure::Method::Mallett,
                    Method::Arctic2026Beta04 => exposure::Method::Arctic,
                },
            })
        })?
        .map_err(Error::Sensitivity)?;
        let mut recipe = Self {
            color,
            method: params.method,
            auto_exposure_enabled: params.auto_exposure_enabled,
            meter: params.meter,
            manual_exposure_ev: params.manual_exposure_ev,
            film_format_long_edge_mm: params.film_format_long_edge_mm,
            band_pass_active: params.filter_override,
            uv,
            ir,
            apply_hanatos_window: apply_window,
            apply_hanatos_surface: apply_surface,
            hanatos_spectral_blur: blur,
            hanatos_window: window,
            hanatos_surface_rgb: surface,
            projection_illuminant: if hanatos {
                ProjectionIlluminant::ProfileReference
            } else {
                ProjectionIlluminant::D65
            },
            input_compression_active: compression,
            input_hull_hash: if tc { resources.input_hull_hash } else { 0 },
            tc_source_hash: if tc { resources.tc_source_hash } else { 0 },
            sensitivity,
            tc_hash: 0,
            hash: 0,
        };
        let reference_label = profile.info().reference_illuminant();
        let projection_label = if hanatos { reference_label } else { "D65" };
        if tc {
            recipe.tc_hash = recipe.hash_tc(
                resources.reference_illuminant,
                reference_label,
                projection_label,
            );
        }
        recipe.hash = recipe.hash_input(reference_label, projection_label);
        if recipe.sensitivity.hash() == 0 || (tc && recipe.tc_hash == 0) || recipe.hash == 0 {
            return Err(Error::ZeroIdentity);
        }
        Ok(recipe)
    }
    fn hash_tc(&self, reference: &[f32; 81], reference_label: &str, projection_label: &str) -> u64 {
        let mut h = hash::FNV_OFFSET;
        hash::update_bytes(&mut h, &2_u32.to_le_bytes());
        hash::update_bytes(&mut h, &[method_byte(self.method)]);
        hash::update_bytes(&mut h, &self.tc_source_hash.to_le_bytes());
        hash::update_bytes(&mut h, &self.sensitivity.hash().to_le_bytes());
        hash_float_bits(&mut h, reference);
        hash_float_bits(&mut h, self.color.projection_white_xyz());
        hash::update_bytes(&mut h, reference_label.as_bytes());
        hash::update_bytes(&mut h, projection_label.as_bytes());
        if matches!(self.method, Method::Hanatos2025) {
            hash_float_bits(&mut h, &[self.hanatos_spectral_blur]);
            hash::update_bytes(&mut h, &[u8::from(self.apply_hanatos_surface)]);
            if self.apply_hanatos_surface {
                hash_float_bits(&mut h, self.hanatos_surface_rgb.as_flattened());
            }
        }
        hash::update_bytes(&mut h, &[u8::from(self.input_compression_active)]);
        if self.input_compression_active {
            hash::update_bytes(&mut h, &self.input_hull_hash.to_le_bytes());
        }
        h
    }
    fn hash_input(&self, reference_label: &str, projection_label: &str) -> u64 {
        let mut h = hash::FNV_OFFSET;
        hash::update_bytes(&mut h, &space_tag(self.color.space()).to_le_bytes());
        hash::update_bytes(
            &mut h,
            &[
                u8::from(self.color.decode_cctf()),
                method_byte(self.method),
                u8::from(self.auto_exposure_enabled),
                meter_byte(self.meter),
            ],
        );
        hash_float_bits(
            &mut h,
            &[self.manual_exposure_ev, self.film_format_long_edge_mm],
        );
        hash::update_bytes(&mut h, &[u8::from(self.band_pass_active)]);
        hash_float_bits(&mut h, &self.uv);
        hash_float_bits(&mut h, &self.ir);
        hash_float_bits(&mut h, self.color.rgb_to_xyz());
        hash_float_bits(&mut h, self.color.xyz_adapt());
        hash_float_bits(&mut h, self.color.xyz_to_linear_srgb());
        hash_float_bits(&mut h, self.color.nominal_white_xyz());
        hash_float_bits(&mut h, self.color.projection_white_xyz());
        hash::update_bytes(&mut h, reference_label.as_bytes());
        hash::update_bytes(&mut h, projection_label.as_bytes());
        hash::update_bytes(&mut h, &self.sensitivity.hash().to_le_bytes());
        if matches!(self.method, Method::Mallett2019) {
            hash_float_bits(&mut h, &[self.sensitivity.mallett_green_scale()]);
        } else {
            hash::update_bytes(&mut h, &self.tc_hash.to_le_bytes());
            hash::update_bytes(&mut h, &self.tc_source_hash.to_le_bytes());
            hash::update_bytes(&mut h, &[u8::from(self.input_compression_active)]);
            if self.input_compression_active {
                hash::update_bytes(&mut h, &self.input_hull_hash.to_le_bytes());
            }
        }
        h
    }
    pub fn color(&self) -> &color::InputColor {
        &self.color
    }
    pub fn method(&self) -> Method {
        self.method
    }
    pub fn auto_exposure_enabled(&self) -> bool {
        self.auto_exposure_enabled
    }
    pub fn meter(&self) -> Meter {
        self.meter
    }
    pub fn manual_exposure_ev(&self) -> f32 {
        self.manual_exposure_ev
    }
    pub fn film_format_long_edge_mm(&self) -> f32 {
        self.film_format_long_edge_mm
    }
    pub fn band_pass_active(&self) -> bool {
        self.band_pass_active
    }
    pub fn uv(&self) -> &[f32; 3] {
        &self.uv
    }
    pub fn ir(&self) -> &[f32; 3] {
        &self.ir
    }
    pub fn apply_hanatos_window(&self) -> bool {
        self.apply_hanatos_window
    }
    pub fn apply_hanatos_surface(&self) -> bool {
        self.apply_hanatos_surface
    }
    pub fn hanatos_spectral_blur(&self) -> f32 {
        self.hanatos_spectral_blur
    }
    pub fn hanatos_window(&self) -> &[f32; 4] {
        &self.hanatos_window
    }
    pub fn hanatos_surface_rgb(&self) -> &[[f32; 15]; 3] {
        &self.hanatos_surface_rgb
    }
    pub fn projection_illuminant(&self) -> ProjectionIlluminant {
        self.projection_illuminant
    }
    pub fn input_compression_active(&self) -> bool {
        self.input_compression_active
    }
    pub fn input_hull_hash(&self) -> u64 {
        self.input_hull_hash
    }
    pub fn tc_source_hash(&self) -> u64 {
        self.tc_source_hash
    }
    pub fn sensitivity(&self) -> &exposure::Sensitivity {
        &self.sensitivity
    }
    pub fn tc_hash(&self) -> u64 {
        self.tc_hash
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

#[cfg(feature = "test-support")]
pub mod testing {
    use super::{Error, Operation};
    use std::cell::Cell;
    thread_local! {
        static COLOR: Cell<Option<(u32, u32, u32)>> = const { Cell::new(None) };
        static EXPOSURE: Cell<Option<(u32, u32, u32)>> = const { Cell::new(None) };
        static CONSUMED: Cell<Option<(u32, u32)>> = const { Cell::new(None) };
    }
    pub fn arm_color(operation: u32, count: u32, fault: u32) {
        COLOR.set(Some((operation, count, fault)));
    }
    pub fn clear_color() {
        COLOR.set(None);
    }
    pub fn color_fault(operation: u32) -> Result<(), u32> {
        if let Some((selected, remaining, fault)) = COLOR.get()
            && selected == operation
        {
            COLOR.set(if remaining > 1 {
                Some((selected, remaining - 1, fault))
            } else {
                None
            });
            if remaining == 1 {
                if fault == 2 {
                    panic!("color production boundary fault");
                }
                return Err(2);
            }
        }
        Ok(())
    }
    pub fn arm_exposure(operation: u32, count: u32, fault: u32) {
        EXPOSURE.set(Some((operation, count, fault)));
        CONSUMED.set(None);
    }
    pub fn clear_exposure() {
        EXPOSURE.set(None);
        CONSUMED.set(None);
    }
    pub fn exposure_consumed(operation: u32, fault: u32) -> bool {
        CONSUMED.take() == Some((operation, fault))
    }
    pub fn exposure_fault(operation: u32) -> Option<u32> {
        if let Some((selected, remaining, fault)) = EXPOSURE.get()
            && selected == operation
        {
            EXPOSURE.set(if remaining > 1 {
                Some((selected, remaining - 1, fault))
            } else {
                None
            });
            if remaining == 1 {
                CONSUMED.set(Some((operation, fault)));
                return Some(fault);
            }
        }
        None
    }
    pub(super) fn check(operation: Operation) -> Result<(), Error> {
        let category = match operation {
            Operation::InputMatrices => color_fault(5),
            Operation::Cat16 => color_fault(1),
            Operation::Cat02 => color_fault(3),
            Operation::Sensitivity => match exposure_fault(2) {
                Some(2) => panic!("exposure production boundary fault"),
                Some(1) => Err(2),
                Some(3) => Err(4),
                Some(4) => Err(3),
                _ => Ok(()),
            },
        };
        category.map_err(|category| Error::Boundary(operation, category))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn profile() -> profile::FilmProfile {
        profile::FilmProfile::new(
            profile::load_film_source(
                &Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../Resources/profiles/kodak_portra_400.json"),
            )
            .unwrap(),
        )
        .unwrap()
    }
    fn params(method: Method) -> Params {
        Params {
            input_space: color::InputSpace::SrgbRec709,
            decode_cctf: true,
            method,
            auto_exposure_enabled: false,
            meter: Meter::CenterWeighted,
            manual_exposure_ev: 0.0,
            film_format_long_edge_mm: 35.0,
            filter_override: false,
            uv: [1.0, 413.123456789012, 9.123456789012],
            ir: [1.0, 679.123456789012, 17.123456789012],
            apply_hanatos_window: true,
            apply_hanatos_surface: true,
            input_compression: true,
        }
    }
    fn resources(reference: &[f32; 81]) -> Resources<'_> {
        Resources {
            reference_illuminant: reference,
            reference_white_xyz: None,
            integrated_d65_xyz: None,
            reconstructed_reference_white: None,
            tc_source_hash: 0,
            input_hull_hash: 0,
        }
    }
    #[test]
    fn completed_mallett_owns_fixed_values_and_inactive_defaults() {
        let profile = profile();
        let recipe = {
            let reference = [1.0; 81];
            Recipe::new(&profile, params(Method::Mallett2019), resources(&reference)).unwrap()
        };
        drop(profile);
        assert_eq!(recipe.uv()[0].to_bits(), 0);
        assert_eq!(
            recipe.uv()[1].to_bits(),
            (413.123456789012_f64 as f32).to_bits()
        );
        assert!(!recipe.input_compression_active());
        assert_eq!(
            (
                recipe.tc_hash(),
                recipe.tc_source_hash(),
                recipe.input_hull_hash()
            ),
            (0, 0, 0)
        );
        assert!(recipe.apply_hanatos_window());
        assert!(!recipe.apply_hanatos_surface());
        assert_eq!(*recipe.hanatos_window(), [0.0; 4]);
        assert_eq!(*recipe.hanatos_surface_rgb(), [[0.0; 15]; 3]);
        assert!(recipe.color().to_conversion().xyz_adaptation.is_some());
        assert_ne!(recipe.hash(), 0);
    }
    #[test]
    fn projection_white_precedes_source_and_hull_requirements() {
        let profile = profile();
        let reference = [1.0; 81];
        assert!(matches!(
            Recipe::new(&profile, params(Method::Hanatos2025), resources(&reference)),
            Err(Error::MissingProjectionWhite)
        ));
        let mut borrowed = resources(&reference);
        let white = [0.95, 1.0, 1.09];
        borrowed.reference_white_xyz = Some(&white);
        assert!(matches!(
            Recipe::new(&profile, params(Method::Hanatos2025), borrowed),
            Err(Error::MissingTcSource(Method::Hanatos2025))
        ));
    }
    #[test]
    fn active_hull_precedes_missing_reconstruction() {
        let profile = profile();
        let reference = [1.0; 81];
        let white = [0.95, 1.0, 1.09];
        let mut borrowed = resources(&reference);
        borrowed.reference_white_xyz = Some(&white);
        borrowed.tc_source_hash = 1;
        assert!(matches!(
            Recipe::new(&profile, params(Method::Hanatos2025), borrowed),
            Err(Error::MissingInputHull)
        ));
    }

    #[test]
    fn missing_hanatos_arrays_and_reference_fail_at_their_ordered_consumers() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let original = root.join("Resources/profiles/kodak_portra_400.json");
        let mut document: serde_json::Value =
            serde_json::from_slice(&std::fs::read(original).unwrap()).unwrap();
        let data = document["data"].as_object_mut().unwrap();
        data.remove("hanatos2025_adaptation_window_params");
        data.remove("hanatos2025_adaptation_surface_params");
        let preset = match (cfg!(target_os = "windows"), cfg!(debug_assertions)) {
            (true, true) => "windows-clang-debug",
            (true, false) => "windows-clang-release",
            (false, true) => "linux-debug",
            (false, false) => "linux-release",
        };
        let scratch = root.join("out/validation").join(preset).join("film-input");
        std::fs::create_dir_all(&scratch).unwrap();
        let path = scratch.join(format!("missing-arrays-{}.json", std::process::id()));
        std::fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();
        let source = profile::load_film_source(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        let missing = profile::FilmProfile::new(source).unwrap();
        let reference = [1.0; 81];
        #[cfg(feature = "test-support")]
        {
            testing::arm_color(5, 1, 1);
            assert!(matches!(
                Recipe::new(&missing, params(Method::Hanatos2025), resources(&reference)),
                Err(Error::Boundary(Operation::InputMatrices, 2))
            ));
        }
        assert!(matches!(
            Recipe::new(&missing, params(Method::Hanatos2025), resources(&reference)),
            Err(Error::MissingHanatosWindow)
        ));
        let mut authored = params(Method::Hanatos2025);
        authored.apply_hanatos_window = false;
        assert!(matches!(
            Recipe::new(&missing, authored, resources(&reference)),
            Err(Error::MissingHanatosSurface)
        ));
        let mut authored = params(Method::Hanatos2025);
        authored.apply_hanatos_window = false;
        authored.apply_hanatos_surface = false;
        assert!(matches!(
            Recipe::new(&missing, authored, resources(&reference)),
            Err(Error::MissingProjectionWhite)
        ));
        let complete = profile();
        let white = [0.95, 1.0, 1.09];
        for (reference, kind) in [
            ([-1.0; 81], exposure::ErrorKind::ReferenceIlluminant),
            ([0.0; 81], exposure::ErrorKind::BandPassResponse),
        ] {
            let mut borrowed = resources(&reference);
            borrowed.reference_white_xyz = Some(&white);
            borrowed.tc_source_hash = 1;
            let mut authored = params(Method::Hanatos2025);
            authored.input_compression = false;
            let Err(Error::Sensitivity(error)) = Recipe::new(&complete, authored, borrowed) else {
                panic!("reference computation must precede missing reconstruction");
            };
            assert_eq!(error.kind(), kind);
        }
    }
    #[test]
    fn identity_matrices_are_completed_values() {
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let matrices = color::InputMatrices {
            rgb_to_xyz: identity,
            nominal_white_xyz: [1.0; 3],
            xyz_to_linear_srgb: identity,
            d65_white_xyz: [1.0; 3],
        };
        let color = color::InputColor::from_matrices(
            color::InputSpace::SrgbRec709,
            false,
            matrices,
            identity,
            [1.0; 3],
        );
        assert_eq!(*color.rgb_to_xyz(), identity);
        assert_eq!(*color.xyz_adapt(), identity);
        assert_eq!(*color.xyz_to_linear_srgb(), identity);
        assert_eq!(color.to_conversion().xyz_adaptation, Some(identity));
    }
}
