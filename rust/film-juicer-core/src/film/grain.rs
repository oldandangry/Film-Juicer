//! Film-Juicer visual grain from captured resolved controls, borrowing completed development.

use crate::hash;
use crate::profile::Polarity;
use super::Development;

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub active: bool,
    pub sublayers_active: bool,
    pub particle_area_um2: f32,
    pub particle_scale_cmy: [f32; 3],
    pub particle_scale_layers: [f32; 3],
    pub particle_density_min_cmy: [f32; 3],
    pub uniformity_cmy: [f32; 3],
    pub correlation_sigma_px: f32,
    pub dye_cloud_blur_um: f32,
    pub micro_structure: [f32; 2],
    pub sublayer_count: i32,
    pub amplitude: f32,
    pub chroma_mix: f32,
    pub chroma_shared_weight: f32,
    pub chroma_independent_weight: f32,
    pub coarse_weight: f32,
    pub mid_weight: f32,
    pub size_mix_scale: f32,
    pub clump_temporal_mix: f32,
    pub clump_morph_period_sec: f32,
    pub debug_view: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Controls,
    MissingLayers,
    ZeroIdentity,
}

#[derive(Debug)]
pub struct Recipe {
    active: bool,
    sublayers_active: bool,
    axis_finite_cmy: [bool; 3],
    axis_block_prefix_max_cmy: [[f32; 16]; 3],
    particle_area_um2: f32,
    particle_scale_cmy: [f32; 3],
    particle_scale_layers: [f32; 3],
    particle_density_min_cmy: [f32; 3],
    uniformity_cmy: [f32; 3],
    correlation_sigma_px: f32,
    dye_cloud_blur_um: f32,
    micro_structure: [f32; 2],
    sublayer_count: i32,
    amplitude: f32,
    chroma_mix: f32,
    chroma_shared_weight: f32,
    chroma_independent_weight: f32,
    coarse_weight: f32,
    mid_weight: f32,
    size_mix_scale: f32,
    clump_temporal_mix: f32,
    clump_morph_period_sec: f32,
    debug_view: i32,
    fine_weight: f32,
    layers_hash: u64,
    hash: u64,
}

fn positive(value: f32) -> bool {
    value.is_finite() && value > 0.0
}
fn nonnegative(value: f32) -> bool {
    value.is_finite() && value >= 0.0
}

fn axis_search_metadata(
    curves_cmy: &[[f32; 3]],
    polarity: Polarity,
) -> ([bool; 3], [[f32; 16]; 3]) {
    let mut finite_cmy = [false; 3];
    let mut prefix_cmy = [[0.0; 16]; 3];
    if curves_cmy.len() != 256 {
        return (finite_cmy, prefix_cmy);
    }
    let positive = polarity == Polarity::Positive;
    for (channel, finite) in finite_cmy.iter_mut().enumerate() {
        let mut prefix = f32::NEG_INFINITY;
        *finite = true;
        for (sample, row) in curves_cmy.iter().enumerate() {
            let axis = if positive {
                -row[channel]
            } else {
                row[channel]
            };
            if !axis.is_finite() {
                *finite = false;
                break;
            }
            // Native std::max keeps its first operand on ordered ties, including zero signs.
            if prefix < axis {
                prefix = axis;
            }
            if sample % 16 == 15 {
                prefix_cmy[channel][sample / 16] = prefix;
            }
        }
    }
    (finite_cmy, prefix_cmy)
}

fn hash_floats(identity: &mut u64, values: &[f32]) {
    for value in values {
        hash::update_bytes(identity, &value.to_bits().to_le_bytes());
    }
}
fn recipe_hash(recipe: &Recipe) -> u64 {
    if !recipe.active {
        return 0;
    }
    let mut identity = hash::FNV_OFFSET;
    hash::update_bytes(
        &mut identity,
        &[u8::from(recipe.active), u8::from(recipe.sublayers_active)],
    );
    hash_floats(&mut identity, &[recipe.particle_area_um2]);
    hash_floats(&mut identity, &recipe.particle_scale_cmy);
    hash_floats(&mut identity, &recipe.particle_scale_layers);
    hash_floats(&mut identity, &recipe.particle_density_min_cmy);
    hash_floats(&mut identity, &recipe.uniformity_cmy);
    hash_floats(
        &mut identity,
        &[recipe.correlation_sigma_px, recipe.dye_cloud_blur_um],
    );
    hash_floats(&mut identity, &recipe.micro_structure);
    hash::update_bytes(&mut identity, &recipe.sublayer_count.to_le_bytes());
    hash_floats(
        &mut identity,
        &[
            recipe.amplitude,
            recipe.chroma_mix,
            recipe.chroma_shared_weight,
            recipe.chroma_independent_weight,
            recipe.fine_weight,
            recipe.mid_weight,
            recipe.coarse_weight,
            recipe.size_mix_scale,
            recipe.clump_temporal_mix,
            recipe.clump_morph_period_sec,
        ],
    );
    hash::update_bytes(&mut identity, &recipe.debug_view.to_le_bytes());
    hash::update_bytes(&mut identity, &recipe.layers_hash.to_le_bytes());
    identity
}

impl Recipe {
    pub fn new(development: &Development, params: Params) -> Result<Self, Error> {
        #[cfg(any(test, feature = "test-support"))]
        let zero_identity = test_support::take();
        if !params.active {
            return Ok(Self {
                active: false,
                sublayers_active: false,
                axis_finite_cmy: [false; 3],
                axis_block_prefix_max_cmy: [[0.0; 16]; 3],
                particle_area_um2: 0.0,
                particle_scale_cmy: [0.0; 3],
                particle_scale_layers: [0.0; 3],
                particle_density_min_cmy: [0.0; 3],
                uniformity_cmy: [0.0; 3],
                correlation_sigma_px: 0.0,
                dye_cloud_blur_um: 0.0,
                micro_structure: [0.0; 2],
                sublayer_count: 0,
                amplitude: 0.0,
                chroma_mix: 0.0,
                chroma_shared_weight: 0.0,
                chroma_independent_weight: 0.0,
                coarse_weight: 0.0,
                mid_weight: 0.0,
                size_mix_scale: 1.0,
                clump_temporal_mix: 0.0,
                clump_morph_period_sec: 0.0,
                debug_view: 0,
                fine_weight: 0.0,
                layers_hash: 0,
                hash: 0,
            });
        }
        if !positive(params.particle_area_um2)
            || !params.particle_scale_cmy.iter().copied().all(positive)
            || !params.particle_scale_layers.iter().copied().all(positive)
            || !params
                .particle_density_min_cmy
                .iter()
                .copied()
                .all(nonnegative)
            || !params.uniformity_cmy.iter().copied().all(nonnegative)
            || !nonnegative(params.correlation_sigma_px)
            || !nonnegative(params.dye_cloud_blur_um)
            || !params.micro_structure.iter().copied().all(nonnegative)
            || params.sublayer_count <= 0
            || !nonnegative(params.amplitude)
            || !params.chroma_mix.is_finite()
            || !nonnegative(params.chroma_shared_weight)
            || !nonnegative(params.chroma_independent_weight)
            || !nonnegative(params.coarse_weight)
            || !nonnegative(params.mid_weight)
            || !params.size_mix_scale.is_finite()
            || params.size_mix_scale < 1.0
            || !nonnegative(params.clump_temporal_mix)
            || !nonnegative(params.clump_morph_period_sec)
        {
            return Err(Error::Controls);
        }
        if params.sublayers_active && development.layers_hash() == 0 {
            return Err(Error::MissingLayers);
        }
        let (axis_finite_cmy, axis_block_prefix_max_cmy) = if params.sublayers_active {
            axis_search_metadata(development.normalized_curves_cmy(), development.polarity())
        } else {
            ([false; 3], [[0.0; 16]; 3])
        };
        let mut mid_weight = params.mid_weight.clamp(0.0, 1.0);
        let mut coarse_weight = params.coarse_weight.clamp(0.0, 1.0);
        let remainder = 1.0 - mid_weight - coarse_weight;
        let mut fine_weight = if 0.0 < remainder { remainder } else { 0.0 };
        let sum = fine_weight + mid_weight + coarse_weight;
        if sum <= 0.0 || !sum.is_finite() {
            return Err(Error::Controls);
        }
        let inverse_sum = 1.0 / sum;
        fine_weight *= inverse_sum;
        mid_weight *= inverse_sum;
        coarse_weight *= inverse_sum;
        let mut recipe = Self {
            active: true,
            sublayers_active: params.sublayers_active,
            axis_finite_cmy,
            axis_block_prefix_max_cmy,
            particle_area_um2: params.particle_area_um2,
            particle_scale_cmy: params.particle_scale_cmy,
            particle_scale_layers: params.particle_scale_layers,
            particle_density_min_cmy: params.particle_density_min_cmy.map(|v| v.clamp(0.0, 1.0)),
            uniformity_cmy: params.uniformity_cmy.map(|v| v.clamp(0.0, 1.0)),
            correlation_sigma_px: params.correlation_sigma_px,
            dye_cloud_blur_um: params.dye_cloud_blur_um,
            micro_structure: params.micro_structure,
            sublayer_count: params.sublayer_count,
            amplitude: params.amplitude,
            chroma_mix: params.chroma_mix.clamp(0.0, 1.0),
            chroma_shared_weight: params.chroma_shared_weight.clamp(0.0, 1.0),
            chroma_independent_weight: params.chroma_independent_weight.clamp(0.0, 1.0),
            coarse_weight,
            mid_weight,
            size_mix_scale: params.size_mix_scale,
            clump_temporal_mix: params.clump_temporal_mix.clamp(0.0, 0.30),
            clump_morph_period_sec: params.clump_morph_period_sec.clamp(5.0, 60.0),
            debug_view: params.debug_view.clamp(0, 6),
            fine_weight,
            layers_hash: if params.sublayers_active {
                development.layers_hash()
            } else {
                0
            },
            hash: 0,
        };
        let identity = recipe_hash(&recipe);
        #[cfg(any(test, feature = "test-support"))]
        let identity = if zero_identity { 0 } else { identity };
        if identity == 0 {
            return Err(Error::ZeroIdentity);
        }
        recipe.hash = identity;
        Ok(recipe)
    }
    pub fn active(&self) -> bool {
        self.active
    }
    pub fn sublayers_active(&self) -> bool {
        self.sublayers_active
    }
    pub fn axis_finite_cmy(&self) -> &[bool; 3] {
        &self.axis_finite_cmy
    }
    pub fn axis_block_prefix_max_cmy(&self) -> &[[f32; 16]; 3] {
        &self.axis_block_prefix_max_cmy
    }
    pub fn particle_area_um2(&self) -> f32 {
        self.particle_area_um2
    }
    pub fn particle_scale_cmy(&self) -> &[f32; 3] {
        &self.particle_scale_cmy
    }
    pub fn particle_scale_layers(&self) -> &[f32; 3] {
        &self.particle_scale_layers
    }
    pub fn particle_density_min_cmy(&self) -> &[f32; 3] {
        &self.particle_density_min_cmy
    }
    pub fn uniformity_cmy(&self) -> &[f32; 3] {
        &self.uniformity_cmy
    }
    pub fn correlation_sigma_px(&self) -> f32 {
        self.correlation_sigma_px
    }
    pub fn dye_cloud_blur_um(&self) -> f32 {
        self.dye_cloud_blur_um
    }
    pub fn micro_structure(&self) -> &[f32; 2] {
        &self.micro_structure
    }
    pub fn sublayer_count(&self) -> i32 {
        self.sublayer_count
    }
    pub fn amplitude(&self) -> f32 {
        self.amplitude
    }
    pub fn chroma_mix(&self) -> f32 {
        self.chroma_mix
    }
    pub fn chroma_shared_weight(&self) -> f32 {
        self.chroma_shared_weight
    }
    pub fn chroma_independent_weight(&self) -> f32 {
        self.chroma_independent_weight
    }
    pub fn coarse_weight(&self) -> f32 {
        self.coarse_weight
    }
    pub fn mid_weight(&self) -> f32 {
        self.mid_weight
    }
    pub fn size_mix_scale(&self) -> f32 {
        self.size_mix_scale
    }
    pub fn clump_temporal_mix(&self) -> f32 {
        self.clump_temporal_mix
    }
    pub fn clump_morph_period_sec(&self) -> f32 {
        self.clump_morph_period_sec
    }
    pub fn debug_view(&self) -> i32 {
        self.debug_view
    }
    pub fn fine_weight(&self) -> f32 {
        self.fine_weight
    }
    pub fn layers_hash(&self) -> u64 {
        self.layers_hash
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    //! Call-bounded reserved-zero rejection witness, absent from production.
    use std::cell::Cell;
    thread_local! { static ZERO: Cell<bool> = const { Cell::new(false) }; }
    pub fn zero_next_identity() {
        ZERO.set(true);
    }
    pub(super) fn take() -> bool {
        ZERO.replace(false)
    }
}

#[cfg(test)]
mod tests {
    use super::{Params, Recipe, Error, axis_search_metadata, recipe_hash};
    use crate::assets::Assets;
    use crate::film::{self, Development};
    use crate::profile::Polarity;
    use crate::route::ScanRoute;
    use serde_json::{Value, json};
    use std::path::PathBuf;
    use std::sync::Arc;

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }
    fn fixture(kind: &str) -> Value {
        let preset = match (cfg!(target_os = "windows"), cfg!(debug_assertions)) {
            (true, true) => "windows-clang-debug",
            (true, false) => "windows-clang-release",
            (false, true) => "linux-debug",
            (false, false) => "linux-release",
        };
        let directory = root().join("tests/film/grain-fixtures");
        let manifest: Value =
            serde_json::from_slice(&std::fs::read(directory.join("manifest.json")).unwrap())
                .unwrap();
        serde_json::from_slice(
            &std::fs::read(directory.join(manifest["presets"][preset][kind].as_str().unwrap()))
                .unwrap(),
        )
        .unwrap()
    }
    fn floats<const N: usize>(v: &Value) -> [f32; N] {
        std::array::from_fn(|i| f32::from_bits(v[i].as_u64().unwrap() as u32))
    }
    fn params(v: &Value) -> Params {
        Params {
            active: v["active"].as_bool().unwrap(),
            sublayers_active: v["sublayers_active"].as_bool().unwrap(),
            particle_area_um2: floats::<1>(&v["particle_area_um2"])[0],
            particle_scale_cmy: floats(&v["particle_scale_cmy"]),
            particle_scale_layers: floats(&v["particle_scale_layers"]),
            particle_density_min_cmy: floats(&v["particle_density_min_cmy"]),
            uniformity_cmy: floats(&v["uniformity_cmy"]),
            correlation_sigma_px: floats::<1>(&v["correlation_sigma_px"])[0],
            dye_cloud_blur_um: floats::<1>(&v["dye_cloud_blur_um"])[0],
            micro_structure: floats(&v["micro_structure"]),
            sublayer_count: v["sublayer_count"].as_i64().unwrap() as i32,
            amplitude: floats::<1>(&v["amplitude"])[0],
            chroma_mix: floats::<1>(&v["chroma_mix"])[0],
            chroma_shared_weight: floats::<1>(&v["chroma_shared_weight"])[0],
            chroma_independent_weight: floats::<1>(&v["chroma_independent_weight"])[0],
            coarse_weight: floats::<1>(&v["coarse_weight"])[0],
            mid_weight: floats::<1>(&v["mid_weight"])[0],
            size_mix_scale: floats::<1>(&v["size_mix_scale"])[0],
            clump_temporal_mix: floats::<1>(&v["clump_temporal_mix"])[0],
            clump_morph_period_sec: floats::<1>(&v["clump_morph_period_sec"])[0],
            debug_view: v["debug_view"].as_i64().unwrap() as i32,
        }
    }
    fn project(recipe: &Recipe) -> Value {
        json!({
            "active": recipe.active(), "sublayers_active": recipe.sublayers_active(),
            "axis_finite_cmy": recipe.axis_finite_cmy(),
            "axis_block_prefix_max_cmy": recipe.axis_block_prefix_max_cmy().map(|v| v.map(f32::to_bits)),
            "particle_area_um2": [recipe.particle_area_um2().to_bits()],
            "particle_scale_cmy": recipe.particle_scale_cmy().map(f32::to_bits),
            "particle_scale_layers": recipe.particle_scale_layers().map(f32::to_bits),
            "particle_density_min_cmy": recipe.particle_density_min_cmy().map(f32::to_bits),
            "uniformity_cmy": recipe.uniformity_cmy().map(f32::to_bits),
            "correlation_sigma_px": [recipe.correlation_sigma_px().to_bits()],
            "dye_cloud_blur_um": [recipe.dye_cloud_blur_um().to_bits()],
            "micro_structure": recipe.micro_structure().map(f32::to_bits),
            "sublayer_count": recipe.sublayer_count(),
            "amplitude": [recipe.amplitude().to_bits()],
            "chroma_mix": [recipe.chroma_mix().to_bits()],
            "chroma_shared_weight": [recipe.chroma_shared_weight().to_bits()],
            "chroma_independent_weight": [recipe.chroma_independent_weight().to_bits()],
            "coarse_weight": [recipe.coarse_weight().to_bits()],
            "mid_weight": [recipe.mid_weight().to_bits()],
            "size_mix_scale": [recipe.size_mix_scale().to_bits()],
            "clump_temporal_mix": [recipe.clump_temporal_mix().to_bits()],
            "clump_morph_period_sec": [recipe.clump_morph_period_sec().to_bits()],
            "debug_view": recipe.debug_view(),
            "fine_weight": [recipe.fine_weight().to_bits()], "layers_hash": recipe.layers_hash(), "hash": recipe.hash(),
        })
    }
    fn route(tag: u64) -> ScanRoute {
        match tag {
            0 => ScanRoute::NegativeDirect,
            1 => ScanRoute::NegativePrint,
            2 => ScanRoute::PositiveDirect,
            3 => ScanRoute::PositivePrint,
            _ => panic!("fixture route"),
        }
    }
    fn development(assets: &Assets, row: &Value, layers_required: bool) -> Development {
        Development::new(
            assets.film(row["key"].as_str().unwrap()).unwrap(),
            film::Params {
                gamma_factor: floats::<1>(&row["gamma"])[0],
                layers_required,
                requested_route: route(row["route"].as_u64().unwrap()),
            },
        )
        .unwrap()
    }

    #[test]
    fn exact_native_controls_with_the_same_completed_development() {
        let captured = fixture("values");
        assert_eq!(captured["controls"].as_array().unwrap().len(), 758);
        let assets = Assets::new(root().join("Resources"));
        let row = &captured["products"][2];
        let with_layers = development(&assets, row, true);
        let without_layers = development(&assets, row, false);
        for row in captured["controls"].as_array().unwrap() {
            let development = if row["source_layers_hash"].as_u64().unwrap() == 0 {
                &without_layers
            } else {
                &with_layers
            };
            assert_eq!(
                development.layers_hash(),
                row["source_layers_hash"].as_u64().unwrap()
            );
            let result = Recipe::new(development, params(&row["controls"]));
            if row["success"].as_bool().unwrap() {
                assert_eq!(project(&result.unwrap()), row["grain"], "{}", row["id"]);
            } else {
                assert_eq!(
                    result.unwrap_err(),
                    if row["id"] == "missing-layers" {
                        Error::MissingLayers
                    } else {
                        Error::Controls
                    },
                    "{}",
                    row["id"]
                );
            }
        }
        let mut controls = params(&row["controls"]);
        controls.particle_area_um2 = 0.0;
        assert_eq!(
            Recipe::new(&without_layers, controls).unwrap_err(),
            Error::Controls
        );
    }

    #[test]
    fn exact_native_normalized_total_axis_helper_domain() {
        let captured = fixture("values");
        assert_eq!(captured["metadata"].as_array().unwrap().len(), 144);
        for row in captured["metadata"].as_array().unwrap() {
            if row["polarity"] == 2 {
                // Unsupported native polarity has no representation in a completed Development.
                assert_eq!(row["finite"], json!(([false; 3])));
                assert_eq!(row["prefix"], json!(([[0u32; 16]; 3])));
                continue;
            }
            let curves: Vec<[f32; 3]> = row["normalized"]
                .as_array()
                .unwrap()
                .iter()
                .map(floats)
                .collect();
            let polarity = if row["polarity"] == 0 {
                Polarity::Negative
            } else {
                Polarity::Positive
            };
            let (finite, prefix) = axis_search_metadata(&curves, polarity);
            assert_eq!(json!(finite), row["finite"]);
            assert_eq!(json!(prefix.map(|v| v.map(f32::to_bits))), row["prefix"]);
        }
    }

    #[test]
    fn exact_native_products_and_genuine_variable_rows_and_overflow() {
        for (kind, count) in [("values", 36), ("synthetic", 28)] {
            let captured = fixture(kind);
            assert_eq!(captured["products"].as_array().unwrap().len(), count);
            for row in captured["products"].as_array().unwrap() {
                let directory = if let Some(name) = row["source"].as_str() {
                    root()
                        .join("tests/film/grain-fixtures/synthetic")
                        .join(name)
                } else {
                    root().join("Resources")
                };
                let assets = Assets::new(directory);
                let controls = params(&row["controls"]);
                let development =
                    development(&assets, row, controls.active && controls.sublayers_active);
                assert_eq!(
                    development.hash(),
                    row["development_hash"].as_u64().unwrap()
                );
                assert_eq!(
                    development.normalized_curves_hash(),
                    row["normalized_hash"].as_u64().unwrap()
                );
                if kind == "synthetic" {
                    assert!(
                        development
                            .authored_curves_cmy()
                            .iter()
                            .flatten()
                            .all(|v| v.is_finite())
                    );
                    assert_eq!(
                        development
                            .normalized_curves_cmy()
                            .iter()
                            .flatten()
                            .any(|v| !v.is_finite()),
                        row["nonfinite_normalized"].as_bool().unwrap()
                    );
                }
                let recipe = Recipe::new(&development, controls).unwrap();
                drop(development);
                assets.release_cached_payloads().unwrap();
                drop(assets);
                assert_eq!(project(&recipe), row["grain"], "{}", row["key"]);
            }
        }
    }

    #[test]
    fn fixed_result_retains_no_profile_or_development_and_excludes_metadata_from_hash() {
        let captured = fixture("values");
        let row = &captured["products"][2];
        let assets = Assets::new(root().join("Resources"));
        let profile = assets.film("kodak_portra_400").unwrap();
        let development = Development::new(
            Arc::clone(&profile),
            film::Params {
                gamma_factor: 1.0,
                layers_required: true,
                requested_route: ScanRoute::NegativeDirect,
            },
        )
        .unwrap();
        let before = Arc::strong_count(&profile);
        let mut recipe = Recipe::new(&development, params(&row["controls"])).unwrap();
        assert_eq!(Arc::strong_count(&profile), before);
        drop(development);
        assets.release_cached_payloads().unwrap();
        drop(profile);
        assert_eq!(project(&recipe), row["grain"]);
        let identity = recipe.hash();
        recipe.axis_finite_cmy.fill(false);
        recipe.axis_block_prefix_max_cmy[0][0] = 123.0;
        assert_eq!(recipe_hash(&recipe), identity);
    }

    #[test]
    fn zero_active_identity_is_rejected_without_leaking_into_another_call() {
        let captured = fixture("values");
        let row = &captured["products"][2];
        let assets = Assets::new(root().join("Resources"));
        let development = development(&assets, row, true);
        let mut controls = params(&row["controls"]);
        super::test_support::zero_next_identity();
        assert_eq!(
            Recipe::new(&development, controls).unwrap_err(),
            Error::ZeroIdentity
        );
        super::test_support::zero_next_identity();
        controls.active = false;
        controls.particle_area_um2 = f32::NAN;
        let inactive = Recipe::new(&development, controls).unwrap();
        assert_eq!(inactive.size_mix_scale().to_bits(), 1.0f32.to_bits());
        assert_eq!(inactive.hash(), 0);
        assert_eq!(
            project(&Recipe::new(&development, params(&row["controls"])).unwrap()),
            row["grain"]
        );
    }
}
