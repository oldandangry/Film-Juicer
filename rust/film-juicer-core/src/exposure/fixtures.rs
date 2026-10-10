//! Fixed-axis witnesses for the retained color/spectrum fixtures, never rendering.

use crate::color;

const DWG_WHITE: [f32; 3] = [0.950455, 1.0, 1.089058];

pub struct Spectrum {
    pre_xyz: [f32; 3],
    consumer_white: [f32; 3],
    post_adapt_xyz: [f32; 3],
    samples: [f32; 81],
}
impl Spectrum {
    pub fn pre_xyz(&self) -> &[f32; 3] {
        &self.pre_xyz
    }
    pub fn consumer_white(&self) -> &[f32; 3] {
        &self.consumer_white
    }
    pub fn post_adapt_xyz(&self) -> &[f32; 3] {
        &self.post_adapt_xyz
    }
    pub fn samples(&self) -> &[f32; 81] {
        &self.samples
    }
}

fn finite(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}
fn finite_positive(value: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    }
}
fn nonnegative(value: f32) -> f32 {
    if value > 0.0 { value } else { 0.0 }
}

pub fn hanatos_spectrum(
    rgb_dwg: [f32; 3],
    spectra: &[f32; 192 * 192 * 81],
    reference_white: [f32; 3],
) -> Spectrum {
    let pre_xyz = color::dwg_to_xyz(rgb_dwg).map(finite);
    let mut consumer_white = std::array::from_fn(|i| {
        if reference_white[i].is_finite() {
            reference_white[i]
        } else {
            DWG_WHITE[i]
        }
    });
    if consumer_white[1] <= 0.0 {
        consumer_white = DWG_WHITE;
    }
    let post_adapt_xyz = color::adapt_cat02(
        pre_xyz,
        color::Whites {
            source_xyz: DWG_WHITE,
            destination_xyz: consumer_white,
        },
    )
    .map(finite);
    let brightness = finite(post_adapt_xyz[0] + post_adapt_xyz[1] + post_adapt_xyz[2]);
    let denominator = brightness.max(1e-10);
    let x = (post_adapt_xyz[0] / denominator).clamp(0.0, 1.0);
    let y = (post_adapt_xyz[1] / denominator).clamp(0.0, 1.0);
    let quad_denominator = (1.0 - x).max(1e-10);
    let quad_x = ((1.0 - x) * (1.0 - x)).clamp(0.0, 1.0);
    let quad_y = (y / quad_denominator).clamp(0.0, 1.0);
    let fx = quad_x * 191.0;
    let fy = quad_y * 191.0;
    let x0 = (fx.floor() as usize).min(191);
    let y0 = (fy.floor() as usize).min(191);
    let x1 = (x0 + 1).min(191);
    let y1 = (y0 + 1).min(191);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let samples = std::array::from_fn(|k| {
        let sample = |x, y| spectra[(x * 192 + y) * 81 + k];
        let v0 = sample(x0, y0) * (1.0 - tx) + sample(x1, y0) * tx;
        let v1 = sample(x0, y1) * (1.0 - tx) + sample(x1, y1) * tx;
        finite(brightness * (v0 * (1.0 - ty) + v1 * ty))
    });
    Spectrum {
        pre_xyz,
        consumer_white,
        post_adapt_xyz,
        samples,
    }
}

pub struct TablesInput<'a> {
    pub rgb_dwg: [f32; 3],
    pub reference_white: [f32; 3],
    pub s_inverse: [f32; 9],
    pub ax: &'a [f32; 81],
    pub ay: &'a [f32; 81],
    pub az: &'a [f32; 81],
}

pub fn tables_spectrum(input: TablesInput<'_>) -> Spectrum {
    let pre_xyz = color::dwg_to_xyz(input.rgb_dwg).map(finite_positive);
    let mut consumer_white = input.reference_white.map(finite_positive);
    if consumer_white[1] <= 0.0 {
        consumer_white = DWG_WHITE;
    }
    let post_adapt_xyz = color::adapt_cat02(
        pre_xyz,
        color::Whites {
            source_xyz: DWG_WHITE,
            destination_xyz: consumer_white,
        },
    )
    .map(nonnegative);
    let target_scale = if post_adapt_xyz[1] > 0.0 {
        post_adapt_xyz[1]
    } else {
        pre_xyz[1]
    };
    let s = input.s_inverse;
    let xyz = post_adapt_xyz;
    let cx = nonnegative(s[0] * xyz[0] + s[1] * xyz[1] + s[2] * xyz[2]);
    let cy = nonnegative(s[3] * xyz[0] + s[4] * xyz[1] + s[5] * xyz[2]);
    let cz = nonnegative(s[6] * xyz[0] + s[7] * xyz[1] + s[8] * xyz[2]);
    let mut samples = [0.0_f32; 81];
    let mut y_recon = 0.0_f64;
    for (i, out) in samples.iter_mut().enumerate() {
        let basis = cx * nonnegative(input.ax[i])
            + cy * nonnegative(input.ay[i])
            + cz * nonnegative(input.az[i]);
        *out = if basis > 1e-6 { basis } else { 1e-6 };
        y_recon += f64::from(*out) * f64::from(input.ay[i]);
    }
    if y_recon > 1e-20 && target_scale > 0.0 {
        let scale = (f64::from(target_scale) / y_recon) as f32;
        for sample in &mut samples {
            *sample = nonnegative(scale * *sample);
        }
    } else if target_scale <= 0.0 {
        samples.fill(0.0);
    }
    Spectrum {
        pre_xyz,
        consumer_white,
        post_adapt_xyz,
        samples,
    }
}

pub fn mallett_raw(
    linear_srgb: [f32; 3],
    basis_rgb: &[[f32; 3]; 81],
    illuminant: &[f32; 81],
    sensitivity_rgb: &[[f32; 3]; 81],
) -> [f32; 3] {
    super::mallett_raw(linear_srgb, basis_rgb, illuminant, sensitivity_rgb)
}
