//! Callback-local geometry and optics prepared from the admitted complete state.
#![forbid(unsafe_code)]

use std::panic::{AssertUnwindSafe, catch_unwind};

use film_juicer_core::{
    frame::Domain,
    optics::{self, diffusion_frame},
    recipe,
    route::ScanRoute,
};

use super::sys;

pub(super) struct Error {
    pub(super) diagnostic: String,
}

fn extent(domain: Domain) -> sys::FjExtent {
    sys::FjExtent {
        x: domain.x,
        y: domain.y,
        width: domain.width,
        height: domain.height,
    }
}

fn domain(value: sys::FjExtent) -> Domain {
    Domain {
        x: value.x,
        y: value.y,
        width: value.width,
        height: value.height,
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

fn diffusion_input(value: &optics::Diffusion) -> diffusion_frame::StageInput {
    diffusion_frame::StageInput {
        domain: match value.domain() {
            optics::Domain::FilmLinearExposure => 0,
            optics::Domain::PrintLinearExposure => 1,
        },
        family: match value.family() {
            optics::Family::Glimmerglass => 0,
            optics::Family::BlackProMist => 1,
            optics::Family::ProMist => 2,
            optics::Family::Cinebloom => 3,
        },
        group_weights: *value.group_weights(),
        group_lambda_um: *value.group_lambda_um(),
        warmth: value.warmth(),
        spatial_scale: value.spatial_scale(),
        scatter_fraction: value.scatter_fraction(),
        hash: value.hash(),
    }
}

fn empty_diffusion() -> sys::FjDiffusion {
    sys::FjDiffusion {
        scatter_fraction: 0.0,
        lambda_px: [0.0; 9],
        red_weights: [0.0; 9],
        green_weights: [0.0; 9],
        blue_weights: [0.0; 9],
        radius_px: 0,
        sample_hash: 0,
        hash: 0,
    }
}

fn diffusion_stage(value: &diffusion_frame::Stage) -> Result<sys::FjDiffusion, Error> {
    let fields = value.sample().fields();
    let components = diffusion_frame::Components::new(fields).map_err(optics_error)?;
    Ok(sys::FjDiffusion {
        scatter_fraction: value.scatter_fraction(),
        lambda_px: *components.lambda_px(),
        red_weights: *components.red_weights(),
        green_weights: *components.green_weights(),
        blue_weights: *components.blue_weights(),
        radius_px: fields.radius_px,
        sample_hash: fields.hash,
        hash: value.hash(),
    })
}

const EMPTY_HALATION_GAUSSIAN: sys::FjHalationGaussian = sys::FjHalationGaussian {
    kind: 0,
    radius: 0,
    fir_weights: [0.0; 19],
    yvv: [0.0; 4],
};

const EMPTY_HALATION_CHANNEL: sys::FjHalationChannel = sys::FjHalationChannel {
    core: EMPTY_HALATION_GAUSSIAN,
    tail: [EMPTY_HALATION_GAUSSIAN; optics::HALATION_TAIL_COMPONENTS],
    bounce: [EMPTY_HALATION_GAUSSIAN; optics::HALATION_MAX_BOUNCES],
    total_strength: 0.0,
    tail_weight: 0.0,
};

fn halation_gaussian(value: &optics::HalationGaussian) -> sys::FjHalationGaussian {
    sys::FjHalationGaussian {
        kind: match value.kind() {
            optics::HalationGaussianKind::Identity => sys::FJ_HALATION_IDENTITY,
            optics::HalationGaussianKind::FirReflect => sys::FJ_HALATION_FIR_REFLECT,
            optics::HalationGaussianKind::YvvReplicate => sys::FJ_HALATION_YVV_REPLICATE,
            optics::HalationGaussianKind::YvvReplicateFactored => {
                sys::FJ_HALATION_YVV_REPLICATE_FACTORED
            }
        },
        radius: value.radius(),
        fir_weights: *value.fir_weights(),
        yvv: *value.yvv(),
    }
}

pub(crate) fn halation_projection(value: &optics::HalationFrame) -> sys::FjScatterHalation {
    sys::FjScatterHalation {
        recipe_hash: value.recipe_hash(),
        scatter_amount: value.scatter_amount(),
        bounce_count: value.bounce_count() as u32,
        bounce_weights: *value.bounce_weights(),
        renormalize: u32::from(value.renormalize()),
        channels: std::array::from_fn(|channel| {
            let source = &value.channels()[channel];
            sys::FjHalationChannel {
                core: halation_gaussian(source.core()),
                tail: source.tail().each_ref().map(halation_gaussian),
                bounce: source.bounce().each_ref().map(halation_gaussian),
                total_strength: source.total_strength(),
                tail_weight: source.tail_weight(),
            }
        }),
    }
}

pub(crate) fn empty_halation() -> sys::FjScatterHalation {
    sys::FjScatterHalation {
        recipe_hash: 0,
        scatter_amount: 0.0,
        bounce_count: 0,
        bounce_weights: [0.0; optics::HALATION_MAX_BOUNCES],
        renormalize: 0,
        channels: [EMPTY_HALATION_CHANNEL; 3],
    }
}

fn optics_error(error: diffusion_frame::Error) -> Error {
    Error {
        diagnostic: error.diagnostic().to_owned(),
    }
}

fn halation_error(error: optics::HalationFrameError) -> Error {
    let diagnostic = match error {
        optics::HalationFrameError::PixelSize => {
            "ResourceDescriptorMismatch component=scatter_halation field=pixel_size_um"
        }
        optics::HalationFrameError::BounceCount => {
            "ResourceDescriptorMismatch component=scatter_halation field=bounce_count"
        }
        optics::HalationFrameError::AdvancedRadius {
            field,
            channel,
            index,
            sigma_px,
        } => {
            return Error {
                diagnostic: format!(
                    "ResourceDescriptorMismatch component=scatter_halation field={} channel={channel} index={index:?} sigma_px={sigma_px} reason=advanced_radius",
                    field.name(),
                ),
            };
        }
        optics::HalationFrameError::Gaussian {
            field: optics::HalationFrameField::ScatterCore,
            ..
        } => "ResourceDescriptorMismatch component=scatter_halation field=scatter_core",
        optics::HalationFrameError::Gaussian {
            field: optics::HalationFrameField::ScatterTail,
            ..
        } => "ResourceDescriptorMismatch component=scatter_halation field=scatter_tail",
        optics::HalationFrameError::Gaussian {
            field: optics::HalationFrameField::Bounce,
            ..
        } => "ResourceDescriptorMismatch component=scatter_halation field=back_reflection_bounce",
    };
    Error {
        diagnostic: diagnostic.to_owned(),
    }
}

pub(super) fn prepare(
    state: &recipe::RenderState,
    full_frame: sys::FjExtent,
) -> Result<super::edge::PreparedFramePolicy, Error> {
    let geometry = catch_unwind(AssertUnwindSafe(|| {
        crate::cuda::geometry::checked_geometry(
            domain(full_frame),
            state.input().film_format_long_edge_mm(),
        )
    }))
    .map_err(|_| Error {
        diagnostic: "frame geometry preparation panicked".to_owned(),
    })?;

    let optics_recipe = state.optics();
    let route = route_tag(state.development().route());
    let diffusion_input = diffusion_frame::FrameInput {
        route,
        full_frame: [
            full_frame.x,
            full_frame.y,
            full_frame.width,
            full_frame.height,
        ],
        pixel_size_um: f64::from(geometry.pixel_size_um()),
        camera: diffusion_input(optics_recipe.camera_diffusion()),
        enlarger: diffusion_input(optics_recipe.enlarger_diffusion()),
    };
    let diffusion = catch_unwind(AssertUnwindSafe(|| {
        diffusion_frame::FrameSet::new(diffusion_input)
    }))
    .map_err(|_| Error {
        diagnostic: "diffusion frame preparation panicked".to_owned(),
    })?
    .map_err(optics_error)?;

    let halation_input = optics::HalationFrameInput::from(optics_recipe.scatter_halation());
    let halation = catch_unwind(AssertUnwindSafe(|| {
        optics::HalationFrame::new(halation_input, geometry.pixel_size_um())
    }))
    .map_err(|_| Error {
        diagnostic: "scatter/halation frame preparation panicked".to_owned(),
    })?
    .map_err(halation_error)?;

    let mut projected_optics = sys::FjOptics {
        flags: 0,
        full_frame: extent(geometry.domain()),
        camera: empty_diffusion(),
        enlarger: empty_diffusion(),
        diffusion_hash: 0,
        scatter_halation: empty_halation(),
        highlight_boost: sys::FjHighlightBoost {
            recipe_hash: 0,
            protection: 0.0,
            sharpness: 0.0,
            peak_gain: 0.0,
        },
    };
    if let Some(frame) = diffusion {
        projected_optics.full_frame = sys::FjExtent {
            x: frame.full_frame()[0],
            y: frame.full_frame()[1],
            width: frame.full_frame()[2],
            height: frame.full_frame()[3],
        };
        projected_optics.diffusion_hash = frame.hash();
        if let Some(camera) = frame.camera() {
            projected_optics.flags |= sys::FJ_OPTICS_CAMERA_DIFFUSION;
            projected_optics.camera = diffusion_stage(camera)?;
        }
        if let Some(enlarger) = frame.enlarger() {
            projected_optics.flags |= sys::FJ_OPTICS_ENLARGER_DIFFUSION;
            projected_optics.enlarger = diffusion_stage(enlarger)?;
        }
    } else {
        projected_optics.full_frame = sys::FjExtent {
            x: 0,
            y: 0,
            width: 0,
            height: 0,
        };
    }
    if let Some(frame) = halation {
        projected_optics.flags |= sys::FJ_OPTICS_SCATTER_HALATION;
        projected_optics.scatter_halation = halation_projection(&frame);
    }
    if let Some(boost) = optics_recipe.highlight_boost() {
        projected_optics.full_frame = full_frame;
        projected_optics.flags |= sys::FJ_OPTICS_HIGHLIGHT_BOOST;
        projected_optics.highlight_boost = sys::FjHighlightBoost {
            recipe_hash: boost.hash(),
            protection: boost.protection(),
            sharpness: boost.sharpness(),
            peak_gain: boost.peak_gain(),
        };
    }

    Ok(super::edge::PreparedFramePolicy {
        geometry: crate::cuda::geometry::FrameGeometry {
            full_frame,
            pixel_size_um: geometry.pixel_size_um(),
        },
        optics: projected_optics,
    })
}
