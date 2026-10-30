//! SF_TEMP_BRIDGE_FRAME_POLICY: early callback-local projection for the held render.
//! Remove with the final native preparation consumer, no later than S6.D.
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
    pub(super) category: u32,
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
        family: 0,
        scatter_fraction: 0.0,
        group_weights: [0.0; 3],
        group_lambda_um: [0.0; 3],
        warmth: 0.0,
        spatial_scale: 0.0,
        pixel_size_um: 0.0,
        radius_px: 0,
        sample_hash: 0,
        hash: 0,
    }
}

fn diffusion_stage(value: &diffusion_frame::Stage) -> sys::FjDiffusion {
    let fields = value.sample().fields();
    sys::FjDiffusion {
        family: fields.family,
        scatter_fraction: value.scatter_fraction(),
        group_weights: fields.group_weights,
        group_lambda_um: fields.group_lambda_um,
        warmth: fields.warmth,
        spatial_scale: fields.spatial_scale,
        pixel_size_um: fields.pixel_size_um,
        radius_px: fields.radius_px,
        sample_hash: fields.hash,
        hash: value.hash(),
    }
}

const EMPTY_HALATION_GAUSSIAN: sys::FjHalationGaussian = sys::FjHalationGaussian {
    kind: 0,
    radius: 0,
    fir_weights: [0.0; 19],
    feedforward: 0.0,
    feedback: [0.0; 3],
};

const EMPTY_HALATION_CHANNEL: sys::FjHalationChannel = sys::FjHalationChannel {
    core: EMPTY_HALATION_GAUSSIAN,
    tail: [EMPTY_HALATION_GAUSSIAN; 3],
    bounce: [EMPTY_HALATION_GAUSSIAN; 3],
    total_strength: 0.0,
};

fn halation_gaussian(value: &optics::HalationGaussian) -> sys::FjHalationGaussian {
    sys::FjHalationGaussian {
        kind: match value.kind() {
            optics::HalationGaussianKind::Identity => 0,
            optics::HalationGaussianKind::FirReflect => 1,
            optics::HalationGaussianKind::YvvReplicate => 2,
        },
        radius: value.radius(),
        fir_weights: *value.fir_weights(),
        feedforward: value.yvv()[0],
        feedback: [value.yvv()[1], value.yvv()[2], value.yvv()[3]],
    }
}

fn halation_projection(value: &optics::HalationFrame) -> sys::FjScatterHalation {
    sys::FjScatterHalation {
        recipe_hash: value.recipe_hash(),
        scatter_amount: value.scatter_amount(),
        channels: std::array::from_fn(|channel| {
            let source = &value.channels()[channel];
            sys::FjHalationChannel {
                core: halation_gaussian(source.core()),
                tail: source.tail().each_ref().map(halation_gaussian),
                bounce: source.bounce().each_ref().map(halation_gaussian),
                total_strength: source.total_strength(),
            }
        }),
    }
}

fn empty_halation() -> sys::FjScatterHalation {
    sys::FjScatterHalation {
        recipe_hash: 0,
        scatter_amount: 0.0,
        channels: [EMPTY_HALATION_CHANNEL; 3],
    }
}

fn optics_error(error: diffusion_frame::Error) -> Error {
    Error {
        category: sys::FJ_STATUS_PREPARATION_FAILURE,
        diagnostic: error.diagnostic().to_owned(),
    }
}

fn halation_error(error: optics::HalationFrameError) -> Error {
    let diagnostic = match error {
        optics::HalationFrameError::PixelSize => {
            "ResourceDescriptorMismatch component=scatter_halation field=pixel_size_um"
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
        category: sys::FJ_STATUS_PREPARATION_FAILURE,
        diagnostic: diagnostic.to_owned(),
    }
}

pub(super) fn prepare(
    state: &recipe::RenderState,
    full_frame: sys::FjExtent,
) -> Result<super::edge::FjPreparedFramePolicy, Error> {
    let geometry = catch_unwind(AssertUnwindSafe(|| {
        crate::frame_bridge::checked_geometry(
            domain(full_frame),
            state.input().film_format_long_edge_mm(),
        )
    }))
    .map_err(|_| Error {
        category: sys::FJ_STATUS_INTERNAL_FAILURE,
        diagnostic: "frame geometry preparation panicked".to_owned(),
    })?;

    let optics_recipe = state.optics();
    let route = route_tag(state.development().route());
    #[cfg(feature = "test-support")]
    let diffusion_fault = crate::diffusion_bridge::diffusion_frame_faults::take(3);
    #[cfg(not(feature = "test-support"))]
    let diffusion_fault = 0;
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
        crate::diffusion_bridge::checked_frame_set(diffusion_input, diffusion_fault)
    }))
    .map_err(|_| Error {
        category: sys::FJ_STATUS_INTERNAL_FAILURE,
        diagnostic: "diffusion frame preparation panicked".to_owned(),
    })?
    .map_err(optics_error)?;

    #[cfg(feature = "test-support")]
    let halation_fault = crate::optics_bridge::halation_frame_faults::take();
    #[cfg(not(feature = "test-support"))]
    let halation_fault = 0;
    let halation_input = optics::HalationFrameInput::from(optics_recipe.scatter_halation());
    let halation = catch_unwind(AssertUnwindSafe(|| {
        crate::optics_bridge::checked_halation_frame(
            halation_input,
            geometry.pixel_size_um(),
            halation_fault,
        )
    }))
    .map_err(|_| Error {
        category: sys::FJ_STATUS_INTERNAL_FAILURE,
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
            projected_optics.camera = diffusion_stage(camera);
        }
        if let Some(enlarger) = frame.enlarger() {
            projected_optics.flags |= sys::FJ_OPTICS_ENLARGER_DIFFUSION;
            projected_optics.enlarger = diffusion_stage(enlarger);
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

    Ok(super::edge::FjPreparedFramePolicy {
        geometry: crate::frame_bridge::FjFrameGeometry {
            full_frame,
            pixel_size_um: geometry.pixel_size_um(),
        },
        optics: projected_optics,
        recipe_hash: state.hash(),
    })
}
