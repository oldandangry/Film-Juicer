//! Default image-effect actions and their ordered host reads.
#![forbid(unsafe_code)]

use super::host::{Callback, Status};

#[derive(Clone, Copy)]
pub(super) enum DefaultAction {
    BeginSequence,
    EndSequence,
    Identity,
    RegionOfDefinition,
    RegionsOfInterest,
    FramesNeeded,
    ClipPreferences,
    TimeDomain,
    PurgeCaches,
    SyncPrivateData,
    BeginEdit,
    EndEdit,
}

impl DefaultAction {
    pub(super) fn from_bytes(action: &[u8]) -> Option<Self> {
        Some(match action {
            b"OfxImageEffectActionBeginSequenceRender" => Self::BeginSequence,
            b"OfxImageEffectActionEndSequenceRender" => Self::EndSequence,
            b"OfxImageEffectActionIsIdentity" => Self::Identity,
            b"OfxImageEffectActionGetRegionOfDefinition" => Self::RegionOfDefinition,
            b"OfxImageEffectActionGetRegionsOfInterest" => Self::RegionsOfInterest,
            b"OfxImageEffectActionGetFramesNeeded" => Self::FramesNeeded,
            b"OfxImageEffectActionGetClipPreferences" => Self::ClipPreferences,
            b"OfxImageEffectActionGetTimeDomain" => Self::TimeDomain,
            b"OfxActionPurgeCaches" => Self::PurgeCaches,
            b"OfxActionSyncPrivateData" => Self::SyncPrivateData,
            b"OfxActionBeginInstanceEdit" => Self::BeginEdit,
            b"OfxActionEndInstanceEdit" => Self::EndEdit,
            _ => return None,
        })
    }

    pub(super) fn requires_input(self) -> bool {
        matches!(
            self,
            Self::BeginSequence
                | Self::EndSequence
                | Self::Identity
                | Self::RegionOfDefinition
                | Self::RegionsOfInterest
                | Self::FramesNeeded
        )
    }

    pub(super) fn requires_output(self) -> bool {
        matches!(
            self,
            Self::Identity
                | Self::RegionOfDefinition
                | Self::RegionsOfInterest
                | Self::FramesNeeded
                | Self::ClipPreferences
                | Self::TimeDomain
        )
    }
}

pub(super) fn run(callback: &Callback<'_>, action: DefaultAction) -> Result<(), Status> {
    callback.instance()?.require_live()?;
    match action {
        DefaultAction::BeginSequence | DefaultAction::EndSequence => {
            if matches!(action, DefaultAction::BeginSequence) {
                callback.action_double(c"OfxImageEffectPropFrameRange", 0)?;
                callback.action_double(c"OfxImageEffectPropFrameRange", 1)?;
                callback.action_double(c"OfxImageEffectPropFrameStep", 0)?;
            }
            scale(callback)?;
            callback.optional_int(c"OfxImageEffectPropOpenCLEnabled");
            callback.optional_int(c"OfxImageEffectPropCudaEnabled");
            callback.optional_int(c"OfxImageEffectPropMetalEnabled");
            callback.optional_pointer(c"OfxImageEffectPropOpenCLCommandQueue");
            callback.optional_pointer(c"OfxImageEffectPropCudaStream");
            callback.optional_pointer(c"OfxImageEffectPropMetalCommandQueue");
            callback.action_int(c"OfxPropIsInteractive", 0)?;
            callback.optional_int(c"OfxImageEffectPropSequentialRenderStatus");
            callback.optional_int(c"OfxImageEffectPropInteractiveRenderStatus");
        }
        DefaultAction::Identity => {
            callback.action_double(c"OfxPropTime", 0)?;
            scale(callback)?;
            for index in 0..4 {
                callback.action_int(c"OfxImageEffectPropRenderWindow", index)?;
            }
            callback.action_field()?;
        }
        DefaultAction::RegionOfDefinition => {
            scale(callback)?;
            callback.action_double(c"OfxPropTime", 0)?;
        }
        DefaultAction::RegionsOfInterest => {
            scale(callback)?;
            for index in 0..4 {
                callback.action_double(c"OfxImageEffectPropRegionOfInterest", index)?;
            }
            callback.action_double(c"OfxPropTime", 0)?;
        }
        DefaultAction::FramesNeeded => {
            callback.action_double(c"OfxPropTime", 0)?;
        }
        DefaultAction::ClipPreferences
        | DefaultAction::TimeDomain
        | DefaultAction::PurgeCaches
        | DefaultAction::SyncPrivateData
        | DefaultAction::BeginEdit
        | DefaultAction::EndEdit => {}
    }
    Ok(())
}

fn scale(callback: &Callback<'_>) -> Result<(), Status> {
    callback.action_double(c"OfxImageEffectPropRenderScale", 0)?;
    callback.action_double(c"OfxImageEffectPropRenderScale", 1)?;
    Ok(())
}
