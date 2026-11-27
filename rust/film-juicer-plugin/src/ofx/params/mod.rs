//! Filter-context clip definitions and ordered parameter groups.

#![forbid(unsafe_code)]

use std::ffi::CStr;
use film_juicer_core::profile::{Catalog, Role};

use super::host::{Callback, Parameters, Properties, Status};
use super::param_id::{Kind, ParamId};
use super::{ofx_str, sys};

mod color;
mod grain;
mod halation;
mod spatial;

const DEFAULT: &CStr = ofx_str(sys::kOfxParamPropDefault);
const MIN: &CStr = ofx_str(sys::kOfxParamPropMin);
const MAX: &CStr = ofx_str(sys::kOfxParamPropMax);
const DISPLAY_MIN: &CStr = ofx_str(sys::kOfxParamPropDisplayMin);
const DISPLAY_MAX: &CStr = ofx_str(sys::kOfxParamPropDisplayMax);
const EVALUATE_ON_CHANGE: &CStr = ofx_str(sys::kOfxParamPropEvaluateOnChange);
const INCREMENT: &CStr = ofx_str(sys::kOfxParamPropIncrement);
const ANIMATES: &CStr = ofx_str(sys::kOfxParamPropAnimates);
const DIMENSION_LABEL: &CStr = ofx_str(sys::kOfxParamPropDimensionLabel);
const SECRET: &CStr = ofx_str(sys::kOfxParamPropSecret);

// Property writes retain authored order, including the optional hint and
// group-open writes. Multi-component values use increasing OFX indices.
#[derive(Clone, Copy)]
enum Property {
    Parent(&'static CStr),
    Hint(&'static CStr),
    Closed,
    Int(&'static CStr, i32),
    Double(&'static CStr, &'static [f64]),
    Text(&'static CStr, &'static [&'static CStr]),
    Choices(&'static [&'static CStr]),
    StringChoices(&'static [(&'static CStr, &'static CStr)]),
    Profiles(Role),
}

struct Descriptor {
    kind: Kind,
    name: &'static CStr,
    label: &'static CStr,
    properties: &'static [Property],
}

impl Descriptor {
    const fn group(
        name: &'static CStr,
        label: &'static CStr,
        properties: &'static [Property],
    ) -> Self {
        Self {
            kind: Kind::Group,
            name,
            label,
            properties,
        }
    }

    const fn parameter(id: ParamId, label: &'static CStr, properties: &'static [Property]) -> Self {
        Self {
            kind: id.kind(),
            name: id.name(),
            label,
            properties,
        }
    }
}

fn define(
    parameters: &Parameters<'_>,
    descriptors: &[Descriptor],
    catalog: Option<&Catalog>,
) -> Result<(), Status> {
    use Property::*;
    for descriptor in descriptors {
        let p = parameters.define(descriptor.kind.ofx(), descriptor.name)?;
        p.set_string(ofx_str(sys::kOfxPropLabel), 0, descriptor.label)?;
        for &property in descriptor.properties {
            match property {
                Parent(name) => p.set_string(ofx_str(sys::kOfxParamPropParent), 0, name)?,
                Hint(hint) => {
                    let _ = p.set_string(ofx_str(sys::kOfxParamPropHint), 0, hint);
                }
                Closed => {
                    let _ = p.set_int(ofx_str(sys::kOfxParamPropGroupOpen), 0, 0);
                }
                Int(key, value) => p.set_int(key, 0, value)?,
                Double(key, values) => {
                    for (index, &value) in values.iter().enumerate() {
                        p.set_double(key, index as i32, value)?;
                    }
                }
                Text(key, values) => {
                    for (index, &value) in values.iter().enumerate() {
                        p.set_string(key, index as i32, value)?;
                    }
                }
                Choices(labels) => {
                    for label in labels {
                        option(&p, label)?;
                    }
                }
                StringChoices(choices) => {
                    for &(key, label) in choices {
                        string_option(&p, key, label)?;
                    }
                }
                Profiles(role) => {
                    let catalog = catalog.ok_or(Status::FATAL)?;
                    let profiles = match role {
                        Role::Film => catalog.films(),
                        Role::Print => catalog.prints(),
                    };
                    for entry in profiles {
                        catalog_option(&p, entry.key(), entry.label())?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn append_index(properties: &Properties<'_>, property: &CStr) -> Result<i32, Status> {
    let index = properties.dimension(property)?;
    if index < 0 {
        Err(Status::BAD_INDEX)
    } else {
        Ok(index)
    }
}

fn append(properties: &Properties<'_>, property: &CStr, value: &CStr) -> Result<(), Status> {
    let index = append_index(properties, property)?;
    properties.set_string(property, index, value)
}

fn option(properties: &Properties<'_>, value: &CStr) -> Result<(), Status> {
    append(properties, ofx_str(sys::kOfxParamPropChoiceOption), value)
}

fn catalog_option(properties: &Properties<'_>, key: &str, label: &str) -> Result<(), Status> {
    let index = append_index(properties, ofx_str(sys::kOfxParamPropChoiceOption))?;
    let enum_index = append_index(properties, ofx_str(sys::kOfxParamPropChoiceEnum))?;
    if index != enum_index {
        return Err(Status::FAILED);
    }
    properties.set_catalog_string(ofx_str(sys::kOfxParamPropChoiceEnum), index, key)?;
    properties.set_catalog_string(ofx_str(sys::kOfxParamPropChoiceOption), index, label)
}

fn string_option(properties: &Properties<'_>, key: &CStr, label: &CStr) -> Result<(), Status> {
    let index = append_index(properties, ofx_str(sys::kOfxParamPropChoiceOption))?;
    let enum_index = append_index(properties, ofx_str(sys::kOfxParamPropChoiceEnum))?;
    if index != enum_index {
        return Err(Status::FAILED);
    }
    properties.set_string(ofx_str(sys::kOfxParamPropChoiceEnum), index, key)?;
    properties.set_string(ofx_str(sys::kOfxParamPropChoiceOption), index, label)
}

pub(super) fn describe(callback: &Callback<'_>, parameters: &Parameters<'_>) -> Result<(), Status> {
    let catalog = callback.catalog()?;
    let source = callback.define_clip(c"Source")?;
    append(
        &source,
        ofx_str(sys::kOfxImageEffectPropSupportedComponents),
        ofx_str(sys::kOfxImageComponentRGBA),
    )?;
    append(
        &source,
        ofx_str(sys::kOfxImageEffectPropSupportedComponents),
        ofx_str(sys::kOfxImageComponentRGB),
    )?;
    source.set_int(ofx_str(sys::kOfxImageEffectPropSupportsTiles), 0, 0)?;
    source.set_int(ofx_str(sys::kOfxImageClipPropOptional), 0, 0)?;
    let output = callback.define_clip(c"Output")?;
    append(
        &output,
        ofx_str(sys::kOfxImageEffectPropSupportedComponents),
        ofx_str(sys::kOfxImageComponentRGBA),
    )?;
    append(
        &output,
        ofx_str(sys::kOfxImageEffectPropSupportedComponents),
        ofx_str(sys::kOfxImageComponentRGB),
    )?;
    output.set_int(ofx_str(sys::kOfxImageEffectPropSupportsTiles), 0, 0)?;

    define(parameters, color::INPUT, None)?;
    let catalog = Some(catalog.as_deref().ok_or(Status::FATAL)?);
    for group in [
        color::INPUT_COLOR,
        color::PRINT,
        spatial::TUNING,
        spatial::SCANNER,
        halation::HALATION,
    ] {
        define(parameters, group, catalog)?;
    }
    halation::describe_advanced(parameters)?;
    define(parameters, grain::GRAIN, catalog)?;
    define(parameters, grain::FILM_MODEL, catalog)?;
    grain::describe_film(parameters)?;
    for group in [
        grain::GRAIN_CONTROLS,
        spatial::EFFECTS,
        spatial::DIFFUSION,
        color::GLARE,
        color::OUTPUT,
    ] {
        define(parameters, group, catalog)?;
    }
    Ok(())
}
