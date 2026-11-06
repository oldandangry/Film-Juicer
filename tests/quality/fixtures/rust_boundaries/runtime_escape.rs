use crate::runtime::Runtime;
use film_juicer_core::assets::Assets;

fn escape_runtime_assets<'a>(runtime: Runtime) -> &'a Assets {
    runtime.assets()
}
