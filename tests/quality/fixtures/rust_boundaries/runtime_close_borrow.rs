use crate::runtime::Runtime;
use film_juicer_core::assets::Assets;

fn close_runtime_while_assets_are_borrowed(runtime: Runtime) -> usize {
    let assets: &Assets = runtime.assets();
    let _ = runtime.close();
    std::mem::size_of_val(assets)
}
