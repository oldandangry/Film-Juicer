use film_juicer_core::scanner;
fn complete() -> scanner::PostEffects {
    let post = scanner::describe_post_effects(scanner::PostEffectsParams {
        route: film_juicer_core::route::ScanRoute::NegativePrint,
        glare_active: true,
        glare_percent: 2.0,
        glare_roughness: -0.0,
        glare_blur_sigma_px: 0.7,
        lens_blur_sigma_px: 1.25,
        unsharp_sigma_px: 0.7,
        unsharp_amount: 0.8,
    })
    .unwrap();
    let _: &[f32; 6] = post.fields().values();
    let _: &[i32; 3] = post.fields().radii();
    let _: u32 = post.fields().schema();
    let _: u64 = post.hash();
    let _: bool = post.active();
    post
}
