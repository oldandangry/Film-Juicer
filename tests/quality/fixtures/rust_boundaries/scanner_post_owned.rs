use film_juicer_core::scanner;
fn owned() -> scanner::PostEffects {
    let post = {
        let input = scanner::PostEffectsParams {
            route: film_juicer_core::route::ScanRoute::NegativePrint,
            glare_active: true,
            glare_percent: 2.0,
            glare_roughness: -0.0,
            glare_blur_sigma_px: 0.7,
            lens_blur_sigma_px: 1.25,
            unsharp_sigma_px: 0.7,
            unsharp_amount: 0.8,
        };
        scanner::describe_post_effects(input).unwrap()
    };
    let _later = scanner::describe_post_effects(scanner::PostEffectsParams {
        route: film_juicer_core::route::ScanRoute::NegativePrint,
        glare_active: true,
        glare_percent: 2.0,
        glare_roughness: -0.0,
        glare_blur_sigma_px: 0.7,
        lens_blur_sigma_px: 1.25,
        unsharp_sigma_px: 0.7,
        unsharp_amount: 0.8,
    });
    post
}
