use film_juicer_core::scanner;
fn mutate(post: scanner::PostEffects) {
    post.fields().values()[0] = 0.0;
}
