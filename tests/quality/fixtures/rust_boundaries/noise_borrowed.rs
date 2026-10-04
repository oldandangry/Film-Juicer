use crate::asset_noise::{NoiseOwner, NoiseView};
pub fn escape(noise_owner: NoiseOwner) -> NoiseView<'static> { noise_owner.view() }
pub fn release_while_borrowed(noise_owner: NoiseOwner) -> usize {
    let view = noise_owner.view();
    drop(noise_owner);
    view.stbn.len()
}
