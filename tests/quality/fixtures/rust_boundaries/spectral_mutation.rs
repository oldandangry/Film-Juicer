use film_juicer_core::spectral::{Tables, White};
pub fn mutate_tables(tables: &mut Tables) { tables.white_xyz[0] = 1.0; }
pub fn mutate_white(white: &mut White) { white.xyz[0] = 1.0; }
