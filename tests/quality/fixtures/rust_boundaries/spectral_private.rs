use film_juicer_core::spectral::{Tables, White};
pub fn forged_tables(tables: Tables) -> Tables { Tables { ..tables } }
pub fn forged_white(white: White) -> White { White { ..white } }
