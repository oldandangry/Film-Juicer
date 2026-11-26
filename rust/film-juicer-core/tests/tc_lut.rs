use film_juicer_core::reconstruction::{self, Input, Method, TcError};

fn constant_lut(negative: bool) -> reconstruction::FilmTcLut {
    let mut spectra = vec![0.0; reconstruction::SPECTRA_SAMPLE_COUNT];
    for value in spectra.iter_mut().step_by(81) {
        *value = if negative { -1.0 } else { 1.0 };
    }
    let sensitivity = [[1.0, 2.0, 4.0]; 81];
    let spd = [1.0; 81];
    reconstruction::build_tc_lut(Input {
        spectra: spectra.as_slice().try_into().unwrap(),
        sensitivity_rgb: &sensitivity,
        reference_illuminant: &spd,
        projection_white_xyz: [f32::NAN; 3],
        method: Method::Hanatos {
            spectral_blur: 0.0,
            surface_rgb: None,
        },
        input_compression_active: false,
        input_hull: None,
    })
    .unwrap()
}
#[test]
fn owned_result_survives_source_expiry_and_preserves_signed_exposure() {
    let polarities = [(false, "negative"), (true, "positive")];
    let cases_ids: Vec<_> = IntoIterator::into_iter(polarities.iter().map(|(_, id)| *id))
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !cases_ids.is_empty(),
        "empty case set: {}",
        "tc-lut-polarity-results"
    );
    let cases_count = cases_ids.len();
    let mut cases_remaining = cases_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        cases_remaining.len(),
        cases_count,
        "duplicate case IDs: {}",
        "tc-lut-polarity-results"
    );
    for (negative, id) in polarities {
        let row = AsRef::<str>::as_ref(&(id)).to_owned();
        let lut = constant_lut(negative);
        let expected = if negative {
            [-1.0_f32, -2.0, -4.0]
        } else {
            [1.0_f32, 2.0, 4.0]
        };
        for cell in lut.samples().as_chunks::<4>().0 {
            for c in 0..3 {
                assert_eq!(cell[c].to_bits(), expected[c].to_bits());
            }
            assert_eq!(cell[3].to_bits(), 0);
        }
        assert_eq!(
            reconstruction::sample_tc_lut(lut.samples(), [1.0, 0.0, 0.0]).unwrap(),
            expected
        );
        std::thread::scope(|scope| {
            for _ in 0..2 {
                scope.spawn(|| {
                    assert_eq!(
                        reconstruction::sample_tc_lut(lut.samples(), [1.0, 0.0, 0.0]).unwrap(),
                        expected
                    )
                });
            }
        });
        std::thread::spawn(move || drop(lut)).join().unwrap();
        assert!(
            cases_remaining.remove(&row),
            "unexpected or repeated comparison: {row}"
        );
    }
    assert!(
        cases_remaining.is_empty(),
        "missing comparisons: {cases_remaining:?}"
    );
}
#[test]
fn consuming_transfer_preserves_original_allocation() {
    let lut = constant_lut(false);
    let address = lut.samples().as_ptr();
    let samples = lut.into_samples();
    assert_eq!(samples.as_ptr(), address);
    assert_eq!(samples.len(), reconstruction::TC_LUT_SAMPLE_COUNT);
    assert!(samples.capacity() >= samples.len());
}
#[test]
fn coordinate_failure_does_not_reject_defined_brightness_overflow() {
    let lut = constant_lut(false);
    assert_eq!(
        reconstruction::sample_tc_lut(lut.samples(), [-f32::MAX, f32::MAX, 0.0]),
        Err(TcError::UnusableSamplingCoordinates)
    );
    assert_eq!(
        reconstruction::sample_tc_lut(lut.samples(), [f32::MAX; 3]),
        Ok([0.0; 3])
    );
    assert_eq!(
        reconstruction::sample_tc_lut(lut.samples(), [f32::INFINITY, f32::NAN, f32::NEG_INFINITY]),
        Ok([0.0; 3])
    );
}
