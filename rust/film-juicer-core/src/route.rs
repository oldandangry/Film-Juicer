//! Profile polarity and direct/print route resolution.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapturePolarity {
    Negative,
    Positive,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteSelection {
    Direct,
    Print,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScanRoute {
    NegativeDirect,
    NegativePrint,
    PositiveDirect,
    PositivePrint,
}

pub fn resolve(capture_polarity: CapturePolarity, selection: RouteSelection) -> ScanRoute {
    match (capture_polarity, selection) {
        (CapturePolarity::Negative, RouteSelection::Direct) => ScanRoute::NegativeDirect,
        (CapturePolarity::Negative, RouteSelection::Print) => ScanRoute::NegativePrint,
        (CapturePolarity::Positive, RouteSelection::Direct) => ScanRoute::PositiveDirect,
        (CapturePolarity::Positive, RouteSelection::Print) => ScanRoute::PositivePrint,
    }
}

#[cfg(test)]
mod tests {
    use super::{CapturePolarity, RouteSelection, ScanRoute, resolve};

    #[test]
    fn resolves_all_polarity_and_selection_pairs() {
        let cases = [
            (
                CapturePolarity::Negative,
                RouteSelection::Direct,
                ScanRoute::NegativeDirect,
            ),
            (
                CapturePolarity::Negative,
                RouteSelection::Print,
                ScanRoute::NegativePrint,
            ),
            (
                CapturePolarity::Positive,
                RouteSelection::Direct,
                ScanRoute::PositiveDirect,
            ),
            (
                CapturePolarity::Positive,
                RouteSelection::Print,
                ScanRoute::PositivePrint,
            ),
        ];
        let ids = [
            "negative/direct",
            "negative/print",
            "positive/direct",
            "positive/print",
        ];
        let row_cases_ids: Vec<_> = IntoIterator::into_iter(ids)
            .map(|id| AsRef::<str>::as_ref(&id).to_owned())
            .collect();
        assert!(!row_cases_ids.is_empty(), "route-resolution");
        let row_cases_count = row_cases_ids.len();
        let mut row_cases_remaining = row_cases_ids
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            row_cases_remaining.len(),
            row_cases_count,
            "route-resolution"
        );
        for ((polarity, selection, expected), id) in cases.into_iter().zip(ids) {
            let case = AsRef::<str>::as_ref(&(id)).to_owned();
            assert_eq!(resolve(polarity, selection), expected);
            assert!(
                row_cases_remaining.remove(&case),
                "unexpected or repeated comparison: {case}"
            );
        }
        assert!(
            row_cases_remaining.is_empty(),
            "missing comparisons: {row_cases_remaining:?}"
        );
    }
}
