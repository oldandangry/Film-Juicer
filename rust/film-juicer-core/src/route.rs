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
        for (polarity, selection, expected) in cases {
            assert_eq!(resolve(polarity, selection), expected);
        }
    }
}
