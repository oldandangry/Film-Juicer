//! Miss-local reflected-YVV boundary construction. Device residency and the
//! final frame descriptor identity remain native responsibilities.

use crate::hash;

#[derive(Clone, Copy, Debug)]
pub struct Input {
    coefficients: [f64; 9],
    width: i32,
    height: i32,
    accuracy: f64,
    tolerance: f64,
    version: u32,
    sigma_px: f32,
}

impl Input {
    /// Literal coefficients are normalized feedforward/feedback[3], then the
    /// unnormalized denominator, feedforward numerator and feedback numerators[3].
    /// Eligibility (YVV and positive component weight) belongs to the caller.
    pub fn new(
        coefficients: [f64; 9],
        width: i32,
        height: i32,
        accuracy: f64,
        tolerance: f64,
        version: u32,
        sigma_px: f32,
    ) -> Result<Self, Error> {
        if width <= 0 || height <= 0 {
            return Err(Error::new(Reason::Request));
        }
        Ok(Self {
            coefficients,
            width,
            height,
            accuracy,
            tolerance,
            version,
            sigma_px,
        })
    }

    /// Identity alone promises no policy admission or boundary construction.
    pub fn key(&self) -> Key {
        let mut value = hash::FNV_OFFSET;
        hash::update_bytes(&mut value, &self.version.to_le_bytes());
        for coefficient in self.coefficients {
            hash::update_bytes(&mut value, &coefficient.to_bits().to_le_bytes());
        }
        hash::update_bytes(&mut value, &self.width.to_le_bytes());
        hash::update_bytes(&mut value, &self.height.to_le_bytes());
        hash::update_bytes(&mut value, &self.accuracy.to_bits().to_le_bytes());
        hash::update_bytes(&mut value, &self.tolerance.to_bits().to_le_bytes());
        Key(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Key(u64);

impl Key {
    pub fn value(self) -> u64 {
        self.0
    }
}

#[derive(Debug)]
pub struct Axis {
    length: i32,
    terminal_size: i32,
    terminal: [f64; 9],
}

impl Axis {
    pub fn length(&self) -> i32 {
        self.length
    }
    pub fn terminal_size(&self) -> i32 {
        self.terminal_size
    }
    pub fn terminal(&self) -> &[f64; 9] {
        &self.terminal
    }
}

#[derive(Debug)]
pub struct Boundary {
    weights: Vec<f64>,
    horizontal: Axis,
    vertical: Axis,
    width: i32,
    height: i32,
    key: Key,
}

impl Boundary {
    pub fn new(input: Input) -> Result<Self, Error> {
        let (horizontal, horizontal_weights) = axis(&input, input.width, 1)?;
        let (vertical, vertical_weights) = axis(&input, input.height, 2)?;
        let count = horizontal_weights
            .len()
            .checked_add(vertical_weights.len())
            .filter(|&count| addressable::<f64>(count))
            .ok_or_else(|| Error::new(Reason::StorageSize))?;
        let mut weights = reserve(count, 0.0, 7).map_err(Error::new)?;
        weights[..horizontal_weights.len()].copy_from_slice(&horizontal_weights);
        weights[horizontal_weights.len()..].copy_from_slice(&vertical_weights);
        Ok(Self {
            weights,
            horizontal,
            vertical,
            width: input.width,
            height: input.height,
            key: input.key(),
        })
    }
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
    pub fn horizontal(&self) -> &Axis {
        &self.horizontal
    }
    pub fn vertical(&self) -> &Axis {
        &self.vertical
    }
    pub fn width(&self) -> i32 {
        self.width
    }
    pub fn height(&self) -> i32 {
        self.height
    }
    pub fn key(&self) -> Key {
        self.key
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Reason {
    Request = 2,
    AxisSize = 3,
    Policy = 4,
    CausalSingular = 5,
    NonfiniteWeights = 6,
    CausalCertification = 7,
    TailEscape = 8,
    TerminalSingular = 9,
    TerminalCertification = 10,
    StorageSize = 11,
    Allocation = 12,
}

#[derive(Clone, Copy, Debug)]
pub struct Error {
    reason: Reason,
    axis: u32,
    row: u32,
    length: i32,
    sigma_px: f32,
    sum: f64,
    tolerance: f64,
}

impl Error {
    fn new(reason: Reason) -> Self {
        Self {
            reason,
            axis: 0,
            row: 3,
            length: 0,
            sigma_px: 0.0,
            sum: 0.0,
            tolerance: 0.0,
        }
    }
    fn on_axis(reason: Reason, axis: u32, length: i32) -> Self {
        Self {
            axis,
            length,
            ..Self::new(reason)
        }
    }
    pub fn reason(&self) -> Reason {
        self.reason
    }
    pub fn axis(&self) -> u32 {
        self.axis
    }
    pub fn row(&self) -> u32 {
        self.row
    }
    pub fn length(&self) -> i32 {
        self.length
    }
    pub fn sigma_px(&self) -> f32 {
        self.sigma_px
    }
    pub fn sum(&self) -> f64 {
        self.sum
    }
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }
}

#[derive(Clone, Copy, Default)]
struct Dd {
    hi: f64,
    lo: f64,
}

impl Dd {
    const ZERO: Self = Self { hi: 0.0, lo: 0.0 };
    const ONE: Self = Self { hi: 1.0, lo: 0.0 };
    fn from(value: f64) -> Self {
        Self { hi: value, lo: 0.0 }
    }
    fn normalize(hi: f64, lo: f64) -> Self {
        let sum = hi + lo;
        Self {
            hi: sum,
            lo: lo - (sum - hi),
        }
    }
    fn add(self, rhs: Self) -> Self {
        let sum = self.hi + rhs.hi;
        let virtual_rhs = sum - self.hi;
        let error = (self.hi - (sum - virtual_rhs)) + (rhs.hi - virtual_rhs) + self.lo + rhs.lo;
        Self::normalize(sum, error)
    }
    fn negate(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
    fn subtract(self, rhs: Self) -> Self {
        self.add(rhs.negate())
    }
    fn multiply(self, rhs: Self) -> Self {
        let product = self.hi * rhs.hi;
        // This is the original explicit fused product residual, not contraction.
        let error = libm::fma(self.hi, rhs.hi, -product)
            + self.hi * rhs.lo
            + self.lo * rhs.hi
            + self.lo * rhs.lo;
        Self::normalize(product, error)
    }
    fn divide(self, denominator: Self) -> Self {
        let first = self.hi / denominator.hi;
        let remainder = self.subtract(denominator.multiply(Self::from(first)));
        let second = (remainder.hi + remainder.lo) / denominator.hi;
        Self::from(first).add(Self::from(second))
    }
    fn value(self) -> f64 {
        self.hi + self.lo
    }
}

type Matrix = [[Dd; 3]; 3];
type Vector = [Dd; 3];

fn identity() -> Matrix {
    let mut result = [[Dd::ZERO; 3]; 3];
    for (index, row) in result.iter_mut().enumerate() {
        row[index] = Dd::ONE;
    }
    result
}

fn multiply(lhs: &Matrix, rhs: &Matrix) -> Matrix {
    let mut result = [[Dd::ZERO; 3]; 3];
    for (row, output) in result.iter_mut().enumerate() {
        for (column, value) in output.iter_mut().enumerate() {
            for (inner, operand) in lhs[row].iter().enumerate() {
                *value = value.add(operand.multiply(rhs[inner][column]));
            }
        }
    }
    result
}

fn transform(matrix: &Matrix, vector: &Vector) -> Vector {
    let mut result = [Dd::ZERO; 3];
    for (row, value) in result.iter_mut().enumerate() {
        for (column, operand) in vector.iter().enumerate() {
            *value = value.add(matrix[row][column].multiply(*operand));
        }
    }
    result
}

fn invert(mut matrix: Matrix, size: usize) -> Option<Matrix> {
    let mut inverse = identity();
    for column in 0..size {
        let mut pivot = column;
        let mut pivot_magnitude = matrix[pivot][column].value().abs();
        for (row, entries) in matrix.iter().enumerate().take(size).skip(column + 1) {
            let magnitude = entries[column].value().abs();
            if magnitude > pivot_magnitude {
                pivot = row;
                pivot_magnitude = magnitude;
            }
        }
        if !(pivot_magnitude > 0.0 && pivot_magnitude.is_finite()) {
            return None;
        }
        if pivot != column {
            matrix.swap(pivot, column);
            inverse.swap(pivot, column);
        }
        let scale = matrix[column][column];
        for entry in 0..size {
            matrix[column][entry] = matrix[column][entry].divide(scale);
            inverse[column][entry] = inverse[column][entry].divide(scale);
        }
        for row in 0..size {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            for entry in 0..size {
                matrix[row][entry] =
                    matrix[row][entry].subtract(factor.multiply(matrix[column][entry]));
                inverse[row][entry] =
                    inverse[row][entry].subtract(factor.multiply(inverse[column][entry]));
            }
        }
    }
    Some(inverse)
}

fn addressable<T>(count: usize) -> bool {
    count
        .checked_mul(std::mem::size_of::<T>())
        .is_some_and(|bytes| bytes <= isize::MAX as usize)
}

fn reserve<T: Clone>(count: usize, initial: T, site: u32) -> Result<Vec<T>, Reason> {
    #[cfg(feature = "test-support")]
    if test_support::reject(site) {
        return Err(Reason::Allocation);
    }
    #[cfg(not(feature = "test-support"))]
    let _ = site;
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| Reason::Allocation)?;
    result.resize(count, initial);
    Ok(result)
}

fn retained_count(length: usize) -> Option<usize> {
    // Root evaluates 3*length in int; CUDA indexes history*length+sample in int.
    length
        .checked_mul(3)
        .filter(|&count| count <= i32::MAX as usize && addressable::<f64>(count))
}

fn axis(input: &Input, length: i32, direction: u32) -> Result<(Axis, Vec<f64>), Error> {
    let error = |reason| Error::on_axis(reason, direction, length);
    if length <= 0 || !(input.accuracy > 0.0 && input.tolerance > 0.0) || input.version != 1 {
        return Err(error(Reason::Policy));
    }
    let period = length
        .checked_mul(2)
        .ok_or_else(|| error(Reason::AxisSize))? as usize;
    let length = length as usize;
    let count = length
        .checked_mul(3)
        .filter(|&count| addressable::<f64>(count))
        .ok_or_else(|| error(Reason::AxisSize))?;
    if !addressable::<Vector>(period) {
        return Err(error(Reason::AxisSize));
    }
    let mut transition = [[Dd::ZERO; 3]; 3];
    for (coefficient, value) in transition[0].iter_mut().enumerate() {
        *value = Dd::from(input.coefficients[coefficient + 1]);
    }
    transition[1][0] = Dd::ONE;
    transition[2][1] = Dd::ONE;
    let site = if direction == 1 { 1 } else { 4 };
    let mut powers = reserve(period, [Dd::ZERO; 3], site).map_err(error)?;
    powers[0][0] = Dd::from(input.coefficients[0]);
    for exponent in 1..period {
        powers[exponent] = transform(&transition, &powers[exponent - 1]);
    }
    let mut period_power = identity();
    for _ in 0..period {
        period_power = multiply(&transition, &period_power);
    }
    let mut periodic_system = identity();
    for (row, entries) in periodic_system.iter_mut().enumerate() {
        for (column, entry) in entries.iter_mut().enumerate() {
            *entry = entry.subtract(period_power[row][column]);
        }
    }
    let periodic_inverse =
        invert(periodic_system, 3).ok_or_else(|| error(Reason::CausalSingular))?;
    let mut full_weights = reserve(count, 0.0, site + 1).map_err(error)?;
    let mut sums = [Dd::ZERO; 3];
    for sample in 0..length {
        let mut rhs = [Dd::ZERO; 3];
        for (row, entry) in rhs.iter_mut().enumerate() {
            *entry = powers[sample][row].add(powers[period - 1 - sample][row]);
        }
        let state = transform(&periodic_inverse, &rhs);
        for row in 0..3 {
            let value = state[row].value();
            if !value.is_finite() {
                return Err(error(Reason::NonfiniteWeights));
            }
            full_weights[row * length + sample] = value;
            sums[row] = sums[row].add(Dd::from(value));
        }
    }
    for (row, sum) in sums.iter().enumerate() {
        let sum = sum.value();
        if !sum.is_finite() || (sum - 1.0).abs() > input.tolerance {
            return Err(Error {
                row: row as u32,
                sigma_px: input.sigma_px,
                sum,
                tolerance: input.tolerance,
                ..error(Reason::CausalCertification)
            });
        }
    }
    let minimum = length.min(3);
    let mut retained = length;
    let mut tails = [0.0_f64; 3];
    for sample in (minimum..length).rev() {
        for row in 0..3 {
            tails[row] += full_weights[row * length + sample].abs();
        }
        let maximum = tails[0].max(tails[1]).max(tails[2]);
        if maximum >= input.accuracy {
            retained = sample + 1;
            break;
        }
        retained = sample;
    }
    let retained_count = retained_count(retained).ok_or_else(|| error(Reason::AxisSize))?;
    let mut weights = reserve(retained_count, 0.0, site + 2).map_err(error)?;
    for row in 0..3 {
        weights[row * retained..(row + 1) * retained]
            .copy_from_slice(&full_weights[row * length..row * length + retained]);
    }
    let size = length.min(3);
    let mut terminal_system = [[Dd::ZERO; 3]; 3];
    for (diagonal, row) in terminal_system.iter_mut().enumerate().take(size) {
        row[diagonal] = Dd::from(input.coefficients[4]);
    }
    for (row, entries) in terminal_system.iter_mut().enumerate().take(size) {
        let output_index = (length - size + row) as i64;
        for coefficient in 0..3 {
            let reflected = reflect(output_index + coefficient as i64 + 1, length as i64);
            let column = reflected - (length - size) as i64;
            if column < 0 || column >= size as i64 {
                return Err(error(Reason::TailEscape));
            }
            entries[column as usize] =
                entries[column as usize].subtract(Dd::from(input.coefficients[6 + coefficient]));
        }
    }
    let terminal_inverse =
        invert(terminal_system, size).ok_or_else(|| error(Reason::TerminalSingular))?;
    let mut terminal = [0.0; 9];
    for (row, entries) in terminal_inverse.iter().enumerate().take(size) {
        let mut sum = 0.0;
        for (column, entry) in entries.iter().enumerate().take(size) {
            let value = Dd::from(input.coefficients[5]).multiply(*entry).value();
            terminal[row * size + column] = value;
            sum += value;
        }
        if !sum.is_finite() || (sum - 1.0).abs() > input.tolerance {
            return Err(error(Reason::TerminalCertification));
        }
    }
    Ok((
        Axis {
            length: retained as i32,
            terminal_size: size as i32,
            terminal,
        },
        weights,
    ))
}

fn reflect(index: i64, length: i64) -> i64 {
    let period = 2 * length;
    let mut wrapped = index % period;
    if wrapped < 0 {
        wrapped += period;
    }
    if wrapped < length {
        wrapped
    } else {
        period - 1 - wrapped
    }
}

#[cfg(feature = "test-support")]
pub mod test_support {
    use std::cell::Cell;
    thread_local! { static RESERVATION: Cell<u32> = const { Cell::new(0) }; }
    pub struct Reservation;
    impl Reservation {
        pub fn new(site: u32) -> Self {
            RESERVATION.set(site);
            Self
        }
    }
    impl Drop for Reservation {
        fn drop(&mut self) {
            RESERVATION.set(0);
        }
    }
    pub(super) fn reject(site: u32) -> bool {
        RESERVATION.get() == site
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    fn scalar(value: &Value) -> f64 {
        f64::from_bits(value.as_u64().unwrap())
    }
    fn input(row: &Value) -> Result<Input, Error> {
        let c = &row["component"];
        if !(c["operator"] == 3 && f32::from_bits(c["weight"].as_u64().unwrap() as u32) > 0.0) {
            return Err(Error::new(Reason::Request));
        }
        Input::new(
            [
                scalar(&c["feedforward"]),
                scalar(&c["feedback"][0]),
                scalar(&c["feedback"][1]),
                scalar(&c["feedback"][2]),
                scalar(&c["denominator"]),
                scalar(&c["numerator"]),
                scalar(&c["numerators"][0]),
                scalar(&c["numerators"][1]),
                scalar(&c["numerators"][2]),
            ],
            row["width"].as_i64().unwrap() as i32,
            row["height"].as_i64().unwrap() as i32,
            scalar(&c["accuracy"]),
            scalar(&c["tolerance"]),
            c["version"].as_u64().unwrap() as u32,
            f32::from_bits(c["sigma"].as_u64().unwrap() as u32),
        )
    }
    fn captured() -> (Value, Value) {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/dir/boundary-fixtures");
        let preset = match (cfg!(windows), cfg!(debug_assertions)) {
            (false, true) => "linux-debug",
            (false, false) => "linux-release",
            (true, true) => "windows-clang-debug",
            (true, false) => "windows-clang-release",
        };
        let captured: Value =
            serde_json::from_slice(&std::fs::read(root.join(format!("{preset}.json"))).unwrap())
                .unwrap();
        assert_eq!(captured["preset"], preset);
        (
            captured,
            serde_json::from_slice(&std::fs::read(root.join("checked.json")).unwrap()).unwrap(),
        )
    }
    fn words(values: &[f64]) -> Value {
        serde_json::json!(values.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
    }
    #[test]
    fn unchanged_primitives() {
        let (captured, _) = captured();
        assert_eq!(captured["primitives"].as_array().unwrap().len(), 56);
        for row in captured["primitives"].as_array().unwrap() {
            let args = &row["input"]["words"];
            let lhs = Dd {
                hi: scalar(&args[0]),
                lo: scalar(&args[1]),
            };
            let rhs = Dd {
                hi: scalar(&args[2]),
                lo: scalar(&args[3]),
            };
            let result = match row["input"]["operation"].as_u64().unwrap() {
                0 => Dd::normalize(lhs.hi, lhs.lo),
                1 => lhs.add(rhs),
                2 => lhs.negate(),
                3 => lhs.subtract(rhs),
                4 => lhs.multiply(rhs),
                5 => lhs.divide(rhs),
                6 => Dd::from(lhs.value()),
                7 => Dd::from(libm::fma(lhs.hi, rhs.hi, -(lhs.hi * rhs.hi))),
                _ => unreachable!(),
            };
            assert_eq!(
                words(&[result.hi, result.lo]),
                row["words"],
                "{}",
                row["input"]["name"]
            );
        }
    }
    #[test]
    fn unchanged_complete_products_and_ordered_failures() {
        let (captured, checked) = captured();
        assert_eq!(captured["cases"].as_array().unwrap().len(), 114);
        for row in captured["cases"].as_array().unwrap() {
            let name = row["input"]["name"].as_str().unwrap();
            let input = input(&row["input"]);
            if let Ok(input) = &input {
                assert_eq!(input.key().value(), row["key"].as_u64().unwrap(), "{name}");
            }
            let result = input.and_then(Boundary::new);
            assert_eq!(result.is_ok(), row["success"].as_bool().unwrap(), "{name}");
            match result {
                Ok(boundary) => {
                    let mut weights = row["horizontal"]["weights"].as_array().unwrap().clone();
                    weights.extend_from_slice(row["vertical"]["weights"].as_array().unwrap());
                    assert_eq!(words(boundary.weights()), Value::Array(weights), "{name}");
                    for (axis, expected) in [
                        (boundary.horizontal(), &row["horizontal"]),
                        (boundary.vertical(), &row["vertical"]),
                    ] {
                        assert_eq!(
                            i64::from(axis.length()),
                            expected["length"].as_i64().unwrap(),
                            "{name}"
                        );
                        assert_eq!(
                            i64::from(axis.terminal_size()),
                            expected["terminal_size"].as_i64().unwrap(),
                            "{name}"
                        );
                        assert_eq!(words(axis.terminal()), expected["terminal"], "{name}");
                    }
                }
                Err(error) => assert_eq!(
                    serde_json::json!([
                        error.reason as u64,
                        u64::from(error.axis),
                        u64::from(error.row),
                        error.length as u64,
                        u64::from(error.sigma_px.to_bits()),
                        error.sum.to_bits(),
                        error.tolerance.to_bits()
                    ]),
                    checked["ordinary"][name],
                    "{name}"
                ),
            }
        }
    }
    #[test]
    fn signed_consumer_and_axis_order_safety() {
        assert_eq!(
            retained_count(i32::MAX as usize / 3),
            Some((i32::MAX as usize / 3) * 3)
        );
        assert_eq!(retained_count(i32::MAX as usize / 3 + 1), None);
        let mut input = Input::new([0.0; 9], i32::MAX / 2 + 1, 1, 1e-11, 5e-7, 1, 4.0).unwrap();
        let error = Boundary::new(input).unwrap_err();
        assert_eq!((error.reason, error.axis), (Reason::AxisSize, 1));
        input.width = 1;
        input.height = i32::MAX / 2 + 1;
        let error = Boundary::new(input).unwrap_err();
        assert_eq!((error.reason, error.axis), (Reason::CausalCertification, 1));
        input.version = 0;
        let error = Boundary::new(input).unwrap_err();
        assert_eq!((error.reason, error.axis), (Reason::Policy, 1));
        input.version = 1;
        input.coefficients = [0.5, 0.5, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0];
        let error = Boundary::new(input).unwrap_err();
        assert_eq!((error.reason, error.axis), (Reason::AxisSize, 2));
    }
}
