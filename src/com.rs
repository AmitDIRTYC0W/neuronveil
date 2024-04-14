use ndarray::{Array, Dimension};

const FRACTION_BITS: i16 = 4;
const FRACTION: f32 = (1 << FRACTION_BITS) as f32;

pub type Com = i16;

pub(crate) fn f32_to_com<D: Dimension>(a: Array<f32, D>) -> Array<Com, D> {
    (FRACTION * a).map(|&x| x as Com)
}

pub(crate) fn com_to_f32<D: Dimension>(a: Array<Com, D>) -> Array<f32, D> {
    a.map(|&x| x as f32) / FRACTION
}
