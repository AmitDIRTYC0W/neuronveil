use float_eq::assert_float_eq;
use ndarray::{array, Array, Array0, Array1, ArrayView, ArrayView1, Dimension};

pub fn softmax<D: Dimension>(x: &ArrayView<f32, D>) -> Array<f32, D> {
    let exp_values = x.mapv(f32::exp);
    &exp_values / exp_values.sum()
}

#[test]
fn test_softmax_1d() {
    let x = array![1.0, 2.0, 3.0];
    let expected = array![0.09003057, 0.24472847, 0.66524096];

    let result = softmax(&x.view());

    let error = &expected - &result;
    error.for_each(|&e| assert_float_eq!(e, 0f32, ulps <= 10));
}
