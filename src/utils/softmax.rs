use float_eq::assert_float_eq;
use ndarray::{array, Array, Array0, Array1, ArrayView, ArrayView1, Dimension};

pub fn softmax<D: Dimension>(x: &ArrayView<f32, D>) -> Option<Array<f32, D>> {
    // Shift all the values s.t. the minimum is 0. This will improve the precision of the output and mostly prevent NaNs.
    let minimum = x.iter().min_by(|a, b| a.partial_cmp(b).unwrap())?;
    let shifted_x = x - *minimum;

    let exp_values = shifted_x.mapv(f32::exp);
    Some(&exp_values / exp_values.sum())
}

fn test_softmax<D: Dimension>(x: &ArrayView<f32, D>, expected: ArrayView<f32, D>) {
    let result = softmax(x).unwrap();

    for (y_hat, y) in result.iter().zip(expected.iter()) {
        assert_float_eq!(y, y_hat, rmax <= 1e-7);
    }
}

#[test]
fn test_softmax_1d() {
    let x = array![1.0, 2.0, 3.0];
    let expected = array![0.09003057, 0.24472847, 0.66524096];
    test_softmax(&x.view(), expected.view());
}

#[test]
fn test_softmax_small_values() {
    let x = array![-9_999.0, -9_998.0, -9_997.0];
    let expected = array![0.09003057, 0.24472847, 0.66524096];
    test_softmax(&x.view(), expected.view());
}
