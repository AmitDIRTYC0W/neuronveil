use bitvec::order::Lsb0;
use bitvec::view::BitView;
use ndarray::{Array, Array1};
use ring::rand::SecureRandom;

/// Creates a new array of random boolean values.
///
/// # Arguments
///
/// - `shape`: The shape of the output array.
/// - `rng`: A secure random number generator.
///
/// # Returns
///
/// A new array of random boolean values with the specified shape.
pub fn sample(n: usize, rng: &dyn SecureRandom) -> Array1<bool> {
    // Calculate the minimum no. of bytes to generate to cover the amount of bits
    let bytes_to_generate = (n + 7) / 8;

    // Generate these bytes
    let mut bytes: Box<[u8]> = unsafe { Box::new_uninit_slice(bytes_to_generate).assume_init() };
    rng.fill(&mut bytes).unwrap();

    let bits = bytes.as_ref().try_view_bits::<Lsb0>().unwrap();

    Array::from_iter(bits.iter().by_vals().take(n))
}
