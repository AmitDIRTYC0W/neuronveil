use ndarray::{Array, Array1, Array2, Dimension};
use ring::rand;
use ring::rand::SecureRandom;

use crate::Com;

// TODO consider renaming this to 'secret'

pub trait Split {
    type Splitted;

    fn split(&self, rng: &dyn SecureRandom) -> (Self::Splitted, Self::Splitted);
}

impl Split for Array1<Com> {
    type Splitted = Array1<Com>;

    fn split(&self, rng: &dyn SecureRandom) -> (Self::Splitted, Self::Splitted) {
        // Generate a random array
        let first_share = Array1::from_shape_simple_fn(self.len(), || {
            Com::from_le_bytes(rand::generate(rng).unwrap().expose())
        });

        // Choose the second array s.t. the sum of both share is the original value
        let second_share = self - &first_share;

        (first_share, second_share)
    }
}

impl Split for Array2<Com> {
    type Splitted = Array2<Com>;

    fn split(&self, rng: &dyn SecureRandom) -> (Self::Splitted, Self::Splitted) {
        // Generate a random array
        let first_share = Array2::from_shape_simple_fn((self.shape()[0], self.shape()[1]), || {
            Com::from_le_bytes(rand::generate(rng).unwrap().expose())
        });

        // Choose the second array s.t. the sum of both share is the original value
        let second_share = self - &first_share;

        (first_share, second_share)
    }
}
