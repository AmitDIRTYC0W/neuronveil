use ndarray::{Array1, Array2};
use ring::rand::SecureRandom;

use crate::{bit, com, Com};

// TODO consider renaming this to 'secret' or 'share'

pub trait Split {
    type Splitted;

    fn split(&self, rng: &dyn SecureRandom) -> (Self::Splitted, Self::Splitted);
}

impl Split for Array1<Com> {
    type Splitted = Array1<Com>;

    fn split(&self, rng: &dyn SecureRandom) -> (Self::Splitted, Self::Splitted) {
        // Generate a random array
        let first_share = com::sample(self.len(), rng);

        // Choose the second array s.t. the sum of both share is the original value
        let second_share = self - &first_share;

        (first_share, second_share)
    }
}

impl Split for Array2<Com> {
    type Splitted = Array2<Com>;

    fn split(&self, rng: &dyn SecureRandom) -> (Self::Splitted, Self::Splitted) {
        // Generate a random array
        let first_share = com::sample((self.shape()[0], self.shape()[1]), rng);

        // Choose the second array s.t. the sum of both share is the original value
        let second_share = self - &first_share;

        (first_share, second_share)
    }
}

impl Split for Array1<bool> {
    type Splitted = Array1<bool>;

    fn split(&self, rng: &dyn SecureRandom) -> (Self::Splitted, Self::Splitted) {
        // Generate a random array
        // TODO implement RandomlyConstructable to avoid copying
        let first_share = bit::sample(self.len(), rng);

        // Choose the second array s.t. the sum of both share is the original value
        let second_share = self ^ &first_share;

        (first_share, second_share)
    }
}
