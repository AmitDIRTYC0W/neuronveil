use ndarray::{Array1, Array2};
use ring::rand::SecureRandom;
use serde::{Deserialize, Serialize};

use crate::{split::Split, Com};

#[derive(Deserialize, Debug, Clone)]
pub struct DenseLayer {
    weights: Array2<Com>,
    biases: Array1<Com>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DenseLayerShare {
    pub weights_share: Array2<Com>,
    pub biases_share: Array1<Com>,
}

impl Split for DenseLayer {
    type Splitted = DenseLayerShare;

    fn split(&self, rng: &dyn SecureRandom) -> (Self::Splitted, Self::Splitted) {
        let weights_shares = self.weights.split(rng);
        let biases_shares = self.biases.split(rng);

        (
            DenseLayerShare {
                weights_share: weights_shares.0,
                biases_share: biases_shares.0,
            },
            DenseLayerShare {
                weights_share: weights_shares.1,
                biases_share: biases_shares.1,
            },
        )
    }
}
