use ndarray::{Array1, Array2};
use ring::rand::SecureRandom;
use serde::{Deserialize, Serialize};

use crate::shares::{DenseLayerShare, LayerShare, ModelShare};
use crate::split::Split;
use crate::Com;

// #[derive(Deserialize, Debug)]
// pub struct InputLayer {
//     value: String,
// }

#[derive(Deserialize, Debug)]
pub struct DenseLayer {
    weights: Array2<Com>,
    biases: Array1<Com>,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
pub enum Layer {
    // InputLayer(InputLayer),
    DenseLayer(DenseLayer),
}

#[derive(Deserialize, Debug)]
pub struct Model {
    layers: Vec<Layer>,
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

impl Split for Layer {
    type Splitted = LayerShare;

    fn split(&self, rng: &dyn SecureRandom) -> (Self::Splitted, Self::Splitted) {
        match self {
            Layer::DenseLayer(dense_layer) => {
                let shares = DenseLayer::split(dense_layer, rng);
                (
                    LayerShare::DenseLayerShare(shares.0),
                    LayerShare::DenseLayerShare(shares.1),
                )
            }
        }
    }
}

impl Split for Model {
    type Splitted = ModelShare;

    fn split(&self, rng: &dyn SecureRandom) -> (Self::Splitted, Self::Splitted) {
        self.layers.iter().map(|layer| layer.split(rng)).unzip()
    }
}
