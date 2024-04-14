use ndarray::{Array1, Array2};
use ring::rand::SecureRandom;
use serde::{Deserialize, Serialize};

use crate::split::Split;
use crate::Com;

#[derive(Deserialize, Debug, Clone)]
pub struct DenseLayer {
    weights: Array2<Com>,
    biases: Array1<Com>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum Layer {
    DenseLayer(DenseLayer),
}

#[derive(Deserialize, Debug, Clone)]
pub struct Model {
    layers: Vec<Layer>,
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

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum LayerShare {
    DenseLayerShare(DenseLayerShare),
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

#[derive(Default, Serialize, Deserialize, Debug)]
pub struct ModelShare {
    pub layer_shares: Vec<LayerShare>,
}

impl Extend<LayerShare> for ModelShare {
    fn extend<T: IntoIterator<Item = LayerShare>>(&mut self, iter: T) {
        self.layer_shares.extend(iter)
    }
}

impl Split for Model {
    type Splitted = ModelShare;

    fn split(&self, rng: &dyn SecureRandom) -> (Self::Splitted, Self::Splitted) {
        self.layers.iter().map(|layer| layer.split(rng)).unzip()
    }
}
