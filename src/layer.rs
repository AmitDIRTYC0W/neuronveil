pub mod dense_layer;
pub mod relu;

use crate::{message::IO, split::Split, Com};
use dense_layer::{DenseLayer, DenseLayerShare};
use ndarray::Array1;
use relu::{ReLULayer, ReLULayerShare};
use ring::rand::SecureRandom;
use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum Layer {
    DenseLayer(DenseLayer),
    ReLULayer(ReLULayer), // TODO ReLULayer shouldn't be a type, just use a union like a union here
}

impl Layer {
    pub fn infer_locally(&self, input: Array1<Com>) -> Array1<Com> {
        match self {
            Layer::DenseLayer(dense_layer) => dense_layer.infer_locally(input),
            Layer::ReLULayer(relu_layer) => relu_layer.infer_locally(input),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum LayerShare {
    DenseLayerShare(DenseLayerShare),
    ReLULayerShare(ReLULayerShare),
}

impl LayerShare {
    pub async fn infer<const PARTY: bool>(
        &self,
        input_share: Array1<Com>,
        (sender, receiver): IO<'_>,
        rng: &dyn SecureRandom,
    ) -> anyhow::Result<Array1<Com>> {
        match self {
            LayerShare::DenseLayerShare(dense_layer_share) => {
                dense_layer_share
                    .infer::<PARTY>(input_share, (sender, receiver))
                    .await
            }
            LayerShare::ReLULayerShare(relu_layer_share) => {
                relu_layer_share
                    .infer::<PARTY>(input_share, (sender, receiver), rng)
                    .await
            }
        }
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
            Layer::ReLULayer(relu_layer) => {
                let shares = ReLULayer::split(relu_layer, rng);
                (
                    LayerShare::ReLULayerShare(shares.0),
                    LayerShare::ReLULayerShare(shares.1),
                )
            }
        }
    }
}
