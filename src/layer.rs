mod dense_layer;
mod relu;

use std::error::Error;

use ndarray::Array1;
use ring::rand::SecureRandom;
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;

use crate::{message::IO, split::Split, Com};

use dense_layer::{DenseLayer, DenseLayerShare};
use relu::{ReLULayer, ReLULayerShare};

#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum Layer {
    DenseLayer(DenseLayer),
    ReLULayer(ReLULayer),
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
    ) -> Result<Array1<Com>, Box<dyn Error>> {
        match self {
            LayerShare::DenseLayerShare(dense_layer_share) => {
                dense_layer_share
                    .infer::<PARTY>(input_share, (sender, receiver))
                    .await
            }
            LayerShare::ReLULayerShare(relu_layer_share) => {
                relu_layer_share
                    .infer::<PARTY>(input_share, (sender, receiver))
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
