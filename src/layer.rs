mod dense_layer;

use std::error::Error;

use ndarray::Array1;
use ring::rand::SecureRandom;
use serde::{Deserialize, Serialize};

use crate::{split::Split, Com};

use dense_layer::{DenseLayer, DenseLayerShare};

#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum Layer {
    DenseLayer(DenseLayer),
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum LayerShare {
    DenseLayerShare(DenseLayerShare),
}

impl LayerShare {
    pub async fn infer(&self, input_share: Array1<Com>) -> Result<Array1<Com>, Box<dyn Error>> {
        Ok(input_share)
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
