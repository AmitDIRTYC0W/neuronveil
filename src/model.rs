use std::error::Error;

use ndarray::Array1;
use ring::rand::SecureRandom;
use serde::{Deserialize, Serialize};

use crate::{
    layer::{Layer, LayerShare},
    split::Split,
    Com,
};

#[derive(Deserialize, Debug, Clone)]
pub struct Model {
    layers: Vec<Layer>,
}

#[derive(Default, Serialize, Deserialize, Debug)]
pub struct ModelShare {
    pub layer_shares: Vec<LayerShare>,
}

impl ModelShare {
    pub async fn infer(&self, input_share: Array1<Com>) -> Result<Array1<Com>, Box<dyn Error>> {
        let mut activations_share = input_share;

        for layer_share in self.layer_shares.iter() {
            activations_share = layer_share.infer(activations_share).await?;
        }

        Ok(activations_share)
    }
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
