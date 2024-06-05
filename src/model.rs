use std::any::Any;

use anyhow::Context;
use log::debug;
use ndarray::Array1;
use ring::rand::SecureRandom;
use serde::{Deserialize, Serialize};

use crate::{
    layer::{Layer, LayerShare},
    message::IO,
    split::Split,
    Com,
};

#[derive(Deserialize, Debug, Clone)]
pub struct Model {
    pub layers: Vec<Layer>,
}

impl Model {
    pub fn infer_locally(&self, input: Array1<Com>) -> Array1<Com> {
        let mut activations = input;

        for (i, layer) in self.layers.iter().enumerate() {
            debug!("Evaluating layer {}", i);
            activations = layer.infer_locally(activations);
        }

        activations
    }
}

#[derive(Default, Serialize, Deserialize, Debug)]
pub struct ModelShare {
    pub layer_shares: Vec<LayerShare>,
}

impl ModelShare {
    pub async fn infer<const PARTY: bool>(
        &self,
        input_share: Array1<Com>,
        (sender, receiver): IO<'_>,
        rng: &dyn SecureRandom,
    ) -> anyhow::Result<Array1<Com>> {
        let mut activations_share = input_share;

        for (i, layer_share) in self.layer_shares.iter().enumerate() {
            activations_share = layer_share
                .infer::<PARTY>(activations_share, (sender, receiver), rng)
                .await
                .with_context(|| format!("Failed to infer layer {}", i + 1))?;
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
