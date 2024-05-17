use std::error::Error;

use ndarray::{Array1, Array2, Ix1, Ix2};
use ring::rand::SecureRandom;
use serde::{Deserialize, Serialize};

use crate::{
    message::IO, multiplication_triplet_share::MultiplicationTripletShare, split::Split, Com,
};

#[derive(Deserialize, Debug, Clone)]
pub struct DenseLayer {
    weights: Array2<Com>,
    biases: Array1<Com>,
}

impl DenseLayer {
    pub fn infer_locally(&self, input: Array1<Com>) -> Array1<Com> {
        &input.dot(&self.weights) + &self.biases
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DenseLayerShare {
    pub(self) weights_share: Array2<Com>,
    pub(self) biases_share: Array1<Com>,
}

impl DenseLayerShare {
    pub async fn infer<const PARTY: bool>(
        &self,
        input_share: Array1<Com>,
        (sender, receiver): IO<'_>,
    ) -> Result<Array1<Com>, Box<dyn Error>> {
        // let mt =
        //     MultiplicationTripletShare::<Ix1, Ix2>::new(input_share.len(), self.biases_share.len());
        let mt = MultiplicationTripletShare::<Ix1, Ix2>::new(
            self.weights_share.nrows(),
            self.biases_share.len(),
        );
        let product = mt
            .dot_product::<PARTY>(&input_share, &self.weights_share, (sender, receiver))
            .await?;
        Ok(product + &self.biases_share)
    }
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
