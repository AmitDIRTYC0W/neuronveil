use std::error::Error;

use log::info;
use ndarray::Array1;
use ring::rand::SecureRandom;
use serde::{Deserialize, Serialize};

use crate::{bitxa, message::IO, split::Split, Com};

#[derive(Deserialize, Debug, Clone, Copy)]
pub struct ReLULayer {}

#[derive(Serialize, Deserialize, Debug)]
pub struct ReLULayerShare {}

impl ReLULayerShare {
    pub async fn infer<const PARTY: bool>(
        &self,
        input_share: Array1<Com>,
        (sender, receiver): IO<'_>,
        rng: &dyn SecureRandom,
    ) -> Result<Array1<Com>, Box<dyn Error>> {
        info!("input_share: {:?}", input_share);
        bitxa::<PARTY>(
            &input_share,
            &Array1::from_elem(input_share.len(), PARTY),
            (sender, receiver),
            rng,
        )
        .await
        // BitXA(x, DReLU())
        // Ok(input_share)
    }
}

impl Split for ReLULayer {
    type Splitted = ReLULayerShare;

    fn split(&self, _: &dyn SecureRandom) -> (Self::Splitted, Self::Splitted) {
        (ReLULayerShare {}, ReLULayerShare {})
    }
}
