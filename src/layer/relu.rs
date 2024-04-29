use crate::{bitxa, message::IO, split::Split, Com};
use log::info;
use ndarray::Array1;
use ring::rand::SecureRandom;
use serde::{Deserialize, Serialize};
use std::error::Error;

pub(crate) mod drelu;

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
        let drelu_output_share =
            drelu::drelu::<PARTY>(&input_share, (sender, receiver), rng).await?;
        info!("DReLU share: {:#}", drelu_output_share);
        bitxa::<PARTY>(&input_share, &drelu_output_share, (sender, receiver), rng).await
    }
}

impl Split for ReLULayer {
    type Splitted = ReLULayerShare;

    fn split(&self, _: &dyn SecureRandom) -> (Self::Splitted, Self::Splitted) {
        (ReLULayerShare {}, ReLULayerShare {})
    }
}
