use std::error::Error;

use ndarray::Array1;
use ring::rand::SecureRandom;
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;

use crate::{message::IO, split::Split, Com};

#[derive(Deserialize, Debug, Clone)]
pub struct ReLULayer {}

#[derive(Serialize, Deserialize, Debug)]
pub struct ReLULayerShare {}

impl ReLULayerShare {
    pub async fn infer<const PARTY: bool>(
        &self,
        input_share: JoinHandle<Array1<Com>>,
        (sender, receiver): IO<'_>,
    ) -> Result<Array1<Com>, Box<dyn Error + Send + Sync>> {
        // BitXA(x, DReLU())
        Ok(input_share.await?)
    }
}

impl Split for ReLULayer {
    type Splitted = ReLULayerShare;

    fn split(&self, _: &dyn SecureRandom) -> (Self::Splitted, Self::Splitted) {
        (ReLULayerShare {}, ReLULayerShare {})
    }
}
