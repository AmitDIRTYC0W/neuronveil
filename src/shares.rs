use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::Com;

#[derive(Serialize, Deserialize, Debug)]
pub struct DenseLayerShare {
    pub weights_share: Array2<Com>,
    pub biases_share: Array1<Com>,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum LayerShare {
    DenseLayerShare(DenseLayerShare),
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
