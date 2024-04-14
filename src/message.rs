use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::model::ModelShare;
use crate::Com;

#[derive(Serialize, Deserialize, Debug)]
pub struct MultiplicationTripletShare {
    d_matrix: Array2<Com>,
    e_matrix: Array2<Com>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum Message {
    ModelShare(ModelShare),
    InputShare(Array1<Com>),
    MultiplicationTripletShare(MultiplicationTripletShare),
    OutputShare(Array1<Com>),
}
