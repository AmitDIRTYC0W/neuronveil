use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::model::ModelShare;
use crate::Com;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DotProductInteraction {
    pub e_share: Array1<Com>,
    pub f_share: Array2<Com>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HadamardProductInteraction {
    pub e_share: Array1<Com>,
    pub f_share: Array1<Com>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BitXAInteraction {
    // TODO shorten the names here
    pub capital_delta_x_share: Array1<Com>,
    pub capital_delta_y_share: Array1<bool>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum Message {
    ModelShare(ModelShare),
    InputShare(Array1<Com>),
    DotProductInteraction(DotProductInteraction),
    HadamardProductInteraction(HadamardProductInteraction),
    BitXAInteraction(BitXAInteraction),
    OutputShare(Array1<Com>),
}

// TODO replace mpsc::Receiver with a message multiplexing receiver
// TODO maybe use references?
pub(crate) type IO<'a> = (&'a mpsc::Sender<Message>, &'a mut mpsc::Receiver<Message>);
