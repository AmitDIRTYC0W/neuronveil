use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::model::ModelShare;
use crate::Com;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MultiplicationTripletInteraction {
    pub e_share: Array1<Com>,
    pub f_share: Array2<Com>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum Message {
    ModelShare(ModelShare),
    InputShare(Array1<Com>),
    MultiplicationTripletInteraction(MultiplicationTripletInteraction),
    OutputShare(Array1<Com>),
}

// TODO replace mpsc::Receiver with a message multiplexing receiver
// TODO maybe use references?
pub(crate) type IO<'a> = (&'a mpsc::Sender<Message>, &'a mut mpsc::Receiver<Message>);
