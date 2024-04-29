use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::bitxa::BitXAInteraction;
use crate::layer::relu::drelu::{DReLUInteraction, DReLUKey};
use crate::model::ModelShare;
use crate::unexpected_message_error::UnexpectedMessageError;
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

// TODO move to other place
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DDCFKey {
    pub alpha: Array1<Com>,
    pub invert: Array1<bool>,
}

// TODO move to other place
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SignedComparisonKeys {
    pub ddcf_keys: DDCFKey,
    pub r_shares: Array1<bool>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum Message {
    ModelShare(ModelShare),
    InputShare(Array1<Com>),
    DotProductInteraction(DotProductInteraction),
    HadamardProductInteraction(HadamardProductInteraction),
    DReLUKey(DReLUKey),
    DReLUInteraction(DReLUInteraction),
    BitXAInteraction(BitXAInteraction),
    OutputShare(Array1<Com>),
}

macro_rules! impl_message_conversions {
    ($message_type:ident) => {
        impl From<$message_type> for Message {
            fn from(value: $message_type) -> Self {
                Message::$message_type(value)
            }
        }

        impl TryFrom<Message> for $message_type {
            type Error = Box<UnexpectedMessageError>;

            fn try_from(value: Message) -> Result<Self, Self::Error> {
                if let Message::$message_type(contents) = value {
                    Ok(contents)
                } else {
                    Err(Box::new(UnexpectedMessageError {}))
                }
            }
        }
    };
}

impl_message_conversions!(DReLUInteraction);
impl_message_conversions!(BitXAInteraction);

// TODO replace mpsc::Receiver with a message multiplexing receiver
pub(crate) type IO<'a> = (&'a mpsc::Sender<Message>, &'a mut mpsc::Receiver<Message>);
