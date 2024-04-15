use std::error::Error;

use ndarray::Array1;
use ring::rand::SecureRandom;

use crate::{
    com::{com_to_f32, f32_to_com},
    message::{Message, IO},
    split::Split,
    unexpected_message_error::UnexpectedMessageError,
    Com,
};

pub async fn infer(
    (sender, receiver): IO<'_>,
    input: Array1<f32>,
    rng: &dyn SecureRandom,
) -> Result<Array1<f32>, Box<dyn Error>> {
    // Convert the input from float to Com
    let input_com = f32_to_com(input);

    // Split the input into shares
    let input_shares = input_com.split(rng);

    // Ok(input)
    Ok(com_to_f32(
        infer_raw((sender, receiver), input_shares).await?,
    ))
}

pub async fn infer_raw(
    (sender, mut receiver): IO<'_>,
    input_shares: (Array1<Com>, Array1<Com>),
) -> Result<Array1<Com>, Box<dyn Error>> {
    sender.send(Message::InputShare(input_shares.1)).await?;

    // Wait for the model share
    if let Some(model_share_message) = receiver.recv().await {
        if let Message::ModelShare(model_share) = model_share_message {
            // Infer the model
            model_share.infer(input_shares.0).await
        } else {
            Err(Box::new(UnexpectedMessageError {}))
        }
    } else {
        Err(Box::new(UnexpectedMessageError {}))
    }
}
