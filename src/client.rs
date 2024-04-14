use ndarray::Array1;
use ring::rand::SecureRandom;

use crate::{
    com::{com_to_f32, f32_to_com},
    message::{Message, IO},
    split::Split,
    Com,
};

pub async fn infer(
    (sender, receiver): IO<'_>,
    input: Array1<f32>,
    rng: &dyn SecureRandom,
) -> Result<Array1<f32>, tokio::sync::mpsc::error::SendError<Message>> {
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
) -> Result<Array1<Com>, tokio::sync::mpsc::error::SendError<Message>> {
    sender.send(Message::InputShare(input_shares.1)).await?;

    let _receive_model_share = receiver.recv().await;

    Ok(input_shares.0)
}
