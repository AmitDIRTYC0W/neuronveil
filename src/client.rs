use std::error::Error;

use ndarray::Array1;
use ring::rand::SecureRandom;

use crate::{
    com::{com_to_f32, f32_to_com},
    message::{Message, IO},
    model::ModelShare,
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
    (sender, receiver): IO<'_>,
    input_shares: (Array1<Com>, Array1<Com>),
) -> Result<Array1<Com>, Box<dyn Error>> {
    // Send the server an input share
    sender.send(Message::InputShare(input_shares.1)).await?;

    // Wait for the model share
    let model_share_message: Message;
    if let Some(message) = receiver.recv().await {
        model_share_message = message;
    } else {
        return Err(Box::new(UnexpectedMessageError {}));
    }

    // Verify the message is indeed a model share
    let model_share: ModelShare;
    if let Message::ModelShare(contents) = model_share_message {
        model_share = contents;
    } else {
        return Err(Box::new(UnexpectedMessageError {}));
    }

    // Infer the model
    let our_output_share = model_share
        .infer::<false>(input_shares.0, (sender, receiver))
        .await?;

    // Wait for output share
    let output_share_message: Message;
    if let Some(message) = receiver.recv().await {
        output_share_message = message;
    } else {
        return Err(Box::new(UnexpectedMessageError {}));
    }

    // Verify the message is indeed an output share
    let their_output_share: Array1<Com>;
    if let Message::OutputShare(contents) = output_share_message {
        their_output_share = contents;
    } else {
        return Err(Box::new(UnexpectedMessageError {}));
    }

    // Reconstruct the output
    Ok(our_output_share + their_output_share)
}
