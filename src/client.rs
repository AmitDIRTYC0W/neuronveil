use std::error::Error;

use ndarray::Array1;
use ring::rand::SecureRandom;

use crate::{
    com::f32_to_com,
    message::{Message, IO},
    model::ModelShare,
    reconstruct::Reconstruct as _,
    split::Split,
    unexpected_message_error::UnexpectedMessageError,
    Com,
};

/// Performs client-side inference of a privacy-preserving neural network.
///
/// This function takes an input array of floats, representing the input to the neural network,
/// and returns the inferred output. The inference is performed securely and privately using
/// secure multi-party computation techniques, considering a semi-honest adversary.
///
/// # Parameters
/// - `(sender, receiver)`: A sender and a receiver for asynchronous communication with the server. Messages may arrive out-of-order.
/// - `input`: The input vector.
/// - `rng`: A secure random number generator for secure computation.
///
/// # Returns
/// Array of floats representing the inferred output of the neural network.
///
/// # Errors
/// Returns an error if communication with the server fails or unexpected messages are received.
pub async fn infer(
    (sender, receiver): IO<'_>,
    input: Array1<f32>,
    rng: &dyn SecureRandom,
) -> Result<Array1<f32>, Box<dyn Error>> {
    // Convert the input from float to Com
    let input_com = input.mapv(Com::from_num);

    // Split the input into shares
    let input_shares = input_com.split(rng);

    // Ok(com_to_f32(input_com))
    Ok(infer_raw((sender, receiver), input_shares, rng)
        .await?
        .mapv(Com::to_num::<f32>))
}

/// Facilitates client-side communication for inference on a privacy-preserving neural network.
///
/// This function operates on secret-shared values (input shares) and returns the inferred output share.
/// It handles the secure and private communication with the server during the inference process.
///
/// # Parameters
/// - `(sender, receiver)`: A sender and a receiver for asynchronous communication with the server. Messages may arrive out-of-order.
/// - `input_shares`: The already-splitted input shares.
///
/// # Returns
/// The raw inferred output vector.
///
/// # Errors
/// Returns an error if communication with the server fails or unexpected messages are encountered.
pub async fn infer_raw(
    (sender, receiver): IO<'_>,
    input_shares: (Array1<Com>, Array1<Com>),
    rng: &dyn SecureRandom,
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
        .infer::<false>(input_shares.0, (sender, receiver), rng)
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
    Ok(Array1::<Com>::reconstruct((
        &our_output_share,
        &their_output_share,
    )))
}
