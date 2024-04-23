use std::error::Error;

use ndarray::Array1;
use ring::rand::SecureRandom;

use crate::message::Message;
use crate::message::IO;
use crate::model::ModelShare;
use crate::unexpected_message_error::UnexpectedMessageError;
use crate::Com;

pub async fn infer(
    (sender, receiver): IO<'_>,
    model_shares: (ModelShare, ModelShare),
    rng: &dyn SecureRandom,
) -> Result<(), Box<dyn Error>> {
    // FIXME: this runs sequentially even though I can easily parallelise this

    // Send the client a model share
    sender.send(Message::ModelShare(model_shares.1)).await?;

    // Wait for the input share
    let input_share_message: Message;
    if let Some(message) = receiver.recv().await {
        input_share_message = message;
    } else {
        return Err(Box::new(UnexpectedMessageError {}));
    }

    // Verify the message is indeed an input share
    // TODO this may be merged with on paragraph above
    let input_share: Array1<Com>;
    if let Message::InputShare(contents) = input_share_message {
        input_share = contents;
    } else {
        return Err(Box::new(UnexpectedMessageError {}));
    }

    // Infer the model
    let output_share = model_shares
        .0
        .infer::<true>(input_share, (sender, receiver), rng)
        .await?;

    // Send the output share back to the client
    sender.send(Message::OutputShare(output_share)).await?;

    Ok(())
}
