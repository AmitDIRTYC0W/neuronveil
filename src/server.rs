use std::error::Error;

use crate::message::Message;
use crate::message::IO;
use crate::model::ModelShare;
use crate::unexpected_message_error::UnexpectedMessageError;

pub async fn infer(
    (sender, mut receiver): IO<'_>,
    model_shares: (ModelShare, ModelShare),
) -> Result<(), Box<dyn Error>> {
    // FIXME: this runs sequentially even though I can easily parallelise this

    // Send the client a model share whilst listening
    sender.send(Message::ModelShare(model_shares.1)).await?;

    // Wait for the input share
    if let Some(input_share_message) = receiver.recv().await {
        if let Message::InputShare(input_share) = input_share_message {
            // Infer the model
            model_shares.0.infer(input_share).await?;
            Ok(())
        } else {
            Err(Box::new(UnexpectedMessageError {}))
        }
    } else {
        Err(Box::new(UnexpectedMessageError {}))
    }
}
