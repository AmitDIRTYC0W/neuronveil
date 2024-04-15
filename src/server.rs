use std::error::Error;

use ndarray::Array1;

use crate::message::Message;
use crate::message::IO;
use crate::model::ModelShare;
use crate::unexpected_message_error::UnexpectedMessageError;
use crate::Com;

pub async fn infer(
    (sender, mut receiver): IO<'_>,
    model_shares: (ModelShare, ModelShare),
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
    let input_share: Array1<Com>;
    if let Message::InputShare(contents) = input_share_message {
        input_share = contents;
    } else {
        return Err(Box::new(UnexpectedMessageError {}));
    }

    // Infer the model
    let output_share = model_shares.0.infer(input_share).await?;

    // Send the output share back to the client
    sender.send(Message::OutputShare(output_share)).await?;

    Ok(())
}
