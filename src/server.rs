use tokio::sync::mpsc;

use crate::message::Message;
use crate::model::ModelShare;

type IO<'a> = (&'a mpsc::Sender<Message>, mpsc::Receiver<Message>);

pub async fn infer(
    (sender, mut receiver): IO<'_>,
    model_shares: (ModelShare, ModelShare),
) -> Result<(), tokio::sync::mpsc::error::SendError<Message>> {
    // Send the client a model share whilst listening
    sender.send(Message::ModelShare(model_shares.1)).await?;
    infer_independent((sender, receiver), model_shares.0).await?;

    Ok(())
}

async fn infer_independent<E>(
    (_sender, mut receiver): IO<'_>,
    _model_share: ModelShare,
) -> Result<(), E> {
    // Wait for the input share
    let _receive_input_share = receiver.recv().await;

    println!("Got here!");
    Ok(())
}
