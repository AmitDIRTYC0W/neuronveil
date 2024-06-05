use std::{error::Error, ops::Add};

use ndarray::{ArrayBase, Dimension, RawData};

use crate::{
    message::{Message, IO},
    unexpected_message_error::UnexpectedMessageError,
};

pub trait Reconstruct {
    type Reconstructed;

    fn reconstruct(shares: (&Self, &Self)) -> Self::Reconstructed;
}

pub trait ReconstructOnline: Reconstruct + TryFrom<Message> + Into<Message> + Clone {
    async fn reconstruct_mutually(
        self,
        (sender, receiver): IO<'_>,
    ) -> anyhow::Result<Self::Reconstructed>
    where
        <Self as TryFrom<Message>>::Error: 'static + Error,
        <Self as TryFrom<Message>>::Error: Send,
        <Self as TryFrom<Message>>::Error: Sync,
    {
        // Send our share to the adversary
        sender.send(self.clone().into()).await?;

        // Receive the shares from the other party
        let their_share_message = receiver
            .recv()
            .await
            .ok_or(Box::new(UnexpectedMessageError {}))?;
        let their_share = Self::try_from(their_share_message)?;

        // Reconstruct the secret
        Ok(Self::reconstruct((&self, &their_share)))
    }
}

impl<S, D> Reconstruct for ArrayBase<S, D>
where
    S: RawData,
    D: Dimension,
    for<'a> &'a ArrayBase<S, D>: Add<&'a ArrayBase<S, D>, Output = ArrayBase<S, D>>,
{
    type Reconstructed = ArrayBase<S, D>;

    fn reconstruct(shares: (&Self, &Self)) -> Self::Reconstructed {
        shares.0 + shares.1
    }
}
