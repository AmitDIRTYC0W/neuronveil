use std::error::Error;

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
    ) -> Result<Self::Reconstructed, Box<dyn Error>>
    where
        <Self as TryFrom<Message>>::Error: 'static + Error,
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
