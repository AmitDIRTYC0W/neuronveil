use std::error::Error;

use ndarray::{Array1, Array2, Ix};

use crate::{
    com,
    message::{Message, MultiplicationTripletInteraction, IO},
    unexpected_message_error::UnexpectedMessageError,
    Com,
};

// TODO make a version that can multiply x and y of any dimensions

pub(crate) struct MultiplicationTripletShare {
    a_share: Array1<Com>,
    b_share: Array2<Com>,
    ab_share: Array1<Com>,
}

impl MultiplicationTripletShare {
    /// Multiplication using Beaver's triplets (Donald Beaver. Efficient
    /// Multiparty Protocols Using Circuit Randomization. CRYPTO 1991.) extended to matrices.
    ///
    /// # Warnings
    /// Multiplication triplets shall not be re-used. To multiply a new pair, generate a triplet.
    pub(crate) async fn multiply<const PARTY: bool>(
        &self,
        x_share: &Array1<Com>,
        y_share: &Array2<Com>,
        (sender, receiver): IO<'_>,
    ) -> Result<Array1<Com>, Box<dyn Error>> {
        // 'Mask' x_share and y_share as e_share and f_share
        let our_ef_shares = MultiplicationTripletInteraction {
            e_share: x_share - &self.a_share,
            f_share: y_share - &self.b_share,
        };

        // Send our e and f shares to the other party
        sender
            .send(Message::MultiplicationTripletInteraction(
                our_ef_shares.clone(),
            ))
            .await?; // TODO this can be easily parallelised

        // Receive the e and f shares of the other party
        let their_ef_shares: MultiplicationTripletInteraction;
        if let Some(Message::MultiplicationTripletInteraction(shares)) = receiver.recv().await {
            their_ef_shares = shares;
        } else {
            return Err(Box::new(UnexpectedMessageError {}));
        }

        // Reconstruct e and f
        let e = our_ef_shares.e_share + their_ef_shares.e_share;
        let f = our_ef_shares.f_share + their_ef_shares.f_share;

        // Complete the calculation
        Ok(if PARTY {
            com::adjust_product(e.dot(&f) + self.a_share.dot(&f) + e.dot(&self.b_share))
                + &self.ab_share
        } else {
            com::adjust_product(self.a_share.dot(&f) + e.dot(&self.b_share)) + &self.ab_share
        })
    }

    pub(crate) fn new(k: Ix, m: Ix) -> Self {
        MultiplicationTripletShare {
            a_share: Array1::<Com>::zeros(k),
            b_share: Array2::<Com>::zeros((k, m)),
            ab_share: Array1::<Com>::zeros(m),
        }
    }
}
