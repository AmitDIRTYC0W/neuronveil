use anyhow::bail;
use ndarray::{Array, Array1, Array2, Dimension, Ix, Ix1, Ix2};

use crate::{
    message::{DotProductInteraction, HadamardProductInteraction, Message, IO},
    unexpected_message_error::UnexpectedMessageError,
    Com,
};

pub(crate) struct MultiplicationTripletShare<DimA: Dimension, DimB: Dimension> {
    a_share: Array<Com, DimA>,
    b_share: Array<Com, DimB>,
    ab_share: Array1<Com>,
}

impl MultiplicationTripletShare<Ix1, Ix1> {
    /// Performs Hadamard (element-wise) product operation using Beaver's multiplication triplets.
    ///
    /// # Parameters
    /// - `x_share`: a share of the first operand
    /// - `y_share`: a share of the second operand
    /// - `(sender, receiver)`: A sender and a receiver for asynchronous communication with the server. Messages may arrive out-of-order.
    ///
    /// # Returns
    /// A share of the Hadamard product
    ///
    /// # Warnings
    /// Multiplication triplets shall not be re-used. To multiply a new pair, generate a triplet.
    pub(crate) async fn hadamard_product<const PARTY: bool>(
        &self,
        x_share: &Array1<Com>,
        y_share: &Array1<Com>,
        (sender, receiver): IO<'_>,
    ) -> anyhow::Result<Array1<Com>> {
        // 'Mask' x_share and y_share as e_share and f_share
        let our_ef_shares = HadamardProductInteraction {
            e_share: x_share - &self.a_share,
            f_share: y_share - &self.b_share,
        };

        // Send our e and f shares to the other party
        sender
            .send(Message::HadamardProductInteraction(our_ef_shares.clone()))
            .await?; // TODO this can be easily parallelised

        // Receive the e and f shares of the other party
        // TODO maybe use Optional::some_cool_func for this?
        let their_ef_shares: HadamardProductInteraction;
        if let Some(Message::HadamardProductInteraction(shares)) = receiver.recv().await {
            their_ef_shares = shares;
        } else {
            bail!(UnexpectedMessageError {});
        }

        // Reconstruct e and f
        let e = our_ef_shares.e_share + their_ef_shares.e_share;
        let f = our_ef_shares.f_share + their_ef_shares.f_share;

        // Complete the computation
        // TODO take the common part out
        Ok(if PARTY {
            &e * &f + &self.a_share * &f + &e * &self.b_share + &self.ab_share
        } else {
            &self.a_share * &f + &e * &self.b_share + &self.ab_share
        })
    }

    pub(crate) fn new(n: Ix) -> Self {
        MultiplicationTripletShare {
            a_share: Array1::<Com>::zeros(n),
            b_share: Array1::<Com>::zeros(n),
            ab_share: Array1::<Com>::zeros(n),
        }
    }
}

impl MultiplicationTripletShare<Ix1, Ix2> {
    /// Multiplication using Beaver's triplets (Donald Beaver. Efficient
    /// Multiparty Protocols Using Circuit Randomization. CRYPTO 1991.) extended to matrices.
    ///
    /// # Warnings
    /// Multiplication triplets shall not be re-used. To multiply a new pair, generate a triplet.
    pub(crate) async fn dot_product<const PARTY: bool>(
        &self,
        x_share: &Array<Com, Ix1>,
        y_share: &Array<Com, Ix2>,
        (sender, receiver): IO<'_>,
    ) -> anyhow::Result<Array1<Com>> {
        // 'Mask' x_share and y_share as e_share and f_share
        let our_ef_shares = DotProductInteraction {
            e_share: x_share - &self.a_share,
            f_share: y_share - &self.b_share,
        };

        // Send our e and f shares to the other party
        sender
            .send(Message::DotProductInteraction(our_ef_shares.clone()))
            .await?; // TODO this can be easily parallelised

        // Receive the e and f shares of the other party
        let their_ef_shares: DotProductInteraction;
        if let Some(Message::DotProductInteraction(shares)) = receiver.recv().await {
            their_ef_shares = shares;
        } else {
            bail!(UnexpectedMessageError {});
        }

        // Reconstruct e and f
        let e = our_ef_shares.e_share + their_ef_shares.e_share;
        let f = our_ef_shares.f_share + their_ef_shares.f_share;

        // Complete the calculation
        Ok(if PARTY {
            e.dot(&f) + self.a_share.dot(&f) + e.dot(&self.b_share) + &self.ab_share
        } else {
            self.a_share.dot(&f) + e.dot(&self.b_share) + &self.ab_share
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
