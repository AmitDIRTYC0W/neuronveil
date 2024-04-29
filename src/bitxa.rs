use std::error::Error;

use crate::com;
use crate::message::BitXAInteraction;
use crate::message::Message;
use crate::message::IO;
use crate::multiplication_triplet_share::MultiplicationTripletShare;
use crate::unexpected_message_error::UnexpectedMessageError;
use bitvec::prelude::*;
use log::info;
use ndarray::Array;
use ndarray::Array1;
use ndarray::Ix1;
use ring::rand::SecureRandom;

///
pub async fn bitxa<const PARTY: bool>(
    x_share: &Array1<Com>,
    y_share: &Array1<bool>,
    (sender, receiver): IO<'_>,
    rng: &dyn SecureRandom,
) -> Result<Array1<Com>, Box<dyn Error>> {
    let masked_boolean_delta_y_share = bit::sample(x_share.len(), rng);

    let masked_arithmatic_delta_y_share = masked_boolean_delta_y_share.mapv(|b| {
        if b {
            Com::from_num(1)
        } else {
            Com::from_num(0)
        }
    });
    let (e_share, f_share) = if PARTY {
        (
            Array1::zeros(x_share.len()),
            masked_arithmatic_delta_y_share,
        )
    } else {
        (
            masked_arithmatic_delta_y_share,
            Array1::zeros(x_share.len()),
        )
    };

    let mt = MultiplicationTripletShare::<Ix1, Ix1>::new(x_share.len()); // TODO these should be generated in advance!
    let ef_share = mt
        .hadamard_product::<PARTY>(&e_share, &f_share, (sender, receiver))
        .await?;

    let arithmatic_delta_y_share = &e_share + &f_share - (ef_share * 2);

    // info!(
    //     "masked boolean delta y share: {:?}",
    //     masked_boolean_delta_y_share
    // );
    // FIXME this shouldn't be necessary
    // BUG it might not be secure
    let delta_x_share = com::sample(x_share.len(), rng) / 256;
    info!("delta x share: {:#}", delta_x_share);
    info!(
        "delta y share: {:#} ({:#} respectively)",
        arithmatic_delta_y_share, masked_boolean_delta_y_share
    );

    let mt2 = MultiplicationTripletShare::<Ix1, Ix1>::new(x_share.len()); // TODO these should be generated in advance!

    let delta_z_share = mt2
        .hadamard_product::<PARTY>(
            &delta_x_share,
            &arithmatic_delta_y_share,
            (sender, receiver),
        )
        .await?;

    info!("delta z share: {:#}", delta_z_share);

    // NOTE the online stage starts here
    // Send our shares of Δx and Δy to the other party
    let our_capital_delta_shares = BitXAInteraction {
        capital_delta_x_share: x_share + &delta_x_share,
        capital_delta_y_share: y_share ^ &masked_boolean_delta_y_share,
    };
    sender
        .send(Message::BitXAInteraction(our_capital_delta_shares.clone()))
        .await?;

    // Receive the Δx and Δy shares of the other party
    let their_capital_delta_shares: BitXAInteraction;
    if let Some(Message::BitXAInteraction(shares)) = receiver.recv().await {
        their_capital_delta_shares = shares;
    } else {
        return Err(Box::new(UnexpectedMessageError {}));
    }

    // Reconstruct Δx and Δy
    let capital_delta_x = our_capital_delta_shares.capital_delta_x_share
        + their_capital_delta_shares.capital_delta_x_share;
    let boolean_capital_delta_y = our_capital_delta_shares.capital_delta_y_share
        ^ their_capital_delta_shares.capital_delta_y_share;

    // This is akin to Δ′y
    let arithmatic_capital_delta_y = boolean_capital_delta_y.mapv(|b| {
        if b {
            Com::from_num(1)
        } else {
            Com::from_num(0)
        }
    });

    // Complete the computation
    // TODO merge adjust_product calls
    let t = &arithmatic_capital_delta_y * &capital_delta_x;
    let without_bt = &delta_z_share * (&arithmatic_capital_delta_y * 2) - &delta_z_share
        + &arithmatic_delta_y_share * (&capital_delta_x - &t * 2)
        - &arithmatic_capital_delta_y * &delta_x_share;
    // let without_bt = com::adjust_product(
    //     &arithmatic_delta_y_share * (&capital_delta_x - &t * Com(Wrapping(2)))
    //         - &arithmatic_capital_delta_y * (&delta_x_share + &delta_z_share * Com(Wrapping(2))),
    // ) - &delta_z_share;
    Ok(if PARTY { t + without_bt } else { without_bt })
}
