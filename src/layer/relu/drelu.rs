use std::error::Error;

use log::info;
use ndarray::Array1;
use ring::rand::SecureRandom;
use serde::{Deserialize, Serialize};

use crate::{
    bit, com,
    message::{Message, SignedComparisonKeys, IO},
    reconstruct::{Reconstruct, ReconstructOnline},
    signed_comparison::generate_signed_comparison_keys,
    split::Split as _,
    unexpected_message_error::UnexpectedMessageError,
    Com,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DReLUInteraction {
    pub masked_x_share: Array1<Com>,
}

impl Reconstruct for DReLUInteraction {
    type Reconstructed = Array1<Com>;

    fn reconstruct(shares: (&Self, &Self)) -> Self::Reconstructed {
        &shares.0.masked_x_share + &shares.1.masked_x_share
    }
}

impl ReconstructOnline for DReLUInteraction {}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DReLUKey {
    pub r_in_1_share: Array1<Com>,
    pub r_in_2: Array1<Com>,
    pub r_out_share: Array1<bool>,
    pub signed_comparison_key: SignedComparisonKeys,
    // NOTE maybe the r_out shares from SignedComparisonKeys can be re-used for r_out_share
}

pub async fn drelu<const PARTY: bool>(
    x_share: &Array1<Com>,
    (sender, receiver): IO<'_>,
    rng: &dyn SecureRandom,
) -> Result<Array1<bool>, Box<dyn Error>> {
    // Deal/receive DReLU keys
    // TODO technically, the signed comparison keys should also be here
    // but actually maybe it's better to do it with promises and such
    let key = if PARTY {
        // Sample random key
        // TODO use RandomConstructible
        // let r_in_1 = com::sample(x_share.len(), rng);
        // let r_in_2 = com::sample(x_share.len(), rng);

        let r_in_1 = Array1::<Com>::zeros(x_share.len());
        let r_in_2 = Array1::<Com>::zeros(x_share.len());
        let r_out = bit::sample(x_share.len(), rng);

        let r_in_1_shares = r_in_1.split(rng);
        let r_out_shares = r_out.split(rng);

        let signed_comparison_keys =
            generate_signed_comparison_keys(r_in_1, r_in_2.clone(), r_out, rng);

        let our_key = DReLUKey {
            r_in_1_share: r_in_1_shares.0,
            r_in_2: r_in_2.clone(),
            r_out_share: r_out_shares.0,
            signed_comparison_key: signed_comparison_keys.0,
        };
        let their_key = DReLUKey {
            r_in_1_share: r_in_1_shares.1,
            r_in_2,
            r_out_share: r_out_shares.1,
            signed_comparison_key: signed_comparison_keys.1,
        };

        // Send one key to the other party
        sender.send(Message::DReLUKey(their_key)).await?;

        our_key
    } else {
        // Wait for the adversary to send us a key
        if let Some(Message::DReLUKey(our_key)) = receiver.recv().await {
            our_key
        } else {
            return Err(Box::new(UnexpectedMessageError {}));
        }
    };

    info!("DReLU Key: {:#?}", key);

    // NOTE the online stage starts here
    let masked_x_share = x_share + key.r_in_1_share;

    let masked_x = DReLUInteraction { masked_x_share }
        .reconstruct_mutually((sender, receiver))
        .await?;

    let signed_comparison_keys = key
        .signed_comparison_key
        .evaluate::<PARTY>(masked_x, key.r_in_2 /* , (sender, receiver), rng */)
        .await;

    Ok(signed_comparison_keys ^ key.r_out_share ^ PARTY)
}
