use crate::bit;
use crate::com;
use crate::message::IO;
use crate::multiplication_triplet_share::MultiplicationTripletShare;
use crate::reconstruct::Reconstruct;
use crate::reconstruct::ReconstructOnline;
use crate::Com;
use anyhow::Context as _;
use log::debug;
use ndarray::Array1;
use ndarray::Ix1;
use ring::rand::SecureRandom;
use serde::Deserialize;
use serde::Serialize;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BitXAInteraction {
    // TODO shorten the names here
    pub capital_delta_x_share: Array1<Com>,
    pub capital_delta_y_share: Array1<bool>,
}

impl Reconstruct for BitXAInteraction {
    type Reconstructed = CapitalDeltas;

    fn reconstruct(shares: (&Self, &Self)) -> Self::Reconstructed {
        CapitalDeltas {
            x: &shares.0.capital_delta_x_share + &shares.1.capital_delta_x_share,
            y: &shares.0.capital_delta_y_share ^ &shares.1.capital_delta_y_share,
        }
    }
}

impl ReconstructOnline for BitXAInteraction {}

#[derive(Debug, Clone)]
pub struct CapitalDeltas {
    pub x: Array1<Com>,
    pub y: Array1<bool>,
}

/// Directly multiply a bit by an integer.
///
/// This is a vectorised implementation of Algorithm no. 1 from
/// [FssNN: Communication-Efficient Secure Neural Network Training via Function Secret Sharing](https://eprint.iacr.org/2023/073.pdf).
///
/// # Arguments
///
/// - `x_share`: Arithmatic values share
/// - `y_share`: Boolean values share
/// - `(sender, receiver)`: A sender and a receiver for asynchronous communication with the server. Messages may arrive out-of-order.
/// - `rng`: A secure random number generator for secure computation.
///
/// # Returns
///
/// A share of the product of x and y
// FIXME this function is long
pub async fn bitxa<const PARTY: bool>(
    x_share: &Array1<Com>,
    y_share: &Array1<bool>,
    (sender, receiver): IO<'_>,
    rng: &dyn SecureRandom,
) -> anyhow::Result<Array1<Com>> {
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
        .await
        .context("Failed to compute Hadamard product")?;

    let arithmatic_delta_y_share = &e_share + &f_share - (ef_share * 2);

    // FIXME BUG this shouldn't be necessary
    let delta_x_share = com::sample(x_share.len(), rng) / 256;
    debug!("delta x share: {:#}", delta_x_share);
    debug!(
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
        .await
        .context("Failed to compute Hadamard product")?;

    debug!("delta z share: {:#}", delta_z_share);

    // NOTE the online stage starts here
    // struct Δx and Δy
    let our_capital_delta_shares = BitXAInteraction {
        capital_delta_x_share: x_share + &delta_x_share,
        capital_delta_y_share: y_share ^ &masked_boolean_delta_y_share,
    };
    let capital_deltas = our_capital_delta_shares
        .reconstruct_mutually((sender, receiver))
        .await
        .context("Failed to reconstruct Δx and Δy")?;

    // This is akin to Δ′y
    let arithmatic_capital_delta_y = capital_deltas.y.mapv(|b| {
        if b {
            Com::from_num(1)
        } else {
            Com::from_num(0)
        }
    });

    // Complete the computation
    // TODO merge adjust_product calls
    let t = &arithmatic_capital_delta_y * &capital_deltas.x;
    let without_bt = &delta_z_share * (&arithmatic_capital_delta_y * 2) - &delta_z_share
        + &arithmatic_delta_y_share * (&capital_deltas.x - &t * 2)
        - &arithmatic_capital_delta_y * &delta_x_share;

    Ok(if PARTY { t + without_bt } else { without_bt })
}
