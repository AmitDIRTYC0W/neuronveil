use std::error::Error;
use std::num::Wrapping;

use crate::multiplication_triplet_share::MultiplicationTripletShare;
use crate::message::IO;
use ndarray::ArrayBase;
use ndarray::IntoDimension;
use ndarray::RawData;
use ndarray::{Array, Dimension, ShapeBuilder};
use ring::rand::SecureRandom;
use bitvec::prelude::*;

use crate::Com;

/// Creates a new array of random boolean values.
///
/// # Arguments
///
/// - `shape`: The shape of the output array.
/// - `rng`: A secure random number generator.
///
/// # Returns
///
/// A new array of random boolean values with the specified shape. The values in the array are with values either `0` or `1`.
#[inline]
fn random_booleans<Sh: ShapeBuilder + IntoDimension>(shape: Sh, rng: &dyn SecureRandom) -> Array<Com, <Sh as IntoDimension>::Dim> {
    // Calculate the minimum no. of bytes to generate to cover the amount of bits
    let bytes_to_generate = (shape.f().size() + 7) / 8;

    // Generate these bytes
    let mut bytes: Box<[u8]> = unsafe { Box::new_uninit_slice(bytes_to_generate).assume_init() };
    rng.fill(&mut bytes).unwrap();

    let bits = bytes.as_ref().view_bits::<Lsb0>();

    let coms_iter = bits.iter().by_vals().map(|bit| Com(Wrapping(if bit { 1 } else { 0 })));
    
    Array::from_iter(coms_iter).into_shape(shape).unwrap()
}

pub fn bitxa<const PARTY: bool, XData, YData, D>(
    x: &ArrayBase<XData, D>,
    y: &ArrayBase<YData, D>,
    (sender, receiver): IO<'_>,
    rng: &dyn SecureRandom,
) -> Result<Com, Box<dyn Error + Send + Sync>> where XData: RawData<Elem = Com>, YData: RawData<Elem = bool>, D: Dimension {
    // create an optimised function for line 16-19
    let masked_boolean_delta_y_shares = random_booleans(x.shape(), rng);

    let e_share = if PARTY { Array::zeros(x.shape()) } else { masked_boolean_delta_y_shares };
    let f_share = if PARTY { masked_boolean_delta_y_shares } else { Array::zeros(x.shape()) };

    let mt = MultiplicationTripletShare::new(input_share.len(), self.biases_share.len()); // TODO these should be generated in advance! (create a new function "prepare" along w/ split and infer)
    let ef_share = mt.multiply::<PARTY>(e_share, f_share, (sender, receiver)); // TODO this should use tensor multiplication!

    let arithmatic_delta_y_share  = e_share + f_share = 2 * ef_share;

    let arithmatic_delta_x_share = /* random n bits (wtf should n be?) */;

    let artihmatic_delta_z_share = mt2.multiply::<PARTY>(arithmatic_delta_x_share, arithmatic_delta_y_share, (sender, receiver));

    // the online stage starts here, the previous part should be in a different function!



    Ok(x)
}
