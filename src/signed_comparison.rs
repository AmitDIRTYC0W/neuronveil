use crate::message::{DDCFKey, SignedComparisonKeys};
use crate::split::Split as _;
use crate::Com;
use fixed::traits::{Fixed, FixedBits};
use fixed::Wrapping;
use log::info;
use ndarray::{Array, Array1, Dimension};
use num_traits::One;
use ring::rand::SecureRandom;

// TODO Consider renaming plural variable names to singular ones

// TODO move this function somewhere else
// fn get_msb<N>(n: N) -> N
// where
//     N: Shr<usize, Output = N> + BitAnd<Output = N> + One,
// {
//     let shift = std::mem::size_of::<N>() * 8 - 1;
//     (n >> shift) & N::one()
// }

fn get_msb<T, D: Dimension>(array: &Array<Wrapping<T>, D>) -> Array<bool, D>
where
    T: Fixed + One,
    // <S::Elem as Fixed>::Bits: BitStore,
{
    // array.map(|x| x.to_bits().view_bits::<Msb0>()[0])
    let shift = <<T as Fixed>::Bits as FixedBits>::BITS - 1;
    array.map(|x| (x.0 >> shift) & T::one() == 1)
}

// pub fn generate_signed_comparison_keys<const LAMBDA: u8>(
pub fn generate_signed_comparison_keys(
    r1_ins: Array1<Com>,
    r2_ins: Array1<Com>,
    r_out: Array1<bool>,
    rng: &dyn SecureRandom,
) -> (SignedComparisonKeys, SignedComparisonKeys) {
    let r = r2_ins - r1_ins /* + (1 << 16) */; // NOTE is the (1 << 16) even necessary?
    let alpha = &r & Com::from_bits(0x7FFF); // TODO I shouldn't write 0x7FFF manually
    let invert = get_msb(&r) /* ^ true */; // WARNING maybe invert should be inverted
    let dual_comparison_function = DDCFKey { alpha, invert };
    let ddcf_keys = (dual_comparison_function.clone(), dual_comparison_function);

    // Sample random r0 and r1 s.t. they share r_out
    let r_shares = r_out.split(rng);

    (
        SignedComparisonKeys {
            ddcf_keys: ddcf_keys.0,
            r_shares: r_shares.0,
        },
        SignedComparisonKeys {
            ddcf_keys: ddcf_keys.1,
            r_shares: r_shares.1,
        },
    )
}

impl SignedComparisonKeys {
    /// Signed integer comparison gate 'Comp' as described in Algorithm no. 2
    /// from (FssNN: Communication-Efficient Secure Neural Network Training via Function Secret Sharing)[https://eprint.iacr.org/2023/073.pdf]
    pub async fn evaluate<const PARTY: bool>(
        &self,
        masked_x: Array1<Com>,
        masked_y: Array1<Com>,
        // (sender, receiver): IO<'_>,
        // rng: &dyn SecureRandom,
    ) -> Array1<bool> {
        let z = masked_x - masked_y;

        info!("Z: {:#?}", z);
        // TODO I shouldn't write 0x7FFF manually
        let point = (!&z) & Com::from_bits(0x7FFF); // z^{(n - 1)} = 2^{n-1} - z_{[0, n-1)} - 1

        // Evaluate the DDCF
        // TODO should be an actual implementation🤡
        let m_shares = Array1::from_iter(
            point
                .into_iter()
                .zip(&self.ddcf_keys.alpha)
                .map(|(a, b)| a < *b),
        ) ^ &self.ddcf_keys.invert;
        // let m_shares = (point < self.ddcf_keys.alpha) ^ self.ddcf.invert;

        // Finish the calculation
        let v_share_without_b = &m_shares ^ &self.r_shares;
        if PARTY {
            v_share_without_b ^ get_msb(&z)
        } else {
            v_share_without_b
        }
    }
}
