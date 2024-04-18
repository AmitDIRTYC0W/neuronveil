use std::{num::Wrapping, ops};

use ndarray::{Array, Dimension, ScalarOperand};
use num_traits::identities;

/// The number of bits used for the fractional part of the fixed-point number.
const FRACTION_BITS: i16 = 4;

/// The scaling factor used to convert between floating-point and fixed-point representations.
/// This is calculated as 2^FRACTION_BITS, which is the maximum value that can be represented
/// in the fractional part of the fixed-point number.
const FRACTION: f32 = (1 << FRACTION_BITS) as f32;

/// A fixed-point number that is used for communication (hence the name 'Com') and upon which cryptography is performed.
/// The fixed-point number is represented using a 16-bit signed integer, with the number of bits used for the fractional part
/// defined by the `FRACTION_BITS` constant.
#[derive(Copy, Clone, Debug)]
pub struct Com(pub Wrapping<i16>);

/// Converts an array of `f32` values to an array of `Com` values. This truncates the value.
#[inline]
pub(crate) fn f32_to_com<D: Dimension>(a: Array<f32, D>) -> Array<Com, D> {
    (FRACTION * a).map(|&x| Com(Wrapping(x as i16)))
}

/// Converts an array of `Com` values to an array of `f32` values.
#[inline]
pub(crate) fn com_to_f32<D: Dimension>(a: Array<Com, D>) -> Array<f32, D> {
    a.map(|&x| x.0 .0 as f32) / FRACTION
}

/// Adjusts the product of to `Com` values to maintain the correct scaling of the fixed-point representation.
///
/// The `adjust_product` function should be called on the result of a multiplication operation involving two `Com`
/// values. It divides the product by the scaling factor `FRACTION` to ensure that the final result has the
/// correct fixed-point representation.
///
/// # Example
/// ```rust
/// let a: Com = Com(Wrapping(0x1000));
/// let b: Com = Com(Wrapping(0x2000));
/// let product = a * b;
/// let adjusted_product = adjust_product(product);
/// ```
///
/// In this example, the `product` of `a` and `b` would have a value of `0x4000000`, which is too large to
/// fit in the 16-bit `Com` representation. The `adjust_product` function divides the `product` by `FRACTION`
/// (which is `16` in this case) to produce the correct fixed-point result.
#[inline]
pub(crate) fn adjust_product<T: ops::Div<Com>>(a: T) -> <T as ops::Div<Com>>::Output {
    a / Com(Wrapping(1 << FRACTION_BITS))
}

impl identities::Zero for Com {
    fn zero() -> Self {
        Com(<Wrapping<i16> as identities::Zero>::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl identities::One for Com {
    fn one() -> Self {
        Com(<Wrapping<i16> as identities::One>::one())
    }

    fn is_one(&self) -> bool {
        self.0.is_one()
    }
}

// TODO enable this when concat_idents becomes stable
// macro_rules! impl_com_identities {
//     ($trait:ident, $func:ident) => {
//         impl identities::$trait for Com {
//             fn $func() -> Self {
//                 Com(<Wrapping<i16> as identities::$trait>::$func())
//             }

//             fn concat_idents!(is_, $func)(&self) -> bool {
//                 concat_idents!(self.0.is_, $func)()
//             }
//         }
//     };
// }

// impl_com_identities!(Zero, zero);
// impl_com_identities!(One, one);

macro_rules! impl_com_ops {
    ($trait:ident, $func:ident) => {
        impl ops::$trait for Com {
            type Output = Self;

            fn $func(self, rhs: Self) -> Self::Output {
                Com(self.0.$func(rhs.0))
            }
        }
    };
}

impl_com_ops!(Add, add);
impl_com_ops!(Sub, sub);
impl_com_ops!(Mul, mul);
impl_com_ops!(Div, div);

impl ScalarOperand for Com {}

impl serde::Serialize for Com {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Com {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Com(Wrapping::<i16>::deserialize(deserializer)?))
    }
}
