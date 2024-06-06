use fixed::{FixedI32, Wrapping};
use ndarray::{Array, ShapeBuilder};
use ring::rand::{self, SecureRandom};

/// A fixed-point number that is used for communication (hence the name 'Com') and upon which cryptography is performed.
/// The fixed-point number is represented using a 16-bit signed integer, with the number of bits used for the fractional part
/// defined by the `FRACTION_BITS` constant.
// pub type Com = Wrapping<FixedI16<4>>;
pub type Com = Wrapping<FixedI32<2>>;

pub(crate) fn sample<Sh: ShapeBuilder>(shape: Sh, rng: &dyn SecureRandom) -> Array<Com, Sh::Dim> {
    // TODO implement RandomlyConstructable to avoid copying
    Array::from_shape_simple_fn(shape, || {
        Com::from_le_bytes(rand::generate(rng).unwrap().expose())
    })
}
