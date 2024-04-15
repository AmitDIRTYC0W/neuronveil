use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub struct UnexpectedMessageError {}

impl Error for UnexpectedMessageError {}

impl fmt::Display for UnexpectedMessageError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Unexpected message",)
    }
}
