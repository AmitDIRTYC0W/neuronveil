#![feature(new_uninit)]

mod com;
pub mod split;
mod unexpected_message_error;
pub use com::Com; // TODO shouldn't be pub
pub(crate) mod bit;
mod bitxa;
pub mod client;
mod layer;
pub mod message;
pub mod model;
mod multiplication_triplet_share;
pub mod server;
pub(crate) use bitxa::bitxa;
pub(crate) mod reconstruct;
pub(crate) mod signed_comparison;
