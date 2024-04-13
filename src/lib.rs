mod com;
pub mod split;
pub use com::Com; // TODO shouldn't be pub
pub mod message;
pub mod model;
pub mod server;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
