mod com;
mod shares;
pub mod split;
pub use com::Com; // TODO shouldn't be pub
pub mod model;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
