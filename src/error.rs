//! Error type for the public API. Construction of the core types ([`crate::domain::Subgroup`],
//! [`crate::code::ReedSolomon`]) validates inputs once, so the analysis kernels can assume
//! well-formed parameters.

/// Errors returned by the public API.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0} is not prime")]
    NotPrime(u64),
    #[error("subgroup order {s} does not divide p - 1 = {pm1}")]
    OrderDoesNotDivide { s: u64, pm1: u64 },
    #[error("parameter out of range: {0}")]
    OutOfRange(String),
    #[error("operation requires {0}")]
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, Error>;
