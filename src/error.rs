//! Error type for the public API. Construction of the core types
//! ([`crate::domain::Subgroup`], [`crate::code::ReedSolomon`]) validates
//! inputs once, so the analysis kernels can assume well-formed parameters.

/// Errors returned by the public API.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The claimed field characteristic is not prime.
    #[error("{0} is not prime")]
    NotPrime(u64),
    /// The requested subgroup order does not divide `p - 1`.
    #[error("subgroup order {s} does not divide p - 1 = {pm1}")]
    OrderDoesNotDivide {
        /// Requested subgroup order.
        s: u64,
        /// `p - 1` for the given prime.
        pm1: u64,
    },
    /// A numeric parameter is outside its valid range.
    #[error("parameter out of range: {0}")]
    OutOfRange(String),
    /// The operation is not supported at these parameters (e.g. an engine's
    /// size limit); the message names the requirement.
    #[error("operation requires {0}")]
    Unsupported(String),
}

/// Convenience alias used across the crate.
pub type Result<T> = std::result::Result<T, Error>;
