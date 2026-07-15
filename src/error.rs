//! Error type for the public API. Construction of the core types
//! ([`crate::domain::MultiplicativeSubgroup`], [`crate::rs::code::ReedSolomon`]) validates
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
    /// size limit); the message names the requirement. Never used for I/O or
    /// input syntax — see [`Error::Io`] and [`Error::MalformedInput`].
    #[error("operation requires {0}")]
    Unsupported(String),
    /// An ingest input file could not be read.
    #[error("reading {path}: {source}")]
    Io {
        /// The path that failed.
        path: String,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// An ingest input (GPU JSON shard or binary dump) is malformed; the
    /// message locates the defect (byte offset for JSON).
    #[error("malformed ingest input: {0}")]
    MalformedInput(String),
}

/// Convenience alias used across the crate.
pub type Result<T> = std::result::Result<T, Error>;
