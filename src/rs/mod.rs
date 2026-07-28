//! Generic Reed–Solomon codes and the list-decoding **discovery** machinery —
//! the `RS[F_p, D, k]` layer over any evaluation domain, decoupled from the
//! smooth-subgroup structure: the code ([`code`]), exact/sampled list decoding
//! ([`decode`]), bottom-up cluster growth and optimization ([`cluster`]), and
//! the graded structure diagnostic ([`classify`]), together with the
//! dual (quotient) view of the same code — the vanishing-syndrome
//! geometry [`vs`] pairing with the primal [`code`].

pub mod classify;
pub mod cluster;
pub mod code;
pub mod decode;
pub mod linalg;
pub mod moments;
pub mod vs;
