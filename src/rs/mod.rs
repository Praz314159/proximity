//! Generic Reed–Solomon codes and the list-decoding **discovery** machinery —
//! the `RS[F_p, D, k]` layer over any evaluation domain, decoupled from the
//! smooth-subgroup structure: the code ([`code`](crate::rs::code)),
//! exact/sampled list decoding ([`decode`](crate::rs::decode)), bottom-up
//! cluster growth and optimization ([`cluster`](crate::rs::cluster)), and
//! the graded structure diagnostic ([`classify`](crate::rs::classify)),
//! together with the dual (quotient) view of the same code — the
//! vanishing-syndrome geometry [`vs`](crate::rs::vs) pairing with the
//! primal [`code`](crate::rs::code).

pub mod classify;
pub mod cluster;
pub mod code;
pub mod decode;
pub mod descent;
pub mod linalg;
pub mod moments;
pub mod vs;
