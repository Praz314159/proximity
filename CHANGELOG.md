# Changelog

## Unreleased

- **error**: `Error` is now `#[non_exhaustive]`; module docs state the
  crate-wide error policy (all public-API fallibility flows through
  `Error`; internal invariants use `expect` with messages naming the
  invariant; no bare `unwrap` on library paths). `EvalDomain::from_points`
  reports a composite modulus as `Error::NotPrime` instead of a generic
  `OutOfRange` string, so callers can match the variant.
- **py**: one central `Error -> PyErr` mapper — I/O failures raise
  `IOError`, engine/regime limits (`Error::Unsupported`) raise
  `NotImplementedError` (previously `ValueError`), validation failures
  raise `ValueError`.
- **internal**: every remaining `unwrap` on a library path replaced by an
  invariant-naming `expect` or a restructure (array destructuring for the
  MITM half-tables, `total_cmp` for the CLI's float sort — no NaN panic).

- **domain**: `EvalDomain::from_points` now validates the full construction
  contract — `p` prime and every point reduced `< p` — closing a silent
  wrong-answer path on the generic-domain decode API (issue #6).
- **py**: the long-running discovery bindings (`list_decode`,
  `anneal_pencil`, `optimize_pencil`) and the p-scale DP bindings
  (`bucket_dist_q1/q2`, `dist_stats_q1`) now release the GIL for the whole
  computation; `buckets_e` additionally fans out over `lams` with rayon
  (issue #7).
- **decode**: the exact list decoder parallelizes over the leading index of
  the information set (rayon; branch merge reproduces the serial lex order
  exactly), batches all interpolation inversions with Montgomery's trick
  (two Fermat exponentiations per combination instead of ~(n-k)(k+1)), and
  clones codewords only when new instead of once per passing combination
  (issue #8).
- **field / decode**: `checked_binom` (`None` on u64 overflow); the exact
  decoder's cap checks use it, so an oversized instance (e.g. C(128, 64))
  returns `Unsupported` instead of aborting the process from FFI (issue #9).
- **py**: the discovery layer is now fully exposed — `optimize_word` (warm-start
  greedy climb from a given word), `pencil_seed`, `decode_profile` (decode +
  the full `classify::structure` profile in one call), plus theorem-word
  constructors `c5_word`, `top_word` (Theorem B_mult), `word_from_syndrome`,
  and `gs_class_counts`. `list_decode` and `optimize_word` return members as
  `(L, n)` uint64 arrays; `buckets_e` returns a uint64 array (issue #10).
- **rs::moments / rs::linalg (new)**: the syndrome/moment layer as audited
  kernels — `moment_cloud` (lex-ordered complement e-vectors), `cut_counts`
  (streaming bulk cut sizes), `cut_max_sparse` (the exhaustive 3-/4-support
  certification kernel, rayon; reproduces 698 @ (1,2,6)/p=257 and 3074 @
  (1,3,5,7)/p=97 as golden pins) — plus dense F_p linear algebra
  (`rref_mod`, `nullspace_mod`, `reduce_mod_span`, batch `inv_mod`,
  `e_syms`, `dd_rows`) with the divided-difference identity pinned against
  the theorem word (issue #11). `batch_inv` promoted to `field`.
- Docs/tests polish: CONTRIBUTING architecture guide synced to the actual
  rs/ + smooth/ layout (was pre-split); coverage tests for
  `Radius::from_delta`, `grow_from_pencil`, `sample_list`,
  `second_moment`, `signed_roundtrip`, `primes_one_mod`;
  `top_elementary_symmetric` moved to `field` (re-exported from rs::code);
  `census_direct` takes `cmax: i64` (matching mitm); `Error::Io` maps to
  `IOError` in Python; `cosets` guards the `2^t` shift; stale
  `badset_from_gpu_json` docstring sentence removed; thiserror 2 (issue #12).
- **rung**: `top_word`, `word_from_syndrome`, `gs_class_counts` in Rust with
  golden pins (810 at 65537 AND the accident prime 97; 715; 17,678,835; the
  e_1-coordinate cut = 70).

## v0.4.0 (2026-07-05)

- **norms::ingest**: `norms_ingest` moved under its parent domain as
  `norms::ingest` (Rust path change only; the Python API is unchanged).
  Streaming ingestion of GPU norm-table shards (JSON and per-weight binary
  dumps) with Galois normalization and provenance flags.
- **field**: Montgomery arithmetic behind `is_prime` and Brent-variant
  Pollard rho (12.6x on hard semiprimes); `primes_one_mod` sweep iterator.
- **error**: `Io` and `MalformedInput` variants — ingest I/O and parse
  failures no longer masquerade as `Unsupported`.
- Idiomatic pass: `#[must_use]` across the pure API, missing derives
  (`HalfTables`, `NormTable`, `Certificate`), `census::direct` takes
  `cmax: i64` (matching `mitm`), shared `census::kernel_side` enumeration
  guaranteeing census/decomposition agreement by construction.

## v0.3.0 (2026-07-04) — public-release candidate

- **attack**: threshold calculator — `best_attack` (quantized-ladder optimum),
  `antipodal_attack` (survey Table-5 baseline, reproduced before improvement),
  `hyperbola_ceiling`, `elias_delta_star`; CLI `vanish attack`.
- **certify**: tiered, p-independent structural certificates for the q=1
  landscape (census-based; exact inflated-bucket anatomy otherwise); CLI
  `vanish certify`.
- **toy**: exact soundness of the survey's Section-6 toy protocol via the
  winning-set identity (`Omega` = occupied buckets); CLI `vanish toy`.
- Hardening: `#![forbid(unsafe_code)]`, complete public docs
  (`#![warn(missing_docs)]` clean), MSRV 1.77, seeded randomized property
  tests, full error-path coverage, CLI integration tests, Python docstrings +
  `vanish.pyi` stubs in the wheel, pytest suite in CI.

## v0.2.0 (2026-07-04)

- Restructure around core machinery: `field` → `domain::Subgroup` →
  `code::ReedSolomon` → analyses (`buckets::{dp, mitm}`, `census`);
  Result-based APIs; CLI binary; dual MIT/Apache-2.0; CONTRIBUTING.md
  validation contract; GitHub Actions CI. Crate renamed to `vanish`.

## v0.1.0 (2026-07-04)

- Initial kernels: q=1/q=2 bucket DPs, kernel censuses, any-q MitM buckets,
  bucket decomposition; PyO3 bindings; golden test suite.
