# Changelog

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
