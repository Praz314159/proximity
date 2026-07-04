# proximity / bucketlab

Fast exact kernels for computationally exploring **proximity gaps, correlated
agreement, and list decoding near capacity** for the smooth-domain Reed–Solomon
codes used in SNARKs (the setting of the Proximity Prize survey, ePrint
2026/680). Rust core + PyO3 bindings; Python is the lab bench, Rust carries the
hot loops.

The central objects are *buckets*: for the order-s subgroup mu_s of F_p*, the
bucket at lambda = (lambda_1..lambda_q) counts r-subsets S of mu_s whose top
elementary symmetric functions e_1(S)..e_q(S) equal lambda. Via the
useful-family framework (Appendix C of the survey), bucket sizes are exactly
the list sizes of extremal words beyond the Johnson radius, kernel censuses
govern their arithmetic inflation, and occupied-bucket supports are exact
winning sets for the survey's Section-6 toy protocol. This crate computes all
of these exactly.

## Kernels

| function | what it computes | exactness | scale |
|---|---|---|---|
| `bucket_dist_q1(p, s, r)` | full distribution N(λ) of e₁-buckets over F_p | exact u64 (s ≤ 64) | p to ~10⁸ (memory-bound) |
| `bucket_dist_q2(p, s, r)` | full joint (e₁,e₂) distribution | exact u64 | p ≤ ~700 |
| `census_direct(p, s, cmax, wmax)` | kernel vectors, coeffs in [−c,c], weight ≤ w | exact | C(s/2,w)(2c)^w ops |
| `census_mitm(p, s, cmax)` | full census by weight (MitM halves) | exact | s ≤ 32 at c=2 |
| `bucket_e / buckets_e(p, s, r, q, λ)` | exact single buckets, **any q** | exact | s ≤ 32 |
| `rung_lambda_e(p, s, r, q)` | Theorem-A rung λ (exp20c conventions) | — | — |
| `decompose_bucket_q1(p, s, r, λ)` | ε-class decomposition (the anatomy law) | exact | s ≤ 32 |

## Build & test

```
cargo test --release            # golden + property suite, no Python needed
cargo run --release --example bench_sweep
maturin build --release --features python && pip install target/wheels/*.whl
```

## Measured throughput (Apple M-series)

- Full sweep, all 1,622 primes ≡ 1 mod 32 below 3×10⁵, DP + max + census ≤ 4
  per prime: **15 s total (9.3 ms/prime)** — the exp17 campaign was ~260 primes
  in minutes.
- Single DP: the rotated-add kernel is memory-bandwidth-bound, so numpy is
  already near-optimal per prime (0.036 s vs 0.044 s at s=32, p=180001);
  the wins are the Python-loop kernels (census/MitM/decomposition: 10–100×)
  and rayon parallelism across primes.
- Whole golden test suite (six s=32 DPs, q=2 joint DP, censuses,
  decomposition): 0.31 s.

## Validation contract

Every kernel is pinned to exhaustively-verified values from an independent
Python/numpy reference implementation (`tests/golden.rs`): maxN at 12 primes
across s=16/32, the q=2 joint extrema at p=97, censuses at p=89633/65537, rung
buckets for q ≤ 8, the p=77569 decomposition to the unit with its weight
profile, plus property tests (mass = C(s,r), dilation invariance, DP↔MitM
agreement). The q≥2 sign convention (c_i = (−1)^i e_i) is documented at the
triangularization site — it was a real bug once, caught by exactly the DP
cross-check now encoded here. Optimization passes must keep this suite green.

## Roadmap (in leverage order)

1. **Adversarial-prime stress tests**: factor norms/resultants of chosen
   ±1-polynomials, measure buckets at the large prime factors directly.
2. **q=3 exhaustive distributions** at p ≤ ~450 (p³ grid DP — the natural next
   kernel, ~minutes here vs infeasible in numpy).
3. **s=64 crossover measured buckets**: MitM at 2³² half-subsets — u32-encoded,
   radix-partitioned sort-join, ~17 GB streamed; unlocks measured (not
   predicted) values at p ≈ 3×10⁹.
4. CRT dual-residue counts (two ~62-bit primes) for exact s ≥ 128 DPs.
5. Montgomery/Barrett reduction in the MitM inner loops; wgpu census kernel if
   CPU saturates.
