# vanish — exact kernels for smooth-domain Reed–Solomon codes

[![CI](https://github.com/Praz314159/proximity/actions/workflows/ci.yml/badge.svg)](https://github.com/Praz314159/proximity/actions)

A Rust toolkit (with Python bindings) of exact computational kernels for
Reed–Solomon codes on smooth multiplicative domains: list decoding,
syndrome-cut geometry, bucket and value-map censuses, cyclotomic-ring
arithmetic, and per-prime certification.

The design premise: *cheap exact computation* is a first-class research
instrument. Every number the toolkit produces is either a certified exact
count or clearly labeled otherwise, and every kernel is pinned to
independently-verified golden values (see [Validation](#validation)).

## The objects

For a prime `p ≡ 1 (mod s)`, the domain is the order-`s` subgroup
`mu_s <= F_p^*`, and the code is `RS_k` evaluated on it. The library
computes with these through a small set of exact objects:

- **Words, two ways.** The primal view (`rs::code`) is words as functions:
  codewords, agreement sets, decoding. The dual view (`rs::vs`,
  `VsSpace`) is the same information through the quotient `F_p^s / RS_k`:
  a word matters only through its syndrome `b`, subsets pair with
  syndromes through elementary symmetric functions of complements, and
  list questions become incidence questions about the **cut**
  `Z(b) = { |S| = r : <b, e(comp S)> = 0 }`. `VsSpace` is the crate's
  convention authority: its certificate (subset ranking, moment rows,
  domain order, syndrome signs) is what every accelerated or external
  view must reproduce before its numbers are believed.
- **Buckets.** The bucket at `lambda = (lambda_1, ..., lambda_q)` counts
  the `r`-subsets `S` of `mu_s` with `e_i(S) = lambda_i`, `i <= q`. By
  the exactness theorem, buckets are exact list sizes of the frozen-head
  words `x^r - lambda_1 x^{r-1} + ... ± lambda_q x^{r-q}` beyond the
  Johnson radius; their structural maxima follow the quantized ladder,
  and their arithmetic inflation at a given prime is a weighted count of
  kernel vectors (`census::kernel`).
- **The ring.** `Z[zeta_s]` in exact arithmetic (`ring::Cyclo`): census
  values, norms at every height (`norm_mod` / `norm_i128` / `norm_crt`),
  the fold, and the fold units with certified alpha tables. What is
  computed in the ring is prime-independent: it descends to every good
  prime at once, and a bucket coincidence at `p` is exactly a norm
  divisible by `p` — the accident criterion behind the per-prime
  certificates (`smooth::certify`).
- **Decoding.** `rs::decode` enumerates the actual list of *any* word at
  a radius, over any evaluation domain — the measurement that the
  counting kernels are pinned against.

## Architecture

Bottom-up, each layer depending only on those below (`lib.rs` carries the
same map as the authoritative rustdoc):

| module | object | role |
|---|---|---|
| `field` | `F_p` scalars | mulmod/powmod, Montgomery Miller–Rabin, generators, Brent–Pollard-rho factorization |
| `domain` | `MultiplicativeSubgroup`, `EvalDomain` | the validated core objects: `mu_s`, cosets, dilation; generic evaluation domains |
| `ring` | `Z[zeta_s]` | `Cyclo` (negacyclic half-basis; norms at every height), the fold, fold units + certified alpha tables, exact negacyclic NTT |
| `rs::code` | `ReedSolomon` | the primal view: radii, frozen-head words, ladder values |
| `rs::decode` | lists | exact and sampled list decoding of arbitrary words, any domain |
| `rs::vs` | `VsSpace` | the dual view: syndromes, cuts, strata, cliques — and the convention certificate |
| `rs::moments` | the moment cloud | cloud materialization, bulk cut counts, exhaustive sparse-support certifications |
| `rs::cluster`, `rs::classify` | discovery | cluster growth around a moving center; the graded bucket-vs-entropy diagnostic |
| `rs::linalg` | dense `F_p` linear algebra | row reduction, nullspaces, batch inversion, divided-difference rows — the audited copy |
| `census` | every counting kernel | by what is counted: `buckets` (DP + `p`-independent MitM), `kernel`, `value` (exact ring censuses), `valuemap` (fibers mod `p`), `skeleton`, over the shared `join` layer |
| `smooth::rung` | ladder + closed forms | quantized-ladder combinatorics; `top_word`, GS-class counts |
| `smooth::norms` | bad sets | cyclotomic norms → complete per-prime accident inventories; `norms::ingest` streams GPU norm tables |
| `smooth::certify` | certificates | tiered `p`-independent proofs that buckets are exactly structural |
| `toy`, `attack` | applications | toy-protocol soundness; the attack-radius calculator |
| `gpu/` | campaign drivers | Python; each certifies itself against the `VsSpace` certificate before use |

The cost split is the strategic point: use `dp` only when you need the max
over *all* `lambda`; use the `p`-independent `mitm` engines to ask targeted
questions at primes of any magnitude; use the ring to answer a question at
every prime at once.

## Usage

**Rust:**

```rust
use vanish::{census::buckets, domain::MultiplicativeSubgroup};

let sg = MultiplicativeSubgroup::new(3457, 32)?;
let dist = buckets::dp::distribution_q1(&sg, 16)?;       // all buckets, exact
let (max, lambda) = dist.max();                          // 220134 at lambda = 0
let t = buckets::mitm::HalfTables::build(&sg, 16, 2)?;   // p-independent engine
let rung = vanish::smooth::rung::rung_lambda(&sg, 16, 2)?;
assert_eq!(t.bucket(&rung)?, 422);                       // exact q=2 list size
```

**Python** (all lines below run as shown):

```python
import vanish, numpy as np

d = np.asarray(vanish.bucket_dist_q1(89633, 32, 16))     # full exact distribution
vanish.bucket_e(3457, 32, 16, vanish.rung_lambda_e(3457, 32, 16, 2))  # -> 422

p, dom = 65537, list(vanish.subgroup(65537, 16))
w = [(pow(x, 7, p) + pow(x, 15, p)) % p for x in dom]    # a frozen-head word
vanish.list_decode(p, dom, 7, w, 8).shape                # -> (809, 16): its exact list

q, domq = 97, list(vanish.subgroup(97, 16))              # the dual view at p = 97
wq = [(pow(x, 7, q) + pow(x, 15, q)) % q for x in domq]
vs = vanish.VsSpace(q, 16, 7)
list(vs.strata_counts(vs.syndrome(wq)))                  # -> [0, 256, 416, 128, 9]

vanish.Cyclo([3 if i % 3 == 0 else -3 for i in range(32)]).norm_crt()
# exact 25-digit norm at s = 64, past the i128 range
```

**CLI** (`cargo install --path .` or `cargo run --release --bin vanish --`):

```
vanish info      --p 3457 --s 32
vanish rung      --p 3457 --s 32 --r 16 --q 2
vanish bucket    --p 89633 --s 32 --r 16 --lam 0
vanish decompose --p 77569 --s 32 --r 16 --lam 0
vanish census    --p 89633 --s 32 --cmax 2
vanish sweep     --s 32 --r 16 --pmax 300000 --csv > landscape.csv
vanish toy       --p 5767169 --s 16 --r 8
vanish certify   --p 1568247649 --s 32 --r 16
vanish attack    --n 2097152 --k 1048576 --list-bits 57.93 --base-bits 31
```

**Pods / remote campaigns.** Two routes to `vanish` on a bare Linux pod:
build on the pod (`curl https://sh.rustup.rs -sSf | sh -s -- -y &&
pip install maturin --break-system-packages`, copy the repo,
`pip install . --break-system-packages`), or cross-build a manylinux wheel
locally (`pip install maturin[zig] && rustup target add
x86_64-unknown-linux-gnu && maturin build --release --zig --target
x86_64-unknown-linux-gnu`) and `scp` the wheel.

## Performance

Apple M-series, release build: a full landscape campaign — every prime
`p = 1 mod 32` below 300k (1,622 primes), exact q=1 distribution + max +
low-weight census each — runs in ~15 s (`examples/bench_sweep.rs`). The whole
golden test suite runs in ~0.3 s. Single full DPs are memory-bandwidth-bound;
the MitM engines answer single-bucket questions in milliseconds at any `p`;
the exact decoder is output-sensitive and parallel.

## Validation

`cargo test --release` runs the golden + property suite: pinned
exhaustively-verified values (bucket maxima at 12 primes, joint-grid extrema,
censuses, rung buckets through q=8, a to-the-unit bucket decomposition, the
skeleton-census pins at levels 32/64 — whose level-64 census reproduced the
independently measured `N(128) = 3,758,482,820` exactly) plus invariants
(mass = `C(s,r)`, dilation symmetry, DP↔MitM agreement, primal↔dual
syndrome identities). Accelerated views (GPU, external mirrors) must
reproduce the `VsSpace` convention certificate before their numbers are
accepted. CI enforces fmt, clippy `-D warnings`, the suite, CLI smoke tests,
and Python-binding parity on every push. The contract for new kernels is in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Roadmap

Tracked as GitHub issues; the current slate: the certified volumes engine
port (#29), the rs/pencil descent layer (#30), sign-resolved
marked-complement enumeration (#31), the per-prime cleanliness certifier
`certify_clean(p, level)` (#2), norms & bad-set extensions (#1), the
`s = 64` MitM sort-join and q=3 grid DP (#4), GPU kernels (#15–#18),
Montgomery arithmetic in hot loops, criterion benches.

## License

MIT or Apache-2.0, at your option.
