# proximity — exact tools for proximity gaps & list decoding near capacity

A Rust toolkit (with Python bindings) for computationally exploring
**proximity gaps, correlated agreement, and list decoding near capacity** for
the smooth-domain Reed–Solomon codes used in SNARKs — the setting of the
Proximity Prize survey (Arnon–Boneh–Fenzi, ePrint 2026/680).

The design premise: in this problem area, *cheap exact computation* is a
first-class research instrument. Every number the toolkit produces is either a
certified exact count or clearly labeled otherwise, and every kernel is pinned
to independently-verified golden values (see [Validation](#validation)).

## The objects

For a prime `p` and the order-`s` multiplicative subgroup `mu_s <= F_p^*`, the
**bucket** at `lambda = (lambda_1, ..., lambda_q)` counts the `r`-subsets
`S` of `mu_s` with elementary symmetric values `e_i(S) = lambda_i`, `i <= q`.
Buckets are the computational heart of the landscape:

- they are **exact list sizes** of the extremal ("C.5-form") words beyond the
  Johnson radius — the words behind the known list-decoding lower bounds near
  capacity;
- their **occupied supports** are exact winning sets for the survey's Section-6
  toy protocol (soundness = occupied / p for the canonical attack pair);
- their **structural maxima** follow the quantized ladder
  `M_struct(s, r, q) = C(s/2^t - [r0 != 0], floor(r/2^t))`,
  `t = ceil(log2(q+1))`;
- their arithmetic inflation is a weighted count of **kernel census** vectors
  (`sum v_i w^i = 0 mod p`), arriving in dilation orbits of size `s` and
  confined by the norm law `N(v) <= (sum v_i^2)^{s/4}` to primes below
  `~w^{s/4}` — small-weight accidents cannot reach the structural regime.

## Architecture

Bottom-up, each layer depending only on those below:

| module | object | role |
|---|---|---|
| `field` | `F_p` scalars | mulmod/powmod, Miller–Rabin, generators, Pollard-rho factorization |
| `domain` | `Subgroup` | the validated core object: `mu_s`, cosets, dilation structure |
| `code` | `ReedSolomon` | radii (capacity/Johnson), C.5 window, rung words, ladder values |
| `buckets` | distributions & queries | `dp`: full distributions (cost ~ `p`); `mitm`: single buckets at any `q` and decompositions (**`p`-independent** — interrogate primes of any size) |
| `census` | kernel vectors | direct (weight-capped, any `s`) and MitM (full, `s <= 32`) engines |

The cost split is the strategic point: use `dp` only when you need the max over
*all* `lambda`; use the `p`-independent `mitm` engines to ask targeted
questions at primes of any magnitude.

## Usage

**Rust:**

```rust
use bucketlab::{domain::Subgroup, buckets};

let sg = Subgroup::new(3457, 32)?;
let dist = buckets::dp::distribution_q1(&sg, 16)?;      // all buckets, exact
let (max, lambda) = dist.max();                         // 220134 at lambda = 0
let t = buckets::mitm::HalfTables::build(&sg, 16, 2)?;  // p-independent engine
let rung = bucketlab::code::rung_lambda(&sg, 16, 2)?;
assert_eq!(t.bucket(&rung)?, 422);                      // exact q=2 list size
```

**CLI** (`cargo install --path .` or `cargo run --release --bin bucketlab --`):

```
bucketlab info      --p 3457 --s 32
bucketlab rung      --p 3457 --s 32 --r 16 --q 2
bucketlab bucket    --p 89633 --s 32 --r 16 --lam 0
bucketlab decompose --p 77569 --s 32 --r 16 --lam 0
bucketlab census    --p 89633 --s 32 --cmax 2
bucketlab sweep     --s 32 --r 16 --pmax 300000 --csv > landscape.csv
```

**Python** (`pip install maturin && maturin build --release --features python
&& pip install target/wheels/*.whl`):

```python
import bucketlab as bl, numpy as np
d = np.asarray(bl.bucket_dist_q1(89633, 32, 16))   # full exact distribution
bl.bucket_e(3457, 32, 16, bl.rung_lambda_e(3457, 32, 16, 2))   # -> 422
```

## Performance

Apple M-series, release build: a full landscape campaign — every prime
`p = 1 mod 32` below 300k (1,622 primes), exact q=1 distribution + max +
low-weight census each — runs in ~15 s (`examples/bench_sweep.rs`). The whole
golden test suite (six s=32 distributions, a q=2 joint grid, censuses,
decompositions) runs in ~0.3 s. Single full DPs are memory-bandwidth-bound;
the MitM engines answer single-bucket questions in milliseconds at any `p`.

## Validation

`cargo test --release` runs the golden + property suite: pinned
exhaustively-verified values (bucket maxima at 12 primes, joint-grid extrema,
censuses, rung buckets through q=8, a to-the-unit bucket decomposition) plus
invariants (mass = `C(s,r)`, dilation symmetry, DP↔MitM agreement). CI enforces
fmt, clippy `-D warnings`, the suite, CLI smoke tests, and Python-binding
parity on every push. The contract for new kernels is in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Roadmap

Tracked as GitHub issues: norms & bad-set enumeration, the spectrum module
(character sums / dilated Gauss periods), toy-protocol winning-set tools,
q=3 grid DP, CRT dual-residue counts for `s >= 128`, the `s = 64` MitM
sort-join, Montgomery arithmetic, criterion benches.

## License

MIT or Apache-2.0, at your option.
