# Contributing

This toolkit's value is *exactness*: every number it produces is either a
certified exact count or clearly labeled otherwise. Contributions are welcome —
new kernels especially (see the open issues) — under one non-negotiable rule:

## The validation contract

Every kernel (new or optimized) must ship with:

1. **Golden pins** — at least one parameter point where the kernel's output is
   checked against an *independent* computation (a brute-force enumeration, a
   second algorithm, or a published exhaustively-verified value). Add them to
   `tests/golden.rs`. Never pin a value produced by the kernel under test.
2. **Property tests** — the invariants that must hold identically:
   - total mass = `C(s, r)` for distributions;
   - dilation invariance (`N(g·lambda) = N(lambda)` for `g` in the subgroup);
   - cross-engine agreement wherever two kernels overlap (DP ↔ MitM,
     direct ↔ MitM census).
3. **A cost note** in the module docs: what the cost scales with (`p`?
   `2^{s/2}`? `C(s/2, w)(2c)^w`?), so users pick the right engine.

Why so strict: a sign-convention bug in the q≥2 triangularization once
produced plausible-but-wrong buckets and was caught *only* because a sampled
value exceeded an exhaustive cross-check. The suite is the institutional memory
of that lesson.

## Workflow

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test --release
```

All three must pass; CI enforces them. For Python-binding changes also run
`maturin build --release --features python` and check the bindings import.

## Architecture guide

Layers, bottom-up (each depends only on those below); this mirrors the
`lib.rs` module doc, which is the authoritative map:

- `field` — scalar `F_p` arithmetic, primality, factorization, binomials,
  batch inversion.
- `domain` — `MultiplicativeSubgroup` (`mu_s <= F_p^*`: cosets, dilation) and
  the generic `EvalDomain` an RS code sits on. Construction validates once;
  kernels assume well-formed inputs.
- `rs/` — generic Reed-Solomon + the list-decoding **discovery** layer, over
  any evaluation domain: `code` (the code, C.5 words), `decode` (exact and
  sampled list decoding), `cluster` (pencil seeds, greedy/anneal search),
  `classify` (graded structure profiles), `moments` (the moment cloud and
  syndrome-cut kernels), `linalg` (dense F_p linear algebra). No subgroup
  structure leaks in here.
- `smooth/` — the smooth-subgroup program: `buckets` (DP and MitM engines),
  `rung` (ladder combinatorics, rung/theorem words), `census`, `norms`
  (+ `norms::ingest`), `certify`.
- Applications: `toy` (Section-6 soundness), `attack` (threshold calculator),
  and the `py` bindings.

New analyses should follow the same shape: validate at construction, return
`Result`, keep hot loops allocation-free, parallelize with rayon at the
outermost natural loop, and release the GIL in any binding that computes for
longer than a bincount.

## Good first contributions

See the issue tracker; the standing wishlist includes: norms & bad-set
enumeration, the spectrum module (character sums / Gauss periods via FFT),
toy-protocol winning-set tools, q=3 grid DP, CRT dual-residue counts for
s ≥ 128, the s = 64 MitM sort-join, Montgomery arithmetic in hot loops, and
criterion benchmarks.

## License

Dual MIT/Apache-2.0; contributions are accepted under the same terms.
