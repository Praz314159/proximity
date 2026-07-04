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

Layers, bottom-up (each depends only on those below):

- `field` — scalar arithmetic over `F_p`, primality, factorization.
- `domain` — `Subgroup`: the validated core object every analysis takes.
- `code` — Reed–Solomon on a subgroup, radii, ladder combinatorics, rung words.
- `buckets`, `census` — the analyses. Full-distribution engines scale with `p`;
  meet-in-the-middle engines are `p`-independent.

New analyses should follow the same shape: take `&Subgroup` (validate nothing
downstream), return `Result`, keep hot loops allocation-free, parallelize with
rayon at the outermost natural loop.

## Good first contributions

See the issue tracker; the standing wishlist includes: norms & bad-set
enumeration, the spectrum module (character sums / Gauss periods via FFT),
toy-protocol winning-set tools, q=3 grid DP, CRT dual-residue counts for
s ≥ 128, the s = 64 MitM sort-join, Montgomery arithmetic in hot loops, and
criterion benchmarks.

## License

Dual MIT/Apache-2.0; contributions are accepted under the same terms.
