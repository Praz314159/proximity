# Design sketch: a `Cyclo` (negacyclic ring) type

**Status:** IMPLEMENTED (2026-07-24) as src/ring/{mod,cyclo,ntt}.rs — fold primitive + Cyclo + radix-2 negacyclic NTT with two-prime exact CRT products; census hot loop migrated onto fold with no bench regression; python surface vanish.Cyclo / vanish.fold. MIGRATION COMPLETE (2026-07-25): norms.rs embedding-exponent reduction now routes through ring::fold (hoisted per-support fold tables, half-basis power tables; enumeration stays flat per caveat 1), and the certify path is glued to the ring by tests — norm_table ≡ Cyclo::norm_i128 entry-for-entry at s=8, census ≡ the Cyclo::eval_at kernel, certificate degradation at an accident prime ≡ Cyclo::norm_mod divisibility. Remaining from this doc: HEXL-style preconditioned butterflies as the known perf upgrade. Purpose: check whether one type cleanly
absorbs the existing fold conventions in `domain` / `census` / `norms` /
`certify` before we refactor.

## Why

The recurring bug class in this program is the *negacyclic fold*: representing an
element of `Z[zeta_s]` on the half-basis `{1, zeta, ..., zeta^{s/2-1}}` and
mis-handling the relation `zeta^{s/2} = -1` when an exponent crosses `s/2` (or
`s`). The lab log records this as the "third sign bug" (exp31c: exponents mod 32
not 16, `zeta^16 = -1` lost) and a "fourth fold bug" (exp33: eta-index `= e/2`).
Every instance is exponent-reduction + sign on the cyclotomic side. A type that
makes the fold impossible to get wrong targets exactly that class.

Key fact that makes this clean: for `s` a power of two,
`Z[zeta_s] = Z[x]/(Phi_s(x)) = Z[x]/(x^{s/2}+1)` — the negacyclic ring. The
half-basis *is* the power basis, and the fold *is* the ring relation.

## Sketch

```rust
/// An element of Z[zeta_s] = Z[x]/(x^{s/2}+1), s a power of two.
/// Coefficients on the half-basis {1, zeta, ..., zeta^{s/2-1}}; the relation
/// zeta^{s/2} = -1 is enforced by every constructor, so exponent-reduction sign
/// errors are structurally impossible.
pub struct Cyclo {
    coeffs: Vec<i64>, // length s/2; element = sum coeffs[i] * zeta^i
}

impl Cyclo {
    fn half(&self) -> usize { self.coeffs.len() }

    /// zeta^exp with the fold: reduce exp mod s, then fold the top half with a
    /// sign. THE operation the bugs kept getting wrong — defined once.
    pub fn monomial(half: usize, exp: usize) -> Self {
        let s = 2 * half;
        let e = exp % s;
        let mut c = vec![0i64; half];
        if e < half { c[e] = 1 } else { c[e - half] = -1 }
        Cyclo { coeffs: c }
    }

    pub fn add(&self, o: &Cyclo) -> Cyclo { /* coeff-wise */ }
    pub fn negate(&self) -> Cyclo { /* coeff-wise */ }

    /// Negacyclic convolution: zeta^{s/2} = -1 wraps the overflow with a sign.
    pub fn mul(&self, o: &Cyclo) -> Cyclo { /* wrap i+j >= half -> -coeff */ }

    /// Dilation by zeta^d (the orbit action census counts in size-s orbits).
    pub fn dilate(&self, d: usize) -> Cyclo { self.mul(&Cyclo::monomial(self.half(), d)) }

    /// The map to F_p that census/certify vanish-test against: sum c_i w^i.
    pub fn eval_at(&self, w: u64, p: u64) -> u64 { /* one Horner pass */ }

    /// Field norm N = prod over embeddings; the anticorrelation law bounds it by
    /// (sum c_i^2)^{s/4}. Centralizes what norms.rs computes ad hoc.
    pub fn norm(&self) -> i128 { /* resultant(coeffs, x^{half}+1) */ }

    pub fn weight(&self) -> usize { self.coeffs.iter().filter(|&&c| c != 0).count() }
    pub fn sq_sum(&self) -> i128 { self.coeffs.iter().map(|&c| (c as i128).pow(2)).sum() }
}
```

## Absorption analysis

| Current site | Concept | Absorbed? |
|---|---|---|
| `domain::pow_table(s/2)` | half-basis powers `[w^0..w^{s/2-1}]` | **Yes** — these *are* the images of `{1..zeta^{s/2-1}}`; `eval_at` consumes them. The half-basis convention gets a name. |
| `census` kernel enumeration | vectors `v`, bounded coeffs, `sum v_i w^i = 0 (mod p)` | **Yes** — "bounded-coeff `Cyclo` in the kernel of `eval_at`." Census *is* the kernel of the eval map. |
| `census` dilation orbits (size `s`) | multiply by `zeta^d` | **Yes** — `Cyclo::dilate`; currently implicit. |
| `norms` `N(v)` + `N(v) <= (Σv²)^{s/4}` | field norm + anticorrelation bound | **Yes** — `Cyclo::norm` / `sq_sum`; the law becomes a property of the type. |
| `certify` eps-differences `{-1,0,1}^{s/2}` | merge vectors, vanish mod `p` | **Yes** — same ring, same eval test. |
| `buckets::mitm` `c_i = (-1)^i e_i` | vanishing-poly / symmetric-function sign | **No** — different ring (`F_p[Y]`, subset elements). Already has its own institutionalized convention. Leave it. |

**Verdict: it cleanly absorbs the entire accident/cyclotomic side** — which is
precisely where every logged fold bug lived — **and correctly does *not* reach
into the bucket side**, whose signs are a separate concern already handled.

## Two caveats that decide the shape

1. **Keep it out of the hot loops.** `census::mitm` and `buckets::mitm`
   enumerate `~(2c+1)^{s/4}` items with flat `u64` accumulators and *zero
   allocation*. A `Cyclo { Vec<i64> }` per item would be a severe regression.
   So `Cyclo` is the layer for **construction, orbit/norm reasoning, `certify`,
   and the Python boundary** — not the inner enumeration. The kernels stay flat.
   To still get one source of truth for the fold, factor the single primitive

   ```rust
   #[inline]
   fn fold(half: usize, exp: usize) -> (usize, i64) { // (index, sign)
       let e = exp % (2 * half);
       if e < half { (e, 1) } else { (e - half, -1) }
   }
   ```

   and have *both* `Cyclo::monomial` and the flat loops call it. That kills the
   bug class without paying allocation.

2. **The real payoff is at the vanish↔experiment boundary.** vanish's Rust
   already gets the fold right (implicit half-basis discipline, golden-tested).
   The bugs were in the Python experiments re-deriving it. So the win is
   exposing `Cyclo` (and `fold`) through the pyo3 bindings so campaigns *call*
   the correct fold instead of reimplementing it — the concrete form of "kernels
   in vanish, campaigns call vanish."

## Recommendation

Worth doing, scoped tightly: introduce `src/ring.rs` with `Cyclo` + the shared
`fold` primitive; migrate `norms`/`certify`/orbit reasoning and the Python
surface onto it; refactor `census`/`buckets` inner loops to call `fold` but keep
their flat representation. Do **not** make `Cyclo` the census element type. This
is the one abstraction (besides the `ListOracle` seam) that pays for itself,
because it retires a demonstrated, repeated bug class rather than adding
structure for its own sake.
```
