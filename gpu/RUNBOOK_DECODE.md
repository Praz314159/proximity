# Pod session — decode experiments (2026-07-16)

Setup: RunPod A100, `pip install cupy-cuda12x numpy` + the vanish wheel.
Ship this `gpu/` directory including `w16_mult.npy/.json`, `w32_mult.npy/.json`
(built locally by `construct_word.py`; rebuildable on-pod, needs only vanish).

Every experiment below has a pre-registered prediction. Run in order.

## Gate (mandatory, ~2 min)

```
python decode_gpu.py --validate
```
Must print PASS on all three: bucket x^8 (70), shifted, and **mult-word
(cpu = gpu = count = predicted = 715, frozen=True)**. The mult-word row also
exercises the count kernel (lex-first dedup) end to end. No PASS → stop.

## Experiment 1 — multiplicative exactness at s=32 (~10 min)

```
python decode_gpu.py --p 2130706433 --s 32 --k 16 --t 17 \
    --word-file w32_mult.npy --count --out-ids w32_members_sample.npy
```
**Prediction: distinct list size = 17,678,835** (= C(32,17)/32, the exact
class count; see w32_mult.json). Rank identity already verified at s=32
during construction (kernel dim 17 = k+1).

Post-check on the sampled IDs (host, seconds): expand via `_interp_full`,
verify every member's agreement set has size exactly 17 and exponent-sum
== 0 (c*) mod 32. Readings:
- exact hit -> multiplicative Theorem B holds at scale; q=1 sup at s=32 KNOWN
- low       -> realizability degrades with s; theory truncates at s=16
- high      -> something beyond the class exists (most interesting)

## Experiment 2 — q-decay curve at s=32, rate 1/2 (~1–3 h)

```
python concentration.py --p 2130706433 --s 32 --cells "16:18,16:19,16:20,14:16" --nseed 16
```
(16:17 deliberately absent — its optimizer pool is C(32,16)≈601M, infeasible;
Experiment 1 covers it by construction.) Output columns include the growth
law (median vs n/(n−t)) and exact-t fraction per cell.

Context values: s=16 q-decay measured 810 (q=1) → 14 (q=2) → ≲6 (q=3);
additive ladder m_struct(32,18,2)=70-scale; exponent-GS-2 classes proved
DEAD at s=16 (kernel = code exactly). Key readings per cell:
- K(16:18) >> 70 and unfrozen -> the q=2 tail has an unidentified mechanism
- K(16:18) ~ 70-scale        -> additive structural floor rules the tail
Extract members of the best cell for invariant checks (frozen exp-sum,
exp-sum^2, additive e1) — the s=16 lesson: test MULTIPLICATIVE invariants.

## Experiment 3 — accident tail (~1 h)

```
python concentration.py --p 77569 --s 32 --cells "16:18,16:19" --nseed 16
```
(77569 = accident-rich s=32 prime, badset_s32; 89633 as alternate.)
q=1 is provably p-free. Question: is the q>=2 tail p-uniform too?
- same tail as Exp 2 -> the defense is a p-free combinatorics problem
- inflated tail      -> accidents re-enter the proof as the tail error term

## Notes

- Count mode: lex-first on-device dedup — counts are exact with NO emit cap;
  sampled IDs use FNV hash & sample_mask (default 0xFF ≈ 1/256 → ~69k ids at
  17.7M, scap 2^20 ample). Algorithm validated by exact CPU mirror at s=16.
- decode_gpu emit-mode cap 2^26 applies only to the legacy kernel; do NOT use
  emit mode at q=1/s=32 (17 threads hit each codeword → ~300M raw hits).
- All predictions and construction diagnostics: w32_mult.json.
