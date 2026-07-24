# GPU runbook

Two historical pod runbooks, merged 2026-07-24. Campaign
drivers now live in `proximity_explorations/experiments/`
(campaigns are local; this directory ships only
library-grade kernels and engines).

---

## Part I — bad-set / norms campaigns


## Pod

- GPU: one A100 80GB (or 4090 for w <= 10; ~3x slower, ~5x cheaper).
- Template: RunPod PyTorch 2.x / CUDA 12.x image.
- Disk: 50 GB is plenty (outputs are MBs — norms are heavily degenerate).

## Setup (on the pod)

```bash
pip install cupy-cuda12x maturin
git clone https://github.com/Praz314159/proximity.git vanish && cd vanish
maturin build --release --features python && pip install target/wheels/vanish-*.whl
cd .. && mkdir campaign && cd campaign
# copy norms_gpu.py here (scp/rsync from local gpu/)
```

## Run order (gates are mandatory)

1. **Correctness gate** — must PASS before anything else is believed:
   ```bash
   python norms_gpu.py --validate
   ```
   Reproduces the exhaustively verified s=32 w<=6 anticorrelation profile
   on the GPU path (both vs hardcoded pins and vs the CPU Rust module).

2. **Campaign, weight-staged** (each stage resumable/independent):
   ```bash
   python norms_gpu.py --s 64 --w 8  --out norms_s64_w8.json    # ~minutes
   python norms_gpu.py --s 64 --w 10 --out norms_s64_w10.json   # ~1-2 h
   ```
   Multi-GPU pod (e.g. 4x A100): one process per GPU, sharded by support —
   same total cost, 4x the wall clock. Shards partition the work exactly;
   merge by summing counts per norm (each vector lands in exactly one shard):
   ```bash
   for i in 0 1 2 3; do
     CUDA_VISIBLE_DEVICES=$i python norms_gpu.py --s 64 --w 10 \
       --shard $i --nshard 4 --out /workspace/w10_shard$i.json &
   done; wait
   python norms_gpu.py --s 64 --w 12 --out norms_s64_w12.json   # ~day; optional
   ```
   Watch the per-weight progress lines; uniques should stay << vector count
   (degeneracy is the expected signature; a uniques explosion means a bug).

3. **Bring results home** (scp the JSONs), then factor + bad-set locally:
   the unique norms are few enough for CPU `vanish.factor`; Galois
   normalization and p^2 fallback follow the same rules as vanish::norms
   (valuation / 32; census fallback where p^2 divides — flag those primes,
   s=64 census fallback needs the weight-capped direct census).

## Outputs feeding the research

- Complete s=64 bad set to w <= 10 (12): the worst-per-prime orbit census
  at the next scale — Pillar 1's key data point.
- N_max(w) profile at s=64: the anticorrelation law's s-growth.
- Cost estimate: A100 at ~$2.5/h — the whole campaign is O($10).

## Zoo-scaling campaign (2026-07-22, data-first program)

Goal: extend every coordinate-family series to s = 32 (601M subsets) and
sample the dense bulk — series for formula-hunting, not certificates.
Driver: `zoo_campaign.py` (numpy fallback lets `--validate` run anywhere).

Decode primes with mu_32: 2130706433 (KB), 77569, 65537, 97, 193, 257,
449, 577. The cloud census sweeps ALL 599 primes p = 1 mod 32 below 1e5
(the accident spectrum is a number-theoretic object: jump-density
statistics need the full spectrum, not a band) — spectroscopy-grade
coverage matching the CPU sweep at s <= 24.

```bash
python decode_gpu.py --validate          # decoder gate
python zoo_campaign.py --validate        # campaign gate (CPU-checkable)
python - <<'EOF' > primes32.txt
for p in range(33, 100000, 32):
    d = 2
    while d*d <= p and p % d: d += 1
    if d*d > p: print(p)
EOF
echo 2130706433 >> primes32.txt; echo 77569 >> primes32.txt
# single GPU (~30-120 s/prime => ~8-20 h), or shard on 4x A100:
#   split -n l/4 primes32.txt shard_ ; one loop per CUDA_VISIBLE_DEVICES
while read P; do
  python zoo_campaign.py --cloud --s 32 --p $P --out cloud_s32_$P.json
done < primes32.txt
for P in 2130706433 77569 65537 97 193 257 449 577; do
  python zoo_campaign.py --zoo --s 32 --p $P --depth 5 --out zoo_s32_$P.json
done                                      # q <= 5 = the full strip at s=32
python zoo_campaign.py --bulk --s 32 --p 65537 --n 20000 --out bulk_s32_65537.json
python zoo_campaign.py --bulk --s 32 --p 577 --n 20000 --out bulk_s32_577.json
# (bulk at KB is empty: E[cut] = C(32,16)/p < 1; use mid-size primes)
# native-dense odd-cell search (the SH/purification frontier):
python concentration.py --p 2130706433 --s 32 --cells "14:17,14:18,14:19,15:17"
```

Budget: cloud sweep (599 primes) ~8-20 h, zoo+depth ~3-4 h, bulk
~2-4 h, concentration ~4-8 h => ~1.5-2 A100-days single-GPU, or ~8-12 h
wall on 4x A100 (shard the cloud prime list across CUDA_VISIBLE_DEVICES;
dedicate one GPU to concentration while shards run). Bring JSONs home into
experiments/landscape/ (they extend census_zoo_scaling.json).

## Launch discipline (2026-07-22, learned the hard way)

1. **Warm the JIT solo, then shard.** Run one small decode (or
   `--validate`) in a single process before fanning out; three processes
   JIT-compiling the same RawKernel simultaneously can stall on the
   kernel cache. For shard fleets set a per-shard cache:
   `CUPY_CACHE_DIR=/workspace/.cupy_$i`.
2. **Always `python -u`, always per-item log lines with timings.** A
   buffered or silent worker is indistinguishable from a hung one.
3. **Wrap workers in `timeout` with one retry.** The failure mode is
   stall, not crash — a time ceiling per decode (e.g. 10x the expected
   cell time) converts hangs into visible retries.
4. **Gates must exercise the high-hit path.** The cp.unique(axis=0)
   stall survived every gate because random words produce zero hits and
   skip dedup; the composed fold-ladder word at (32,15,17) (~583k hits,
   ~5 s) is the canonical high-hit regression cell.
5. **`faulthandler` in every driver** (`import faulthandler;
   faulthandler.register(signal.SIGUSR1)`): `kill -USR1 <pid>` dumps
   Python stacks when py-spy is blocked by the container's ptrace
   policy.


---

## Part II — decode experiments


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
