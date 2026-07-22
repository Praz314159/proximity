# RunPod runbook — s=64 bad-set campaign

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
449, 577. Cloud census additionally sweeps every p = 1 mod 32 below 3000
(~25 more) — minutes each, and the large-prime accident discovery at
s = 22/24 says thin large-p coverage cannot identify structural values.

```bash
python decode_gpu.py --validate          # decoder gate
python zoo_campaign.py --validate        # campaign gate (CPU-checkable)
for P in 2130706433 77569 65537 97 193 257 353 449 577 641 769 929 1153          1217 1249 1409 1601 1697 2113 2273 2593 2657 2689 2753 3041; do
  python zoo_campaign.py --cloud --s 32 --p $P --out cloud_s32_$P.json
done                                      # ~minutes each on A100
for P in 2130706433 77569 65537 97 193 257 449 577; do
  python zoo_campaign.py --zoo --s 32 --p $P --depth 5 --out zoo_s32_$P.json
done                                      # q <= 5 = the full strip at s=32
python zoo_campaign.py --bulk --s 32 --p 65537 --n 20000 --out bulk_s32_65537.json
python zoo_campaign.py --bulk --s 32 --p 577 --n 20000 --out bulk_s32_577.json
# (bulk at KB is empty: E[cut] = C(32,16)/p < 1; use mid-size primes)
# native-dense odd-cell search (the SH/purification frontier):
python concentration.py --p 2130706433 --s 32 --cells "14:17,14:18,14:19,15:17"
```

Budget: cloud sweep ~2-3 h, zoo+depth ~3-4 h, bulk ~2-4 h, concentration
~4-8 h => ~1.5 A100-days single-GPU, or ~6 h wall on 4x A100 (shard the
cloud/zoo prime loops across CUDA_VISIBLE_DEVICES; dedicate one GPU to
concentration). Bring JSONs home into
experiments/landscape/ (they extend census_zoo_scaling.json).
