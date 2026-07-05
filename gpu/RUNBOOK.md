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
