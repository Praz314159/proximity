# gpu/ — accelerated views of the vanish kernels

Library-grade GPU code only; campaign drivers live in the (local)
`proximity_explorations/experiments/` tree and *call* these modules.

| module | role |
|---|---|
| `decode_gpu.py` | the decode kernels: templated register-resident list decoder (Barrett reduction, bitmask membership), pool counters, `validate()` gates. The brute-force oracle. |
| `core_residual_gpu.py` | the s=64 attack instrument: core enumeration + Koetter/Roth-Ruckenstein residual decodes, one thread per core. Reaches cells `decode_gpu` cannot (C(32,11) cores, not C(64,31) information sets). Gates: `--selfcheck` (CPU mirror vs the Rust decoder, any machine), `--validate` (kernel vs mirror, on the pod). |
| `cloud_engine.py` | the certified moment-cloud engine (issue #16): C1 builder + certificate verification, C2 cut counters, C3 strata, C6 value histograms; `light` mode with `verify_pins`. Never believe an engine number before its gate. |
| `norms_gpu.py` | GPU norm/census kernels for the bad-set campaigns. |
| `decode_ref.py` | slow reference implementations the kernels are A/B-checked against. |
| `RUNBOOK.md` | pod setup + gate-first run procedures. |

Discipline: every module exposes a gate (`--validate` / `--selfcheck`)
that must pass on the target machine before campaign output is trusted;
the gates check against `vanish`'s Rust authority (certificates, pins).
