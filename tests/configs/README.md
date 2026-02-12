# test config overlays

last updated: 2026-02-12

TOML overlay files for E2E API tests. Each file contains **only the fields that differ** from `config.toml.example`. The test framework deep-merges them.

## how it works

```
config.toml.example (base)
    + tests/configs/<overlay>.toml (test overrides)
    + config.toml (real model paths only)
    = merged TOML -> Config.from_toml() -> RuntimeConfig
```

Model paths come from the real `config.toml` so tests use actual model locations without hardcoding them.

## available overlays

| File | Pipeline | Resolution | Steps | Purpose |
|------|----------|------------|-------|---------|
| `flux2_smoke.toml` | FLUX.2 | 256x256 | 2 | Fastest validation |
| `flux2_standard.toml` | FLUX.2 | 512x512 | 4 | Quality validation |
| `ltx2_smoke.toml` | LTX-2 | 256x384, 9 frames | 4 | Fastest video |
| `ltx2_standard.toml` | LTX-2 | 512x768, 33 frames | 12 | Quality video |
| `zimage_smoke.toml` | Z-Image | 256x256 | 9 (turbo) | Fastest Z-Image |

## adding a new overlay

1. Create `<pipeline>_<tier>.toml` in this directory
2. Include only fields that differ from `config.toml.example`
3. Always set `default_pipeline = "none"` (on-demand loading)
4. Always set `compile = false` (avoid warmup overhead in tests)
