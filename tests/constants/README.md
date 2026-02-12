last updated: 2026-02-12

# test constants module

Single source of truth for generation parameters used across all test infrastructure.

## architecture

```
tests/constants/<pipeline>.py  (canonical parameter definitions)
        |                    |
        v                    v
tests/configs/*.toml    tests/backends/protocol.py
(E2E API tests)         (integration/pipeline tests)
        |                    |
        v                    v
config_factory.py       Backend.generate_video()
(RuntimeConfig)         (GenerationConfig)
```

TOML overlays encode the same values as static TOML (for config factory parsing).
Protocol.py constructs GenerationConfig objects from the constants.
The validation test (`tests/unit/test_config_consistency.py`) catches any drift.

## pipeline modules

| Module | Reference Source | Tiers |
|--------|-----------------|-------|
| `ltx2.py` | `coderef/LTX-2/.../constants.py` | SMOKE, STANDARD (distilled) + FULL_SMOKE, FULL_REFERENCE (full model) |
| `flux2.py` | Our validated configs | SMOKE, STANDARD, REFERENCE |
| `zimage.py` | Our validated configs | SMOKE, STANDARD_TURBO, STANDARD_BASE, REFERENCE_BASE |

## adding a new pipeline

1. Create `tests/constants/<pipeline>.py` with reference values and tier dicts
2. Add TOML overlays in `tests/configs/<pipeline>_<tier>.toml`
3. Import in `tests/constants/__init__.py`
4. Add consistency assertions in `tests/unit/test_config_consistency.py`
5. Run: `uv run pytest tests/unit/test_config_consistency.py -v`

## two tier families (LTX-2)

LTX-2 has two families because distilled and full models use different step counts:

- **Distilled** (SMOKE, STANDARD): 4-12 steps, used by TOML overlays / E2E API tests
- **Full model** (FULL_SMOKE, FULL_REFERENCE): 30-40 steps, used by protocol.py / backend comparison

Both share resolution, CFG, seed, and FPS from the reference repo.
