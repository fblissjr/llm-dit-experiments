# documentation checklist

*last updated: 2025-12-22*

## documentation structure

Documentation is organized in tiers for efficient Claude agent consumption:

### tier 1: CLAUDE.md (~300 lines)
Read at every cold start. Contains:
- Critical rules (no emojis, use uv, etc.)
- Architecture overview
- Key parameters (tables only)
- Directory structure
- Documentation map (pointers to detailed docs)

**Keep CLAUDE.md lean.** If adding content, consider if it belongs in a reference doc or guide instead.

### tier 2: docs/reference/
Specs and facts - load when working on specific areas:

| File | Content |
|------|---------|
| `cli_flags.md` | All CLI flags by category |
| `api_endpoints.md` | REST API reference |
| `configuration.md` | DRY config principles, TOML wiring |
| `resolution.md` | VAE constraints, presets |
| `dype.md` | High-resolution generation |
| `long_prompts.md` | 1504 token limit, compression |
| `chat_templates.md` | Qwen3/VL template formats |
| `documentation_checklist.md` | This file |

### tier 3: docs/guides/
How-to docs - load when working on features:

| File | Content |
|------|---------|
| `vl_conditioning.md` | Vision conditioning (single source of truth) |
| `prompt_rewriting.md` | Qwen3 prompt expansion |
| `lora.md` | LoRA loading and fusion |
| `distributed.md` | Mac encode, CUDA generate |
| `profiler.md` | Performance testing |

### tier 4: internal/
Research and development - load for context on past decisions:

| Path | Content |
|------|---------|
| `index.md` | Map of all internal docs |
| `SESSION_CONTINUITY.md` | Current state, blockers, next steps |
| `GUIDING_PRINCIPLES.md` | Architectural north star |
| `research/` | Hypotheses, experiments, findings |
| `log/` | Session logs (log_YYYY-MM-DD.md) |

### where to put new content

| Content Type | Location |
|--------------|----------|
| New CLI flag | `docs/reference/cli_flags.md` |
| New API endpoint | `docs/reference/api_endpoints.md` |
| New feature guide | `docs/guides/<feature>.md` |
| Research notes | `internal/research/<topic>.md` |
| Session work | `internal/log/log_YYYY-MM-DD.md` |
| Architecture change | CLAUDE.md (brief) + relevant docs |

### avoiding duplication

- Each topic should have ONE source of truth
- Use pointers (links) instead of copying content
- When consolidating, update all references to point to the canonical location
- VL conditioning example: consolidated from 4 places into `docs/guides/vl_conditioning.md`

---

## after implementing any feature or significant change

| File | When to Update |
|------|----------------|
| `CHANGELOG.md` | Every feature, fix, or breaking change (semantic versioning) |
| `README.md` | New user-facing features, quick start examples |
| `CLAUDE.md` | New directories, architecture changes |
| `pyproject.toml` | New dependencies, version bumps |
| `config.toml.example` | New configurable parameters (with comments) |
| `experiments/README.md` | Experiment-related features or tools |
| `internal/log/log_YYYY-MM-DD.md` | Every session (create dated file) |
| `docs/*.md` | Feature-specific documentation |

## checklist for new features

1. Add entry to `CHANGELOG.md` under appropriate version
2. Update `README.md` if user-facing
3. Update `CLAUDE.md` Directory Structure if new files/directories
4. Update relevant `docs/*.md` or `experiments/README.md`
5. Create/update `internal/log/log_YYYY-MM-DD.md` with session details
6. Update `internal/SESSION_CONTINUITY.md` with current state

## for new configurable parameters

Follow DRY Configuration Principles (see `docs/reference/configuration.md`):

1. Add to `config.toml.example` with descriptive comment
2. Add to Config dataclass in `src/llm_dit/config.py`
3. Add CLI argument in `src/llm_dit/cli.py` (`create_argument_parser()`)
4. Add to `RuntimeConfig` in `src/llm_dit/cli.py`
5. Wire in `load_runtime_config()` (TOML -> RuntimeConfig)
6. Wire in `src/llm_dit/startup.py` (RuntimeConfig -> Backend configs)
7. Expose in `web/server.py` and `web/index.html` if user-facing
8. Run DRY config test: `uv run pytest tests/unit/test_dry_config.py -v`
9. Update `docs/reference/cli_flags.md` with new flag

## end of session

1. Create/update `internal/log/log_YYYY-MM-DD.md` with session summary
2. Update `internal/SESSION_CONTINUITY.md` with:
   - Current focus
   - Recent decisions
   - Known blockers
   - Next steps
3. If significant changes: Update CHANGELOG.md
